#!/usr/bin/env python3
"""Generate puzzles.json with an expanded pool of words and prompt templates per lobe,
supporting both live procedural LLM generation from any seed and offline fallback."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Xenova" / "albert-base-v2"
PUZZLES_OUT = ROOT / "puzzles.json"
CORPUS_OUT = ROOT / "scripts" / "categories_corpus.json"

# 40 categories: (id, label, lobe, list_of_prompts, candidate_word_pool)
# Each category has 3 distinct prompt templates and 6-10 single-token candidate words.
# In LIVE LLM mode, the browser picks 4 mutually-compatible categories and a prompt template
# for each from the seed, runs 4 forward passes on the fly, and lets the LLM itself select
# the 4 words per lobe with the highest synaptic dominance!
RAW_CATEGORIES: list[tuple[str, str, str, list[str], list[str]]] = [
    (
        "ice_cream",
        "ICE CREAM FLAVORS",
        "GUSTATORY CORTEX // FROZEN SWEET",
        [
            "the best ice cream flavor is [MASK].",
            "she ordered a double scoop of [MASK] ice cream.",
            "at the parlor, the sweetest ice cream flavor is [MASK].",
        ],
        ["vanilla", "mint", "chocolate", "strawberry", "caramel", "raspberry", "coconut", "fudge"],
    ),
    (
        "capitals",
        "EUROPEAN CITIES",
        "HIPPOCAMPUS // GEO-COORDINATES",
        [
            "in europe, a major capital city is [MASK].",
            "the international flight landed in the european city of [MASK].",
            "across europe, travelers visited the historic city of [MASK].",
        ],
        [
            "madrid",
            "paris",
            "rome",
            "berlin",
            "london",
            "vienna",
            "athens",
            "warsaw",
            "prague",
            "brussels",
            "dublin",
            "lisbon",
        ],
    ),
    (
        "breakfast",
        "COOKED AT BREAKFAST",
        "ENTERIC NERVE // SKILLET GREASE",
        [
            "for breakfast i cooked eggs, toast, and [MASK].",
            "the morning breakfast plate came with eggs and [MASK].",
            "at the diner, she ordered breakfast eggs with [MASK].",
        ],
        ["bacon", "cheese", "butter", "sausage", "potatoes", "ham", "beans", "toast"],
    ),
    (
        "instruments",
        "CONCERT INSTRUMENTS",
        "AUDITORY CORTEX // HARMONIC",
        [
            "in the orchestra, the musician played the [MASK].",
            "on the concert stage, the soloist performed on the [MASK].",
            "at rehearsal, the virtuoso tuned the [MASK].",
        ],
        ["piano", "violin", "clarinet", "flute", "cello", "oboe", "trumpet", "harp", "trombone", "guitar", "saxophone"],
    ),
    (
        "colors",
        "PAINT PIGMENTS",
        "OCCIPITAL LOBE // CHROMA-V1",
        [
            "the painter mixed a bright shade of [MASK] paint.",
            "across the canvas, the artist brushed vivid [MASK] pigment.",
            "the neon sign glowed with a brilliant [MASK] hue.",
        ],
        ["purple", "yellow", "green", "red", "blue", "orange", "pink", "crimson", "scarlet", "indigo", "turquoise"],
    ),
    (
        "planets",
        "SOLAR SYSTEM",
        "PARIETAL LOBE // ORBITAL",
        [
            "in our solar system, the planet [MASK] orbits the sun.",
            "through the telescope, the astronomer observed the planet [MASK].",
            "the deep space probe entered orbit around the planet [MASK].",
        ],
        ["mars", "jupiter", "neptune", "venus", "saturn", "mercury", "pluto"],
    ),
    (
        "metals",
        "FORGED METALS",
        "METALLURGIC SYNAPSE // ORE",
        [
            "the heavy blade was forged from solid [MASK].",
            "inside the foundry, molten [MASK] poured into the mold.",
            "the armor plating was hammered from thick [MASK].",
        ],
        ["steel", "copper", "brass", "iron", "bronze", "titanium", "aluminum", "nickel"],
    ),
    (
        "weather",
        "STORMY WEATHER",
        "BAROMETRIC GANGLIA // TEMPEST",
        [
            "the weather forecast warned of heavy [MASK] tonight.",
            "across the coastline swept a violent [MASK] storm.",
            "the ship battled through blinding [MASK] and gales.",
        ],
        ["rain", "snow", "wind", "fog", "hail", "thunder", "lightning", "blizzard", "frost"],
    ),
    (
        "fruits",
        "ORCHARD FRUITS",
        "FRUCTOSE RECEPTOR // HARVEST",
        [
            "from the fruit basket she picked a ripe [MASK].",
            "in the summer orchard hung a juicy ripe [MASK].",
            "he sliced a fresh sweet [MASK] into the fruit bowl.",
        ],
        ["pear", "mango", "peach", "apple", "banana", "grape", "cherry", "melon"],
    ),
    (
        "farm_animals",
        "BARNYARD LIVESTOCK",
        "FAUNA CORTEX // LIVESTOCK",
        [
            "in the barn, the farmer fed the [MASK].",
            "across the green pasture grazed a herd of [MASK].",
            "at dawn on the ranch, the farmhands tended the [MASK].",
        ],
        ["sheep", "goats", "horses", "pigs", "cattle", "calves", "ducks", "geese"],
    ),
    (
        "gems",
        "CROWN TREASURE",
        "CRYSTALLINE CORTEX // LUX",
        [
            "the ancient treasure chest glittered with [MASK] and diamonds.",
            "the royal necklace was inlaid with precious [MASK] and pearls.",
            "inside the vault gleamed rare [MASK] and rubies.",
        ],
        ["gold", "silver", "jade", "amber", "ivory", "coral", "crystal", "emerald"],
    ),
    (
        "drinks",
        "TAVERN SPIRITS",
        "DOPAMINE RECEPTOR // ETHANOL",
        [
            "at the tavern, the bartender poured a glass of cold [MASK].",
            "in the pub, the sailors toasted with mugs of [MASK].",
            "from the oak barrel, the innkeeper tapped fresh [MASK].",
        ],
        ["beer", "wine", "vodka", "whiskey", "rum", "brandy", "gin", "ale", "champagne"],
    ),
    (
        "hot_brews",
        "MORNING CUPS",
        "ADENOSINE BLOCK // CAFFEINE",
        [
            "in the morning, she sipped a hot cup of [MASK].",
            "to warm his hands, he held a steaming mug of [MASK].",
            "the cafe served freshly brewed hot [MASK].",
        ],
        ["coffee", "tea", "cocoa", "cider", "milk", "broth"],
    ),
    (
        "weapons",
        "MELEE WEAPONS",
        "AMYGDALA // LETHAL COMBAT",
        [
            "the warrior drew his sharp [MASK] in battle.",
            "in the arena, the gladiator swung his deadly [MASK].",
            "the knight unsheathed his bloodstained [MASK].",
        ],
        ["sword", "axe", "dagger", "spear", "blade", "mace", "lance", "halberd"],
    ),
    (
        "organs",
        "VITAL ORGANS",
        "AUTONOMIC VISCERA // VIVISECTION",
        [
            "the surgeon cut open the torso to transplant the [MASK].",
            "during the autopsy, the coroner examined the ruptured [MASK].",
            "on the operating table, the scalpel exposed the bleeding [MASK].",
        ],
        ["lungs", "kidney", "liver", "heart", "stomach", "spleen", "bladder", "pancreas"],
    ),
    (
        "trees",
        "FOREST TREES",
        "DENDRITIC ARBOR // CANOPY",
        [
            "in the ancient forest grew tall [MASK] trees.",
            "the logging camp harvested timber from giant [MASK] trees.",
            "wind rustled through the branches of the old [MASK] tree.",
        ],
        ["pine", "oak", "cedar", "willow", "maple", "birch", "cypress", "elm", "redwood"],
    ),
    (
        "sports",
        "VARSITY SPORTS",
        "MOTOR CORTEX // STADIUM",
        [
            "in high school, the varsity team played [MASK].",
            "at the stadium, the crowd cheered the championship [MASK] match.",
            "the college athlete won a scholarship for playing [MASK].",
        ],
        ["basketball", "baseball", "football", "soccer", "volleyball", "tennis", "hockey", "rugby", "golf"],
    ),
    (
        "clothes",
        "WINTER GEAR",
        "THERMAL CORTEX // INSULATION",
        [
            "before stepping into the blizzard, he put on his warm [MASK].",
            "to survive the freezing expedition, she packed a thick [MASK].",
            "bundled against the frost, the hiker wore a heavy [MASK].",
        ],
        ["jacket", "gloves", "boots", "sweater", "coat", "scarf", "cloak", "hoodie"],
    ),
    (
        "months",
        "CALENDAR MONTHS",
        "PINEAL GLAND // ANNUAL CYCLE",
        [
            "the festival is held every year in the month of [MASK].",
            "the calendar marked the deadline in the month of [MASK].",
            "the annual harvest begins each year in [MASK].",
        ],
        ["october", "june", "december", "april", "july", "august", "january", "november", "september", "february"],
    ),
    (
        "days",
        "DAYS OF THE WEEK",
        "SUPRACHIASMATIC NUCLEUS // WEEK",
        [
            "the office meeting was scheduled for next [MASK] morning.",
            "the weekly broadcast airs every [MASK] night.",
            "the courthouse docket is heard on [MASK] afternoon.",
        ],
        ["tuesday", "monday", "thursday", "friday", "wednesday", "saturday", "sunday"],
    ),
    (
        "emotions",
        "DARK EMOTIONS",
        "LIMBIC SYSTEM // TRAUMA",
        [
            "his mind was consumed by overwhelming [MASK] and grief.",
            "in the interrogation room, her voice shook with [MASK] and dread.",
            "the trauma left a deep scar of [MASK] and anguish.",
        ],
        ["anger", "fear", "guilt", "sorrow", "shame", "rage", "despair", "envy", "hatred"],
    ),
    (
        "furniture",
        "PARLOR SEATING",
        "PROPRIOCEPTIVE LOBE // POSTURE",
        [
            "in the living room, he sat down on the leather [MASK].",
            "beside the fireplace stood an upholstered [MASK].",
            "the parlor was furnished with a wooden [MASK] and table.",
        ],
        ["couch", "sofa", "bench", "chair", "stool", "recliner", "armchair", "mattress"],
    ),
    (
        "vehicles",
        "HIGHWAY VEHICLES",
        "LOCOMOTOR PLEXUS // ASPHALT",
        [
            "in heavy traffic, the driver parked his [MASK] on the street.",
            "down the highway sped a roaring [MASK].",
            "outside the garage stood a customized [MASK].",
        ],
        ["car", "truck", "jeep", "motorcycle", "van", "bus", "tractor", "ambulance", "limousine"],
    ),
    (
        "birds",
        "BIRDS OF PREY",
        "AVIAN OPTIC TECTUM // RAPTOR",
        [
            "high above the canyon soared a wild [MASK].",
            "from the cliffside nest swooped a hunting [MASK].",
            "across the mountain ridge glided a feathered [MASK].",
        ],
        ["eagle", "owl", "falcon", "hawk", "vulture", "raven", "crow", "condor"],
    ),
    (
        "sea_life",
        "OCEAN GIANTS",
        "PELAGIC GANGLIA // ABYSSAL",
        [
            "deep beneath the ocean waves swam a giant [MASK].",
            "the submarine sonar detected a massive marine [MASK].",
            "near the coral reef glided a wild [MASK].",
        ],
        ["octopus", "whale", "turtle", "shark", "dolphin", "squid", "seal", "lobster"],
    ),
    (
        "bugs",
        "CREEPING INSECTS",
        "ARTHROPOD NERVE // CHITIN",
        [
            "under the rotting log crawled a tiny [MASK] and a spider.",
            "in the damp soil burrowed a wriggling [MASK].",
            "the entomologist pinned a specimen of a [MASK] to the board.",
        ],
        ["worm", "moth", "beetle", "roach", "cricket", "locust", "mosquito", "hornet", "wasp"],
    ),
    (
        "royalty",
        "ROYAL COURT",
        "PREFRONTAL HIERARCHY // THRONE",
        [
            "on the royal throne sat the crowned [MASK] of the realm.",
            "the heralds bowed before the reigning [MASK].",
            "inside the castle reigned the royal [MASK].",
        ],
        ["king", "queen", "prince", "emperor", "monarch", "princess", "duke", "sultan"],
    ),
    (
        "family",
        "BLOOD RELATIVES",
        "OXYTOCIN CIRCUIT // KINSHIP",
        [
            "at the family reunion, she hugged her [MASK].",
            "around the holiday table sat her [MASK] and cousins.",
            "in the family portrait stood his [MASK].",
        ],
        ["brother", "sister", "father", "mother", "uncle", "aunt", "grandmother", "grandfather", "nephew", "niece"],
    ),
    (
        "rooms",
        "HOUSEHOLD ROOMS",
        "ENTORHINAL GRID // FLOORPLAN",
        [
            "she walked down the stairs into the [MASK].",
            "the blueprint showed a spacious [MASK] on the first floor.",
            "he locked the door to the upstairs [MASK].",
        ],
        ["kitchen", "bathroom", "bedroom", "cellar", "attic", "hallway", "garage", "parlor", "basement"],
    ),
    (
        "sciences",
        "ACADEMIC SCIENCES",
        "NEOCORTEX // EMPIRICAL LOGIC",
        [
            "at the university, the professor taught a course in [MASK].",
            "the doctoral candidate published a thesis in [MASK].",
            "the nobel prize was awarded for research in [MASK].",
        ],
        [
            "physics",
            "mathematics",
            "psychology",
            "biology",
            "chemistry",
            "astronomy",
            "geology",
            "genetics",
            "sociology",
        ],
    ),
    (
        "spices",
        "CULINARY SPICES",
        "OLFACTORY BULB // SEASONING",
        [
            "the chef seasoned the soup with salt, pepper, and [MASK].",
            "in the mortar, the cook ground fresh [MASK] and herbs.",
            "the recipe called for a pinch of dried [MASK].",
        ],
        ["garlic", "cinnamon", "basil", "ginger", "thyme", "oregano", "parsley", "rosemary", "mustard"],
    ),
    (
        "fabrics",
        "WOVEN TEXTILES",
        "SOMATOSENSORY CORTEX // THREAD",
        [
            "the dress was tailored from fine woven [MASK].",
            "the merchant unrolled a bolt ofdyed [MASK] fabric.",
            "the royal curtains were draped in heavy [MASK].",
        ],
        ["linen", "wool", "silk", "cotton", "satin", "velvet", "lace", "denim", "canvas", "cashmere"],
    ),
    (
        "bones",
        "HUMAN SKELETON",
        "OSTEOLOGICAL MATRIX // MARROW",
        [
            "the autopsy revealed a brutal fracture in the [MASK].",
            "in the catacombs lay a bleached human [MASK].",
            "the orthopedic surgeon bolted a titanium plate to the [MASK].",
        ],
        ["skull", "pelvis", "spine", "jaw", "femur", "rib", "collarbone", "ankle", "wrist"],
    ),
    (
        "crimes",
        "FELONY CHARGES",
        "FORENSIC ENGRAM // INDICTMENT",
        [
            "the prosecutor charged the syndicate boss with [MASK].",
            "the grand jury indicted the conspirators for [MASK].",
            "the fugitive was sentenced to prison for [MASK].",
        ],
        ["robbery", "murder", "treason", "theft", "extortion", "bribery", "arson", "kidnapping", "fraud", "forgery"],
    ),
    (
        "monsters",
        "CRYPT HORRORS",
        "NIGHTMARE GANGLIA // UNDEAD",
        [
            "in the cursed graveyard lurked a bloodthirsty [MASK].",
            "from the gothic crypt crawled aflesh-eating [MASK].",
            "the villagers fled in terror from the undead [MASK].",
        ],
        ["demon", "ghost", "zombie", "vampire", "werewolf", "ghoul", "banshee", "skeleton", "witch"],
    ),
    (
        "genres",
        "FICTION GENRES",
        "WERNICKE'S AREA // STORYBOOK",
        [
            "the library shelf was packed with [MASK] novels.",
            "the bestselling author wrote pulp [MASK] stories.",
            "the magazine published serialized [MASK] fiction.",
        ],
        ["romance", "horror", "mystery", "fantasy", "thriller", "detective", "adventure", "crime"],
    ),
    (
        "elements",
        "CHEMICAL ELEMENTS",
        "MOLECULAR SYNAPSE // VALENCE",
        [
            "in the laboratory, the chemist analyzed a sample of pure [MASK].",
            "the periodic table lists the atomic weight of [MASK].",
            "inside the pressurized tank was compressed [MASK].",
        ],
        ["sodium", "calcium", "oxygen", "nitrogen", "carbon", "hydrogen", "sulfur", "potassium", "chlorine"],
    ),
    (
        "dances",
        "STAGE DANCES",
        "CEREBELLUM // CHOREOGRAPHY",
        [
            "across the stage floor, the performers danced a [MASK].",
            "the orchestra struck up the rhythm of a lively [MASK].",
            "at the masquerade ball, the couples twirled in a [MASK].",
        ],
        [
            "waltz",
            "jig",
            "tango",
            "ballet",
            "polka",
            "samba",
            "rumba",
            "mambo",
            "disco",
            "salsa",
            "swing",
            "tap",
            "reel",
        ],
    ),
    (
        "flowers",
        "GREENHOUSE FLOWERS",
        "BOTANICAL CORTEX // PETAL",
        [
            "in the greenhouse bloomed a fragrant [MASK] flower.",
            "the botanist sketched the petals of a wild [MASK].",
            "in the spring garden blossomed a delicate [MASK].",
        ],
        ["orchid", "tulip", "lotus", "lily", "rose", "violet", "daisy", "poppy", "iris"],
    ),
    (
        "chants",
        "SACRED CHANTS",
        "TEMPORAL GYRUS // RITUAL",
        [
            "in the ancient temple, the monks chanted a sacred [MASK].",
            "beneath the cathedral dome echoed a solemn [MASK].",
            "the high priest recited an ancient [MASK].",
        ],
        ["hymn", "chant", "psalm", "mantra", "prayer", "blessing", "proverb", "ballad"],
    ),
    (
        "tools",
        "WORKSHOP TOOLS",
        "PARIETAL CORTEX // HARDWARE",
        [
            "the carpenter reached into the toolbox for a [MASK].",
            "in the workshop, the mechanic tightened the bolt with a [MASK].",
            "on the workbench lay a heavy steel [MASK] and pliers.",
        ],
        ["hammer", "wrench", "screwdriver", "pliers", "chisel", "saw", "drill", "clamp", "mallet", "shovel"],
    ),
    (
        "kitchenware",
        "COOKING VESSELS",
        "INSULAR CORTEX // CAST-IRON",
        [
            "on the stove simmered a heavy cast-iron [MASK].",
            "the chef poured boiling stock into the copper [MASK].",
            "in the kitchen sink soaked a greasy [MASK] and ladle.",
        ],
        ["skillet", "kettle", "pot", "pan", "cauldron", "wok", "teapot", "pitcher", "bowl", "platter"],
    ),
    (
        "reptiles",
        "COLD-BLOODED SCALES",
        "REPTILIAN BRAINSTEM // VENOM",
        [
            "basking on the sunny swamp rock was a scaly [MASK].",
            "through the marsh grass slithered a venomous [MASK].",
            "inside the glass terrarium coiled a spotted [MASK].",
        ],
        ["snake", "lizard", "crocodile", "alligator", "cobra", "viper", "python", "gecko", "iguana", "toad", "frog"],
    ),
    (
        "predators",
        "APEX PREDATORS",
        "PREDATORY STRIATUM // FANG",
        [
            "through the dark wilderness prowled a hungry [MASK].",
            "at midnight the pack of [MASK] stalked their prey.",
            "in the jungle shadows crouched a snarling [MASK].",
        ],
        ["wolf", "bear", "fox", "tiger", "lion", "leopard", "panther", "coyote", "jaguar", "cougar", "hyena"],
    ),
    (
        "structures",
        "STONE CITADELS",
        "PARAHIPPOCAMPAL // MONOLITH",
        [
            "atop the rocky crag stood a medieval stone [MASK].",
            "the pilgrims traveled across the valley to the ancient [MASK].",
            "behind thick granite walls rose the fortified [MASK].",
        ],
        ["castle", "temple", "palace", "cathedral", "chapel", "fortress", "monastery", "shrine", "tower", "citadel"],
    ),
    (
        "currency",
        "HARD CURRENCY",
        "NUCLEUS ACCUMBENS // MINT",
        [
            "the foreign exchange counter traded gold and [MASK] bills.",
            "the teller counted out a stack of crisp [MASK] notes.",
            "the merchant accepted payment in silver [MASK].",
        ],
        ["dollar", "pound", "franc", "peso", "yen", "coin", "cent", "penny", "dime", "euro"],
    ),
    (
        "medical",
        "SURGICAL TRAY",
        "MEDULLA OBLONGATA // STERILE",
        [
            "the trauma nurse handed the surgeon a sterile [MASK].",
            "on the stainless surgical tray gleamed a bloody [MASK].",
            "the paramedic tore open a packet of [MASK] for the wound.",
        ],
        [
            "scalpel",
            "syringe",
            "bandage",
            "needle",
            "gauze",
            "catheter",
            "splint",
            "forceps",
            "swab",
            "mask",
            "pill",
            "vaccine",
        ],
    ),
    (
        "desserts",
        "BAKERY PASTRIES",
        "SEROTONIN LOOP // CONFECTION",
        [
            "from the bakery display case she chose a frosted [MASK].",
            "for dessert the baker served warm [MASK] with cream.",
            "on the porcelain plate sat a freshly baked [MASK].",
        ],
        ["cake", "pie", "cookie", "tart", "pastry", "muffin", "brownie", "donut", "biscuit", "pudding"],
    ),
]

SURGICAL_CASES = [
    (
        "CASE #001 // PRE-FRONTAL LOBOTOMY",
        "PATIENT: ALBERT-V2 (11.8M PARAMS) — SYMPTOM: MELTED FLAVOR & GEOGRAPHY SYNAPSES",
    ),
    ("CASE #002 // TREPANATION OVERDRIVE", "PATIENT: ALBERT-V2 — SYMPTOM: ACUTE SENSORY & MOTOR CROSS-TALK"),
    ("CASE #003 // CORPUS CALLOSUM SEVERANCE", "PATIENT: ALBERT-V2 — SYMPTOM: TEMPORAL & AUTONOMIC HEMORRHAGE"),
    ("CASE #004 // AMYGDALA EXCISION", "PATIENT: ALBERT-V2 — SYMPTOM: LETHAL TRAUMA & CRYPT HALLUCINATIONS"),
    ("CASE #005 // VENTRICULAR DRAINAGE", "PATIENT: ALBERT-V2 — SYMPTOM: TOXIC DOPAMINE & CAFFEINE SPILLAGE"),
    ("CASE #006 // CORTICAL DECOMPRESSION", "PATIENT: ALBERT-V2 — SYMPTOM: SEVERE FAUNA & PELAGIC DELUSIONS"),
    ("CASE #007 // SYNAPTIC CAUTERIZATION", "PATIENT: ALBERT-V2 — SYMPTOM: ROYALTY & ACADEMIC MEGALOMANIA"),
    ("CASE #008 // DEEP THALAMIC PROBE", "PATIENT: ALBERT-V2 — SYMPTOM: SKELETAL & CRIMINAL OSSIFICATION"),
    ("CASE #009 // SUBDURAL EVACUATION", "PATIENT: ALBERT-V2 — SYMPTOM: BOTANICAL & OLFACTORY SEIZURE"),
    ("CASE #010 // PINEAL GLAND EXTRACTION", "PATIENT: ALBERT-V2 — SYMPTOM: RITUALISTIC & HARMONIC PSYCHOSIS"),
]


def generate() -> None:
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model_fp16.onnx"))
    mask_id = tok.token_to_id("[MASK]")

    # Verify global uniqueness of pool words across categories and single-token status
    seen_words: dict[str, str] = {}
    categories_out = []

    for cid, label, lobe, prompts, pool in RAW_CATEGORIES:
        valid_words = []
        for w in pool:
            if w in seen_words:
                raise ValueError(f"Duplicate word {w!r} in {cid} and {seen_words[w]}")
            ids = tok.encode(w, add_special_tokens=False).ids
            if len(ids) != 1:
                continue  # skip multi-token words cleanly
            seen_words[w] = cid
            valid_words.append(
                {
                    "word": w,
                    "tokenId": int(ids[0]),
                    "tokenStr": tok.id_to_token(ids[0]),
                }
            )
        if len(valid_words) < 5:
            raise ValueError(f"Category {cid} has fewer than 5 single-token words: {len(valid_words)}")

        # Score primary prompt (prompts[0]) on all valid_words to sort them by natural model affinity
        enc = tok.encode(prompts[0])
        ids = np.array([enc.ids], dtype=np.int64)
        out = sess.run(
            None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        )[0]
        logits = out[0, enc.ids.index(mask_id)].astype(np.float32)
        for we in valid_words:
            we["ownLogit"] = round(float(logits[we["tokenId"]]), 2)
        valid_words.sort(key=lambda x: -x["ownLogit"])

        categories_out.append(
            {
                "id": cid,
                "label": label,
                "lobe": lobe,
                "prompt": prompts[0],
                "prompts": prompts,
                "words": valid_words[:4],  # Canonical top-4 for static fallback deck
                "wordPool": valid_words,  # Full pool (6-12 words) for live procedural LLM synthesis!
            }
        )

    # Compute logits of every category's primary prompt on ALL pool words across all categories
    n_cats = len(categories_out)
    all_pool_entries = [(ci, we["word"], we["tokenId"]) for ci, c in enumerate(categories_out) for we in c["wordPool"]]
    pool_word_to_idx = {w: idx for idx, (_, w, _) in enumerate(all_pool_entries)}
    pool_tids = [tid for _, _, tid in all_pool_entries]

    full_pool_logits = np.zeros((n_cats, len(all_pool_entries)), dtype=np.float32)
    all_tids = [we["tokenId"] for c in categories_out for we in c["words"]]
    logit_matrix = np.zeros((n_cats, n_cats * 4), dtype=np.float32)

    for ci, c in enumerate(categories_out):
        enc = tok.encode(c["prompt"])
        ids = np.array([enc.ids], dtype=np.int64)
        out = sess.run(
            None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        )[0]
        logits = out[0, enc.ids.index(mask_id)].astype(np.float32)
        for wi, tid in enumerate(all_tids):
            logit_matrix[ci, wi] = float(logits[tid])
        for pwi, tid in enumerate(pool_tids):
            full_pool_logits[ci, pwi] = float(logits[tid])

    margin = 1.4
    compat = np.zeros((n_cats, n_cats), dtype=bool)
    pair_margin_matrix = np.zeros((n_cats, n_cats), dtype=np.float32)
    for i in range(n_cats):
        own_i = logit_matrix[i, i * 4 : (i + 1) * 4]
        min_i = float(np.min(own_i))
        for j in range(n_cats):
            if i == j:
                continue
            other_on_i = logit_matrix[i, j * 4 : (j + 1) * 4]
            own_j = logit_matrix[j, j * 4 : (j + 1) * 4]
            other_on_j = logit_matrix[j, i * 4 : (i + 1) * 4]
            gap_ij = min(min_i - float(np.max(other_on_i)), float(np.min(own_j)) - float(np.max(other_on_j)))
            pair_margin_matrix[i, j] = gap_ij
            if gap_ij > margin:
                compat[i, j] = True

    def is_valid_quad(quad: tuple[int, int, int, int]) -> bool:
        return all(bool(compat[a, b]) for a, b in itertools.combinations(quad, 2))

    def quad_min_margin(quad: tuple[int, int, int, int]) -> float:
        return min(float(pair_margin_matrix[a, b]) for a, b in itertools.combinations(quad, 2))

    p0_quad = (0, 1, 3, 5)
    assert is_valid_quad(p0_quad), "Puzzle 0 quad must be valid!"

    rng = random.Random(19960420)
    seen_quads = {p0_quad}
    all_quads_list = [p0_quad]
    usage_counts = {i: (1 if i in p0_quad else 0) for i in range(n_cats)}
    attempts = 0
    while len(all_quads_list) < 128 and attempts < 80000:
        attempts += 1
        weights = [1.0 / ((usage_counts[i] + 1) ** 2) for i in range(n_cats)]
        chosen = []
        pool = list(range(n_cats))
        while len(chosen) < 4:
            w_pool = [weights[i] for i in pool]
            pick = rng.choices(pool, weights=w_pool, k=1)[0]
            chosen.append(pick)
            pool.remove(pick)
        quad = tuple(sorted(chosen))
        if quad not in seen_quads and is_valid_quad(quad):
            seen_quads.add(quad)
            all_quads_list.append(quad)
            for c in quad:
                usage_counts[c] += 1

    # Keep Puzzle 0 as the opening case, and sort remaining 127 cases from widest margin (easiest/cleanest)
    # to tightest margin (most semantically entangled seed phrases!)
    rest_quads = sorted(all_quads_list[1:], key=lambda q: -quad_min_margin(q))
    all_quads_list = [p0_quad] + rest_quads

    # Build compatibility adjacency list & pairwise margin map for live procedural generation in browser
    # Sorted by pair margin ascending (most entangled/similar first, most orthogonal last)
    compat_map = {}
    pair_margins_out = {}
    for i in range(n_cats):
        cid_i = categories_out[i]["id"]
        compat_neighbors = [j for j in range(n_cats) if compat[i, j]]
        compat_neighbors.sort(key=lambda j: float(pair_margin_matrix[i, j]))
        compat_map[cid_i] = [categories_out[j]["id"] for j in compat_neighbors]
        pair_margins_out[cid_i] = {
            categories_out[j]["id"]: round(float(pair_margin_matrix[i, j]), 2) for j in compat_neighbors
        }

    pool_logits_out = {
        categories_out[ci]["id"]: [round(float(x), 1) for x in full_pool_logits[ci]] for ci in range(n_cats)
    }

    puzzles = []
    for pi, quad in enumerate(all_quads_list):
        case_title, case_note = SURGICAL_CASES[pi % len(SURGICAL_CASES)]
        case_num = f"CASE #{pi + 1:03d}"
        title = f"{case_num} // {case_title.split('//')[1].strip()}"
        word_indices = [c * 4 + k for c in quad for k in range(4)]
        p_logits = []
        for c in quad:
            row = [round(float(logit_matrix[c, wi]), 2) for wi in word_indices]
            p_logits.append(row)
        puzzles.append(
            {
                "id": pi + 1,
                "title": title,
                "symptom": case_note,
                "minMargin": round(quad_min_margin(quad), 2),
                "categories": [categories_out[c]["id"] for c in quad],
                "fallbackLogits": p_logits,
            }
        )

    total_pool_words = sum(len(c["wordPool"]) for c in categories_out)
    total_prompts = sum(len(c["prompts"]) for c in categories_out)

    payload = {
        "version": 2,
        "modelId": "Xenova/albert-base-v2",
        "dtype": "fp16",
        "maskToken": "[MASK]",
        "maskTokenId": int(mask_id),
        "categories": {c["id"]: c for c in categories_out},
        "compatibility": compat_map,
        "pairMargins": pair_margins_out,
        "poolWordIndex": pool_word_to_idx,
        "poolLogits": pool_logits_out,
        "puzzles": puzzles,
    }

    CORPUS_OUT.write_text(json.dumps(RAW_CATEGORIES, indent=2) + "\n", encoding="utf-8")
    PUZZLES_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(categories_out)} lobes ({total_prompts} prompts, {total_pool_words} pool tokens) "
        f"and {len(puzzles)} fallback cases to {PUZZLES_OUT}"
    )


if __name__ == "__main__":
    generate()
