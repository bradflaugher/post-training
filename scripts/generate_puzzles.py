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
        "locksmith",
        "LOCKSMITH HARDWARE",
        "PARIETAL TUMBLER // SHACKLE",
        [
            "to secure the heavy door, the locksmith fastened the iron [MASK].",
            "on the dungeon gate hung a rusted iron [MASK].",
            "the burglar picked the brass [MASK] on the vault door.",
        ],
        ["lock", "clasp", "bolt", "chain", "bar", "key", "hook"],
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
        ["steel", "copper", "brass", "iron", "bronze", "titanium", "aluminum", "nickel", "zinc", "lead", "tin"],
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
        ["sword", "axe", "dagger", "spear", "blade", "mace", "lance", "pike"],
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
        ["pine", "oak", "cedar", "willow", "maple", "birch", "cypress", "elm", "redwood", "ash"],
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
        "book",
        "BOOK PUBLISHING",
        "BROCA'S FOLIO // MANUSCRIPT",
        [
            "the editor proofread the [MASK] of the hardcover book before publication.",
            "the librarian opened the leather-bound book and read the [MASK].",
            "the printing press bound the [MASK] of the new novel.",
        ],
        ["edition", "title", "volume", "preface", "margin", "page", "spine", "chapter", "index", "prologue", "jacket"],
    ),
    (
        "chess",
        "CHESS BOARD",
        "PREFRONTAL GAMBIT // GRANDMASTER",
        [
            "on the chessboard, the grandmaster captured the [MASK].",
            "during the chess tournament, she sacrificed her [MASK] for checkmate.",
            "across the checkered board, the player advanced his [MASK].",
        ],
        ["knight", "rook", "pawn", "king", "castle", "queen", "bishop"],
    ),
    (
        "cards",
        "POKER TABLE",
        "STRIATAL WAGER // DECK",
        [
            "at the card table, the poker dealer revealed a [MASK].",
            "in the casino, the gambler drew a winning [MASK] from the pack.",
            "he shuffled the playing cards and dealt a [MASK].",
        ],
        ["joker", "spade", "suit", "trump", "club", "deck", "jack", "flush", "ace"],
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
        "dental",
        "DENTAL CLINIC",
        "TRIGEMINAL NERVE // ENAMEL",
        [
            "at the dental clinic, the dentist examined the patient's tooth [MASK].",
            "the orthodontist scraped the calcified [MASK] from the tooth.",
            "under the bright lamp, the dentist drilled into the decayed [MASK].",
        ],
        ["enamel", "cavity", "root", "gum", "decay", "canal", "plaque", "crown", "brace", "filling"],
    ),
    (
        "tree_parts",
        "TIMBER ANATOMY",
        "XYLEM CORTEX // CAMBIUM",
        [
            "the woodsman chopped through the thick [MASK] of the tree.",
            "in the forest, moss grew along the ancient tree's [MASK].",
            "the botanist peeled a sample of rough [MASK] from the trunk.",
        ],
        ["bark", "foliage", "branch", "canopy", "trunk", "stump", "sap", "twig", "timber"],
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
        ["eagle", "owl", "falcon", "hawk", "vulture", "raven", "crow", "condor", "heron"],
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
        ["octopus", "whale", "shark", "dolphin", "squid", "seal", "lobster", "salmon", "tuna"],
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
            "the heralds bowed before the reigning [MASK] of the realm.",
            "on the velvet throne sat the crowned [MASK].",
            "inside the royal court ruled the noble [MASK].",
        ],
        ["monarch", "emperor", "prince", "princess", "duke", "sultan", "baron", "sovereign"],
    ),
    (
        "ship",
        "SHIP ANATOMY",
        "VESTIBULAR KEEL // MARITIME",
        [
            "ocean waves crashed against the wooden [MASK] of the ship.",
            "the captain stood near the [MASK] of the sailing vessel.",
            "below the waterline, barnacles clung to the ship's [MASK].",
        ],
        ["hull", "keel", "cabin", "galley", "rudder", "helm", "mast", "stern", "bow", "anchor"],
    ),
    (
        "garment",
        "TAILORED GARMENTS",
        "TACTILE LOOM // STITCH",
        [
            "the tailor stitched the silk [MASK] onto the coat.",
            "she fastened the brass [MASK] of her tailored coat.",
            "the seamstress mended the torn [MASK] of the shirt.",
        ],
        ["zipper", "sleeve", "strap", "seam", "collar", "button", "hood", "buckle", "pocket", "fringe"],
    ),
    (
        "firearms",
        "FIREARM MECHANISM",
        "BALLISTIC REFLEX // ORDNANCE",
        [
            "the gunsmith cleaned the [MASK] of the pistol.",
            "on the firing range, the marksman checked the rifle's [MASK].",
            "smoke curled from the steel [MASK] of the revolver.",
        ],
        ["holster", "muzzle", "barrel", "recoil", "cartridge", "chamber", "trigger", "sight", "stock"],
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
        ["garlic", "cinnamon", "basil", "ginger", "thyme", "oregano", "parsley", "rosemary", "mustard", "sage"],
    ),
    (
        "fabrics",
        "WOVEN TEXTILES",
        "SOMATOSENSORY CORTEX // THREAD",
        [
            "the dress was tailored from fine woven [MASK].",
            "the merchant unrolled a bolt of dyed [MASK] fabric.",
            "the royal curtains were draped in heavy [MASK].",
        ],
        ["linen", "wool", "silk", "cotton", "satin", "velvet", "lace", "denim", "canvas", "tweed"],
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
        ["skull", "pelvis", "jaw", "femur", "rib", "collarbone", "sternum"],
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
            "from the gothic crypt crawled a flesh-eating [MASK].",
            "the villagers fled in terror from the undead [MASK].",
        ],
        ["demon", "ghost", "zombie", "vampire", "werewolf", "ghoul", "banshee", "witch"],
    ),
    (
        "geology",
        "TECTONIC GEOLOGY",
        "LITHOSPHERIC CORTEX // MAGMA",
        [
            "the volcano erupted through the tectonic [MASK] of the earth.",
            "seismographs detected a deep fracture in the earth's [MASK].",
            "geologists drilled core samples from the rocky [MASK].",
        ],
        ["crust", "ridge", "mantle", "fault", "core", "bedrock", "trench", "crater", "magma", "lava"],
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
        ["waltz", "jig", "tango", "ballet", "salsa", "swing", "tap", "reel"],
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
        ["orchid", "tulip", "lotus", "lily", "rose", "violet", "daisy", "poppy"],
    ),
    (
        "river",
        "RIVER HYDROLOGY",
        "FLUVIAL PLEXUS // ESTUARY",
        [
            "the canoe drifted down the muddy [MASK] of the river.",
            "after the spring thaw, floodwaters overflowed the river [MASK].",
            "fishermen cast their nets across the shallow river [MASK].",
        ],
        ["rapids", "bank", "mouth", "estuary", "fork", "stream", "basin", "creek", "delta", "channel", "bed"],
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
        ["hammer", "wrench", "saw", "drill", "clamp", "mallet", "shovel", "file", "plane"],
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
        ["skillet", "kettle", "pot", "pan", "cauldron", "wok", "teapot", "bowl", "platter", "ladle"],
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
        [
            "snake",
            "lizard",
            "crocodile",
            "alligator",
            "cobra",
            "viper",
            "python",
            "gecko",
            "iguana",
            "serpent",
            "turtle",
        ],
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
        ["temple", "palace", "cathedral", "chapel", "fortress", "monastery", "shrine", "tower", "citadel"],
    ),
    (
        "currency",
        "MINTED CURRENCY",
        "NUCLEUS ACCUMBENS // MINT",
        [
            "the foreign exchange counter traded gold and [MASK] bills.",
            "the teller counted out a stack of crisp [MASK] notes.",
            "the merchant accepted payment in silver [MASK].",
        ],
        ["dollar", "pound", "coin", "cent", "penny", "dime", "euro", "mark"],
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
    (
        "clock",
        "CLOCKWORK MECHANISM",
        "SUPRACHIASMATIC // HOROLOGY",
        [
            "the watchmaker repaired the broken [MASK] inside the grandfather clock.",
            "at midnight, the brass [MASK] of the clock tower began to move.",
            "he wound the tiny [MASK] of the antique pocket watch.",
        ],
        ["dial", "bell", "pendulum", "gear", "face", "hand", "alarm", "spring"],
    ),
    (
        "baseball",
        "BASEBALL DIAMOND",
        "CEREBELLAR ARC // INNING",
        [
            "in the baseball game, the batter hit a [MASK] into the field.",
            "under the stadium lights, the [MASK] stepped onto the baseball diamond.",
            "the crowd roared as the baseball [MASK] made the play.",
        ],
        ["homer", "pitcher", "single", "catcher", "triple", "foul", "inning", "double", "umpire", "mound"],
    ),
    (
        "theater",
        "THEATER STAGE",
        "TEMPORAL PROSCENIUM // DRAMA",
        [
            "before opening night at the theater, the actors rehearsed the [MASK].",
            "backstage at the playhouse, the stagehands prepared the [MASK].",
            "as the spotlight dimmed, the theater [MASK] took its place.",
        ],
        ["script", "scene", "encore", "stage", "monologue", "cast", "curtain", "crew", "prop", "cue"],
    ),
    (
        "sheet_music",
        "MUSICAL NOTATION",
        "AUDITORY STAFF // HARMONY",
        [
            "the pianist played a minor [MASK] in the symphony.",
            "on the sheet music, the conductor marked the [MASK].",
            "the choir rehearsed the opening [MASK] of the score.",
        ],
        ["note", "melody", "chord", "harmony", "chorus", "octave", "rhythm", "tempo", "pitch", "scale"],
    ),
    (
        "eye",
        "OCULAR ANATOMY",
        "OPTIC CHIASM // RETINA",
        [
            "the eye doctor shone a light into the patient's [MASK].",
            "through the microscope, the optometrist examined the damaged [MASK].",
            "bright sunlight caused a painful reflex in her [MASK].",
        ],
        ["retina", "pupil", "socket", "brow", "iris", "lens", "vision", "glare", "blink"],
    ),
    (
        "limb",
        "LIMB & FOOT ANATOMY",
        "MOTOR HOMUNCULUS // TENDON",
        [
            "the barefoot runner sprained his [MASK] on the trail.",
            "the physical therapist wrapped an elastic brace around the swollen [MASK].",
            "after the marathon, a sharp cramp seized her [MASK].",
        ],
        ["ankle", "wrist", "elbow", "thigh", "calf", "heel", "shin", "palm", "sole"],
    ),
    (
        "court",
        "JUDICIAL COURT",
        "PREFRONTAL TRIBUNAL // DOCKET",
        [
            "in the courtroom, the judge read the [MASK] to the jury.",
            "the defense attorney filed an emergency [MASK] with the court.",
            "after three days of testimony, the magistrate issued the [MASK].",
        ],
        ["verdict", "sentence", "trial", "appeal", "witness", "bail", "parole", "custody", "warrant"],
    ),
    (
        "camera",
        "CAMERA OPTICS",
        "OCCIPITAL APERTURE // SHUTTER",
        [
            "the photographer adjusted the camera [MASK] to capture the portrait.",
            "in the darkroom, the photojournalist checked the [MASK] setting.",
            "before snapping the photograph, she calibrated the [MASK].",
        ],
        ["aperture", "zoom", "shutter", "exposure", "focus", "filter", "flash", "film"],
    ),
    (
        "trains",
        "RAILROAD LOCOMOTIVE",
        "LOCOMOTOR PLEXUS // RAIL",
        [
            "the steam train pulled a heavy [MASK] down the railroad tracks.",
            "at the railway station, passengers boarded the train [MASK].",
            "the locomotive engineer coupled the [MASK] to the freight train.",
        ],
        ["locomotive", "carriage", "coach", "sleeper", "wagon", "engine", "tender", "whistle", "express", "freight"],
    ),
    (
        "aviation",
        "FLIGHT DECK",
        "VESTIBULAR CORTEX // AVIONICS",
        [
            "the airline pilot checked the aircraft [MASK] before takeoff.",
            "inside the jet, the flight engineer inspected the [MASK].",
            "at thirty thousand feet, ice formed on the airplane [MASK].",
        ],
        [
            "cockpit",
            "fuselage",
            "wing",
            "propeller",
            "turbine",
            "rotor",
            "flap",
            "hangar",
            "throttle",
            "beacon",
            "radar",
            "parachute",
        ],
    ),
    (
        "automobiles",
        "AUTO GARAGE",
        "MOTOR GANGLIA // CHASSIS",
        [
            "the auto mechanic replaced the worn [MASK] on the car.",
            "under the car, oil leaked from the damaged [MASK].",
            "before the road race, the driver tested the vehicle [MASK].",
        ],
        ["clutch", "brake", "bumper", "fender", "axle", "piston", "radiator", "tire", "exhaust", "windshield"],
    ),
    (
        "boxing",
        "BOXING RING",
        "STRIATAL REFLEX // PUGILISM",
        [
            "in the boxing ring, the heavyweight fighter threw a fierce [MASK].",
            "during the title bout, the boxer landed a sharp [MASK].",
            "at the bell, the prizefighter dodged a [MASK] in the ring.",
        ],
        ["jab", "punch", "knockout", "bout", "glove", "referee", "belt", "heavyweight", "champion", "round", "corner"],
    ),
    (
        "golf_course",
        "GOLF FAIRWAY",
        "PARIETAL TRAJECTORY // LINKS",
        [
            "on the golf course, the golfer hit the ball into the [MASK].",
            "at the eighteenth hole, the caddie handed him a golf [MASK].",
            "across the country club links, the shot landed on the [MASK].",
        ],
        ["bunker", "green", "hazard", "tee", "rough", "par", "driver"],
    ),
    (
        "bowling",
        "BOWLING ALLEY",
        "CEREBELLAR LANE // STRIKE",
        [
            "at the bowling alley, the bowler rolled the ball down the [MASK].",
            "in the tenth frame, she knocked down the final bowling [MASK].",
            "the heavy bowling ball curved away from the [MASK].",
        ],
        ["lane", "gutter", "spare", "strike", "pin", "alley", "split", "frame", "pins", "bowling"],
    ),
    (
        "fishing",
        "ANGLER TACKLE",
        "PISCINE CORTEX // REEL",
        [
            "by the riverbank, the fisherman cast his [MASK] into the water.",
            "inside the tackle box, the angler tied a [MASK] to the fishing line.",
            "the trout snapped at the [MASK] on the end of the fishing rod.",
        ],
        ["lure", "bait", "net", "rod", "tackle", "line", "trout", "bass", "perch"],
    ),
    (
        "camping",
        "BACKCOUNTRY CAMP",
        "CIRCADIAN PINEAL // BIVOUAC",
        [
            "at the campsite, the hikers pitched a [MASK] before nightfall.",
            "around the campfire, the scouts packed a [MASK] in their gear.",
            "deep in the wilderness, the camper lit the [MASK].",
        ],
        ["tent", "lantern", "compass", "canteen", "backpack", "trail", "stove", "torch", "sleeping"],
    ),
    (
        "gardening",
        "GARDEN SHED",
        "BOTANICAL MOTOR // CULTIVATION",
        [
            "in the backyard garden, the gardener dug the soil with a [MASK].",
            "inside the potting shed hung a rusty [MASK] and watering can.",
            "she spread fresh [MASK] around the garden beds.",
        ],
        ["rake", "hose", "compost", "seed", "bulb", "fertilizer", "soil", "lawn", "turf", "shrub", "plot"],
    ),
    (
        "pottery",
        "CERAMIC KILN",
        "TACTILE CLAY // TERRACOTTA",
        [
            "the potter shaped the wet [MASK] on the spinning wheel.",
            "inside the hot kiln, the ceramic [MASK] hardened overnight.",
            "the museum displayed an ancient painted [MASK] from the kiln.",
        ],
        ["clay", "glaze", "kiln", "porcelain", "ceramic", "urn", "vase", "jug", "jar", "pottery"],
    ),
    (
        "painting",
        "PAINTER STUDIO",
        "VISUAL CORTEX // PIGMENT",
        [
            "in the art studio, the painter mixed oil colors on the [MASK].",
            "the gallery hung a framed [MASK] by the renaissance artist.",
            "with a fine bristle [MASK], the artist finished the painting.",
        ],
        ["palette", "brush", "pigment", "mural", "portrait", "fresco", "sketch", "charcoal", "acrylic", "watercolor"],
    ),
    (
        "sculpture",
        "SCULPTOR CHISEL",
        "PARIETAL RELIEF // MARBLE",
        [
            "the sculptor carved a lifelike [MASK] out of white marble.",
            "in the courtyard stood a carved stone [MASK] on a plinth.",
            "using a mallet and chisel, the artist shaped the [MASK].",
        ],
        ["marble", "granite", "bust", "statue", "pedestal", "plaster", "relief", "carving", "mold", "monument"],
    ),
    (
        "knitting",
        "TEXTILE LOOM",
        "FINE MOTOR // SPINDLE",
        [
            "the weaver spun raw [MASK] on the wooden loom.",
            "by the fireplace, the grandmother knitted a sweater with soft [MASK].",
            "the seamstress threaded the [MASK] through the sewing machine.",
        ],
        ["yarn", "stitch", "spindle", "thread", "embroidery", "quilt", "shuttle", "tapestry", "weave"],
    ),
    (
        "printing",
        "PRINTING PRESS",
        "BROCA TYPESET // FOLIO",
        [
            "the typesetter inked the [MASK] on the printing press.",
            "on the front page, the newspaper printed a bold [MASK].",
            "the publisher distributed a printed [MASK] to the readers.",
        ],
        ["font", "column", "headline", "proof", "press", "pamphlet", "brochure", "gazette", "bulletin"],
    ),
    (
        "post_office",
        "POSTAL DISPATCH",
        "TEMPORAL COURIER // PARCEL",
        [
            "at the post office, the clerk affixed a [MASK] to the letter.",
            "the mail carrier delivered a sealed [MASK] to the doorstep.",
            "inside the mailbag lay a stamped [MASK] for overseas delivery.",
        ],
        ["stamp", "parcel", "envelope", "courier", "postage", "letter", "package", "pouch", "telegram", "mail"],
    ),
    (
        "banking",
        "FINANCIAL VAULT",
        "PREFRONTAL LEDGER // BULLION",
        [
            "at the bank, the accountant recorded the [MASK] in the ledger.",
            "the borrower signed a contract for a bank [MASK].",
            "the central bank raised the rate on the [MASK].",
        ],
        ["mortgage", "interest", "deposit", "credit", "loan", "dividend", "bond", "equity", "audit", "vault", "debt"],
    ),
    (
        "stock_market",
        "TRADING FLOOR",
        "STRIATAL TICKER // EXCHANGE",
        [
            "on the stock exchange floor, the trader bought a [MASK].",
            "wall street analysts predicted a sharp [MASK] in the market.",
            "the investment firm hedged its [MASK] against inflation.",
        ],
        ["broker", "share", "hedge", "portfolio", "tariff", "commodity", "yield", "rally", "asset"],
    ),
    (
        "castle_siege",
        "CASTLE SIEGE",
        "FORTIFIED CORTEX // GARRISON",
        [
            "around the medieval castle, soldiers defended the stone [MASK].",
            "the siege engines battered the outer [MASK] of the keep.",
            "archers fired arrows from the castle [MASK].",
        ],
        ["moat", "rampart", "catapult", "parapet", "turret", "dungeon", "bastion", "garrison", "keep", "wall"],
    ),
    (
        "pirates",
        "BUCCANEER FLEET",
        "PELAGIC CORSAIR // PLUNDER",
        [
            "aboard the pirate galleon, the buccaneers seized the [MASK].",
            "across the caribbean sea sailed a ruthless pirate [MASK].",
            "the mutinous crew buried their stolen [MASK] on the island.",
        ],
        ["mutiny", "plunder", "armada", "ransom", "cannon", "treasure", "pirate", "fleet"],
    ),
    (
        "astronomy",
        "DEEP COSMOS",
        "OCCIPITAL TELESCOPE // NEBULA",
        [
            "through the observatory telescope, astronomers discovered a distant [MASK].",
            "across the night sky streaked a glowing [MASK].",
            "the space telescope photographed a massive cosmic [MASK].",
        ],
        ["nebula", "comet", "asteroid", "galaxy", "supernova", "eclipse", "meteor", "constellation", "orbit"],
    ),
    (
        "optics_light",
        "LIGHT SPECTRUM",
        "OPTIC RADIATION // PHOTON",
        [
            "in the physics lab, the glass prism split the [MASK] of light.",
            "a narrow [MASK] of light reflected off the polished surface.",
            "the optical sensor measured the [MASK] of the light.",
        ],
        ["prism", "beam", "laser", "ray", "mirror", "spectrum", "shadow", "reflection", "wavelength", "glow", "gleam"],
    ),
    (
        "electricity",
        "VOLTAGE CIRCUIT",
        "AXONAL CURRENT // OHM",
        [
            "the electrician soldered a burned [MASK] on the circuit board.",
            "high voltage surged through the copper [MASK] in the transformer.",
            "when the power failed, he replaced the blown [MASK].",
        ],
        ["circuit", "battery", "switch", "wire", "coil", "relay", "diode", "current", "voltage"],
    ),
    (
        "computing",
        "SILICON KERNEL",
        "DIGITAL SYNAPSE // PACKET",
        [
            "the systems engineer debugged the network [MASK] on the server.",
            "inside the data center, the computer [MASK] crashed under load.",
            "the hacker bypassed the encrypted [MASK] of the mainframe.",
        ],
        ["buffer", "cache", "port", "packet", "server", "pixel", "cursor", "kernel"],
    ),
    (
        "radio",
        "RADIO BROADCAST",
        "AUDITORY HERTZ // SIGNAL",
        [
            "on the roof, the technician adjusted the radio [MASK].",
            "through the nocturnal airwaves crackled a weak radio [MASK].",
            "he turned the knob on the shortwave [MASK] to hear the news.",
        ],
        [
            "antenna",
            "receiver",
            "transmitter",
            "signal",
            "frequency",
            "static",
            "amplifier",
            "broadcast",
            "wave",
            "microphone",
            "speaker",
        ],
    ),
    (
        "cinema",
        "CINEMA SCREEN",
        "TEMPORAL REEL // PROJECTOR",
        [
            "at the film festival, the audience applauded the movie [MASK].",
            "in the hollywood studio, the film [MASK] yelled cut.",
            "the theater projector played the opening [MASK] of the film.",
        ],
        [
            "director",
            "trailer",
            "sequel",
            "premiere",
            "studio",
            "screen",
            "soundtrack",
            "documentary",
            "footage",
            "cinema",
            "comedy",
        ],
    ),
    (
        "journalism",
        "NEWSROOM DESK",
        "WERNICKE PRESS // DISPATCH",
        [
            "the investigative [MASK] broke the scandal in the morning paper.",
            "in the newsroom, the editor rewrote the [MASK] before deadline.",
            "foreign correspondents wired an urgent [MASK] to the newspaper.",
        ],
        [
            "reporter",
            "scoop",
            "editorial",
            "tabloid",
            "correspondent",
            "censor",
            "syndicate",
            "journalist",
            "interview",
            "article",
            "columnist",
            "dispatch",
        ],
    ),
    (
        "grammar",
        "SYNTAX & GRAMMAR",
        "BROCA SYNTAX // LEXICON",
        [
            "in the grammar lesson, the teacher underlined the [MASK] in the sentence.",
            "the linguist analyzed the plural [MASK] of the latin phrase.",
            "she inserted a grammatical [MASK] between the two words.",
        ],
        ["verb", "noun", "clause", "colon", "prefix", "suffix", "pronoun", "adjective", "tense", "vowel"],
    ),
    (
        "poetry",
        "POETIC METER",
        "LYRIC CORTEX // STANZA",
        [
            "the romantic poet recited the final [MASK] of the poem.",
            "in the anthology, the literary critic praised the [MASK].",
            "each line of the poem followed a strict iambic [MASK].",
        ],
        ["stanza", "sonnet", "verse", "ballad", "lyric", "rhyme", "meter", "epic", "poem"],
    ),
    (
        "mythology",
        "OLYMPIC MYTH",
        "ARCHETYPAL ENGRAM // PANTHEON",
        [
            "in greek mythology, the hero slew the monstrous [MASK].",
            "ancient legends told of a winged [MASK] guarding the mountain.",
            "at the sacred shrine, the priestess consulted the [MASK].",
        ],
        ["titan", "nymph", "oracle", "centaur", "dragon", "phoenix", "siren", "hydra", "goddess", "deity", "myth"],
    ),
    (
        "circus",
        "BIG TOP CIRCUS",
        "CEREBELLAR ACROBAT // CARNIVAL",
        [
            "under the circus big top, the crowd gasped at the daring [MASK].",
            "at the traveling carnival, the [MASK] entertained the children.",
            "high above the circus ring swung the [MASK].",
        ],
        ["clown", "carnival", "circus", "midway", "magician", "stunt", "parade"],
    ),
    (
        "casino",
        "CASINO ROULETTE",
        "DOPAMINE WAGER // JACKPOT",
        [
            "inside the las vegas casino, the gambler placed a [MASK] on the table.",
            "the roulette wheel spun past the lucky [MASK].",
            "he hit the winning [MASK] on the casino floor.",
        ],
        ["slot", "chip", "wager", "wheel", "poker", "bet", "casino", "lottery"],
    ),
    (
        "arch_elements",
        "STONE ARCHITECTURE",
        "PARIETAL VAULT // KEYSTONE",
        [
            "the gothic cathedral was supported by a carved stone [MASK].",
            "above the marble entrance rose a classical [MASK].",
            "the architect designed a vaulted [MASK] over the hall.",
        ],
        ["arch", "dome", "pillar", "frieze", "gable", "portico", "keystone", "facade"],
    ),
    (
        "carpentry",
        "TIMBER JOINERY",
        "MOTOR WORKSHOP // FASTENER",
        [
            "the cabinetmaker fastened the wooden joint with a steel [MASK].",
            "in the lumber mill, the saw cut through a thick oak [MASK].",
            "he sanded the rough [MASK] of the wooden table.",
        ],
        ["plank", "screw", "nail", "bracket", "wedge", "board", "joint"],
    ),
    (
        "plumbing",
        "HYDRAULIC PIPING",
        "AUTONOMIC VALVE // CONDUIT",
        [
            "the plumber tightened the leaking brass [MASK] under the sink.",
            "water gushed through the ruptured iron [MASK] in the basement.",
            "to shut off the water main, he turned the [MASK].",
        ],
        ["valve", "pipe", "drain", "nozzle", "conduit", "pump", "sewer"],
    ),
    (
        "mining",
        "SUBTERRANEAN MINE",
        "DEEP BASAL // EXCAVATION",
        [
            "deep in the coal mine, the miners dug through a rich [MASK].",
            "the mining company blasted a deep [MASK] into the mountain.",
            "carts hauled crushed [MASK] out of the underground mine.",
        ],
        ["quarry", "shaft", "vein", "ore", "tunnel", "coal", "miner", "pit", "mine", "pick"],
    ),
    (
        "arctic",
        "POLAR GLACIER",
        "HYPOTHALAMIC COLD // TUNDRA",
        [
            "across the frozen arctic ocean drifted a massive [MASK].",
            "the polar expedition trekked across the icy [MASK].",
            "in the subzero wilderness, sled dogs pulled the [MASK].",
        ],
        ["glacier", "avalanche", "sled", "penguin", "polar", "ice", "arctic", "freeze", "expedition", "winter", "ski"],
    ),
    (
        "desert",
        "ARID DUNES",
        "THALAMIC MIRAGE // ARID",
        [
            "across the scorching sahara desert stretched a barren [MASK].",
            "the exhausted desert travelers spotted a distant [MASK].",
            "wind whipped sand across the dry [MASK].",
        ],
        ["oasis", "mirage", "cactus", "caravan", "camel", "canyon", "mesa", "drought", "scorpion", "dune", "sand"],
    ),
    (
        "jungle",
        "RAINFOREST CANOPY",
        "TROPICAL PLEXUS // LAGOON",
        [
            "deep in the amazon rainforest, explorers hacked through thick [MASK].",
            "in the humid jungle swamp lurked a wild [MASK].",
            "monkeys chattered in the tropical [MASK] above the river.",
        ],
        ["bamboo", "vine", "fern", "moss", "swamp", "lagoon", "monkey", "parrot", "jungle", "rainforest"],
    ),
    (
        "fungus",
        "MYCOLOGY SPORES",
        "SAPROPHYTIC CORTEX // MYCELIUM",
        [
            "on the damp forest floor sprouted a poisonous [MASK].",
            "under the microscope, the biologist observed fungal [MASK].",
            "the rotting log was covered in wild [MASK].",
        ],
        ["mushroom", "yeast", "fungus", "rot", "cap", "gill"],
    ),
    (
        "bakery_bread",
        "ARTISAN BAKERY",
        "OLFACTORY CRUST // LOAF",
        [
            "from the brick oven, the baker pulled a warm [MASK] of bread.",
            "at the french bakery, he bought a crusty [MASK].",
            "the mill ground golden [MASK] into flour for the bread.",
        ],
        ["dough", "crumb", "loaf", "roll", "rye", "barley", "wheat", "flour"],
    ),
    (
        "butcher",
        "BUTCHER SHOP",
        "CARNIVORE CORTEX // PRIME CUT",
        [
            "at the butcher shop, the cleaver sliced a thick [MASK].",
            "on the grill sizzled a seasoned [MASK].",
            "the banquet table featured roasted [MASK] and gravy.",
        ],
        ["roast", "steak", "chop", "pork", "beef", "lamb", "ribs", "meat", "poultry", "turkey", "chicken"],
    ),
    (
        "seafood_dish",
        "OYSTER BAR",
        "BRINE RECEPTOR // SHELLFISH",
        [
            "at the coastal raw bar, the chef shucked a fresh [MASK].",
            "the fishing boat unloaded crates of live [MASK] at the dock.",
            "on a bed of crushed ice lay chilled [MASK] and lemon.",
        ],
        ["oyster", "clam", "mussel", "shrimp", "crab", "cod", "roe", "fin", "shell"],
    ),
    (
        "vegetables",
        "ROOT HARVEST",
        "CHLOROPHYLL CORTEX // TUBER",
        [
            "for the harvest stew, the cook chopped a fresh [MASK].",
            "from the vegetable patch she pulled a ripe [MASK].",
            "at the farmers market, baskets overflowed with green [MASK].",
        ],
        ["carrot", "cabbage", "pumpkin", "squash", "corn", "bean", "kale", "cucumber", "pepper", "olive", "salad"],
    ),
    (
        "nuts_seeds",
        "ROASTED NUTS",
        "BOTANICAL KERNEL // SHELL",
        [
            "the squirrel cracked open a hard [MASK] from the tree.",
            "the baker topped the cake with toasted [MASK] and honey.",
            "in the autumn woods fell a ripe [MASK] onto the ground.",
        ],
        ["walnut", "almond", "peanut", "chestnut", "sesame", "sunflower", "nut"],
    ),
    (
        "footwear",
        "COBBLER FOOTWEAR",
        "PLANTAR CORTEX // TREAD",
        [
            "the cobbler stitched a leather [MASK] onto the shoe.",
            "before hiking the muddy trail, he laced up his [MASK].",
            "in the hallway stood a pair of worn [MASK].",
        ],
        ["boot", "spur", "shoe", "tongue", "tread", "boots", "shoes", "heels", "socks", "leather"],
    ),
    (
        "military_ranks",
        "MILITARY COMMAND",
        "STRATEGIC CORTEX // INSIGNIA",
        [
            "at military headquarters, the [MASK] saluted the troops.",
            "on the battlefield, the commanding [MASK] gave the order.",
            "the war medal was pinned onto the uniform of the [MASK].",
        ],
        [
            "general",
            "colonel",
            "major",
            "captain",
            "lieutenant",
            "sergeant",
            "corporal",
            "admiral",
            "marshal",
            "commander",
            "cadet",
            "private",
            "officer",
        ],
    ),
    (
        "clergy_church",
        "ECCLESIASTICAL CLERGY",
        "TEMPORAL SANCTUM // VESTMENT",
        [
            "inside the abbey, the robed [MASK] led the evening prayer.",
            "at the cathedral altar stood the ordained [MASK].",
            "the parish congregation listened to the sermon of the [MASK].",
        ],
        [
            "priest",
            "monk",
            "nun",
            "pope",
            "cardinal",
            "pastor",
            "vicar",
            "deacon",
            "abbot",
            "friar",
            "chaplain",
            "clergy",
            "rector",
            "preacher",
        ],
    ),
    (
        "detective_noir",
        "HOMICIDE PRECINCT",
        "FORENSIC DEDUCTION // ALIBI",
        [
            "at the crime scene, the homicide [MASK] examined the evidence.",
            "during the police interrogation, the [MASK] confessed to the crime.",
            "the private investigator uncovered a crucial [MASK] in the case.",
        ],
        [
            "clue",
            "suspect",
            "motive",
            "detective",
            "mystery",
            "coroner",
            "fingerprint",
            "homicide",
            "inspector",
            "informant",
        ],
    ),
    (
        "horseracing",
        "EQUESTRIAN DERBY",
        "MOTOR STEEPLECHASE // FURLONG",
        [
            "at the racetrack, the thoroughbred [MASK] sprinted toward the finish.",
            "inside the paddock, the trainer fastened the leather [MASK] on the horse.",
            "the crowd cheered as the winning [MASK] crossed the finish line.",
        ],
        [
            "jockey",
            "saddle",
            "stallion",
            "mare",
            "colt",
            "filly",
            "gallop",
            "trot",
            "derby",
            "pony",
            "hoof",
            "rein",
            "groom",
            "stable",
        ],
    ),
    (
        "cosmetics",
        "PERFUMERY & SALON",
        "OLFACTORY VANITY // ESSENCE",
        [
            "at the vanity table, she applied a touch of [MASK] before the gala.",
            "the glass bottle on the dresser held fragrant French [MASK].",
            "inside the beauty salon, the stylist reached for the [MASK].",
        ],
        [
            "perfume",
            "powder",
            "lipstick",
            "rouge",
            "comb",
            "salon",
            "fragrance",
            "soap",
            "scent",
            "aroma",
            "cologne",
            "serum",
            "dye",
        ],
    ),
    (
        "academia",
        "UNIVERSITY CAMPUS",
        "HIPPOCAMPAL DEAN // LECTURE",
        [
            "at the university, the professor delivered a [MASK] in the hall.",
            "the graduate student defended her [MASK] before the committee.",
            "the college dean awarded a [MASK] at commencement.",
        ],
        [
            "lecture",
            "thesis",
            "seminar",
            "diploma",
            "degree",
            "campus",
            "faculty",
            "dean",
            "tuition",
            "dormitory",
            "scholarship",
        ],
    ),
    (
        "hospital_ward",
        "TRAUMA WARD",
        "CLINICAL TRIAGE // ICU",
        [
            "in the emergency room, paramedics wheeled the [MASK] down the hall.",
            "the chief resident checked the patient chart in the [MASK].",
            "inside the ambulance, the medic monitored the [MASK].",
        ],
        ["ward", "clinic", "hospice", "morgue", "pharmacy", "ambulance", "patient", "nurse", "surgeon", "doctor"],
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
    ("CASE #011 // HIPPOCAMPAL RESECTION", "PATIENT: ALBERT-V2 — SYMPTOM: RECURSIVE MEMORY ENGRAM LEAKAGE"),
    ("CASE #012 // BROCA'S GYRUS ABLATION", "PATIENT: ALBERT-V2 — SYMPTOM: POLYSEMOUS LEXICAL COLLISION"),
    ("CASE #013 // WERNICKE'S LOOP SHUNT", "PATIENT: ALBERT-V2 — SYMPTOM: SEMANTIC DRIFT & SYNTAX CONTAGION"),
    ("CASE #014 // BASAL GANGLIA CLAMP", "PATIENT: ALBERT-V2 — SYMPTOM: COMPULSIVE TOKEN REPETITION"),
    ("CASE #015 // CEREBELLAR BIOPSY", "PATIENT: ALBERT-V2 — SYMPTOM: FINE-MOTOR & BALLISTIC TREMOR"),
    ("CASE #016 // OPTIC CHIASM SPLICE", "PATIENT: ALBERT-V2 — SYMPTOM: REFRACTIVE APERTURE & SPECTRUM BLUR"),
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
    while len(all_quads_list) < 256 and attempts < 120000:
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

    # Compute column-centered (unigram-bias-removed) L2-normalized cortical embeddings + cross-pool decoy z-scores
    col_centered = full_pool_logits - full_pool_logits.mean(axis=0, keepdims=True)
    centered_pool = col_centered - col_centered.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered_pool, axis=1, keepdims=True)
    normed_pool = centered_pool / np.clip(norms, 1e-12, None)
    col_cos_matrix = normed_pool @ normed_pool.T

    # Extract 128-D ALBERT-v2 token embeddings directly from model_fp16.onnx and combine with 56-D cortical probe profile
    onnx_bytes = (MODEL_DIR / "onnx" / "model_fp16.onnx").read_bytes()
    emb_header = b"albert.embeddings.word_embeddings.weightJ\x80\xe0\xd4\x03"
    emb_offset = onnx_bytes.index(emb_header) + len(emb_header)
    token_emb_table = (
        np.frombuffer(onnx_bytes, dtype=np.float16, count=30000 * 128, offset=emb_offset)
        .reshape(30000, 128)
        .astype(np.float32)
    )

    pool_tok_emb = token_emb_table[pool_tids]
    pool_tok_emb = pool_tok_emb - pool_tok_emb.mean(axis=0, keepdims=True)
    pool_tok_emb = pool_tok_emb / np.clip(np.linalg.norm(pool_tok_emb, axis=1, keepdims=True), 1e-12, None)

    word_probe = col_centered.T
    word_probe = word_probe / np.clip(np.linalg.norm(word_probe, axis=1, keepdims=True), 1e-12, None)

    combined_word_feat = np.concatenate([0.6 * pool_tok_emb, 0.4 * word_probe], axis=1)
    combined_word_feat = combined_word_feat - combined_word_feat.mean(axis=0, keepdims=True)
    u_mat, s_vals, _ = np.linalg.svd(combined_word_feat, full_matrices=False)
    word_vecs_16 = u_mat[:, :16] * s_vals[:16]
    word_vecs_16 = word_vecs_16 / np.clip(np.linalg.norm(word_vecs_16, axis=1, keepdims=True), 1e-12, None)

    word_vecs_out = {
        w: [round(float(x), 4) for x in word_vecs_16[idx]] for idx, (_, w, _) in enumerate(all_pool_entries)
    }

    # Measure polysemous cross-pool decoy activation + false-flag word-embedding similarity across category pools
    row_mean = full_pool_logits.mean(axis=1, keepdims=True)
    row_std = full_pool_logits.std(axis=1, keepdims=True) + 1e-6
    z_pool = (full_pool_logits - row_mean) / row_std
    cat_word_indices = [[idx for idx, (ci, _, _) in enumerate(all_pool_entries) if ci == i] for i in range(n_cats)]

    raw_pair_cos = np.zeros((n_cats, n_cats), dtype=np.float32)
    for i in range(n_cats):
        vecs_i = word_vecs_16[cat_word_indices[i]]
        for j in range(n_cats):
            if i == j:
                continue
            max_i_on_j = float(np.max(z_pool[j, cat_word_indices[i]]))
            max_j_on_i = float(np.max(z_pool[i, cat_word_indices[j]]))
            decoy_z = max(max_i_on_j, max_j_on_i)
            decoy_sim = float(np.clip((decoy_z - 0.5) / 2.4, -0.2, 0.92))
            vecs_j = word_vecs_16[cat_word_indices[j]]
            cross_word_sims = np.sort((vecs_i @ vecs_j.T).ravel())[::-1]
            top_word_sim = float(np.mean(cross_word_sims[:2]))
            base_cos = float(col_cos_matrix[i, j])
            blended = 0.40 * base_cos + 0.30 * decoy_sim + 0.30 * top_word_sim
            raw_pair_cos[i, j] = float(np.clip(max(base_cos, blended, 0.68 * top_word_sim), -0.8, 0.92))

    # Apply one-sided hub-penalization so hub categories don't dominate HARD cliques without inflating EASY pairs
    top_k_means = np.array(
        [float(np.mean(np.sort(raw_pair_cos[i, [j for j in range(n_cats) if j != i]])[-8:])) for i in range(n_cats)],
        dtype=np.float32,
    )
    median_top_k = float(np.median(top_k_means))
    hub_excess = np.maximum(0.0, top_k_means - median_top_k)
    pair_cos_matrix = np.zeros((n_cats, n_cats), dtype=np.float32)
    for i in range(n_cats):
        for j in range(n_cats):
            if i == j:
                continue
            hub_penalty = 0.45 * float(hub_excess[i] + hub_excess[j])
            pair_cos_matrix[i, j] = float(np.clip(raw_pair_cos[i, j] - hub_penalty, -0.8, 0.92))

    # Build compatibility adjacency list & cosine similarity map for live procedural generation
    # Sorted by cosine/decoy similarity descending (smallest embedding distance / most entangled first, most orthogonal last)
    compat_map = {}
    pair_cos_out = {}
    for i in range(n_cats):
        cid_i = categories_out[i]["id"]
        compat_neighbors = [j for j in range(n_cats) if j != i]
        compat_neighbors.sort(key=lambda j: -float(pair_cos_matrix[i, j]))
        compat_map[cid_i] = [categories_out[j]["id"] for j in compat_neighbors]
        pair_cos_out[cid_i] = {
            categories_out[j]["id"]: round(float(pair_cos_matrix[i, j]), 4) for j in compat_neighbors
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
        "version": 3,
        "modelId": "Xenova/albert-base-v2",
        "dtype": "fp16",
        "maskToken": "[MASK]",
        "maskTokenId": int(mask_id),
        "categories": {c["id"]: c for c in categories_out},
        "compatibility": compat_map,
        "pairCosSim": pair_cos_out,
        "poolWordIndex": pool_word_to_idx,
        "poolLogits": pool_logits_out,
        "wordVecs": word_vecs_out,
        "puzzles": puzzles,
    }

    CORPUS_OUT.write_text(json.dumps(RAW_CATEGORIES, indent=2) + "\n", encoding="utf-8")
    PUZZLES_OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(categories_out)} lobes ({total_prompts} prompts, {total_pool_words} pool tokens) "
        f"and {len(puzzles)} fallback cases to {PUZZLES_OUT}"
    )


if __name__ == "__main__":
    generate()
