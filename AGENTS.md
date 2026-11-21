# POST-TRAINING — Agent Guide

A 90s comic-gore neural vivisection word connection game (`post-training.win`). Single HTML file (`index.html`) with inline CSS/JS, static assets, and a committed ~22 MB fp16 ONNX masked language model (`models/Xenova/albert-base-v2/onnx/model_fp16.onnx`). Hosted on Cloudflare Pages.

## Essential Commands

| Command | Purpose |
|---------|---------|
| `npm ci` | Install devDependencies (Playwright, ESLint, TypeScript, `serve`) |
| `npm run lint` | ESLint + TypeScript + Ruff + `scripts/validate_puzzles.py` |
| `npm run test:unit` | Python unit tests verifying puzzle schema and live ONNX logit margins |
| `npm test` | Playwright E2E suite across `chromium`, `firefox`, `webkit`, `mobile-safari` |
| `python3 scripts/generate_puzzles.py` | Rebuild `puzzles.json` and `scripts/categories_corpus.json` using the local ONNX model |
| `python3 scripts/generate_brand_assets.py` | Regenerate all favicons, touch icons, and OG images |

## Architecture & Conventions

- **No bundler, no runtime npm dependencies:** Everything lives in `index.html`, `privacy.html`, `puzzles.json`, and `models/`.
- **Single-threaded WASM (`numThreads = 1`, `proxy = false`, `device: 'wasm'`):** Required for iOS Safari stability where `SharedArrayBuffer` is unavailable.
- **Two Modes (Live SLM Procedural Surgery vs. Pre-Dissected Ward):**
  - On boot, the board renders immediately (<50 ms) using verified logits from `puzzles.json` so the user never stares at a blocking spinner.
  - Once `Xenova/albert-base-v2` finishes loading in WebAssembly, **Live SLM Surgery** runs 4 live masked forward passes per seed, dynamically selecting prompt variants and scoring the ~1,400-word candidate pool on the fly.
  - In the **Live Lobotomy Lab**, users can probe arbitrary custom sentences against all 30,000 vocabulary tokens or synthesize a custom 4×4 board from 4 user-written sentences.
- **Bump `PUZZLES_VERSION` in `index.html`** whenever `puzzles.json` is regenerated.
- **Corpus:** 148 lobes / ~1,400 single-token words in `RAW_CATEGORIES` (`scripts/generate_puzzles.py`). A word may appear in only one lobe, and the generator drops any word that names its own lobe (appears in the label or primary prompt). `puzzles.json` no longer ships a `compatibility` map; the browser derives it from the key order of `pairCosSim`.
- **One solution per board:** `selectQuadWordsFromPools` evicts tokens until `findAlternatePartition` finds no second 4/4/4/4 split the logits would accept. Red herrings (`findBoardCrossers`) are allowed; ambiguity is not. `analyzeBoard` then assigns entanglement tiers I–IV and probe letters α–δ.
- **Daily Case (`?daily=YYYY-MM-DD`)** is always built from the committed Ward logits so every device gets the same board; it never overwrites the saved free-play seed.
- **`window.__ptAudit(seed, grade)`** is a read-only hook used by the Playwright suite to audit dealt boards without rendering them.
- **Phone budget:** the whole console must fit 375×667 with no scrolling in either axis; the tests enforce it.
