# POST-TRAINING

**[post-training.win](https://post-training.win/)** · **[Privacy Policy](https://post-training.win/privacy.html)**

A 90s comic-gore word connection game powered by a tiny on-device Small Language Model (`Xenova/albert-base-v2`, 11.8M parameters, ~22 MB fp16 ONNX) running **100% client-side in your browser** via WebAssembly.

Inspired by NYT Connections, edge keyboard next-word prediction models, and the visceral black-ink / crimson / acid-green comic book aesthetic of Kevin Eastman & Simon Bisley's 1996 *TMNT: Bodycount*.

---

## How It Plays

Our patient — a compact 11.8M-parameter transformer running inside your browser — has suffered catastrophic synaptic entanglement. Four **Cortical Sentence Probes** (each with a `[MASK]` slot) are firing at once, and **16 vocabulary tokens** are floating loose on the operating table.

1. **Select 4 Tokens & Amputate:** Just like NYT Connections, group the 4 tokens on the 4×4 grid that belong to one category and hit **🩸 AMPUTATE** (or press `Enter`).
2. **Declassify the Hidden SLM Probes:** While you solve, the 4 cortical sentence probes and token probabilities are encrypted. Each time you amputate a category, its cortical lobe banner rips open to reveal the exact **sentence completion probe** (for example: `"the best ice cream flavor is [MASK]."`) along with the language model's **softmax probability (`%`)** and **raw logit (`z`)** for each of the 4 words!
3. **One Token Off Warning:** If 3 of your 4 selected tokens belong to the same lobe, the cranial monitor flashes **"3/4 SYNAPSES ALIGNED — ONE TOKEN OFF!"**.
4. **Two Operating Modes:**
   - **LIVE SLM SURGERY (On-Device Procedural):** Every seed rolls 4 mutually compatible cortical lobes and a prompt variant for each, runs 4 real-time ONNX forward passes in your browser, and lets the language model select its own highest-margin tokens from a 331-word pool on the fly.
   - **PRE-DISSECTED WARD (`?lite=1`):** Instant verified logit tables so you can play immediately while the 22 MB ONNX model boots (or on ultra-low-memory mobile tabs).
5. **🔬 Live Lobotomy Lab:** Open the Lab at any time to type **any custom sentence** and inspect both the current board's rankings and the model's raw top predictions across its entire **30,000-token vocabulary** — or write **4 custom sentences of your own** and let the SLM synthesize a playable 4×4 board from scratch!

---

## Development & Testing

```bash
npm ci                                   # Install Playwright, ESLint, TypeScript, serve
npm run serve                            # Start local server on http://localhost:3000
npm run lint                             # ESLint + TypeScript + Ruff + puzzle validator
npm run test:unit                        # Python unit + ONNX verification tests
npm test                                 # Full Playwright E2E suite (Chromium, Firefox, WebKit, Mobile Safari)
```

---

## Privacy

**POST-TRAINING** runs 100% on your device. There is no backend server, no analytics, no cookies, and no telemetry. Only `localStorage` and `sessionStorage` are used to save your seed, streak, and UI preferences locally. See [privacy.html](privacy.html).

## License

Code: [MIT](LICENSE). Model weights (`models/Xenova/albert-base-v2/`): Apache 2.0.
