from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUZZLES_PATH = ROOT / "puzzles.json"
MODEL_DIR = ROOT / "models" / "Xenova" / "albert-base-v2"

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class PuzzleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))

    def test_schema_and_uniqueness(self) -> None:
        cats = self.data["categories"]
        self.assertGreaterEqual(len(cats), 40)
        seen = set()
        for cid, c in cats.items():
            self.assertEqual(cid, c["id"])
            self.assertEqual(len(c["words"]), 4)
            self.assertGreaterEqual(len(c["wordPool"]), 5)
            for p in c["prompts"]:
                self.assertIn("[MASK]", p)
            for we in c["wordPool"]:
                self.assertNotIn(we["word"], seen)
                seen.add(we["word"])

    def test_all_64_puzzles_have_strict_logit_separation(self) -> None:
        for p in self.data["puzzles"]:
            logits = p["fallbackLogits"]
            self.assertEqual(len(logits), 4)
            for i in range(4):
                row = logits[i]
                own = row[i * 4 : (i + 1) * 4]
                others = [row[j] for j in range(16) if not (i * 4 <= j < (i + 1) * 4)]
                self.assertGreater(
                    min(own),
                    max(others) + 1.0,
                    f"Puzzle {p['id']} lobe {i} lacks strict margin",
                )

    @unittest.skipUnless(HAS_ONNX, "onnxruntime + tokenizers not installed")
    def test_live_onnx_model_matches_puzzles(self) -> None:
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model_fp16.onnx"))
        mask_id = self.data["maskTokenId"]

        # Verify Case #001 (which includes ice_cream) directly on the live ONNX model
        p0 = self.data["puzzles"][0]
        self.assertIn("ice_cream", p0["categories"])
        cats = [self.data["categories"][cid] for cid in p0["categories"]]
        all_tids = [we["tokenId"] for c in cats for we in c["words"]]

        for ci, c in enumerate(cats):
            enc = tok.encode(c["prompt"])
            ids = np.array([enc.ids], dtype=np.int64)
            out = sess.run(
                None,
                {
                    "input_ids": ids,
                    "attention_mask": np.ones_like(ids),
                    "token_type_ids": np.zeros_like(ids),
                },
            )[0]
            idx = enc.ids.index(mask_id)
            logits = out[0, idx].astype(np.float32)
            row = [float(logits[tid]) for tid in all_tids]
            own = row[ci * 4 : (ci + 1) * 4]
            others = [row[j] for j in range(16) if not (ci * 4 <= j < (ci + 1) * 4)]
            self.assertGreater(min(own), max(others) + 1.0)


if __name__ == "__main__":
    unittest.main()
