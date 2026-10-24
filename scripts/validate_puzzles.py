#!/usr/bin/env python3
"""Validate puzzles.json and categories_corpus.json schema and invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUZZLES_PATH = ROOT / "puzzles.json"
CORPUS_PATH = ROOT / "scripts" / "categories_corpus.json"
MODEL_PATH = ROOT / "models" / "Xenova" / "albert-base-v2" / "onnx" / "model_fp16.onnx"


def main() -> int:
    if not PUZZLES_PATH.exists():
        print(f"ERROR: Missing {PUZZLES_PATH}", file=sys.stderr)
        return 1
    if not CORPUS_PATH.exists():
        print(f"ERROR: Missing {CORPUS_PATH}", file=sys.stderr)
        return 1
    if not MODEL_PATH.exists():
        print(f"ERROR: Missing ONNX model {MODEL_PATH}", file=sys.stderr)
        return 1

    data = json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))
    assert data.get("version") == 2, f"Unexpected version: {data.get('version')}"
    assert data.get("modelId") == "Xenova/albert-base-v2"
    assert data.get("dtype") == "fp16"
    assert data.get("maskToken") == "[MASK]"

    categories = data.get("categories", {})
    assert len(categories) >= 40, f"Expected >= 40 categories, got {len(categories)}"

    seen_words = set()
    for cid, cat in categories.items():
        assert cat["id"] == cid
        assert "[MASK]" in cat["prompt"], f"Prompt missing [MASK] in {cid}"
        assert len(cat.get("prompts", [])) >= 3, f"Expected >= 3 prompts in {cid}"
        for p in cat["prompts"]:
            assert "[MASK]" in p, f"Prompt variant missing [MASK] in {cid}: {p}"
        words = cat.get("words", [])
        assert len(words) == 4, f"Expected 4 canonical words in {cid}, got {len(words)}"
        pool = cat.get("wordPool", [])
        assert len(pool) >= 5, f"Expected >= 5 pool words in {cid}, got {len(pool)}"
        for we in pool:
            w = we["word"]
            assert w not in seen_words, f"Duplicate word across categories: {w}"
            seen_words.add(w)
            assert isinstance(we["tokenId"], int) and we["tokenId"] > 0

    puzzles = data.get("puzzles", [])
    assert len(puzzles) >= 64, f"Expected >= 64 fallback puzzles, got {len(puzzles)}"
    for p in puzzles:
        cats = p["categories"]
        assert len(cats) == 4 and len(set(cats)) == 4
        logits = p["fallbackLogits"]
        assert len(logits) == 4
        for rowIdx, row in enumerate(logits):
            assert len(row) == 16
            own_slice = row[rowIdx * 4 : (rowIdx + 1) * 4]
            other_vals = [row[i] for i in range(16) if not (rowIdx * 4 <= i < (rowIdx + 1) * 4)]
            assert min(own_slice) > max(other_vals), (
                f"Puzzle {p['id']} row {rowIdx} ({cats[rowIdx]}) own min {min(own_slice)} <= other max {max(other_vals)}"
            )

    print(
        f"OK: {len(categories)} lobes, {len(seen_words)} unique single-token words, {len(puzzles)} verified surgical cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
