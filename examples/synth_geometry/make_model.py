#!/usr/bin/env python3
"""Generate the synthetic "model" — a deterministic weight matrix W.

Stands in for a real model's unembedding matrix (the *subject* of archetype-C
geometric analysis), without any real LLM or multi-GB weights — same "fake
model" spirit as the doubler's ``synthetic-noop-model``. Generated from a fixed
``model_seed`` so it is reproducible and content-addressable; written as JSON so
the pure-stdlib executor can load it from its ``--models`` dir.

    python make_model.py [--out <dir>] [--rows 256] [--cols 64] [--seed 20260601]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

MODEL_FILENAME = "W.json"


def build_matrix(rows: int, cols: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "model")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(args.rows, args.cols, args.seed)
    path = args.out / MODEL_FILENAME
    # Stable serialization so the file is content-addressable + reproducible.
    path.write_text(json.dumps(matrix, separators=(",", ":"), sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Wrote {args.rows}x{args.cols} weight matrix to {path}")
    print(f"model_sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
