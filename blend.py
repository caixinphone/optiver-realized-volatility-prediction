"""Combine saved OOF predictions (LGB / CatBoost / NN) and find RMSPE-optimal weights.

  python blend.py --cv kfold
Reports each model's OOF RMSPE plus the best non-negative simplex blend.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "output"


def rmspe(y, p):
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def best_weights(y, preds, step=0.05):
    """Grid search over the simplex (weights sum to 1, non-negative)."""
    n = len(preds)
    grid = np.arange(0, 1 + 1e-9, step)
    best, bw = 1e9, None
    for combo in itertools.product(grid, repeat=n - 1):
        if sum(combo) > 1 + 1e-9:
            continue
        w = list(combo) + [1 - sum(combo)]
        p = sum(wi * pi for wi, pi in zip(w, preds))
        r = rmspe(y, p)
        if r < best:
            best, bw = r, w
    return bw, best


def main(cv):
    y = np.load(OUT / f"oof_y_{cv}.npy")
    names, preds = [], []
    for n in ("lgb", "cat", "nn"):
        f = OUT / f"oof_{n}_{cv}.npy"
        if f.exists():
            names.append(n)
            preds.append(np.load(f))
            print(f"  {n:4s} OOF RMSPE = {rmspe(y, preds[-1]):.5f}")
    if len(preds) < 2:
        print("need >=2 model OOFs to blend"); return
    w, r = best_weights(y, preds)
    print(f"\nBest blend ({cv}): " + "  ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))
    print(f"BLEND OOF RMSPE = {r:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="kfold")
    main(ap.parse_args().cv)
