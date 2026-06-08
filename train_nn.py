"""Run the MLP (stock-embedding) CV; save OOF aligned to the GBM folds for blending.

Uses the same fold split as train.py so the OOF arrays line up.
  python train_nn.py --cv group         # honest GroupKFold
  python train_nn.py --epochs 25        # faster pass
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import nn_model as NN

OUT = Path(__file__).parent / "output"
ID_COLS = ["row_id", "stock_id", "time_id", "target"]


def rmspe(y, yhat):
    return float(np.sqrt(np.mean(np.square((y - yhat) / y))))


def main(args):
    tr = pd.read_parquet(OUT / args.feats).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = tr["target"].to_numpy().astype(np.float32)
    feat_cols = [c for c in tr.columns if c not in ID_COLS]
    Xnum = tr[feat_cols].to_numpy().astype(np.float32)

    codes, uniq = pd.factorize(tr["stock_id"])
    stock_idx = codes.astype(np.int64)
    n_stocks = len(uniq)
    print(f"rows={len(y):,} num_feats={len(feat_cols)} stocks={n_stocks} device={NN.DEVICE}")

    if args.cv == "group":
        splits = list(GroupKFold(n_splits=5).split(Xnum, y, tr["time_id"].to_numpy()))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(Xnum))

    oof = np.zeros(len(y), dtype=np.float32)
    done = np.zeros(len(y), dtype=bool)
    for k, (tri, vai) in enumerate(splits, 1):
        if args.folds and k > args.folds:
            break
        pred, score = NN.train_fold(Xnum, stock_idx, y, tri, vai, n_stocks,
                                    epochs=args.epochs, seed=k)
        oof[vai] = pred; done[vai] = True
        print(f"  fold {k}: NN RMSPE={score:.5f}", flush=True)
    print(f"\nOOF  NN={rmspe(y[done], oof[done]):.5f}  (over {done.sum():,} rows)")
    if not args.folds:
        np.save(OUT / f"oof_nn_{args.cv}.npy", oof)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="train_features.parquet")
    ap.add_argument("--cv", choices=["kfold", "group"], default="kfold")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=0, help="limit #folds for a quick probe (0=all)")
    main(ap.parse_args())
