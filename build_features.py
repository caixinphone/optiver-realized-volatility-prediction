"""Build feature tables (per-bucket + cross-sectional + KNN) for train and test.

  python build_features.py            # all stocks
  python build_features.py --limit 4  # quick smoke test on the first 4 stocks
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

import features as F
from knn_features import add_knn_features

OUT = F.DATA_DIR.parent / "output"


def build_raw(split: str, stock_ids) -> pd.DataFrame:
    rows = []
    for sid in tqdm(stock_ids, desc=f"{split} bucket-features"):
        f = F.features_for_stock(int(sid), split)
        if f is not None:
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def main(limit):
    OUT.mkdir(exist_ok=True)
    train_csv = pd.read_csv(F.DATA_DIR / "train.csv")
    stock_ids = sorted(train_csv.stock_id.unique())
    if limit:
        stock_ids = stock_ids[:limit]

    t0 = time.time()
    tr = build_raw("train", stock_ids)
    print(f"  raw bucket features: {tr.shape}  ({time.time()-t0:.0f}s)")
    tr = F.add_cross_sectional(tr)
    tr = add_knn_features(tr)
    tr = tr.merge(train_csv[["stock_id", "time_id", "target"]], on=["stock_id", "time_id"])
    suffix = f"_lim{limit}" if limit else ""
    tr.to_parquet(OUT / f"train_features{suffix}.parquet")
    print(f"train_features{suffix}: {tr.shape} -> done in {time.time()-t0:.0f}s")

    test_csv_path = F.DATA_DIR / "test.csv"
    if test_csv_path.exists() and not limit:
        test_ids = sorted(pd.read_csv(test_csv_path).stock_id.unique())
        te = build_raw("test", test_ids)
        te = F.add_cross_sectional(te)
        te = add_knn_features(te)
        te.to_parquet(OUT / "test_features.parquet")
        print(f"test_features: {te.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
