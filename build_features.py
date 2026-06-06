"""Build feature tables for train and test, save to output/{train,test}_features.parquet.

Usage:
  python build_features.py            # all stocks
  python build_features.py --limit 8  # quick subset for a smoke test
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

import features as F

OUT = F.DATA_DIR.parent / "output"


def build(split: str, stock_ids) -> pd.DataFrame:
    rows = []
    for sid in tqdm(stock_ids, desc=f"{split} features"):
        f = F.features_for_stock(int(sid), split)
        if f is not None:
            rows.append(f)
    df = pd.concat(rows, ignore_index=True)
    return F.add_cross_aggregates(df)


def main(limit):
    OUT.mkdir(exist_ok=True)
    train_csv = pd.read_csv(F.DATA_DIR / "train.csv")
    stock_ids = sorted(train_csv.stock_id.unique())
    if limit:
        stock_ids = stock_ids[:limit]

    tr = build("train", stock_ids)
    tr = tr.merge(train_csv[["stock_id", "time_id", "target"]], on=["stock_id", "time_id"])
    tr.to_parquet(OUT / "train_features.parquet")
    print(f"train_features: {tr.shape} -> {OUT/'train_features.parquet'}")

    test_csv_path = F.DATA_DIR / "test.csv"
    if test_csv_path.exists():
        test_ids = sorted(pd.read_csv(test_csv_path).stock_id.unique())
        te = build("test", test_ids)
        te.to_parquet(OUT / "test_features.parquet")
        print(f"test_features: {te.shape} -> {OUT/'test_features.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only first N stocks")
    main(ap.parse_args().limit)
