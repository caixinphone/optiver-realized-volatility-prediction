"""V6 = V2 + AR volatility features along the PRICE-reconstructed time order (stassl).

  python build_v6_priceorder.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import recover_order as RO

OUT = Path(__file__).parent / "output"


def main():
    t0 = time.time()
    train_csv = pd.read_csv(RO.DATA_DIR / "train.csv")
    stock_ids = sorted(train_csv.stock_id.unique())

    pm = RO.mean_price_matrix("train", stock_ids)
    print(f"price matrix {pm.shape}  ({time.time()-t0:.0f}s)")
    tid_pos = RO.recover_order(pm)
    print(f"recovered order for {len(tid_pos)} time_ids  ({time.time()-t0:.0f}s)")

    df = pd.read_parquet(OUT / "train_features_v2.parquet")
    df = RO.add_ar_features(df, tid_pos)
    df.to_parquet(OUT / "train_features_v6.parquet")
    new = [c for c in df.columns if c.startswith(("prev_", "next_", "roll5", "rv1_over", "prev_next")) or c == "tid_pos"]
    print(f"train_features_v6: {df.shape}  AR cols: {new}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
