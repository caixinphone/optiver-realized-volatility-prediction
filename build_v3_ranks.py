"""V3 = V2 + rank-within-time_id features (the 1st-place "Nearest Neighbors" trick).

Ranking a feature within its time_id removes the market-wide absolute level at that
instant -- the part a model can only *memorise* per time_id (and that does NOT transfer
to the unseen private time_ids) -- and keeps the cross-sectional *relative position*,
which generalises. This is what shrinks the KFold<->GroupKFold gap and lifts private.

Computed from the existing V2 columns, so no raw rebuild needed.
  python build_v3_ranks.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "output"

# drift-prone / regime features worth ranking within each time_id
RANK_FEATS = [
    "book_rv1", "book_rv", "book_rv2", "book_rv1_w300", "book_rv1_w450",
    "trade_rv", "trade_rv_w300",
    "book_total_volume_sum", "book_total_volume_mean",
    "trade_size_sum", "trade_seconds_in_bucket_count", "trade_order_count_sum",
    "book_spread_mean", "book_imbalance_ratio_mean", "book_wap_balance_mean",
]


def add_time_ranks(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in RANK_FEATS if c in df.columns]
    g = df.groupby("time_id")
    ranks = {f"{c}_trank": g[c].rank(pct=True) for c in feats}
    return df.join(pd.DataFrame(ranks))


def main():
    for split in ("train", "test"):
        p = OUT / f"{split}_features_v2.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = add_time_ranks(df)
        df.to_parquet(OUT / f"{split}_features_v3.parquet")
        print(f"{split}_features_v3: {df.shape}  (+{len([c for c in df.columns if c.endswith('_trank')])} rank feats)")


if __name__ == "__main__":
    main()
