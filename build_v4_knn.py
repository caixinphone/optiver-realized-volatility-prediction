"""V4 = V2 bucket+cross-sectional features with a *thorough* KNN aggregation block,
replacing V2's basic 4-feature/mean-only KNN. This is the 1st-place "Nearest Neighbors"
booster done properly: many base features, neighbour mean AND std, larger k, both axes,
plus own/neighbour ratios. Recomputed from the V2 parquet (bucket cols already there).

  python build_v4_knn.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

OUT = Path(__file__).parent / "output"

PIVOT_FEAT = "book_rv1"
# features aggregated over neighbours (realized vols + liquidity/regime signals)
SMOOTH_FEATS = [
    "book_rv1", "book_rv", "book_rv2", "book_rv1_w300", "book_rv1_w450",
    "trade_rv", "trade_rv_w300", "book_total_volume_sum", "trade_size_sum",
    "trade_seconds_in_bucket_count",
]
K_TIME = 40
K_STOCK = 20


def _pivot(df, col):
    m = df.pivot(index="time_id", columns="stock_id", values=col)
    return m.fillna(m.median())


def _z(m):
    z = (m - np.nanmean(m, 0, keepdims=True)) / (np.nanstd(m, 0, keepdims=True) + 1e-9)
    return np.nan_to_num(z)


def add_knn_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.startswith(("knnT_", "knnS_"))
                          or c.endswith("_knnT_ratio")], errors="ignore")
    smooth = [c for c in SMOOTH_FEATS if c in df.columns]
    base = _pivot(df, PIVOT_FEAT)
    tids, sids = base.index.to_numpy(), base.columns.to_numpy()
    new = {}

    # time neighbours: mean + std over the k nearest time_ids
    Zt = _z(np.log1p(base.to_numpy()))
    kt = min(K_TIME, len(tids) - 1)
    idx_t = NearestNeighbors(n_neighbors=kt + 1).fit(Zt).kneighbors(Zt, return_distance=False)[:, 1:]
    for f in smooth:
        M = _pivot(df, f).reindex(index=tids, columns=sids).to_numpy()
        nb = M[idx_t]                                  # [time, k, stock]
        new[f"knnT_{f}_mean"] = pd.DataFrame(nb.mean(1), index=tids, columns=sids).stack()
        new[f"knnT_{f}_std"] = pd.DataFrame(nb.std(1), index=tids, columns=sids).stack()

    # stock neighbours: mean over the k nearest stocks
    Zs = _z(np.log1p(base.T.to_numpy()))
    ks = min(K_STOCK, len(sids) - 1)
    idx_s = NearestNeighbors(n_neighbors=ks + 1).fit(Zs).kneighbors(Zs, return_distance=False)[:, 1:]
    for f in smooth:
        M = _pivot(df, f).reindex(index=tids, columns=sids).T.to_numpy()
        nb = M[idx_s]                                  # [stock, k, time]
        new[f"knnS_{f}_mean"] = pd.DataFrame(nb.mean(1), index=sids, columns=tids).T.stack()

    for k in new:
        new[k].index.names = ["time_id", "stock_id"]
    knn = pd.DataFrame(new).reset_index()
    df = df.merge(knn, on=["time_id", "stock_id"], how="left")
    # own vs neighbour-mean ratios for the realized vols (regime-relative extremity)
    for f in ("book_rv1", "book_rv", "trade_rv"):
        if f in df.columns and f"knnT_{f}_mean" in df.columns:
            df[f"{f}_knnT_ratio"] = df[f] / (df[f"knnT_{f}_mean"] + 1e-9)
            df[f"{f}_knnS_ratio"] = df[f] / (df[f"knnS_{f}_mean"] + 1e-9)
    return df


def main():
    for split in ("train", "test"):
        p = OUT / f"{split}_features_v2.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if split == "test" and len(df) < 5:
            df.to_parquet(OUT / f"{split}_features_v4.parquet"); continue
        df = add_knn_v2(df)
        df.to_parquet(OUT / f"{split}_features_v4.parquet")
        nknn = len([c for c in df.columns if "knn" in c])
        print(f"{split}_features_v4: {df.shape}  knn cols={nknn}")


if __name__ == "__main__":
    main()
