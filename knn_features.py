"""Nearest-neighbour aggregation features — the single biggest lever in this comp.

Idea: represent each time_id by its cross-stock realized-volatility vector, and each
stock by its cross-time vector. Time_ids (resp. stocks) whose vectors are close behave
similarly. Averaging a stock's *current-window* realized vol over its nearest neighbours
denoises the signal and injects market-regime / peer information that a per-bucket model
cannot see. Uses ONLY input features (never the target) -> leak-free, computed identically
on train+test together at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# the feature whose [time_id x stock_id] matrix defines "similarity"
PIVOT_FEAT = "book_rv1"
# features smoothed over neighbours (kept small: this is the costly part)
SMOOTH_FEATS = ("book_rv1", "book_rv", "trade_rv", "book_total_volume_sum")


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    m = df.pivot(index="time_id", columns="stock_id", values=col)
    return m.fillna(m.median())


def _zscore_cols(m: np.ndarray) -> np.ndarray:
    mu = np.nanmean(m, axis=0, keepdims=True)
    sd = np.nanstd(m, axis=0, keepdims=True) + 1e-9
    z = (m - mu) / sd
    return np.nan_to_num(z)


def add_knn_features(df: pd.DataFrame, k_time: int = 20, k_stock: int = 10) -> pd.DataFrame:
    """Append neighbour-mean (and ratio) features along the time and stock axes."""
    smooth = [c for c in SMOOTH_FEATS if c in df.columns]
    pivot_feat = PIVOT_FEAT if PIVOT_FEAT in df.columns else smooth[0]

    # ---- time_id neighbours: each time_id is a vector over stocks --------------
    base = _pivot(df, pivot_feat)                      # [time_id x stock_id]
    tids, sids = base.index.to_numpy(), base.columns.to_numpy()
    Zt = _zscore_cols(np.log1p(base.to_numpy()))
    k = min(k_time, len(tids) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zt)
    idx_t = nn.kneighbors(Zt, return_distance=False)[:, 1:]   # drop self -> [n_time, k]

    new = {}
    for f in smooth:
        M = _pivot(df, f).reindex(index=tids, columns=sids).to_numpy()  # [time, stock]
        nbr_mean = M[idx_t].mean(axis=1)                               # [time, stock]
        sm = pd.DataFrame(nbr_mean, index=tids, columns=sids).stack()
        sm.index.names = ["time_id", "stock_id"]
        new[f"knnT_{f}"] = sm

    # ---- stock neighbours: each stock is a vector over time_ids ----------------
    baseS = base.T                                     # [stock_id x time_id]
    Zs = _zscore_cols(np.log1p(baseS.to_numpy()))
    ks = min(k_stock, len(sids) - 1)
    nns = NearestNeighbors(n_neighbors=ks + 1).fit(Zs)
    idx_s = nns.kneighbors(Zs, return_distance=False)[:, 1:]   # [n_stock, ks]
    for f in smooth:
        M = _pivot(df, f).reindex(index=tids, columns=sids).T.to_numpy()  # [stock, time]
        nbr_mean = M[idx_s].mean(axis=1)                                  # [stock, time]
        sm = pd.DataFrame(nbr_mean, index=sids, columns=tids).T.stack()
        sm.index.names = ["time_id", "stock_id"]
        new[f"knnS_{f}"] = sm

    knn = pd.DataFrame(new).reset_index()
    out = df.merge(knn, on=["time_id", "stock_id"], how="left")
    # ratio of own value to its time-neighbour mean: how extreme is this bucket now
    if f"knnT_{pivot_feat}" in out.columns:
        out[f"{pivot_feat}_knnT_ratio"] = out[pivot_feat] / (out[f"knnT_{pivot_feat}"] + 1e-9)
    return out
