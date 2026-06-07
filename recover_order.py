"""Recover time_id chronological order from the [time_id x stock] mean-PRICE matrix
(stassl's method), then build autoregressive volatility features along that order.

Why price, not volatility: each 10-min window is price-normalised to ~1.0, but the tiny
cross-stock co-movements of the average price form a smooth temporal manifold (the market
state drifts continuously). t-SNE 1D on that matrix recovers the order with good accuracy;
the realized-vol matrix has a much weaker temporal signal. Computed per split (train alone /
hidden test alone), so it transfers to the private test the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

DATA_DIR = Path(__file__).parent / "data"


def mean_price_matrix(split: str, stock_ids, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """[time_id x stock_id] matrix of the per-window average WAP."""
    cols = {}
    for sid in stock_ids:
        p = data_dir / f"book_{split}.parquet" / f"stock_id={sid}"
        if not p.exists():
            continue
        b = pd.read_parquet(p, columns=["time_id", "bid_price1", "ask_price1",
                                        "bid_size1", "ask_size1"])
        wap = (b.bid_price1 * b.ask_size1 + b.ask_price1 * b.bid_size1) / (b.bid_size1 + b.ask_size1)
        cols[sid] = wap.groupby(b.time_id).mean()
    return pd.DataFrame(cols)            # index=time_id, columns=stock_id


def recover_order(price_mat: pd.DataFrame, seed: int = 0, perplexity: int = 50) -> pd.Series:
    """Return a Series time_id -> position in reconstructed time (0..n-1)."""
    M = price_mat.to_numpy()
    M = np.where(np.isnan(M), np.nanmean(M, axis=0, keepdims=True), M)
    M = (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-12)
    n = len(price_mat)
    if n < 10:                          # too few (e.g. placeholder test) -> trivial order
        return pd.Series(np.arange(n), index=price_mat.index, name="tid_pos")
    coord = TSNE(n_components=1, perplexity=min(perplexity, (n - 1) // 3),
                 init="pca", random_state=seed).fit_transform(M).ravel()
    order = np.argsort(coord)
    pos = np.empty(n, dtype=int); pos[order] = np.arange(n)
    return pd.Series(pos, index=price_mat.index, name="tid_pos")


def add_ar_features(df: pd.DataFrame, tid_pos: pd.Series) -> pd.DataFrame:
    """Per stock, along reconstructed time: prev/next window RV & volume, local rolling mean."""
    df = df.join(tid_pos, on="time_id")
    df = df.sort_values(["stock_id", "tid_pos"])
    g = df.groupby("stock_id")
    for col, name in [("book_rv1", "rv1"), ("trade_rv", "trv"), ("book_total_volume_sum", "vol")]:
        if col in df.columns:
            df[f"prev_{name}"] = g[col].shift(1)
            df[f"next_{name}"] = g[col].shift(-1)
    if "book_rv1" in df.columns:
        roll = g["book_rv1"].shift(1).rolling(5, min_periods=1).mean()
        df["roll5_rv1"] = roll.reset_index(level=0, drop=True)
        df["rv1_over_roll5"] = df["book_rv1"] / (df["roll5_rv1"] + 1e-9)
        df["prev_next_mean_rv1"] = (df["prev_rv1"] + df["next_rv1"]) / 2
    return df.sort_index()
