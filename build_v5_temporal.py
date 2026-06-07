"""V5 = V2 + reconstructed-time-order temporal features (1st-place style).

Per-window normalisation removes price drift, so we recover an *approximate* global
time order from the realized-volatility manifold: t-SNE compresses the
[time_id x stock] log-RV matrix to 1D; sorting by that coordinate gives an order in
which neighbouring time_ids share a volatility regime. Along that order we add, per
stock, the realized vol of the previous / next window and a local rolling mean --
the autoregressive vol-clustering signal a per-bucket model cannot otherwise see.

Honest test: does it lower GroupKFold below the 0.213 plateau?
  python build_v5_temporal.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

OUT = Path(__file__).parent / "output"
RV = "book_rv1"


def _pivot(df, col):
    m = df.pivot(index="time_id", columns="stock_id", values=col)
    return m.fillna(m.median())


def add_temporal(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    base = _pivot(df, RV)
    tids = base.index.to_numpy()
    X = np.log1p(base.to_numpy())
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    X = np.nan_to_num(X)
    coord = TSNE(n_components=1, perplexity=50, init="pca", random_state=seed,
                 n_iter=1000).fit_transform(X).ravel()
    order = np.argsort(coord)                       # position in reconstructed time
    pos = np.empty_like(order); pos[order] = np.arange(len(order))
    tid_pos = pd.Series(pos, index=tids, name="tid_pos")

    df = df.join(tid_pos, on="time_id")
    df = df.sort_values(["stock_id", "tid_pos"])
    g = df.groupby("stock_id")
    for col, name in [(RV, "rv1"), ("book_total_volume_sum", "vol")]:
        df[f"prev_{name}"] = g[col].shift(1)
        df[f"next_{name}"] = g[col].shift(-1)
    df["roll3_rv1"] = g[RV].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # ratio of own RV to the local temporal average
    df["rv1_over_roll3"] = df[RV] / (df["roll3_rv1"] + 1e-9)
    return df.sort_index()


def main():
    df = pd.read_parquet(OUT / "train_features_v2.parquet")
    df = add_temporal(df)
    df.to_parquet(OUT / "train_features_v5.parquet")
    new = [c for c in df.columns if c in ("tid_pos", "prev_rv1", "next_rv1", "prev_vol",
                                          "next_vol", "roll3_rv1", "rv1_over_roll3")]
    print(f"train_features_v5: {df.shape}  temporal cols: {new}")


if __name__ == "__main__":
    main()
