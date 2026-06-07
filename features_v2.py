"""V2 feature engineering for Optiver Realized Volatility Prediction.

Two layers:
  1. Per-(stock_id, time_id) bucket features  -> `features_for_stock`
     Richer than v1: wap1/wap2/combined-wap realized vol + realized quarticity,
     two-level spreads, signed order-book imbalance, trade `tau` / avg trade size,
     and recency sub-windows. Log returns are vectorised (no python lambda).
  2. Cross-sectional aggregates                -> `add_cross_sectional`
     For a curated set of base features, aggregate by stock_id (typical level of
     a name) and by time_id (market-wide regime). Uses ONLY input features, never
     the target -> no fold leakage, available identically at test time.

KNN-by-time_id neighbour features live in `knn_features.py` (the big lever).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
# 0 == whole bucket; the rest weight the recent end (closest to the predicted window)
SUB_WINDOWS = (0, 150, 300, 450)


# ----------------------------------------------------------------------------- book
def _book_raw(df: pd.DataFrame) -> pd.DataFrame:
    bs1, bs2, as1, as2 = df.bid_size1, df.bid_size2, df.ask_size1, df.ask_size2
    bp1, bp2, ap1, ap2 = df.bid_price1, df.bid_price2, df.ask_price1, df.ask_price2

    df["wap1"] = (bp1 * as1 + ap1 * bs1) / (bs1 + as1)
    df["wap2"] = (bp2 * as2 + ap2 * bs2) / (bs2 + as2)
    # combined wap weighted by total size on each side (uses both levels)
    tot_bid, tot_ask = bs1 + bs2, as1 + as2
    df["wap"] = (bp1 * tot_ask + ap1 * tot_bid) / (tot_bid + tot_ask)

    for c in ("wap1", "wap2", "wap"):
        lr = np.log(df[c]).groupby(df.time_id).diff()
        df[f"lr_{c}"] = lr
        df[f"sq_{c}"] = lr * lr           # for realized volatility (sum then sqrt)
        df[f"q_{c}"] = df[f"sq_{c}"] ** 2  # for realized quarticity

    df["wap_balance"] = (df.wap1 - df.wap2).abs()
    df["spread"] = ap1 / bp1 - 1.0
    df["price_spread"] = (ap1 - bp1) / ((ap1 + bp1) / 2)
    df["price_spread2"] = (ap2 - bp2) / ((ap2 + bp2) / 2)
    df["bid_spread"] = bp1 / bp2 - 1.0
    df["ask_spread"] = ap1 / ap2 - 1.0
    df["total_volume"] = tot_bid + tot_ask
    df["volume_imbalance"] = (tot_ask - tot_bid).abs()
    df["imbalance_ratio"] = (tot_bid - tot_ask) / (tot_bid + tot_ask)
    return df


_BOOK_SUM = ["sq_wap1", "sq_wap2", "sq_wap", "q_wap1"]  # -> realized vol / quarticity
_BOOK_AGG = {
    "wap_balance": ["mean", "std"],
    "spread": ["mean", "std", "max"],
    "price_spread": ["mean"],
    "price_spread2": ["mean"],
    "bid_spread": ["mean"],
    "ask_spread": ["mean"],
    "total_volume": ["mean", "sum", "std", "max"],
    "volume_imbalance": ["mean", "sum"],
    "imbalance_ratio": ["mean", "std"],
    "seconds_in_bucket": ["count"],
}


def _agg_book(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("time_id")
    sums = g[_BOOK_SUM].sum()
    out = pd.DataFrame(index=sums.index)
    out["rv1"] = np.sqrt(sums["sq_wap1"])
    out["rv2"] = np.sqrt(sums["sq_wap2"])
    out["rv"] = np.sqrt(sums["sq_wap"])
    out["rq1"] = sums["q_wap1"]
    rest = g.agg(_BOOK_AGG)
    rest.columns = ["_".join(c) for c in rest.columns]
    out = out.join(rest)
    out.columns = ["book_" + c for c in out.columns]
    return out


# ----------------------------------------------------------------------------- trade
def _trade_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lr = np.log(df.price).groupby(df.time_id).diff()
    df["sq_price"] = lr * lr
    df["amount"] = df.price * df["size"]
    df["size_per_order"] = df["size"] / df["order_count"]
    return df


_TRADE_SUM = ["sq_price"]
_TRADE_AGG = {
    "size": ["sum", "mean", "max", "std"],
    "order_count": ["sum", "mean", "max"],
    "amount": ["sum", "mean"],
    "size_per_order": ["mean", "max"],
    "seconds_in_bucket": ["count"],  # tau = number of trade ticks
}


def _agg_trade(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("time_id")
    out = pd.DataFrame(index=g.size().index)
    out["rv"] = np.sqrt(g["sq_price"].sum())
    rest = g.agg(_TRADE_AGG)
    rest.columns = ["_".join(c) for c in rest.columns]
    out = out.join(rest)
    out.columns = ["trade_" + c for c in out.columns]
    return out


def _windowed(df: pd.DataFrame, agg_fn) -> pd.DataFrame:
    parts = []
    for s in SUB_WINDOWS:
        sub = df if s == 0 else df[df.seconds_in_bucket >= s]
        a = agg_fn(sub)
        if s:
            a = a.add_suffix(f"_w{s}")
        parts.append(a)
    return parts[0].join(parts[1:], how="left")


def features_for_stock(stock_id: int, split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    bpath = data_dir / f"book_{split}.parquet" / f"stock_id={stock_id}"
    tpath = data_dir / f"trade_{split}.parquet" / f"stock_id={stock_id}"
    if not bpath.exists():
        return None
    feats = _windowed(_book_raw(pd.read_parquet(bpath)), _agg_book)
    if tpath.exists():
        feats = feats.join(_windowed(_trade_raw(pd.read_parquet(tpath)), _agg_trade), how="left")
    feats["stock_id"] = stock_id
    feats = feats.reset_index()
    feats["row_id"] = feats.stock_id.astype(str) + "-" + feats.time_id.astype(str)
    return feats


# -------------------------------------------------------------------- cross-sectional
# Base features to aggregate across the two axes. Realized vols + key liquidity signals.
_CS_BASE = [
    "book_rv1", "book_rv", "book_rv1_w300", "book_rv1_w450",
    "trade_rv", "trade_rv_w300",
    "book_total_volume_sum", "book_spread_mean", "book_imbalance_ratio_mean",
    "trade_size_sum", "trade_seconds_in_bucket_count",
]


def add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stock (typical level) and per-time_id (market regime) aggregates of input
    features. Leak-free: target is never used; identical computation at test time."""
    base = [c for c in _CS_BASE if c in df.columns]
    # market regime: how this bucket compares to the whole market at the same time_id
    g_t = df.groupby("time_id")[base]
    for stat in ("mean", "std"):
        agg = g_t.transform(stat)
        df = df.join(agg.add_suffix(f"_tid_{stat}"))
    # per-stock typical level (and dispersion) across all its time_ids
    g_s = df.groupby("stock_id")[base]
    for stat in ("mean", "std"):
        agg = g_s.transform(stat)
        df = df.join(agg.add_suffix(f"_sid_{stat}"))
    return df
