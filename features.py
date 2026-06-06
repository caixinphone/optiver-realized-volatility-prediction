"""Order-book + trade feature engineering for Optiver Realized Volatility Prediction.

Each (stock_id, time_id) bucket -> one feature row. The signature feature is the bucket's
own realized volatility (same formula as the target), plus sub-window versions that weight
the end of the bucket (closest to the window we must predict).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SUB_WINDOWS = (150, 300, 450)   # seconds_in_bucket >= s : recent-portion features


def realized_volatility(log_returns: pd.Series) -> float:
    return float(np.sqrt(np.nansum(log_returns.to_numpy() ** 2)))


def _wap(bid_p, bid_s, ask_p, ask_s):
    return (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s)


def _book_raw(df: pd.DataFrame) -> pd.DataFrame:
    df["wap1"] = _wap(df.bid_price1, df.bid_size1, df.ask_price1, df.ask_size1)
    df["wap2"] = _wap(df.bid_price2, df.bid_size2, df.ask_price2, df.ask_size2)
    df["log_ret1"] = df.groupby("time_id")["wap1"].transform(lambda s: np.log(s).diff())
    df["log_ret2"] = df.groupby("time_id")["wap2"].transform(lambda s: np.log(s).diff())
    df["wap_balance"] = (df.wap1 - df.wap2).abs()
    df["spread"] = df.ask_price1 / df.bid_price1 - 1.0
    df["price_spread2"] = (df.ask_price2 - df.bid_price2) / ((df.ask_price2 + df.bid_price2) / 2)
    df["total_volume"] = df.bid_size1 + df.bid_size2 + df.ask_size1 + df.ask_size2
    df["volume_imbalance"] = ((df.ask_size1 + df.ask_size2) - (df.bid_size1 + df.bid_size2)).abs()
    return df


def _agg_book(df: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "log_ret1": [realized_volatility],
        "log_ret2": [realized_volatility],
        "wap_balance": ["mean", "std"],
        "spread": ["mean", "std", "max"],
        "price_spread2": ["mean"],
        "total_volume": ["mean", "sum", "std"],
        "volume_imbalance": ["mean", "sum"],
        "seconds_in_bucket": ["count"],
    }
    g = df.groupby("time_id").agg(aggs)
    g.columns = ["book_" + "_".join(c) for c in g.columns]
    return g


def _trade_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_ret"] = df.groupby("time_id")["price"].transform(lambda s: np.log(s).diff())
    df["amount"] = df.price * df["size"]
    return df


def _agg_trade(df: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "log_ret": [realized_volatility],
        "size": ["sum", "mean", "max"],
        "order_count": ["sum", "mean"],
        "amount": ["sum"],
        "seconds_in_bucket": ["count"],
    }
    g = df.groupby("time_id").agg(aggs)
    g.columns = ["trade_" + "_".join(c) for c in g.columns]
    return g


def _windowed(df: pd.DataFrame, agg_fn) -> pd.DataFrame:
    out = agg_fn(df)
    for s in SUB_WINDOWS:
        sub = agg_fn(df[df.seconds_in_bucket >= s]).add_suffix(f"_w{s}")
        out = out.join(sub, how="left")
    return out


def features_for_stock(stock_id: int, split: str) -> pd.DataFrame | None:
    """Build all book+trade features for one stock; index = time_id."""
    book_path = DATA_DIR / f"book_{split}.parquet" / f"stock_id={stock_id}"
    trade_path = DATA_DIR / f"trade_{split}.parquet" / f"stock_id={stock_id}"
    if not book_path.exists():
        return None
    book = _windowed(_book_raw(pd.read_parquet(book_path)), _agg_book)
    feats = book
    if trade_path.exists():
        trade = _windowed(_trade_raw(pd.read_parquet(trade_path)), _agg_trade)
        feats = feats.join(trade, how="left")
    feats["stock_id"] = stock_id
    feats = feats.reset_index()                       # time_id back as column
    feats["row_id"] = feats["stock_id"].astype(str) + "-" + feats["time_id"].astype(str)
    return feats


def add_cross_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Market-regime features: per time_id mean of key vol features across stocks
    (target-independent, so no label leakage). stock_id stays categorical for the model."""
    key = [c for c in ("book_log_ret1_realized_volatility",
                       "trade_log_ret_realized_volatility",
                       "book_total_volume_sum") if c in df.columns]
    for c in key:
        df[c + "_tid_mean"] = df.groupby("time_id")[c].transform("mean")
    return df
