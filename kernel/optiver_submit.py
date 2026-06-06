# Optiver Realized Volatility Prediction — self-contained submission kernel.
# Reads /kaggle/input/optiver-realized-volatility-prediction/, writes submission.csv.
# Order-book microstructure features + sub-windows + LightGBM optimising RMSPE (1/y^2 weights).
import glob
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

# auto-detect the mounted competition data dir (mount name can vary)
print("/kaggle/input ->", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")
_c = glob.glob("/kaggle/input/**/train.csv", recursive=True)
DATA = os.path.dirname(_c[0]) if _c else "/kaggle/input/optiver-realized-volatility-prediction"
print("Using DATA =", DATA)
SUB_WINDOWS = (150, 300, 450)
N_EST = 1400          # ~ median best-iteration from local 5-fold CV (CV RMSPE ~0.2162)


def realized_volatility(s):
    return float(np.sqrt(np.nansum(s.to_numpy() ** 2)))


def _book_raw(df):
    df["wap1"] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (df.bid_size1 + df.ask_size1)
    df["wap2"] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (df.bid_size2 + df.ask_size2)
    df["log_ret1"] = df.groupby("time_id")["wap1"].transform(lambda s: np.log(s).diff())
    df["log_ret2"] = df.groupby("time_id")["wap2"].transform(lambda s: np.log(s).diff())
    df["wap_balance"] = (df.wap1 - df.wap2).abs()
    df["spread"] = df.ask_price1 / df.bid_price1 - 1.0
    df["price_spread2"] = (df.ask_price2 - df.bid_price2) / ((df.ask_price2 + df.bid_price2) / 2)
    df["total_volume"] = df.bid_size1 + df.bid_size2 + df.ask_size1 + df.ask_size2
    df["volume_imbalance"] = ((df.ask_size1 + df.ask_size2) - (df.bid_size1 + df.bid_size2)).abs()
    return df


def _agg_book(df):
    aggs = {"log_ret1": [realized_volatility], "log_ret2": [realized_volatility],
            "wap_balance": ["mean", "std"], "spread": ["mean", "std", "max"],
            "price_spread2": ["mean"], "total_volume": ["mean", "sum", "std"],
            "volume_imbalance": ["mean", "sum"], "seconds_in_bucket": ["count"]}
    g = df.groupby("time_id").agg(aggs)
    g.columns = ["book_" + "_".join(c) for c in g.columns]
    return g


def _trade_raw(df):
    df = df.copy()
    df["log_ret"] = df.groupby("time_id")["price"].transform(lambda s: np.log(s).diff())
    df["amount"] = df.price * df["size"]
    return df


def _agg_trade(df):
    aggs = {"log_ret": [realized_volatility], "size": ["sum", "mean", "max"],
            "order_count": ["sum", "mean"], "amount": ["sum"], "seconds_in_bucket": ["count"]}
    g = df.groupby("time_id").agg(aggs)
    g.columns = ["trade_" + "_".join(c) for c in g.columns]
    return g


def _windowed(df, fn):
    out = fn(df)
    for s in SUB_WINDOWS:
        out = out.join(fn(df[df.seconds_in_bucket >= s]).add_suffix(f"_w{s}"), how="left")
    return out


def features_for_stock(stock_id, split):
    bpath = f"{DATA}/book_{split}.parquet/stock_id={stock_id}"
    tpath = f"{DATA}/trade_{split}.parquet/stock_id={stock_id}"
    if not glob.glob(bpath):
        return None
    feats = _windowed(_book_raw(pd.read_parquet(bpath)), _agg_book)
    if glob.glob(tpath):
        feats = feats.join(_windowed(_trade_raw(pd.read_parquet(tpath)), _agg_trade), how="left")
    feats["stock_id"] = stock_id
    feats = feats.reset_index()
    feats["row_id"] = feats.stock_id.astype(str) + "-" + feats.time_id.astype(str)
    return feats


def build(split, stock_ids):
    rows = [features_for_stock(int(s), split) for s in stock_ids]
    df = pd.concat([r for r in rows if r is not None], ignore_index=True)
    for c in ("book_log_ret1_realized_volatility", "trade_log_ret_realized_volatility",
              "book_total_volume_sum"):
        if c in df.columns:
            df[c + "_tid_mean"] = df.groupby("time_id")[c].transform("mean")
    return df


train_csv = pd.read_csv(f"{DATA}/train.csv")
tr = build("train", sorted(train_csv.stock_id.unique()))
tr = tr.merge(train_csv[["stock_id", "time_id", "target"]], on=["stock_id", "time_id"])

test_csv = pd.read_csv(f"{DATA}/test.csv")
te = build("test", sorted(test_csv.stock_id.unique()))

ID = ["row_id", "stock_id", "time_id", "target"]
feat_cols = [c for c in tr.columns if c not in ID]
tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
te = te.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = tr["target"].to_numpy()

params = dict(objective="rmse", n_estimators=N_EST, learning_rate=0.03, num_leaves=127,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
              min_child_samples=100, reg_lambda=5.0, verbose=-1)
model = lgb.LGBMRegressor(**params).fit(tr[feat_cols], y, sample_weight=1.0 / np.square(y))

pred = model.predict(te[feat_cols])
sub = pd.DataFrame({"row_id": te["row_id"], "target": np.clip(pred, 1e-6, None)})
# align to sample_submission row order/ids
ssub = pd.read_csv(f"{DATA}/sample_submission.csv")
sub = ssub[["row_id"]].merge(sub, on="row_id", how="left").fillna(y.mean())
sub.to_csv("submission.csv", index=False)
print("submission.csv written:", sub.shape)
print(sub.head())
