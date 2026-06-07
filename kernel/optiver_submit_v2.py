# Optiver Realized Volatility Prediction — V2 self-contained submission kernel (GPU).
# Pipeline: rich book+trade bucket features -> cross-sectional (stock/time) aggregates ->
# KNN neighbour smoothing (time & stock axes) -> blend of LightGBM + CatBoost + MLP(stock emb).
# Reads competition data under /kaggle/input, writes submission.csv.
import glob
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("/kaggle/input ->", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")
_c = glob.glob("/kaggle/input/**/train.csv", recursive=True)
DATA = os.path.dirname(_c[0]) if _c else "/kaggle/input/optiver-realized-volatility-prediction"
print("Using DATA =", DATA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)

SUB_WINDOWS = (0, 150, 300, 450)
# Filled from local GroupKFold (honest: predicts unseen time_ids = the private set).
# LGB best_iter under GroupKFold was ~193 -> few trees generalise; more trees just
# memorise training time_ids (that is why v1's 1400-tree fit over-fit to 0.236).
W_LGB, W_CAT, W_NN = 0.70, 0.20, 0.10
N_LGB, N_CAT = 250, 2000
NN_SEEDS = (0, 1, 2)
NN_EPOCHS = 50

# ----------------------------------------------------------------- bucket features
def _book_raw(df):
    bs1, bs2, as1, as2 = df.bid_size1, df.bid_size2, df.ask_size1, df.ask_size2
    bp1, bp2, ap1, ap2 = df.bid_price1, df.bid_price2, df.ask_price1, df.ask_price2
    df["wap1"] = (bp1 * as1 + ap1 * bs1) / (bs1 + as1)
    df["wap2"] = (bp2 * as2 + ap2 * bs2) / (bs2 + as2)
    tot_bid, tot_ask = bs1 + bs2, as1 + as2
    df["wap"] = (bp1 * tot_ask + ap1 * tot_bid) / (tot_bid + tot_ask)
    for c in ("wap1", "wap2", "wap"):
        lr = np.log(df[c]).groupby(df.time_id).diff()
        df[f"sq_{c}"] = lr * lr
        df[f"q_{c}"] = df[f"sq_{c}"] ** 2
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


_BOOK_SUM = ["sq_wap1", "sq_wap2", "sq_wap", "q_wap1"]
_BOOK_AGG = {"wap_balance": ["mean", "std"], "spread": ["mean", "std", "max"],
             "price_spread": ["mean"], "price_spread2": ["mean"], "bid_spread": ["mean"],
             "ask_spread": ["mean"], "total_volume": ["mean", "sum", "std", "max"],
             "volume_imbalance": ["mean", "sum"], "imbalance_ratio": ["mean", "std"],
             "seconds_in_bucket": ["count"]}


def _agg_book(df):
    g = df.groupby("time_id")
    s = g[_BOOK_SUM].sum()
    out = pd.DataFrame(index=s.index)
    out["rv1"], out["rv2"] = np.sqrt(s["sq_wap1"]), np.sqrt(s["sq_wap2"])
    out["rv"], out["rq1"] = np.sqrt(s["sq_wap"]), s["q_wap1"]
    rest = g.agg(_BOOK_AGG); rest.columns = ["_".join(c) for c in rest.columns]
    out = out.join(rest); out.columns = ["book_" + c for c in out.columns]
    return out


def _trade_raw(df):
    df = df.copy()
    lr = np.log(df.price).groupby(df.time_id).diff()
    df["sq_price"] = lr * lr
    df["amount"] = df.price * df["size"]
    df["size_per_order"] = df["size"] / df["order_count"]
    return df


_TRADE_AGG = {"size": ["sum", "mean", "max", "std"], "order_count": ["sum", "mean", "max"],
              "amount": ["sum", "mean"], "size_per_order": ["mean", "max"],
              "seconds_in_bucket": ["count"]}


def _agg_trade(df):
    g = df.groupby("time_id")
    out = pd.DataFrame(index=g.size().index)
    out["rv"] = np.sqrt(g["sq_price"].sum())
    rest = g.agg(_TRADE_AGG); rest.columns = ["_".join(c) for c in rest.columns]
    out = out.join(rest); out.columns = ["trade_" + c for c in out.columns]
    return out


def _windowed(df, fn):
    parts = []
    for s in SUB_WINDOWS:
        sub = df if s == 0 else df[df.seconds_in_bucket >= s]
        a = fn(sub)
        parts.append(a.add_suffix(f"_w{s}") if s else a)
    return parts[0].join(parts[1:], how="left")


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


_CS_BASE = ["book_rv1", "book_rv", "book_rv1_w300", "book_rv1_w450", "trade_rv",
            "trade_rv_w300", "book_total_volume_sum", "book_spread_mean",
            "book_imbalance_ratio_mean", "trade_size_sum", "trade_seconds_in_bucket_count"]


def add_cross_sectional(df):
    base = [c for c in _CS_BASE if c in df.columns]
    for stat in ("mean", "std"):
        df = df.join(df.groupby("time_id")[base].transform(stat).add_suffix(f"_tid_{stat}"))
    for stat in ("mean", "std"):
        df = df.join(df.groupby("stock_id")[base].transform(stat).add_suffix(f"_sid_{stat}"))
    return df


PIVOT_FEAT = "book_rv1"
SMOOTH_FEATS = ("book_rv1", "book_rv", "trade_rv", "book_total_volume_sum")


def _pivot(df, col):
    m = df.pivot(index="time_id", columns="stock_id", values=col)
    return m.fillna(m.median())


def _z(m):
    z = (m - np.nanmean(m, 0, keepdims=True)) / (np.nanstd(m, 0, keepdims=True) + 1e-9)
    return np.nan_to_num(z)


def add_knn_features(df, k_time=20, k_stock=10):
    smooth = [c for c in SMOOTH_FEATS if c in df.columns]
    base = _pivot(df, PIVOT_FEAT)
    tids, sids = base.index.to_numpy(), base.columns.to_numpy()
    new = {}
    if len(tids) > 2:
        Zt = _z(np.log1p(base.to_numpy()))
        k = min(k_time, len(tids) - 1)
        idx_t = NearestNeighbors(n_neighbors=k + 1).fit(Zt).kneighbors(Zt, return_distance=False)[:, 1:]
        for f in smooth:
            M = _pivot(df, f).reindex(index=tids, columns=sids).to_numpy()
            sm = pd.DataFrame(M[idx_t].mean(1), index=tids, columns=sids).stack()
            sm.index.names = ["time_id", "stock_id"]; new[f"knnT_{f}"] = sm
    if len(sids) > 2:
        Zs = _z(np.log1p(base.T.to_numpy()))
        ks = min(k_stock, len(sids) - 1)
        idx_s = NearestNeighbors(n_neighbors=ks + 1).fit(Zs).kneighbors(Zs, return_distance=False)[:, 1:]
        for f in smooth:
            M = _pivot(df, f).reindex(index=tids, columns=sids).T.to_numpy()
            sm = pd.DataFrame(M[idx_s].mean(1), index=sids, columns=tids).T.stack()
            sm.index.names = ["time_id", "stock_id"]; new[f"knnS_{f}"] = sm
    if new:
        df = df.merge(pd.DataFrame(new).reset_index(), on=["time_id", "stock_id"], how="left")
        if f"knnT_{PIVOT_FEAT}" in df.columns:
            df[f"{PIVOT_FEAT}_knnT_ratio"] = df[PIVOT_FEAT] / (df[f"knnT_{PIVOT_FEAT}"] + 1e-9)
    return df


def build(split, stock_ids):
    rows = [features_for_stock(int(s), split) for s in stock_ids]
    df = pd.concat([r for r in rows if r is not None], ignore_index=True)
    df = add_cross_sectional(df)
    return add_knn_features(df)


# --------------------------------------------------------------------------- NN
def rmspe_loss(p, t):
    return torch.sqrt(torch.mean(((t - p) / t) ** 2))


class TabularMLP(nn.Module):
    def __init__(self, n_num, n_stocks, emb=24, hidden=(384, 192, 96), p=0.25):
        super().__init__()
        self.emb = nn.Embedding(n_stocks, emb)
        dims = [n_num + emb] + list(hidden)
        b = []
        for a, c in zip(dims[:-1], dims[1:]):
            b += [nn.Linear(a, c), nn.SiLU(), nn.BatchNorm1d(c), nn.Dropout(p)]
        self.body = nn.Sequential(*b); self.head = nn.Linear(dims[-1], 1)

    def forward(self, x, s):
        return self.head(self.body(torch.cat([x, self.emb(s)], 1))).squeeze(1)


def train_nn_full(Xnum, stock_idx, y, n_stocks, Xtest, stock_test, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xnum.mean(0), Xnum.std(0) + 1e-6
    Xn = np.clip((Xnum - mu) / sd, -10, 10).astype(np.float32)
    Xt = np.clip((Xtest - mu) / sd, -10, 10).astype(np.float32)
    ds = TensorDataset(torch.tensor(Xn), torch.tensor(stock_idx), torch.tensor(y))
    dl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)
    model = TabularMLP(Xnum.shape[1], n_stocks).to(DEVICE)
    with torch.no_grad():
        model.head.bias.fill_(float(np.mean(y)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NN_EPOCHS)
    for ep in range(NN_EPOCHS):
        model.train()
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); rmspe_loss(model(xb, sb), yb).backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xt).to(DEVICE),
                     torch.tensor(stock_test).to(DEVICE)).cpu().numpy()


# --------------------------------------------------------------------------- main
train_csv = pd.read_csv(f"{DATA}/train.csv")
tr = build("train", sorted(train_csv.stock_id.unique()))
tr = tr.merge(train_csv[["stock_id", "time_id", "target"]], on=["stock_id", "time_id"])
te = build("test", sorted(pd.read_csv(f"{DATA}/test.csv").stock_id.unique()))

ID = ["row_id", "stock_id", "time_id", "target"]
feat_cols = [c for c in tr.columns if c not in ID]
tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
te = te.replace([np.inf, -np.inf], np.nan).fillna(0.0)
for c in feat_cols:
    if c not in te.columns:
        te[c] = 0.0
y = tr["target"].to_numpy()
w = 1.0 / np.square(y)
print(f"train {tr.shape}  test {te.shape}  feats={len(feat_cols)}")

lgb_params = dict(objective="rmse", n_estimators=N_LGB, learning_rate=0.025, num_leaves=255,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.5, min_child_samples=200,
                  reg_lambda=8.0, reg_alpha=2.0, verbose=-1)
cat_params = dict(loss_function="RMSE", iterations=N_CAT, learning_rate=0.03, depth=8,
                  l2_leaf_reg=6.0, random_seed=42, verbose=0, allow_writing_files=False,
                  task_type="GPU" if DEVICE == "cuda" else "CPU")

p_lgb = lgb.LGBMRegressor(**lgb_params).fit(tr[feat_cols], y, sample_weight=w).predict(te[feat_cols])
p_cat = CatBoostRegressor(**cat_params).fit(tr[feat_cols], y, sample_weight=w).predict(te[feat_cols])

w_lgb, w_cat, w_nn = W_LGB, W_CAT, W_NN
try:
    codes, uniq = pd.factorize(tr["stock_id"])
    s_map = {s: i for i, s in enumerate(uniq)}
    stock_idx = codes.astype(np.int64)
    stock_test = te["stock_id"].map(s_map).fillna(0).astype(np.int64).to_numpy()
    Xnum = tr[feat_cols].to_numpy().astype(np.float32)
    Xtest = te[feat_cols].to_numpy().astype(np.float32)
    nn_preds = [train_nn_full(Xnum, stock_idx, y.astype(np.float32), len(uniq), Xtest, stock_test, s)
                for s in NN_SEEDS]
    p_nn = np.mean(nn_preds, axis=0)
    print("NN ok; preds:", float(np.mean(p_nn)))
except Exception as e:                    # NN must never sink the whole submission
    print("NN failed, falling back to GBM-only blend:", repr(e))
    p_nn = np.zeros(len(te))
    s = w_lgb + w_cat
    w_lgb, w_cat, w_nn = w_lgb / s, w_cat / s, 0.0

pred = w_lgb * p_lgb + w_cat * p_cat + w_nn * p_nn
sub = pd.DataFrame({"row_id": te["row_id"], "target": np.clip(pred, 1e-6, None)})
ssub = pd.read_csv(f"{DATA}/sample_submission.csv")
sub = ssub[["row_id"]].merge(sub, on="row_id", how="left").fillna(y.mean())
sub.to_csv("submission.csv", index=False)
print("submission.csv written:", sub.shape)
print(sub.head())
