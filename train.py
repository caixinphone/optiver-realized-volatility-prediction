"""Train LGB + CatBoost, optimise RMSPE, report honest CV, blend.

  python train.py                          # KFold (optimistic on this comp)
  python train.py --cv group               # GroupKFold by time_id (honest generalisation)
  python train.py --models lgb --no-test   # LightGBM only

RMSPE is minimised by training an RMSE objective with sample_weight = 1/y^2.
OOF predictions are saved to output/oof_*.npy so the NN blend can reuse them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import lightgbm as lgb
from catboost import CatBoostRegressor

OUT = Path(__file__).parent / "output"
ID_COLS = ["row_id", "stock_id", "time_id", "target"]

LGB_PARAMS = dict(
    objective="rmse", n_estimators=3000, learning_rate=0.025, num_leaves=255,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
    min_child_samples=200, reg_lambda=8.0, reg_alpha=2.0, verbose=-1,
)
CAT_PARAMS = dict(
    loss_function="RMSE", iterations=3000, learning_rate=0.03, depth=8,
    l2_leaf_reg=6.0, random_seed=42, verbose=0, allow_writing_files=False,
)


def rmspe(y, yhat):
    return float(np.sqrt(np.mean(np.square((y - yhat) / y))))


def feval_rmspe(y_true, y_pred):
    return "RMSPE", rmspe(y_true, y_pred), False


def best_blend_weight(y, a, b):
    ws = np.linspace(0, 1, 101)
    scores = [rmspe(y, w * a + (1 - w) * b) for w in ws]
    i = int(np.argmin(scores))
    return float(ws[i]), scores[i]


def main(args):
    tr = pd.read_parquet(OUT / args.feats).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = tr["target"].to_numpy()
    feat_cols = [c for c in tr.columns if c not in ID_COLS]
    X = tr[feat_cols]
    groups = tr["time_id"].to_numpy()
    print(f"train rows={len(X):,}  features={len(feat_cols)}  cv={args.cv}")

    if args.cv == "group":
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X))

    do_lgb = args.models in ("lgb", "both")
    do_cat = args.models in ("cat", "both")
    oof_lgb, oof_cat = np.zeros(len(X)), np.zeros(len(X))
    lgb_iters, cat_iters = [], []
    for k, (tri, vai) in enumerate(splits, 1):
        w = 1.0 / np.square(y[tri])
        wv = 1.0 / np.square(y[vai])

        if do_lgb:
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(X.iloc[tri], y[tri], sample_weight=w,
                  eval_set=[(X.iloc[vai], y[vai])], eval_sample_weight=[wv],
                  eval_metric=feval_rmspe, callbacks=[lgb.early_stopping(150, verbose=False)])
            lgb_iters.append(m.best_iteration_ or LGB_PARAMS["n_estimators"])
            oof_lgb[vai] = m.predict(X.iloc[vai])

        if do_cat:
            c = CatBoostRegressor(**CAT_PARAMS)
            c.fit(X.iloc[tri], y[tri], sample_weight=w,
                  eval_set=(X.iloc[vai], y[vai]), early_stopping_rounds=150, use_best_model=True)
            cat_iters.append(c.get_best_iteration() or CAT_PARAMS["iterations"])
            oof_cat[vai] = c.predict(X.iloc[vai])

        msg = f"  fold {k}:"
        if do_lgb: msg += f" LGB={rmspe(y[vai], oof_lgb[vai]):.5f}"
        if do_cat: msg += f" CAT={rmspe(y[vai], oof_cat[vai]):.5f}"
        print(msg, flush=True)

    r_lgb = rmspe(y, oof_lgb) if do_lgb else None
    r_cat = rmspe(y, oof_cat) if do_cat else None
    if do_lgb:
        print(f"  LGB median best_iter = {int(np.median(lgb_iters))}")
    if do_cat:
        print(f"  CAT median best_iter = {int(np.median(cat_iters))}")
    if do_lgb and do_cat:
        w, r_blend = best_blend_weight(y, oof_lgb, oof_cat)
        print(f"\nOOF  LGB={r_lgb:.5f}  CAT={r_cat:.5f}  BLEND(w_lgb={w:.2f})={r_blend:.5f}")
    else:
        w = 1.0 if do_lgb else 0.0
        print(f"\nOOF  LGB={r_lgb}  CAT={r_cat}")

    if do_lgb: np.save(OUT / f"oof_lgb_{args.cv}.npy", oof_lgb)
    if do_cat: np.save(OUT / f"oof_cat_{args.cv}.npy", oof_cat)
    np.save(OUT / f"oof_y_{args.cv}.npy", y)

    if args.no_test or not (do_lgb and do_cat):
        return
    test_path = OUT / "test_features.parquet"
    if not test_path.exists():
        print("No test_features.parquet -- skip submission."); return
    te = pd.read_parquet(test_path).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_lgb, n_cat = max(int(np.median(lgb_iters)), 50), max(int(np.median(cat_iters)), 50)
    mf = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_lgb}).fit(X, y, sample_weight=1/np.square(y))
    cf = CatBoostRegressor(**{**CAT_PARAMS, "iterations": n_cat}).fit(X, y, sample_weight=1/np.square(y))
    pred = w * mf.predict(te[feat_cols]) + (1 - w) * cf.predict(te[feat_cols])
    sub = pd.DataFrame({"row_id": te["row_id"], "target": np.clip(pred, 1e-6, None)})
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"Wrote submission.csv ({len(sub):,})  LGB n_est={n_lgb} CAT n_est={n_cat}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="train_features.parquet")
    ap.add_argument("--cv", choices=["kfold", "group"], default="kfold")
    ap.add_argument("--models", choices=["lgb", "cat", "both"], default="both")
    ap.add_argument("--no-test", action="store_true")
    main(ap.parse_args())
