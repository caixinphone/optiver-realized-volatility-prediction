"""Train LightGBM to minimise RMSPE, KFold CV, write submission.

RMSPE = sqrt(mean(((y-yhat)/y)^2)). Minimising MSE with sample_weight = 1/y^2 is exactly
minimising MSPE, so we train an RMSE objective with those weights.

Usage: python train.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import lightgbm as lgb

OUT = Path(__file__).parent / "output"
ID_COLS = ["row_id", "stock_id", "time_id", "target"]

LGB_PARAMS = dict(
    objective="rmse", n_estimators=2000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
    min_child_samples=100, reg_lambda=5.0, verbose=-1,
)


def rmspe(y, yhat):
    return float(np.sqrt(np.mean(np.square((y - yhat) / y))))


def feval_rmspe(y_true, y_pred):           # sklearn API: (y_true, y_pred)
    return "RMSPE", rmspe(y_true, y_pred), False


def main():
    tr = pd.read_parquet(OUT / "train_features.parquet").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = tr["target"].to_numpy()
    feat_cols = [c for c in tr.columns if c not in ID_COLS]
    X = tr[feat_cols]
    print(f"train rows={len(X):,}  features={len(feat_cols)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    best_iters = []
    for k, (tri, vai) in enumerate(kf.split(X), 1):
        w = 1.0 / np.square(y[tri])
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X.iloc[tri], y[tri], sample_weight=w,
              eval_set=[(X.iloc[vai], y[vai])], eval_sample_weight=[1.0 / np.square(y[vai])],
              eval_metric=feval_rmspe,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters.append(m.best_iteration_ or LGB_PARAMS["n_estimators"])
        oof[vai] = m.predict(X.iloc[vai])
        print(f"  fold {k}: RMSPE={rmspe(y[vai], oof[vai]):.5f}")
    print(f"\nOOF RMSPE = {rmspe(y, oof):.5f}  (lower is better; ~0.20 is a solid score)")

    test_path = OUT / "test_features.parquet"
    if not test_path.exists():
        print("No test_features.parquet -- skip submission."); return
    n_est = max(int(np.median(best_iters)), 50)
    model = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_est}).fit(
        X, y, sample_weight=1.0 / np.square(y))
    te = pd.read_parquet(test_path).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred = model.predict(te[feat_cols])
    sub = pd.DataFrame({"row_id": te["row_id"], "target": pred})
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"Wrote {OUT/'submission.csv'} ({len(sub):,} rows)  LGB n_est={n_est}")


if __name__ == "__main__":
    main()
