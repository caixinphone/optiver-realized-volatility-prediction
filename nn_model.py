"""Tabular MLP with a stock_id embedding, trained directly on RMSPE.

Model diversity vs the GBMs is where most of the late-stage gain comes from in this
comp. Kept self-contained so the submission kernel can import the same code on GPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os

_OVERRIDE = os.environ.get("NN_DEVICE")
if _OVERRIDE:
    DEVICE = _OVERRIDE
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"            # Apple Metal GPU — faster locally, but BatchNorm can be off
else:
    DEVICE = "cpu"


def rmspe_loss(pred, target):
    return torch.sqrt(torch.mean(((target - pred) / target) ** 2))


def mspe_loss(pred, target):
    # same objective as RMSPE but no sqrt -> smoother, better-scaled gradients
    return torch.mean(((target - pred) / target) ** 2)


class TabularMLP(nn.Module):
    def __init__(self, n_num, n_stocks, emb_dim=24, hidden=(384, 192, 96), p=0.25):
        super().__init__()
        self.emb = nn.Embedding(n_stocks, emb_dim)
        dims = [n_num + emb_dim] + list(hidden)
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.SiLU(), nn.BatchNorm1d(b), nn.Dropout(p)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x_num, x_stock):
        h = self.body(torch.cat([x_num, self.emb(x_stock)], dim=1))
        return self.head(h).squeeze(1)


class _Scaler:
    """log1p the heavy-tailed non-negative columns (realized vols, volumes, counts) so the
    NN sees roughly-Gaussian inputs, then standardise + clip. GBMs don't need this; NNs do."""
    def fit(self, X):
        self.logcols = (np.nanmin(X, axis=0) >= 0) & (np.nanmax(X, axis=0) > 5 * (np.nanmean(X, axis=0) + 1e-9))
        Xl = self._log(X)
        self.mu = np.nanmean(Xl, axis=0)
        self.sd = np.nanstd(Xl, axis=0) + 1e-6
        return self

    def _log(self, X):
        X = X.copy()
        X[:, self.logcols] = np.log1p(np.clip(X[:, self.logcols], 0, None))
        return X

    def transform(self, X):
        return np.clip((self._log(X) - self.mu) / self.sd, -8, 8)


def _loaders(Xn, st, y, idx, bs, shuffle):
    ds = TensorDataset(torch.tensor(Xn[idx], dtype=torch.float32),
                       torch.tensor(st[idx], dtype=torch.long),
                       torch.tensor(y[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def train_fold(Xnum, stock_idx, y, tri, vai, n_stocks, *, epochs=60, bs=2048,
               lr=1e-3, patience=10, seed=0, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = _Scaler().fit(Xnum[tri])
    Xn = sc.transform(Xnum).astype(np.float32)

    model = TabularMLP(Xnum.shape[1], n_stocks).to(DEVICE)
    with torch.no_grad():
        model.head.bias.fill_(float(np.mean(y[tri])))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_dl = _loaders(Xn, stock_idx, y, tri, bs, True)
    va_x = torch.tensor(Xn[vai], dtype=torch.float32, device=DEVICE)
    va_s = torch.tensor(stock_idx[vai], dtype=torch.long, device=DEVICE)

    best, best_pred, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, sb, yb in tr_dl:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = rmspe_loss(model(xb, sb), yb)  # sqrt keeps the 1/y^2-weighted grads O(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tighter: kills rare divergence
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(va_x, va_s).cpu().numpy()
        score = float(np.sqrt(np.mean(((y[vai] - vp) / y[vai]) ** 2)))
        if score < best - 1e-5:
            best, best_pred, wait = score, vp, 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose:
            print(f"    ep{ep:02d} val RMSPE={score:.5f} (best {best:.5f})")
    return best_pred, best
