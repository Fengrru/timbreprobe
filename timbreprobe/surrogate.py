"""Learned patch -> embedding surrogate, and its fidelity metrics (E3).

The surrogate is the "acceleration" half of the oracle/acceleration pair: it
maps patch parameters to the perceptual embedding without rendering audio, so
an optimiser can take gradient steps at a tiny fraction of the cost of the
true renderer.  E3 measures how faithful it is -- R^2 per embedding dimension,
cosine agreement, and (the metric that actually matters for navigation) the
rank correlation between surrogate and true *distances* in embedding space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:  # scipy is present on the Stage 0 machine; numpy fallback below otherwise
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


class Surrogate(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


@dataclass
class SurrogateReport:
    train_mse: float
    val_mse: float
    val_r2_mean: float
    val_cosine: float
    dist_pearson: float
    dist_kendall: float
    history: list


def _euclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise L2 distance via the Gram trick.

    The obvious `(a[:, None] - b[None])` broadcast materialises an
    [N, M, D] temporary -- at N = 1200, D = 85 that is ~500 MB per call and
    three calls in a row thrashed the Stage-0 machine.  This form only ever
    allocates [N, M].
    """
    aa = (a * a).sum(axis=1)[:, None]
    bb = (b * b).sum(axis=1)[None, :]
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _pairwise_distance_agreement(z_true: np.ndarray, z_pred: np.ndarray,
                                 n_points: int = 900, seed: int = 0
                                 ) -> tuple[float, float]:
    """Correlation between true and surrogate pairwise distances."""
    rng = np.random.default_rng(seed)
    n = z_true.shape[0]
    idx = rng.choice(n, size=min(n_points, n), replace=False)
    a, b = z_true[idx], z_pred[idx]
    d_true = _euclid(a, a)
    d_pred = _euclid(b, b)
    iu = np.triu_indices(len(idx), k=1)
    dt, dp = d_true[iu], d_pred[iu]
    pearson = float(np.corrcoef(dt, dp)[0, 1])
    if _stats is not None and len(dt) > 20:
        sub = rng.choice(len(dt), size=min(20000, len(dt)), replace=False)
        kendall = float(_stats.kendalltau(dt[sub], dp[sub]).statistic)
    else:  # pragma: no cover
        kendall = float("nan")
    return pearson, kendall


def train_surrogate(u: np.ndarray, z: np.ndarray, idx_train: np.ndarray,
                    idx_val: np.ndarray, cfg, seed: int = 0
                    ) -> tuple[Surrogate, SurrogateReport]:
    """Train patch -> normalised-embedding regression."""
    torch.manual_seed(seed)
    dev = torch.device("cpu")

    def t(a):  # noqa: E731
        return torch.as_tensor(a, dtype=torch.float32, device=dev)

    xt, yt = t(u[idx_train]), t(z[idx_train])
    xv, yv = t(u[idx_val]), t(z[idx_val])

    model = Surrogate(u.shape[1], z.shape[1], hidden=cfg.hidden, depth=cfg.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    history = []
    n = xt.shape[0]
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for s in range(0, n, cfg.batch):
            sel = perm[s:s + cfg.batch]
            opt.zero_grad()
            loss = lossf(model(xt[sel]), yt[sel])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        model.eval()
        with torch.no_grad():
            val_loss = float(lossf(model(xv), yv))
        history.append({"epoch": epoch, "train_mse": total / n, "val_mse": val_loss})
        if epoch % 50 == 0 or epoch == cfg.epochs - 1:
            print(f"\r  [surrogate] epoch {epoch:4d} train {total / n:.5f} "
                  f"val {val_loss:.5f}", end="", flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    print(flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_v = model(xv)
        val_mse = float(lossf(pred_v, yv))
        train_mse = float(lossf(model(xt), yt))
        var = yv.var(dim=0, unbiased=False).clamp(min=1e-8)
        r2 = float((1.0 - ((pred_v - yv) ** 2).mean(dim=0) / var).mean())
        cos = float(torch.nn.functional.cosine_similarity(pred_v, yv, dim=1).mean())

    rep = SurrogateReport(
        train_mse=train_mse, val_mse=val_mse, val_r2_mean=r2, val_cosine=cos,
        dist_pearson=float("nan"), dist_kendall=float("nan"), history=history,
    )
    return model, rep


def fidelity_report(model: Surrogate, u: np.ndarray, z: np.ndarray,
                    idx: np.ndarray, tag: str = "val") -> dict[str, float]:
    """Full fidelity metrics on a split (R^2, cosine, distance agreement)."""
    with torch.no_grad():
        pred = model(torch.as_tensor(u[idx], dtype=torch.float32)).numpy()
    true = z[idx]
    mse = float(np.mean((pred - true) ** 2))
    var = true.var(axis=0)
    r2 = float(np.mean(1.0 - ((pred - true) ** 2).mean(axis=0) / np.clip(var, 1e-8, None)))
    cos = float(np.mean(np.sum(pred * true, axis=1) /
                        (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-8)))
    dp, dk = _pairwise_distance_agreement(true, pred)
    return {"split": tag, "n": int(len(idx)), "mse": mse, "r2_mean": r2,
            "cosine": cos, "dist_pearson": dp, "dist_kendall": dk}


def save_surrogate(path: Path, model: Surrogate, normalizer: dict,
                   report: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": model.net[0].in_features,
        "d_out": model.net[-1].out_features,
        "mean": normalizer["mean"],
        "std": normalizer["std"],
    }, path / "surrogate.pt")
    with open(path / "surrogate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


def load_surrogate(path: Path, hidden: int = 256, depth: int = 2
                   ) -> tuple[Surrogate, dict]:
    blob = torch.load(path / "surrogate.pt", weights_only=False)
    model = Surrogate(blob["d_in"], blob["d_out"], hidden=hidden, depth=depth)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, {"mean": blob["mean"], "std": blob["std"]}
