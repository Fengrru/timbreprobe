"""Post-hoc diagnostics for an E2 run.

Three questions the discussion chapter depends on, all answered from the same
targets the E2 run used (read from its `targets.npz`):

1. **Where does the residual live?** After gradient navigation, decompose the
   remaining squared distance into the embedding's feature blocks.  If the
   residual concentrates in blocks that respond to *discrete* choices (harmonic
   and odd/even ratios) rather than in smooth spectral blocks, that is direct
   evidence for the "the plateau is a discrete-jump problem" reading.

2. **Do discrete jumps predict failure?** For each target, count how many
   discrete dimensions differ (hardened) between the *start* patch and the
   target patch, and correlate that with the achieved ratio.

3. **Does the monotone accept rule cost anything?** Re-run the identical search
   with `monotone=False` (every candidate accepted, adaptive step kept) at the
   same budget.  Monotone-only descent cannot cross a valley; a free walk can.

    python -m timbreprobe.cli diagnose --run out/e2_smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import features
from . import schema as S
from .config import Config
from .corpus import load_corpus, split_indices
from .navigators import PerceptualSpace, nav_gradient

_BLOCK_ITEMS = list(features.FEATURE_BLOCKS.items())


def _discrete_jumps(u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
    """Count discrete dimensions whose hardened choice index differs."""
    a, b = S.harden(u_a), S.harden(u_b)
    counts = torch.zeros(a.shape[0], dtype=torch.long)
    for i, sp in enumerate(S.SCHEMA):
        if sp.is_choice:
            n = sp.n_choices
            ia = (a[:, i] * (n - 1)).round().long()
            ib = (b[:, i] * (n - 1)).round().long()
            counts += (ia != ib).long()
    return counts


def run_diagnostics(run_dir: Path, corpus_dir: Path, cfg: Config) -> dict:
    run_dir = Path(run_dir)
    tg = np.load(run_dir / "targets.npz", allow_pickle=False)
    t_idx = torch.as_tensor(tg["idx"], dtype=torch.long)
    u_target = torch.as_tensor(tg["u_target"], dtype=torch.float32)
    holdout = torch.as_tensor(tg["holdout"])

    corpus, raw = load_corpus(corpus_dir)
    splits = split_indices(corpus, cfg, seed=cfg.seed)
    stats_t = features.fit_normalizer(torch.as_tensor(raw[splits["train"]]))
    stats = {"mean": stats_t["mean"].numpy(), "std": stats_t["std"].numpy()}
    z_all = torch.as_tensor((raw - stats["mean"]) / stats["std"], dtype=torch.float32)
    u_all = torch.as_tensor(corpus.u, dtype=torch.float32)

    z_target = z_all[t_idx].clone()
    u_pool, z_pool = u_all[splits["train"]].clone(), z_all[splits["train"]].clone()
    dm = torch.cdist(z_target, z_pool, compute_mode="donot_use_mm_for_euclid_dist")
    nn = dm.argmin(dim=1)
    u_start = u_pool[nn].clone()
    d_init = dm.min(dim=1).values

    space = PerceptualSpace(stats, cfg)
    out: dict = {"n_targets": int(u_target.shape[0])}

    # ---- (3) monotone vs free walk, identical budget ----------------------
    results = {}
    for tag, monotone in (("monotone", True), ("free", False)):
        space.patch_renders = 0
        u_end, traj = nav_gradient(u_start, z_target, space, cfg.steps_true,
                                   cfg.lr_opt, model=None, log_every=cfg.log_every,
                                   monotone=monotone)
        d_end = space.dist(u_end, z_target)
        results[tag] = {
            "u_end": u_end,
            "d_end": d_end,
            "ratio": d_end / d_init.clamp(min=1e-8),
            "renders": space.patch_renders,
        }
        print(f"[diagnose] {tag:>9}: ratio median "
              f"{float(results[tag]['ratio'].median()):.3f} "
              f"({space.patch_renders} renders)", flush=True)

    # ---- (1) residual decomposition on the monotone solution -------------
    z_end = space.embed(results["monotone"]["u_end"])
    delta2 = (z_end - z_target) ** 2                      # [M, 85]
    total = delta2.sum(dim=1).clamp(min=1e-12)
    shares = {name: float((delta2[:, blk].sum(dim=1) / total).mean())
              for name, blk in _BLOCK_ITEMS}
    out["residual_share"] = shares
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:5]
    print("[diagnose] residual by block (top 5): "
          + ", ".join(f"{k} {v:.2f}" for k, v in top), flush=True)

    # ---- (2) do discrete jumps predict failure? --------------------------
    jumps = _discrete_jumps(u_start, u_target)
    ratio = results["monotone"]["ratio"]
    from scipy import stats as st
    rho = float(st.spearmanr(jumps.numpy(), ratio.numpy()).statistic)
    by_count = {}
    for c in sorted(set(jumps.tolist())):
        m = (jumps == c).numpy()
        if m.sum():
            by_count[str(c)] = {"n": int(m.sum()), "ratio_median": float(ratio[m].median())}
    out["discrete_jumps"] = {"spearman": rho, "by_count": by_count,
                             "n_values": jumps.tolist()}

    # ---- (4) ablation comparison -----------------------------------------
    r_mono, r_free = results["monotone"]["ratio"], results["free"]["ratio"]
    out["ablation"] = {
        "monotone_ratio_median": float(r_mono.median()),
        "free_ratio_median": float(r_free.median()),
        "free_better_fraction": float((r_free < r_mono).float().mean()),
        "renders_per_run": int(results["monotone"]["renders"]),
        "monotone_ratio_holdout": float(r_mono[holdout].median()) if bool(holdout.any()) else None,
        "free_ratio_holdout": float(r_free[holdout].median()) if bool(holdout.any()) else None,
    }
    print(f"[diagnose] discrete-jump vs ratio spearman: {rho:.3f}", flush=True)
    print(f"[diagnose] free walk better on "
          f"{out['ablation']['free_better_fraction'] * 100:.0f}% of targets", flush=True)

    # ---- figure + json ----------------------------------------------------
    _figure(out, shares, jumps.numpy(), ratio.numpy(), r_free.numpy(),
            run_dir / "figures")
    (run_dir / "diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnose] wrote {run_dir / 'diagnostics.json'}", flush=True)
    return out


def _figure(out: dict, shares: dict, jumps: np.ndarray, ratio: np.ndarray,
            ratio_free: np.ndarray, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ax = axes[0]
    items = sorted(shares.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color="#7fb3d5")
    ax.set_xlabel("share of remaining squared distance")
    ax.set_title("(a) where the residual lives")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=.3)

    ax = axes[1]
    jitter = np.random.default_rng(0).normal(0, .06, size=len(jumps))
    ax.scatter(jumps + jitter, ratio, s=18, alpha=.75, color="#d9534f", zorder=3)
    med = {c: np.median(ratio[jumps == c]) for c in sorted(set(jumps.tolist()))}
    ax.plot(list(med.keys()), list(med.values()), "k.-", zorder=4, label="median")
    ax.axhline(1.0, color="k", linewidth=.8, linestyle="--")
    ax.set_xlabel("discrete dims differing from the start patch")
    ax.set_ylabel("ratio  $d_{final}/d_{init}$")
    ax.set_title(f"(b) discrete jumps vs outcome ($\\rho$={out['discrete_jumps']['spearman']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[2]
    ax.scatter(ratio, ratio_free, s=18, alpha=.75, color="#f0ad4e", zorder=3)
    lim = [0, max(1.1, float(max(ratio.max(), ratio_free.max())))]
    ax.plot(lim, lim, "k--", linewidth=.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ratio, monotone accept/reject")
    ax.set_ylabel("ratio, free walk (ablation)")
    ax.set_title(f"(c) ablation: free better on "
                 f"{out['ablation']['free_better_fraction'] * 100:.0f}% of targets")
    ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "diagnostics.png", dpi=140)
    plt.close(fig)
