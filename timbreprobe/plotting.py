"""Figures for an E2 run (method comparison + search trajectories).

    python -m timbreprobe.cli plot --run out/e2_smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHOD_ORDER = ["retrieval_nearest", "retrieval_unigram64", "grad_surrogate",
                "grad_surrogate_trust", "grad_reground", "cmaes_true", "grad_true"]

CURVE_LEGEND = (("grad_true_true", "grad_true (true obj)", "-"),
                ("cmaes_true_true", "cmaes_true (best-so-far)", "-"),
                ("grad_surrogate_surr", "grad_surrogate (surrogate obj)", "--"),
                ("grad_surrogate_trust_surr", "grad_surrogate_trust (surr obj)", "-."),
                ("grad_reground_surr", "grad_reground (surrogate obj)", ":"))


def make_figures(run_dir: Path) -> Path:
    """Write e2_ratios.png and e2_curves.png into <run_dir>/figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_dir)
    res = json.loads((run / "e2_results.json").read_text(encoding="utf-8"))
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    methods = res["results"]

    # ---- 1. final ratio per method --------------------------------------
    order = [m for m in METHOD_ORDER if m in methods] or list(methods)
    ratios = [methods[m]["ratio_median"] for m in order]
    err_lo = [methods[m]["ratio_median"] - methods[m]["ratio_median_ci95"][0] for m in order]
    err_hi = [methods[m]["ratio_median_ci95"][1] - methods[m]["ratio_median"] for m in order]
    colors = ["#999"] * 2 + ["#7fb3d5"] * 3 + ["#f0ad4e", "#d9534f"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(order)), ratios, yerr=[err_lo, err_hi], capsize=4,
           color=colors[:len(order)])
    ax.axhline(1.0, color="k", alpha=.4, linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("median  d_final / d_init   (lower = better)")
    ax.set_title("E2: distance reduction over the shared retrieval start")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "e2_ratios.png", dpi=140)
    plt.close(fig)

    # ---- 2. search trajectories ------------------------------------------
    curves = np.load(run / "curves.npz")
    d_init = curves["d_init"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, label, style in CURVE_LEGEND:
        if key not in curves:
            continue
        arr = curves[key]                     # [logged steps, n_targets]
        ax.plot(range(arr.shape[0]), np.median(arr, axis=1), style,
                label=label, linewidth=1.8)
    ax.axhline(float(np.median(d_init)), color="k", alpha=.4, linestyle="--",
               label="start (retrieval init)")
    ax.set_xlabel("logged step / generation")
    ax.set_ylabel("median distance to target (objective as labelled)")
    ax.set_title("E2: search trajectories")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "e2_curves.png", dpi=140)
    plt.close(fig)
    return fig_dir
