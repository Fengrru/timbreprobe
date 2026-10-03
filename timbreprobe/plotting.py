"""Figures for an E2 run (method comparison + search trajectories).

    python -m timbreprobe.cli plot --run out/e2_smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHOD_ORDER = ["retrieval_nearest", "retrieval_unigram", "grad_surrogate",
                "grad_surrogate_trust", "grad_reground", "cmaes_true", "grad_true"]


def _resolve(order, methods):
    """Match method names to result keys, tolerating suffixed variants."""
    out = []
    for m in order:
        if m in methods:
            out.append(m)
        else:
            hits = sorted(k for k in methods if k.startswith(m))
            out.extend(hits[:1])
    return out

CURVE_LEGEND = (("grad_true_true", "grad_true (true obj)", "-"),
                ("cmaes_true_true", "cmaes_true (best-so-far)", "-"),
                ("grad_surrogate_surr", "grad_surrogate (surrogate obj)", "--"),
                ("grad_surrogate_trust_surr", "grad_surrogate_trust (surr obj)", "-."),
                ("grad_reground_surr", "grad_reground (surrogate obj)", ":"))


def _pareto(run: Path, fig_dir: Path, n_targets: int, pop: int, log_every: int) -> None:
    """Quality vs true renders: the two true-objective methods as lines, the
    others as points at their total render cost (all y-values are true
    distances, so the axes are comparable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = np.load(run / "curves.npz")
    res = json.loads((run / "e2_results.json").read_text(encoding="utf-8"))
    d_init = float(np.median(curves["d_init"]))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # lines: renders per logging step -> n*(2*i*log_every + 1) for grad_true,
    #        n*(1 + i*pop) for CMA-ES (initial evaluation + pop per generation)
    if "grad_true_true" in curves:
        arr = curves["grad_true_true"]
        x = n_targets * (2 * np.arange(arr.shape[0]) * log_every + 1)
        ax.plot(x, np.median(arr, axis=1), "-", color="#d9534f",
                label="grad_true (oracle)", linewidth=1.9)
    if "cmaes_true_true" in curves:
        arr = curves["cmaes_true_true"]
        x = n_targets * (1 + np.arange(arr.shape[0]) * pop)
        ax.plot(x, np.median(arr, axis=1), "-", color="#f0ad4e",
                label="cmaes_true (best-so-far)", linewidth=1.9)

    # points: surrogate methods (and their final true distances) from the JSON
    for key, colour, label in (("grad_surrogate", "#7fb3d5", "grad_surrogate"),
                               ("grad_surrogate_trust", "#a9cbe3", "grad_surrogate_trust"),
                               ("grad_reground", "#9b8fd0", "grad_reground"),
                               ("retrieval_unigram", "#888888", "retrieval_unigram64")):
        r = res["results"].get(key)
        if r:
            ax.scatter([max(r["renders"], 2)], [r["d_final_median"]], s=42,
                       color=colour, zorder=4,
                       label=f"{label} ({r['renders']} renders)")
    ax.axhline(d_init, color="k", linestyle="--", linewidth=1,
               label=f"retrieval start ({d_init:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("true renders consumed (log)")
    ax.set_ylabel("median true distance to target")
    ax.set_title("E2: quality per render")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "e2_pareto.png", dpi=140)
    plt.close(fig)


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
    order = _resolve(METHOD_ORDER, methods) or list(methods)
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

    # ---- 3. quality vs render budget -------------------------------------
    tg = np.load(run / "targets.npz")
    n_targets = int(tg["d_init"].shape[0])
    pop = int(res["results"].get("cmaes_true", {}).get("population", 10))
    log_every = int(res["config"].get("log_every", 5))
    _pareto(run, fig_dir, n_targets, pop, log_every)
    return fig_dir
