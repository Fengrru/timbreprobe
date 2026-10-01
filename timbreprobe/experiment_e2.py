"""E2 -- navigability of the perceptual space.

Question: given a target embedding (produced by a held-out patch), can an
optimiser recover parameters that reach it, and how do the available search
strategies compare?

All methods start from the same point per target: the nearest *training*
corpus patch (Stage 1's retrieval initialisation), so every number below the
retrieval baseline is the value added by optimisation.  Targets are drawn from
the test split (held-out anchors included) and are never in the training pool.

Reported per method:
  * d_init           distance at the shared start point (== retrieval_nearest)
  * d_final          true distance after search (always evaluated with the real
                     renderer, also for surrogate-based methods)
  * ratio            d_final / d_init (lower is better), with bootstrap CI of
                     the median
  * success@0.5      fraction of targets with ratio < 0.5
  * param L1         hardened parameter recovery error vs the target patch
  * renders          true patch renders consumed (the resource the surrogate
                     exists to save)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from . import features
from . import schema as S
from .config import Config
from .corpus import load_corpus, split_indices
from .navigators import PerceptualSpace, nav_gradient, nav_reground, retrieval_distances, sep_cmaes
from .surrogate import fidelity_report, save_surrogate, train_surrogate


def _bootstrap_ci(x: np.ndarray, stat: str = "median", n: int = 2000, seed: int = 0
                  ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fn = np.median if stat == "median" else np.mean
    vals = [fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _param_l1(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return (S.harden(u_pred) - S.harden(u_true)).abs().mean(dim=1)


def _summarise(name: str, d_final: torch.Tensor, d_init: torch.Tensor,
               u_pred: torch.Tensor, u_target: torch.Tensor,
               renders: int, seconds: float, holdout: torch.Tensor,
               extra: dict | None = None) -> dict:
    ratio = (d_final / d_init.clamp(min=1e-8))
    lo, hi = _bootstrap_ci(ratio.numpy())
    out = {
        "method": name,
        "d_final_median": float(d_final.median()),
        "d_final_mean": float(d_final.mean()),
        "ratio_median": float(ratio.median()),
        "ratio_median_ci95": [lo, hi],
        "ratio_mean": float(ratio.mean()),
        "success@0.5": float((ratio < 0.5).float().mean()),
        "success@0.2": float((ratio < 0.2).float().mean()),
        "param_l1_mean": float(_param_l1(u_pred, u_target).mean()),
        "renders": int(renders),
        "seconds": round(float(seconds), 2),
    }
    for tag, mask in (("indist", ~holdout), ("holdout", holdout)):
        if bool(mask.any()):
            out[f"ratio_median_{tag}"] = float(ratio[mask].median())
            out[f"success@0.5_{tag}"] = float((ratio[mask] < 0.5).float().mean())
    if extra:
        out.update(extra)
    return out


def run_e2(cfg: Config, corpus_dir: Path, out_dir: Path,
           methods: list[str] | None = None) -> dict:
    methods = methods or ["retrieval", "grad_surrogate", "grad_surrogate_trust",
                          "grad_reground", "cmaes_true", "grad_true"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- corpus + perceptual space ---------------------------------------
    corpus, raw = load_corpus(corpus_dir)
    splits = split_indices(corpus, cfg, seed=cfg.seed)
    stats_t = features.fit_normalizer(torch.as_tensor(raw[splits["train"]]))
    stats = {"mean": stats_t["mean"].numpy(), "std": stats_t["std"].numpy()}
    z_all = (raw - stats["mean"]) / stats["std"]
    u_all = torch.as_tensor(corpus.u, dtype=torch.float32)
    z_all_t = torch.as_tensor(z_all, dtype=torch.float32)
    print(f"[e2] corpus {u_all.shape[0]} patches | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])} "
          f"(holdout-anchor {len(splits['holdout'])})", flush=True)

    # ---- surrogate (E3) ---------------------------------------------------
    model, _ = train_surrogate(corpus.u, z_all, splits["train"], splits["val"],
                               cfg, seed=cfg.seed)
    fid = []
    for s in ("val", "test", "holdout"):
        f = fidelity_report(model, corpus.u, z_all, splits[s], tag=s)
        fid.append(f)
        print(f"[e3] {f['split']:>8}: n={f['n']:5d} R2={f['r2_mean']:.3f} "
              f"cos={f['cosine']:.3f} dist_r={f['dist_pearson']:.3f} "
              f"dist_tau={f['dist_kendall']:.3f}", flush=True)
    save_surrogate(out_dir / "surrogate", model, stats,
                   {"fidelity": fid, "config": vars(cfg)})

    # ---- targets and shared start points ---------------------------------
    rng = np.random.default_rng(cfg.seed + 77)
    test_all = splits["test"]
    test_ho = np.array([i for i in test_all if corpus.holdout[i]], dtype=np.int64)
    test_id = np.array([i for i in test_all if not corpus.holdout[i]], dtype=np.int64)
    n_ho = min(cfg.n_targets // 2, len(test_ho))
    n_id = min(cfg.n_targets - n_ho, len(test_id))
    t_idx = np.concatenate([rng.choice(test_ho, n_ho, replace=False),
                            rng.choice(test_id, n_id, replace=False)])
    rng.shuffle(t_idx)
    holdout_mask = torch.as_tensor(corpus.holdout[t_idx])
    u_target = u_all[t_idx].clone()
    z_target = z_all_t[t_idx].clone()
    u_pool = u_all[splits["train"]].clone()
    z_pool = z_all_t[splits["train"]].clone()

    d_nearest, d_unigram = retrieval_distances(z_target, z_pool,
                                               draws=cfg.retrieval_draws, seed=cfg.seed)
    dmat = torch.cdist(z_target, z_pool)
    u_start = u_pool[dmat.argmin(dim=1)].clone()
    d_init = torch.linalg.vector_norm(z_target - z_pool[dmat.argmin(dim=1)], dim=1)
    print(f"[e2] targets={len(t_idx)} ({int(holdout_mask.sum())} holdout-anchor) "
          f"| d_init median={float(d_init.median()):.3f} "
          f"| best-of-{cfg.retrieval_draws} median={float(d_unigram.median()):.3f}", flush=True)

    space = PerceptualSpace(stats, cfg)
    results: dict[str, dict] = {}
    curves: dict[str, np.ndarray] = {}
    target_meta = {"idx": t_idx.tolist(),
                   "holdout": corpus.holdout[t_idx].tolist(),
                   "d_init": d_init.numpy(),
                   "u_target": u_target.numpy()}

    def _run(name: str, fn: Callable[[], tuple]):
        t0 = time.time()
        before = space.patch_renders
        u_pred, d_final, extra = fn()
        res = _summarise(name, d_final, d_init, u_pred, u_target,
                         space.patch_renders - before, time.time() - t0,
                         holdout_mask, extra)
        results[name] = res
        print(f"[e2] {name:>18}: ratio_med={res['ratio_median']:.3f} "
              f"succ@0.5={res['success@0.5']:.2f} renders={res['renders']} "
              f"({res['seconds']:.1f}s)", flush=True)
        return u_pred, d_final

    # ---- retrieval baselines (no renders, corpus already rendered) --------
    if "retrieval" in methods:
        nn_idx = dmat.argmin(dim=1)
        u_pred = u_pool[nn_idx]
        results["retrieval_nearest"] = _summarise(
            "retrieval_nearest", d_nearest, d_init, u_pred, u_target, 0, 0.0,
            holdout_mask)
        results["retrieval_unigram"] = _summarise(
            f"retrieval_unigram{cfg.retrieval_draws}", d_unigram, d_init, u_pred,
            u_target, 0, 0.0, holdout_mask)
        for k in ("retrieval_nearest", "retrieval_unigram"):
            print(f"[e2] {k:>18}: ratio_med={results[k]['ratio_median']:.3f} "
                  f"succ@0.5={results[k]['success@0.5']:.2f} renders=0", flush=True)

    # ---- surrogate gradient ----------------------------------------------
    if "grad_surrogate" in methods:
        def f_sur():
            u, traj = nav_gradient(u_start, z_target, space, cfg.steps_surrogate,
                                   cfg.lr_opt, model=model, log_every=cfg.log_every)
            d = space.dist(u, z_target)
            curves["grad_surrogate_surr"] = torch.stack([t[1] for t in traj]).numpy()
            return u, d, {"objective": "surrogate"}

        _run("grad_surrogate", f_sur)

    # ---- surrogate gradient with a trust region --------------------------
    if "grad_surrogate_trust" in methods:
        def f_sur_trust():
            u, traj = nav_gradient(u_start, z_target, space, cfg.steps_surrogate,
                                   cfg.lr_opt, model=model, log_every=cfg.log_every,
                                   trust_radius=cfg.trust_radius)
            d = space.dist(u, z_target)
            curves["grad_surrogate_trust_surr"] = torch.stack([t[1] for t in traj]).numpy()
            return u, d, {"objective": "surrogate", "trust_radius": cfg.trust_radius}

        _run("grad_surrogate_trust", f_sur_trust)

    # ---- re-grounded surrogate -------------------------------------------
    if "grad_reground" in methods:
        anchor_sel = rng.choice(splits["train"], size=min(256, len(splits["train"])),
                                replace=False)
        def f_reg():
            u, traj, renders = nav_reground(
                u_start, z_target, space, model, cfg,
                u_all[anchor_sel].clone(), z_all_t[anchor_sel].clone(),
                trust_radius=cfg.trust_radius)
            d = space.dist(u, z_target)
            curves["grad_reground_surr"] = torch.stack([t[1] for t in traj]).numpy()
            return u, d, {"objective": "surrogate+reground", "reground_rounds":
                          cfg.steps_surrogate // cfg.reground_every}

        _run("grad_reground", f_reg)

    # ---- true-gradient oracle --------------------------------------------
    if "grad_true" in methods:
        def f_true():
            u, traj = nav_gradient(u_start, z_target, space, cfg.steps_true,
                                   cfg.lr_opt, model=None, log_every=cfg.log_every)
            d = space.dist(u, z_target)
            curves["grad_true_true"] = torch.stack([t[1] for t in traj]).numpy()
            return u, d, {"objective": "true"}

        _run("grad_true", f_true)

    # ---- separable CMA-ES on the true objective --------------------------
    if "cmaes_true" in methods:
        pop = cfg.cmaes_pop
        z_rep = z_target.repeat_interleave(pop, dim=0)

        def f_cma(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] == z_target.shape[0]:
                return space.dist(x, z_target)
            assert x.shape[0] == z_rep.shape[0]
            return space.dist(x, z_rep)

        def f_cma_wrapped(x: torch.Tensor) -> torch.Tensor:
            # population rows are [M, pop, D] flattened -> target row = i // pop
            return f_cma(x)

        def f_cmaes():
            best_x, traj, best_f = sep_cmaes(u_start, f_cma_wrapped, cfg.cmaes_gens,
                                             pop, sigma0=cfg.cmaes_sigma,
                                             log_every=1, seed=cfg.seed)
            d = space.dist(best_x, z_target)
            curves["cmaes_true_true"] = torch.stack([t[1] for t in traj]).numpy()
            return best_x, d, {"objective": "true", "population": pop,
                               "generations": cfg.cmaes_gens}

        # note: the initial f_eval(u_start) inside sep_cmaes would use the
        # [M, D] branch; sep_cmaes is written to only evaluate populations, so
        # the wrapper above only ever sees [M*pop, D] rows.
        _run("cmaes_true", f_cmaes)

    # ---- persist ----------------------------------------------------------
    np.savez_compressed(out_dir / "targets.npz", **target_meta)
    np.savez_compressed(out_dir / "curves.npz",
                        **{k: v for k, v in curves.items()},
                        d_init=d_init.numpy())
    out = {"config": vars(cfg), "split_sizes": {k: int(len(v)) for k, v in splits.items()},
           "surrogate_fidelity": fid, "results": results}
    with open(out_dir / "e2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== E2 summary (median over targets, lower ratio is better) ===", flush=True)
    print(f"{'method':>20} {'ratio_med':>10} {'success@0.5':>12} {'renders':>9} {'sec':>8}")
    for k, r in results.items():
        print(f"{k:>20} {r['ratio_median']:>10.3f} {r['success@0.5']:>12.2f} "
              f"{r['renders']:>9d} {r['seconds']:>8.1f}")
    return out
