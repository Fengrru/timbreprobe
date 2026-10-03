"""Stage 0 command line.

    python -m timbreprobe.cli sanity   [--preset smoke]   # fast correctness checks
    python -m timbreprobe.cli corpus   --preset smoke     # build corpus + embeddings
    python -m timbreprobe.cli e2       --preset smoke     # run the E2 experiment
    python -m timbreprobe.cli all      --preset smoke     # corpus + E2
    python -m timbreprobe.cli e1       --preset smoke     # E1 listening-test material
    python -m timbreprobe.cli e1-analyze --responses res1.json,res2.json
    python -m timbreprobe.cli plot     --run out/e2_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import corpus as CORP
from . import features, synth
from . import schema as S
from .navigators import sep_cmaes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEFAULT_THREADS = 1


def _set_threads() -> None:
    """Thread count matters enormously on the Stage 0 machine.

    Measured on a 4-core Windows box under mixed load (1T vs 4T):
        MLP step      65 ms  vs 681 ms
        render B=64   2.11 s vs 5.03 s
        fwd+bwd B=64  6.91 s vs 31.85 s
    i.e. intra-op thread synchronisation costs far more than the parallelism
    buys for these tensor shapes.  Override with STAGE0_THREADS if your machine
    behaves normally.
    """
    import os
    n = int(os.environ.get("STAGE0_THREADS", DEFAULT_THREADS))
    torch.set_num_threads(n)
    return n


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def render_corpus(u: np.ndarray, cfg: C.Config, tag: str = "") -> np.ndarray:
    """Render patches and return raw (un-normalised) features [N, FRAW]."""
    ut = torch.as_tensor(u, dtype=torch.float32)
    out = []
    t0 = time.time()
    n = ut.shape[0]
    with torch.no_grad():
        for s in range(0, n, cfg.chunk_b):
            chunk = ut[s:s + cfg.chunk_b]
            w = synth.render(chunk, sr=cfg.sr, dur=cfg.dur, chunk_b=cfg.chunk_b)
            z = features.embed(w, synth.reference_f0(chunk), sr=cfg.sr)
            out.append(z.numpy())
            done = min(s + cfg.chunk_b, n)
            print(f"\r  render {tag} {done}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)
    print(flush=True)
    return np.concatenate(out, axis=0)


def save_demo_wavs(u: np.ndarray, cfg: C.Config, path: Path,
                   names: Sequence[str] | None = None, dur: float = 0.6) -> None:
    import soundfile as sf
    path.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        w = synth.render(torch.as_tensor(u, dtype=torch.float32), sr=cfg.sr,
                         dur=dur, chunk_b=cfg.chunk_b).numpy()
    for i, sig in enumerate(w):
        name = names[i] if names is not None else f"patch_{i:03d}"
        sig = sig / max(1e-6, float(np.abs(sig).max())) * 0.9
        sf.write(path / f"{name}.wav", sig.astype(np.float32), cfg.sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# sanity
# ---------------------------------------------------------------------------

def cmd_sanity(args) -> None:
    n = _set_threads()
    print(f"[threads] torch.set_num_threads({n})")
    torch.manual_seed(0)
    from scipy import stats as st
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

    # --- renderer basics ---------------------------------------------------
    u = torch.rand(8, S.D)
    w1, w2 = synth.render(u), synth.render(u)
    check("render is deterministic", bool(torch.equal(w1, w2)))
    rms = float(w1.pow(2).mean(1).sqrt().mean())
    check("render is RMS-normalised", abs(rms - synth.TARGET_RMS) < 1e-6, f"rms={rms:.4f}")
    check("render is finite", bool(torch.isfinite(w1).all()))

    # --- gradient path -----------------------------------------------------
    ug = u.clone().requires_grad_(True)
    z = features.embed(synth.render(ug, chunk_b=None), synth.reference_f0(ug))
    z.pow(2).mean().backward()
    g = ug.grad
    nz = (g.abs() > 1e-12).sum(0)
    check("gradients finite (render->features)", bool(torch.isfinite(g).all()))
    check("gradients reach every dim", int((nz == 0).sum()) == 0,
          f"zero dims={[S.NAMES[i] for i in range(S.D) if nz[i] == 0]}")

    # --- semantic monotonicity --------------------------------------------
    sweep = torch.linspace(0.05, 0.95, 9)
    batch = S.default_patch().repeat(len(sweep), 1)
    batch[:, S.IDX["filter_cutoff"]] = sweep
    zb = features.embed(synth.render(batch), synth.reference_f0(batch))
    rho_cut = float(st.spearmanr(sweep.numpy(), zb[:, 64].numpy()).statistic)
    check("cutoff -> spectral centroid monotone", rho_cut > 0.9, f"spearman={rho_cut:.3f}")

    res = torch.linspace(0.05, 0.95, 9)
    batch = S.default_patch().repeat(len(res), 1)
    batch[:, S.IDX["filter_res"]] = res
    zr = features.embed(synth.render(batch), synth.reference_f0(batch))
    disp = torch.linalg.vector_norm(zr - zr[0:1], dim=1)
    rho_res = float(st.spearmanr(res.numpy(), disp.numpy()).statistic)
    check("resonance monotonically moves the embedding", rho_res > 0.9,
          f"spearman={rho_res:.3f}")

    sl = torch.linspace(0.05, 0.95, 9)
    batch = S.default_patch().repeat(len(sl), 1)
    batch[:, S.IDX["filter_slope"]] = sl
    zs = features.embed(synth.render(batch), synth.reference_f0(batch))
    rho_slope = float(st.spearmanr(sl.numpy(), zs[:, 77].numpy()).statistic)
    check("filter slope -> HF band ratio monotone (decreasing)", rho_slope < -0.9,
          f"spearman={rho_slope:.3f}")

    # --- sep-CMA-ES unit test ---------------------------------------------
    D = 10
    rng = torch.Generator().manual_seed(1)
    tgt = torch.rand(D, generator=rng)

    def sphere(x: torch.Tensor) -> torch.Tensor:
        return ((x - tgt[None, :]) ** 2).sum(dim=1)

    best, _, _ = sep_cmaes(torch.full((1, D), 0.5), sphere, gens=300, pop=10,
                           sigma0=0.3, log_every=100, seed=0)
    f_final = float(sphere(best)[0])
    check("sep-CMA-ES converges on a sphere", f_final < 1e-4, f"final={f_final:.2e}")

    # --- spectral filter vs per-sample SVF reference ----------------------
    n = 2000
    g = torch.Generator().manual_seed(3)
    sig = torch.randn(1, n, generator=g) * 0.3
    fc, q = 800.0, 2.0
    n_frames = 1 + n // synth.FILT_HOP
    cut = torch.full((1, n_frames), fc)
    ones = torch.ones(1, n_frames)
    zeros = torch.zeros(1, n_frames)
    y_spec = synth._spectral_filter(sig, cut, torch.tensor([q]), ones, zeros, zeros,
                                    zeros, synth.SR)
    gg = math.tan(math.pi * fc / synth.SR)
    kk = 1.0 / q
    a1 = torch.full((1, n), 1.0 / (1.0 + gg * (gg + kk)))
    a2 = a1 * gg
    a3 = a2 * gg
    y_ref = synth._svf_loop_eager(sig, a1, a2, a3, torch.tensor([kk]),
                                  torch.ones(1), torch.zeros(1), torch.zeros(1),
                                  torch.zeros(1))
    win = torch.hann_window(256)
    ms = torch.stft(y_spec, 256, 64, window=win, return_complex=True).abs()[0].mean(dim=1)
    mr = torch.stft(y_ref, 256, 64, window=win, return_complex=True).abs()[0].mean(dim=1)
    rho_filt = float(st.spearmanr(ms.numpy(), mr.numpy()).statistic)
    check("spectral filter matches SVF response shape (spearman)", rho_filt > 0.9,
          f"rho={rho_filt:.3f}")

    # --- timing ------------------------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        synth.render(torch.rand(64, S.D))
    fwd = time.time() - t0
    uu = torch.rand(64, S.D, requires_grad=True)
    t0 = time.time()
    features.embed(synth.render(uu, chunk_b=64), synth.reference_f0(uu)).pow(2).mean().backward()
    fwd_bwd = time.time() - t0
    print(f"[info] render fwd B=64 no_grad {fwd:.2f}s | fwd+bwd (incl. features) {fwd_bwd:.2f}s")

    print("\nsanity:", "ALL PASS" if ok else "FAILURES PRESENT")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# corpus / e2
# ---------------------------------------------------------------------------

def cmd_corpus(args) -> None:
    _set_threads()
    cfg = C.get(args.preset)
    out = Path(args.out) if args.out else OUT / f"corpus_{cfg.name}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] preset={cfg.name}: building sampler", flush=True)
    c = CORP.build_corpus(cfg, seed=cfg.seed)
    print(f"[corpus] {c.u.shape[0]} unique patches "
          f"({c.meta['source_counts']}), rendering", flush=True)
    raw = render_corpus(c.u, cfg, tag=cfg.name)
    stats = features.fit_normalizer(torch.as_tensor(raw))
    np.savez(out / "stats.npz", **{k: v.numpy() for k, v in stats.items()})
    CORP.save_corpus(out, c, raw)

    A, anames = CORP.anchor_matrix()
    save_demo_wavs(A, cfg, out / "wav_anchors", names=anames)
    print(f"[corpus] saved to {out} (corpus.npz, corpus_meta.json, stats.npz, "
          f"wav_anchors/)", flush=True)


def cmd_e2(args) -> None:
    _set_threads()
    from .experiment_e2 import run_e2
    cfg = C.get(args.preset)
    corpus_dir = Path(args.corpus) if args.corpus else OUT / f"corpus_{cfg.name}"
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "metric", None):
        cfg.metric = args.metric
    tag = f"_seed{cfg.seed}" if cfg.seed else ""
    tag += "_balanced" if cfg.metric != "uniform" else ""
    default_out = OUT / f"e2_{cfg.name}{tag}"
    out = Path(args.out) if args.out else default_out
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run_e2(cfg, corpus_dir, out, methods=methods)


def cmd_diagnose(args) -> None:
    _set_threads()
    from .diagnose import run_diagnostics
    cfg = C.get(args.preset)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    run_dir = Path(args.run)
    corpus_dir = Path(args.corpus) if args.corpus else OUT / f"corpus_{cfg.name}"
    run_diagnostics(run_dir, corpus_dir, cfg)


def cmd_all(args) -> None:
    cmd_corpus(args)
    cmd_e2(args)


def cmd_e1(args) -> None:
    _set_threads()
    from .e1_listening import build_e1_material
    cfg = C.get(args.preset)
    corpus_dir = Path(args.corpus) if args.corpus else OUT / f"corpus_{cfg.name}"
    out = Path(args.out) if args.out else OUT / f"e1_{cfg.name}"
    build_e1_material(cfg, corpus_dir, out, n_triplets=cfg.e1_n_triplets, seed=cfg.seed)


def cmd_plot(args) -> None:
    from .plotting import make_figures
    for run in [Path(p) for p in args.run.split(",")]:
        fig_dir = make_figures(run)
        print(f"[plot] figures -> {fig_dir}")


def cmd_e1_analyze(args) -> None:
    from .e1_listening import analyze
    material = Path(args.out) if args.out else OUT / f"e1_{C.get(args.preset).name}"
    responses = [Path(p) for p in args.responses.split(",")]
    out = analyze(responses, material)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="timbreprobe")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_san = sub.add_parser("sanity")
    sp_san.add_argument("--preset", default="smoke", choices=list(C.PRESETS))
    sp_san.add_argument("--out", default=None)
    sp_san.set_defaults(func=cmd_sanity)

    for name, fn in (("corpus", cmd_corpus), ("e2", cmd_e2), ("all", cmd_all),
                     ("e1", cmd_e1)):
        sp = sub.add_parser(name)
        sp.add_argument("--preset", default="smoke", choices=list(C.PRESETS))
        sp.add_argument("--out", default=None)
        if name in ("e2", "all", "e1"):
            sp.add_argument("--corpus", default=None)
        if name in ("e2", "all"):
            sp.add_argument("--seed", default=None, type=int,
                            help="override the preset seed (robustness replicates)")
            sp.add_argument("--metric", default=None, choices=["uniform", "balanced"],
                            help="objective weighting (see config.Config.metric)")
            sp.add_argument("--methods", default=None,
                            help="comma list of: retrieval,grad_surrogate,grad_reground,"
                                 "cmaes_true,grad_true")
        sp.set_defaults(func=fn)

    sp_dg = sub.add_parser("diagnose", help="residual / discrete-jump / ablation "
                                             "diagnostics for an E2 run")
    sp_dg.add_argument("--run", required=True, help="run dir, e.g. out/e2_smoke")
    sp_dg.add_argument("--preset", default="smoke", choices=list(C.PRESETS))
    sp_dg.add_argument("--corpus", default=None)
    sp_dg.add_argument("--seed", default=None, type=int)
    sp_dg.set_defaults(func=cmd_diagnose)

    sp_pl = sub.add_parser("plot", help="figures for one or more E2 runs")
    sp_pl.add_argument("--run", required=True,
                       help="comma list of run dirs, e.g. out/e2_smoke")
    sp_pl.add_argument("--preset", default="smoke", choices=list(C.PRESETS))
    sp_pl.add_argument("--out", default=None)
    sp_pl.set_defaults(func=cmd_plot)

    sp_an = sub.add_parser("e1-analyze")
    sp_an.add_argument("--preset", default="smoke", choices=list(C.PRESETS))
    sp_an.add_argument("--out", default=None)
    sp_an.add_argument("--responses", required=True, help="comma list of response JSONs")
    sp_an.set_defaults(func=cmd_e1_analyze)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
