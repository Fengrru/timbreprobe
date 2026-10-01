"""Anchor-stratified corpus construction.

Random uniform sampling of a 31-d synth space is a bad corpus: most of the
volume is perceptually degenerate (nothing audible, everything wide open), so
it teaches the surrogate almost nothing about the perceptual geometry.  The
corpus here is built in three layers:

  * sweeps   -- for every anchor and every parameter, the parameter is moved
                across its range while the anchor is held fixed: this maps the
                *local* perceptual direction of each axis;
  * jitter   -- small Gaussian perturbations around each anchor: dense local
                neighbourhoods, the region optimisation actually traverses
                (choice dims get a much smaller sigma so patches stay close to
                "hard" presets);
  * uniform  -- a modest uniform component plus sparse patches (one or two
                sources active) for global coverage.

Anchors are hand-designed starting points with semantic labels.  The last
`_HOLDOUT_ANCHORS` entries are excluded from surrogate training entirely and
serve as a generalisation probe (their whole neighbourhood is unseen).

Corpus = a *scientific instrument*: it is deliberately neutral and controlled.
The Stage 1 retrieval library will use real presets instead, where bias is a
feature, not a bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import schema as S

HOLDOUT_NAMES = {"resonant_sweep", "octave_stacked", "sub_heavy_pad", "metallic_tri_fm"}

# name -> {param: normalised u value}.  Everything not listed comes from the
# neutral base patch (schema.default_patch).
ANCHORS: dict[str, dict[str, float]] = {
    "bright_saw_lead": {
        "osc1_wave": 1 / 3, "filter_cutoff": 0.72, "filter_res": 0.35,
        "filter_env_amount": 0.55, "fenv_decay": 0.30, "aenv_attack": 0.06,
        "master_drive": 0.10,
    },
    "warm_detuned_pad": {
        "osc2_level": 0.50, "osc2_detune": 0.85, "osc1_detune": 0.15,
        "filter_cutoff": 0.42, "filter_res": 0.20, "aenv_attack": 0.62,
        "aenv_decay": 0.70, "aenv_sustain": 0.85, "lfo_rate": 0.25,
        "lfo_depth": 0.25, "lfo_target": 0.0, "lfo_wave": 0.0,
    },
    "dark_sub_bass": {
        "osc1_wave": 0.0, "osc1_oct": 0.0, "osc1_level": 0.80,
        "sub_level": 0.90, "filter_cutoff": 0.28, "filter_res": 0.15,
    },
    "hollow_square": {
        "osc1_wave": 2 / 3, "osc1_pw": 0.5, "filter_cutoff": 0.55,
        "filter_res": 0.30,
    },
    "narrow_pulse": {
        "osc1_wave": 2 / 3, "osc1_pw": 0.12, "filter_cutoff": 0.62,
        "filter_res": 0.35,
    },
    "fm_brass": {
        "osc2_level": 0.85, "osc2_oct": 1.0, "fm_amount": 0.55,
        "filter_cutoff": 0.55, "filter_env_amount": 0.35, "aenv_attack": 0.25,
    },
    "noisy_breath": {
        "noise_level": 0.85, "noise_color": 0.35, "filter_type": 1.0,
        "filter_cutoff": 0.50, "filter_res": 0.40, "aenv_attack": 0.35,
        "aenv_decay": 0.35, "aenv_sustain": 0.60,
    },
    "pluck_bright": {
        "filter_cutoff": 0.55, "filter_env_amount": 0.92, "fenv_attack": 0.03,
        "fenv_decay": 0.12, "aenv_attack": 0.02, "aenv_decay": 0.28,
        "aenv_sustain": 0.05,
    },
    "resonant_sweep": {
        "filter_res": 0.78, "filter_cutoff": 0.35, "lfo_rate": 0.30,
        "lfo_depth": 0.65, "lfo_target": 0.0, "lfo_wave": 0.0,
        "filter_env_amount": 0.30,
    },
    "soft_flute": {
        "osc1_wave": 0.12, "aenv_attack": 0.45, "filter_res": 0.12,
        "filter_cutoff": 0.60, "lfo_rate": 0.55, "lfo_depth": 0.22,
        "lfo_target": 0.5, "lfo_wave": 0.0,
    },
    "tri_organ": {
        "osc1_wave": 1.0, "osc1_level": 0.85, "filter_cutoff": 0.68,
        "filter_res": 0.10, "aenv_sustain": 0.95, "aenv_decay": 0.80,
    },
    "hp_air": {
        "osc1_wave": 1 / 3, "filter_type": 0.5, "filter_cutoff": 0.58,
        "filter_res": 0.25, "noise_level": 0.12,
    },
    "octave_stacked": {
        "osc1_level": 0.70, "osc2_level": 0.70, "osc2_oct": 1.0,
        "osc1_wave": 2 / 3, "osc2_wave": 0.0, "filter_cutoff": 0.60,
    },
    "sub_heavy_pad": {
        "sub_level": 0.80, "osc1_wave": 1 / 3, "aenv_attack": 0.50,
        "filter_cutoff": 0.40, "filter_res": 0.20, "filter_slope": 0.9,
    },
    "drive_lead": {
        "master_drive": 0.55, "filter_drive": 0.50, "filter_cutoff": 0.60,
        "filter_res": 0.40, "filter_env_amount": 0.40,
    },
    "metallic_tri_fm": {
        "osc2_wave": 0.0, "osc2_oct": 1.0, "osc2_detune": 0.78,
        "fm_amount": 0.75, "filter_cutoff": 0.60,
    },
}


def anchor_matrix() -> tuple[np.ndarray, list[str]]:
    """[A, D] anchor patch matrix and their names (order preserved)."""
    base = S.default_patch()
    rows, names = [], []
    for name, overrides in ANCHORS.items():
        u = base.clone()
        for pname, val in overrides.items():
            u[0, S.IDX[pname]] = float(val)
        rows.append(u[0])
        names.append(name)
    return torch.stack(rows).numpy(), names


def anchor_flags(names: list[str]) -> np.ndarray:
    return np.array([n in HOLDOUT_NAMES for n in names], dtype=bool)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    u: np.ndarray              # [N, D] patches in [0,1]
    source: list[str]          # provenance label per patch
    anchor_idx: np.ndarray     # index of the anchor it was derived from (-1 if none)
    holdout: np.ndarray        # bool, patch belongs to a held-out anchor
    meta: dict


def build_corpus(cfg, seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    A, names = anchor_matrix()
    ho = anchor_flags(names)
    n_anch, D = A.shape

    us: list[np.ndarray] = []
    src: list[str] = []
    aidx: list[int] = []

    # --- layer 1: single-parameter sweeps --------------------------------
    for ai in range(n_anch):
        for dim in range(D):
            for v in cfg.sweep_values:
                u = A[ai].copy()
                u[dim] = v
                us.append(u)
                src.append("sweep")
                aidx.append(ai)

    # --- layer 2: local jitter -------------------------------------------
    sigma = np.array([cfg.sigma_choice if sp.is_choice else cfg.sigma_cont
                      for sp in S.SCHEMA])
    for ai in range(n_anch):
        for _ in range(cfg.n_jitter_per_anchor):
            u = A[ai] + rng.normal(0.0, sigma)
            us.append(np.clip(u, 0.0, 1.0))
            src.append("jitter")
            aidx.append(ai)

    # --- layer 3a: uniform -----------------------------------------------
    for _ in range(cfg.n_uniform):
        us.append(rng.random(D))
        src.append("uniform")
        aidx.append(-1)

    # --- layer 3b: sparse (one or two sources active) --------------------
    src_dims = [S.IDX[n] for n in ("osc1_level", "osc2_level", "sub_level", "noise_level")]
    for _ in range(cfg.n_sparse):
        u = rng.random(D)
        active = rng.choice(len(src_dims), size=int(rng.integers(1, 3)), replace=False)
        for d in src_dims:
            u[d] = 0.0
        for a in active:
            u[src_dims[a]] = float(rng.uniform(0.4, 1.0))
        us.append(u)
        src.append("sparse")
        aidx.append(-1)

    u_all = np.stack(us).astype(np.float32)
    aidx_arr = np.array(aidx, dtype=np.int64)
    holdout = np.where(aidx_arr >= 0, ho[np.clip(aidx_arr, 0, None)], False)

    # dedup on rounded coordinates (keep first occurrence)
    key = np.round(u_all, 4)
    _, keep = np.unique(key, axis=0, return_index=True)
    keep = np.sort(keep)
    u_all = u_all[keep]
    aidx_arr = aidx_arr[keep]
    holdout = holdout[keep]
    src = [src[i] for i in keep]

    meta = {
        "schema_version": S.SCHEMA_VERSION,
        "dim": int(S.D),
        "n_patches": int(u_all.shape[0]),
        "anchors": names,
        "holdout_anchors": sorted(HOLDOUT_NAMES),
        "config": {k: v for k, v in vars(cfg).items()},
        "source_counts": {s: src.count(s) for s in sorted(set(src))},
    }
    return Corpus(u=u_all, source=src, anchor_idx=aidx_arr, holdout=holdout, meta=meta)


def split_indices(corpus: Corpus, cfg, seed: int = 0) -> dict[str, np.ndarray]:
    """train/val/test split.  Held-out anchors never enter train/val."""
    rng = np.random.default_rng(seed + 1234)
    trainable = np.where(~corpus.holdout)[0]
    perm = rng.permutation(trainable)
    n_val = int(len(perm) * cfg.val_frac)
    n_test = int(len(perm) * cfg.test_frac)
    val = perm[:n_val]
    test = perm[n_val:n_val + n_test]
    train = perm[n_val + n_test:]
    holdout = np.where(corpus.holdout)[0]
    return {"train": np.sort(train), "val": np.sort(val),
            "test": np.sort(np.concatenate([test, holdout])),
            "holdout": np.sort(holdout)}


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_corpus(path: Path, corpus: Corpus, raw_feats: np.ndarray,
                u_targets: np.ndarray | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "corpus.npz",
        u=corpus.u,
        raw_feats=raw_feats,
        source=np.array(corpus.source),
        anchor_idx=corpus.anchor_idx,
        holdout=corpus.holdout,
    )
    with open(path / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump(corpus.meta, f, ensure_ascii=False, indent=2)
    if u_targets is not None:
        np.save(path / "targets.npy", u_targets)


def load_corpus(path: Path) -> tuple[Corpus, np.ndarray]:
    data = np.load(path / "corpus.npz", allow_pickle=False)
    with open(path / "corpus_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    corpus = Corpus(
        u=data["u"],
        source=[str(s) for s in data["source"]],
        anchor_idx=data["anchor_idx"],
        holdout=data["holdout"],
        meta=meta,
    )
    return corpus, data["raw_feats"]
