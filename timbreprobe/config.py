"""Configuration presets for Stage 0.

`smoke` runs the whole pipeline end-to-end in tens of minutes on a 4-core CPU
and is the default.  `full` is the same pipeline at paper scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    name: str = "smoke"

    # --- rendering --------------------------------------------------------
    sr: int = 16_000
    dur: float = 0.25
    chunk_b: int = 64
    seed: int = 0

    # --- corpus -----------------------------------------------------------
    sweep_values: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_jitter_per_anchor: int = 120
    n_uniform: int = 500
    n_sparse: int = 150
    sigma_cont: float = 0.10
    sigma_choice: float = 0.03

    # --- surrogate --------------------------------------------------------
    hidden: int = 256
    depth: int = 2
    epochs: int = 150
    batch: int = 256
    lr: float = 1e-3
    patience: int = 40
    val_frac: float = 0.10
    test_frac: float = 0.10

    # --- E2 navigation ----------------------------------------------------
    n_targets: int = 48
    steps_true: int = 60
    steps_surrogate: int = 200
    lr_opt: float = 0.05
    trust_radius: float = 0.35   # L2 ball around the start for surrogate search
    log_every: int = 5
    reground_every: int = 20
    reground_inner: int = 15
    reground_buffer: int = 512
    cmaes_gens: int = 45
    cmaes_pop: int = 10
    cmaes_sigma: float = 0.25
    retrieval_draws: int = 64

    # --- E1 listening test ------------------------------------------------
    e1_n_triplets: int = 120
    # triplet sampling now uses empirical quantiles of the pool distance
    # distribution (see e1_listening.build_e1_material); this field is kept for
    # backwards compatibility of saved configs.
    e1_distance_bins: tuple[tuple[float, float], ...] = (
        (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def smoke() -> Config:
    return Config()


def full() -> Config:
    return Config(
        name="full",
        sweep_values=(0.1, 0.3, 0.5, 0.7, 0.9),
        n_jitter_per_anchor=160,
        n_uniform=800,
        n_sparse=400,
        epochs=300,
        n_targets=128,
        steps_true=120,
        steps_surrogate=300,
        cmaes_gens=40,
        cmaes_pop=12,
        e1_n_triplets=200,
    )


PRESETS = {"smoke": smoke, "full": full}


def get(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {list(PRESETS)}")
    return PRESETS[name]()


def as_dict(cfg: Config) -> dict:
    return asdict(cfg)
