"""Parameter schema for the Stage 0 differentiable subtractive synthesizer.

Every patch is a point in the unit hypercube u in [0,1]^D.  Continuous
parameters are mapped to physical units (linear or log); discrete parameters
(the "choice" kind) are represented as a normalized position over the ordered
choice list and expanded downstream into soft weights.

Design notes
------------
* Choice parameters use *piecewise-linear* blending between adjacent choices
  (triangular basis, weights `max(0, 1 - |p - k|)` normalized).  Unlike a
  softmax kernel this keeps gradients alive at every point of the interval --
  crucially also when a patch sits exactly on a choice, which is the common
  case for preset-like patches.  It also reads as an interpretable "morph
  knob" between neighbouring choices.
* Hardening (`harden`) snaps choice dims to the nearest index and is used only
  for reporting / exporting patches, never inside the gradient path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # 'lin' | 'log' | 'choice'
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple[float, ...] = ()
    unit: str = ""
    group: str = ""

    @property
    def is_choice(self) -> bool:
        return self.kind == "choice"

    @property
    def n_choices(self) -> int:
        return len(self.choices)


# ---------------------------------------------------------------------------
# The schema.  Order defines the index of every dimension; never reorder.
# ---------------------------------------------------------------------------

SCHEMA: list[ParamSpec] = [
    # --- oscillator 1 -----------------------------------------------------
    ParamSpec("osc1_wave", "choice", choices=(0, 1, 2, 3), group="osc1",
              unit="sine|saw|square|triangle"),
    ParamSpec("osc1_oct", "choice", choices=(-1, 0, 1), group="osc1", unit="octave"),
    ParamSpec("osc1_detune", "lin", -50.0, 50.0, unit="cents", group="osc1"),
    ParamSpec("osc1_level", "lin", 0.0, 1.0, group="osc1"),
    ParamSpec("osc1_pw", "lin", 0.15, 0.85, unit="duty", group="osc1"),
    # --- oscillator 2 -----------------------------------------------------
    ParamSpec("osc2_wave", "choice", choices=(0, 1, 2, 3), group="osc2",
              unit="sine|saw|square|triangle"),
    ParamSpec("osc2_oct", "choice", choices=(-1, 0, 1), group="osc2", unit="octave"),
    ParamSpec("osc2_detune", "lin", -50.0, 50.0, unit="cents", group="osc2"),
    ParamSpec("osc2_level", "lin", 0.0, 1.0, group="osc2"),
    ParamSpec("osc2_pw", "lin", 0.15, 0.85, unit="duty", group="osc2"),
    # --- extra sources ----------------------------------------------------
    ParamSpec("sub_level", "lin", 0.0, 1.0, group="src"),
    ParamSpec("noise_level", "lin", 0.0, 1.0, group="src"),
    ParamSpec("noise_color", "lin", 0.0, 1.0, unit="white..pink", group="src"),
    ParamSpec("fm_amount", "lin", 0.0, 2.5, unit="PM index", group="src"),
    # --- filter -----------------------------------------------------------
    ParamSpec("filter_type", "choice", choices=(0, 1, 2), group="filter", unit="lp|hp|bp"),
    ParamSpec("filter_cutoff", "log", 40.0, 7000.0, unit="Hz", group="filter"),
    ParamSpec("filter_res", "log", 0.5, 12.0, unit="Q", group="filter"),
    ParamSpec("filter_env_amount", "lin", -1.0, 1.0, unit="+/-5 oct", group="filter"),
    ParamSpec("filter_drive", "lin", 1.0, 4.0, unit="gain", group="filter"),
    ParamSpec("filter_slope", "lin", 0.0, 1.0, unit="12..24 dB", group="filter"),
    # --- filter envelope --------------------------------------------------
    ParamSpec("fenv_attack", "log", 0.001, 2.0, unit="s", group="fenv"),
    ParamSpec("fenv_decay", "log", 0.005, 4.0, unit="s", group="fenv"),
    ParamSpec("fenv_sustain", "lin", 0.0, 1.0, group="fenv"),
    # --- amplitude envelope ----------------------------------------------
    ParamSpec("aenv_attack", "log", 0.001, 3.0, unit="s", group="aenv"),
    ParamSpec("aenv_decay", "log", 0.005, 6.0, unit="s", group="aenv"),
    ParamSpec("aenv_sustain", "lin", 0.0, 1.0, group="aenv"),
    # --- LFO --------------------------------------------------------------
    ParamSpec("lfo_rate", "log", 0.05, 20.0, unit="Hz", group="lfo"),
    ParamSpec("lfo_depth", "lin", 0.0, 1.0, group="lfo"),
    ParamSpec("lfo_target", "choice", choices=(0, 1, 2), group="lfo", unit="cutoff|pitch|amp"),
    ParamSpec("lfo_wave", "choice", choices=(0, 1, 2, 3), group="lfo", unit="sine|tri|square|saw"),
    # --- output -----------------------------------------------------------
    ParamSpec("master_drive", "lin", 1.0, 8.0, unit="gain", group="output"),
]

NAMES: list[str] = [s.name for s in SCHEMA]
IDX: dict[str, int] = {n: i for i, n in enumerate(NAMES)}
D: int = len(SCHEMA)
SCHEMA_VERSION = "timbreprobe-31d-v1"

# index of the wave choice slot that uses the pulse engine
_WAVE_PULSE_INDEX = 2  # 'square' slot


def get(name: str) -> ParamSpec:
    return SCHEMA[IDX[name]]


def continuous(u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map u [b, D] -> {name: physical value [b]} for continuous dims."""
    out: dict[str, torch.Tensor] = {}
    for i, sp in enumerate(SCHEMA):
        if sp.kind == "lin":
            out[sp.name] = sp.lo + u[:, i] * (sp.hi - sp.lo)
        elif sp.kind == "log":
            out[sp.name] = sp.lo * (sp.hi / sp.lo) ** u[:, i]
    return out


def choices(u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map u [b, D] -> {name: blend weights [b, n]} for discrete dims."""
    out: dict[str, torch.Tensor] = {}
    for i, sp in enumerate(SCHEMA):
        if sp.kind == "choice":
            n = sp.n_choices
            p = (u[:, i] * (n - 1)).clamp(0.0, float(n - 1))
            k = torch.arange(n, dtype=u.dtype, device=u.device)
            w = torch.clamp(1.0 - (p[:, None] - k[None, :]).abs(), min=0.0)
            out[sp.name] = w / w.sum(dim=-1, keepdim=True)
    return out


def octave_multiplier(u: torch.Tensor, name: str) -> torch.Tensor:
    """Blended frequency multiplier for an octave choice dim, [b]."""
    sp = get(name)
    w = choices(u)[name]
    vals = torch.tensor([2.0 ** c for c in sp.choices], dtype=u.dtype, device=u.device)
    return (w * vals[None, :]).sum(dim=-1)


def harden(u: torch.Tensor) -> torch.Tensor:
    """Snap choice dims to the nearest choice index (reporting/export only)."""
    out = u.clone()
    for i, sp in enumerate(SCHEMA):
        if sp.kind == "choice":
            n = sp.n_choices
            idx = (u[:, i] * (n - 1)).round().clamp(0.0, float(n - 1))
            out[:, i] = idx / (n - 1)
    return out


def choice_label(u_row: torch.Tensor, name: str) -> float:
    """Hardened choice value of one patch row (for human-readable dumps)."""
    sp = get(name)
    i = IDX[name]
    n = sp.n_choices
    idx = int(round(float(u_row[i]) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return sp.choices[idx]


def describe(u_row: torch.Tensor) -> dict[str, float]:
    """Physical, human-readable description of one patch row (hardened choices)."""
    u1 = harden(u_row[None, :])
    cont = continuous(u1)
    desc: dict[str, float] = {}
    for sp in SCHEMA:
        if sp.kind == "choice":
            idx = int(round(float(u1[0, IDX[sp.name]]) * (sp.n_choices - 1)))
            desc[sp.name] = sp.choices[max(0, min(sp.n_choices - 1, idx))]
        else:
            desc[sp.name] = round(float(cont[sp.name][0]), 4)
    return desc


def default_patch() -> torch.Tensor:
    """A neutral mid-bright saw patch used as the base for anchor definitions."""
    u = torch.full((1, D), 0.5)
    def setv(name: str, value_norm: float) -> None:
        u[0, IDX[name]] = value_norm
    # choices: saw (index 1 of 4), octave 0 (index 1 of 3), lp filter, lfo cutoff, lfo sine
    setv("osc1_wave", 1.0 / 3.0)
    setv("osc1_oct", 0.5)
    setv("osc1_level", 0.75)
    setv("osc2_wave", 1.0 / 3.0)
    setv("osc2_oct", 0.5)
    setv("osc2_level", 0.0)
    setv("sub_level", 0.0)
    setv("noise_level", 0.0)
    setv("fm_amount", 0.0)
    setv("filter_type", 0.0)          # lp
    setv("filter_cutoff", 0.75)
    setv("filter_res", 0.25)
    setv("filter_env_amount", 0.0)
    setv("filter_slope", 0.5)
    setv("fenv_attack", 0.15)
    setv("fenv_decay", 0.5)
    setv("fenv_sustain", 0.5)
    setv("aenv_attack", 0.1)
    setv("aenv_decay", 0.5)
    setv("aenv_sustain", 0.7)
    setv("lfo_rate", 0.4)
    setv("lfo_depth", 0.0)
    setv("lfo_target", 0.0)           # cutoff
    setv("lfo_wave", 0.0)             # sine
    setv("master_drive", 0.0)
    setv("filter_drive", 0.0)
    return u
