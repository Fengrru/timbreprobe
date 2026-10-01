"""Differentiable subtractive synthesizer (Stage 0 oracle).

Signal chain
------------
    osc1 (+ PM from osc2) + osc2 + sub + noise
        -> tanh drive -> time-varying filter -> VCA (amp env x tremolo)
        -> tanh master drive -> RMS normalise

Everything is differentiable w.r.t. the patch vector u in [0,1]^D, so the
"true gradient" navigator in E2 backprops through this renderer to the
parameters.

Filtering is done in the STFT domain
------------------------------------
A per-sample SVF loop was the first implementation and was measured at ~10 s
per batch *forward-only* on the Stage 0 machine (6400 python-level iterations,
cost independent of batch size, ~60 s with backward).  That is unusable for an
optimisation loop, so the renderer applies the *analytic magnitude response* of
the same TPT state-variable filter family per STFT frame instead:

    |H_lp| = 1 / sqrt((1 - r^2)^2 + (r/Q)^2),   r = f / fc(frame)
    |H_hp| = r^2 * |H_lp|,   |H_bp| = (r/Q) * |H_lp|,   24 dB = 12 dB squared

Per-frame cutoff (4 ms hop), the LP/HP/BP blend, the 12/24 dB blend and the Q
peak all survive; the phase response / ringing does not (zero-phase), so
resonance self-oscillation and ultra-fast spectral transients are absent.  The
amp envelope is applied *after* synthesis, so loudness transients stay
sample-accurate.  `_svf_loop_eager` keeps the per-sample version for an
optional one-off fidelity check.

Analysis/synthesis uses a hand-rolled overlap-add (not torch.istft) whose
window-envelope normalisation is clamped: torch.istft's backward divides by a
near-zero envelope at the padded edges and produced non-finite gradients there
(reproduced on this machine at 0.4 s renders).

Other deliberate choices
------------------------
* Oscillators are additive with a band-limit mask (k * f0 * (1 + 0.6*fm) <
  0.95 * Nyquist): no aliasing at the fixed render pitch.  Harmonic phases use
  sin() uniformly -- the embedding is phase-invariant and the ear is largely
  phase-insensitive for steady tones.
* Each waveform family is unit-RMS normalised so "waveform" is a timbre axis,
  not a loudness axis.
* Noise uses a fixed seed: identical patch -> bit-identical render.
* Output is RMS-normalised, so loudness is not a distance cue (mirrors
  loudness matching in the human listening task).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import schema as S

SR = 16_000
DUR = 0.25
F0_HZ = 261.6255653005986  # C4
KMAX = 24
TARGET_RMS = 0.1
NYQ = SR / 2.0
_TWO_PI = 2.0 * math.pi
FILT_NFFT = 256
FILT_HOP = 64


# ---------------------------------------------------------------------------
# waveform bank
# ---------------------------------------------------------------------------

def _unit_rms(a: torch.Tensor) -> torch.Tensor:
    return a / torch.sqrt((a * a).sum(dim=-1, keepdim=True) / 2.0)


def _static_bank(kmax: int = KMAX) -> torch.Tensor:
    """[3, K] unit-RMS coefficient rows for sine, saw, triangle (k = 1..K)."""
    k = torch.arange(1, kmax + 1, dtype=torch.float32)
    sine = torch.zeros(kmax)
    sine[0] = 1.0
    saw = 1.0 / k
    tri = torch.zeros(kmax)
    tri[0::2] = 1.0 / (k[0::2] ** 2)
    return torch.stack([_unit_rms(sine), _unit_rms(saw), _unit_rms(tri)], dim=0)


_BANK = _static_bank()
_bank_cache: dict = {}


def _bank_for(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, str(device))
    if key not in _bank_cache:
        _bank_cache[key] = _BANK.to(dtype=dtype, device=device)
    return _bank_cache[key]


def _pulse_coeffs(duty: torch.Tensor) -> torch.Tensor:
    """[b, K] unit-RMS coefficients of a zero-mean rectangular pulse train."""
    k = torch.arange(1, KMAX + 1, dtype=duty.dtype, device=duty.device)
    a = (2.0 / (k * math.pi)) * torch.sin(math.pi * k[None, :] * duty[:, None])
    return _unit_rms(a)


def _osc_coeffs(wave_w: torch.Tensor, pw: torch.Tensor, f_base: torch.Tensor,
                bandwidth_scale: torch.Tensor) -> torch.Tensor:
    """Blend waveform families and apply the band-limit mask.  Returns [b, K]."""
    bank = _bank_for(wave_w.dtype, wave_w.device)
    a = (wave_w[:, 0:1] * bank[0]
         + wave_w[:, 1:2] * bank[1]
         + wave_w[:, 2:3] * _pulse_coeffs(pw)
         + wave_w[:, 3:4] * bank[2])
    k = torch.arange(1, KMAX + 1, dtype=f_base.dtype, device=f_base.device)
    limit = 0.95 * NYQ
    x = k[None, :] * f_base[:, None] * bandwidth_scale[:, None]
    mask = torch.sigmoid((limit - x) / (0.05 * NYQ))
    return a * mask


def _harm_sum(phase: torch.Tensor, a: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """sum_k a[:,k] * sin(k * phase[:,t]); chunked over k to bound memory."""
    out: torch.Tensor | None = None
    kmax = a.shape[-1]
    for s in range(0, kmax, chunk):
        e = min(s + chunk, kmax)
        kk = torch.arange(s + 1, e + 1, dtype=phase.dtype, device=phase.device)
        term = (torch.sin(kk[:, None] * phase[:, None, :]) * a[:, s:e, None]).sum(dim=1)
        out = term if out is None else out + term
    assert out is not None
    return out


# ---------------------------------------------------------------------------
# noise buffers (fixed seed -> deterministic renders)
# ---------------------------------------------------------------------------

_noise_cache: dict = {}


def _noise_buffers(t_len: int, dtype: torch.dtype, device: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    key = (t_len, dtype, str(device))
    if key not in _noise_cache:
        g = torch.Generator().manual_seed(20260501)
        white = torch.randn(t_len, generator=g)
        spec = torch.fft.rfft(white)
        f = torch.fft.rfftfreq(t_len).clamp(min=1.0 / t_len)
        pink = torch.fft.irfft(spec / torch.sqrt(f), n=t_len)
        pink = pink / pink.std() * white.std()
        _noise_cache[key] = (white.to(dtype=dtype, device=device),
                             pink.to(dtype=dtype, device=device))
    w, p = _noise_cache[key]
    return w, p


# ---------------------------------------------------------------------------
# modulation sources
# ---------------------------------------------------------------------------

def _lfo_blend(u: torch.Tensor, t: torch.Tensor, cont: dict, cho: dict
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blended LFO and the three scaled modulations, evaluated at times t [n].

    Returns (lfo, cutoff_mult, pitch_mult, tremolo).  The four waveform
    families are blended directly instead of materialising a [b,4,T] stack.
    """
    phase = cont["lfo_rate"][:, None] * t[None, :]
    s = torch.sin(_TWO_PI * phase)
    frac = phase + 0.25
    frac = frac - torch.floor(frac)
    tri = 1.0 - 4.0 * (frac - 0.5).abs()   # closed form, no asin (asin'(+-1) = inf)
    lfo = (cho["lfo_wave"][:, 0:1] * s
           + cho["lfo_wave"][:, 1:2] * tri
           + cho["lfo_wave"][:, 2:3] * torch.tanh(6.0 * s)
           + cho["lfo_wave"][:, 3:4] * (2.0 * (phase - torch.floor(phase)) - 1.0))
    depth = cont["lfo_depth"][:, None]
    tgt = cho["lfo_target"]                                        # [b,3]
    cutoff_mult = 2.0 ** (tgt[:, 0:1] * depth * 3.0 * lfo)
    pitch_mult = 2.0 ** (tgt[:, 1:2] * depth * 7.0 * lfo / 12.0)
    tremolo = 1.0 - tgt[:, 2:3] * depth * 0.5 * (1.0 - lfo)
    return lfo, cutoff_mult, pitch_mult, tremolo


def _envelope(t: torch.Tensor, attack: torch.Tensor, decay: torch.Tensor,
              sustain: torch.Tensor) -> torch.Tensor:
    """Linear attack + exponential decay to sustain.  t [n], attack [b,1]."""
    att = torch.clamp(t[None, :] / attack, max=1.0)
    tau = decay / 5.0
    dec = sustain + (1.0 - sustain) * torch.exp(-torch.clamp(t[None, :] - attack, min=0.0) / tau)
    return torch.where(t[None, :] < attack, att, dec)


# ---------------------------------------------------------------------------
# time-varying filter, STFT domain (analytic TPT SVF magnitude response)
# ---------------------------------------------------------------------------

def _frame_centers(t_len: int, hop: int, dtype, device) -> torch.Tensor:
    n_frames = 1 + t_len // hop
    return torch.arange(n_frames, dtype=dtype, device=device) * hop / SR


def _overlap_add(frames: torch.Tensor, window: torch.Tensor, hop: int,
                 length: int) -> torch.Tensor:
    """frames [b, n_fft, F] -> signal [b, length], clamped-envelope normalisation."""
    b, n_fft, n_frames = frames.shape
    pad = n_fft // 2
    l_padded = (n_frames - 1) * hop + n_fft
    out = F.fold(frames * window[None, :, None], output_size=(1, l_padded),
                 kernel_size=(1, n_fft), stride=(1, hop))
    w2 = (window * window)[None, :, None].expand(b, n_fft, n_frames)
    norm = F.fold(w2, output_size=(1, l_padded), kernel_size=(1, n_fft), stride=(1, hop))
    out = out / norm.clamp(min=1e-3)
    return out.reshape(b, l_padded)[:, pad:pad + length]


def _spectral_filter(x: torch.Tensor, cutoff_fc: torch.Tensor, q: torch.Tensor,
                     wl: torch.Tensor, wh: torch.Tensor, wb: torch.Tensor,
                     slope: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """x [b, T] -> filtered [b, T] via per-frame analytic SVF magnitudes.

    cutoff_fc/wl/wh/wb/slope are per-frame [b, F]; q is [b].
    """
    n_fft, hop = FILT_NFFT, FILT_HOP
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, center=True, return_complex=True)
    mag = spec.abs()                                   # [b, nbins, F]
    freqs = torch.linspace(0.0, sr / 2.0, mag.shape[1], dtype=x.dtype, device=x.device)
    r = freqs[None, None, :] / cutoff_fc[:, :, None].clamp(min=10.0)   # [b, F, nbins]
    rq = r / q[:, None, None].clamp(min=0.1)
    denom = torch.sqrt((1.0 - r * r) ** 2 + rq * rq) + 1e-8
    h12 = (wl[:, :, None] * (1.0 / denom)
           + wh[:, :, None] * ((r * r) / denom)
           + wb[:, :, None] * (rq / denom))
    h24 = h12 * h12                                    # cascade = squared magnitude
    h = (1.0 - slope[:, :, None]) * h12 + slope[:, :, None] * h24
    frames = torch.fft.irfft(spec * h.permute(0, 2, 1), n=n_fft, dim=1)
    return _overlap_add(frames, window, hop, x.shape[1])


# ---------------------------------------------------------------------------
# per-sample reference implementation (used only by the sanity check)
# ---------------------------------------------------------------------------

def _svf_loop_eager(x: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor,
                    a3: torch.Tensor, k: torch.Tensor, wl: torch.Tensor,
                    wh: torch.Tensor, wb: torch.Tensor,
                    slope: torch.Tensor) -> torch.Tensor:
    """Reference 2-stage TPT SVF, sample-by-sample.  Slow; validation only."""
    b, t_len = x.shape
    ic1 = torch.zeros(b, dtype=x.dtype, device=x.device)
    ic2 = torch.zeros(b, dtype=x.dtype, device=x.device)
    jc1 = torch.zeros(b, dtype=x.dtype, device=x.device)
    jc2 = torch.zeros(b, dtype=x.dtype, device=x.device)
    ys: list[torch.Tensor] = []
    for t in range(t_len):
        a1t = a1[:, t]
        a2t = a2[:, t]
        a3t = a3[:, t]
        v0 = x[:, t]
        v3 = v0 - ic2
        v1 = a1t * ic1 + a2t * v3
        v2 = ic2 + a2t * ic1 + a3t * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        y1 = wl * v2 + wh * (v0 - k * v1 - v2) + wb * v1
        v3b = y1 - jc2
        u1 = a1t * jc1 + a2t * v3b
        u2 = jc2 + a2t * jc1 + a3t * v3b
        jc1 = 2.0 * u1 - jc1
        jc2 = 2.0 * u2 - jc2
        y2 = wl * u2 + wh * (y1 - k * u1 - u2) + wb * u1
        ys.append((1.0 - slope) * y1 + slope * y2)
    return torch.stack(ys, dim=1)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def reference_f0(u: torch.Tensor, f0: float = F0_HZ) -> torch.Tensor:
    """Soft level-weighted fundamental of the patch, for feature masks [b]."""
    cont = S.continuous(u)
    f1 = f0 * S.octave_multiplier(u, "osc1_oct") * 2.0 ** (cont["osc1_detune"] / 1200.0)
    f2 = f0 * S.octave_multiplier(u, "osc2_oct") * 2.0 ** (cont["osc2_detune"] / 1200.0)
    lvls = torch.stack([cont["osc1_level"] + cont["sub_level"] + 1e-3,
                        cont["osc2_level"] + 1e-3,
                        cont["noise_level"] + 1e-3], dim=1)
    f0t = torch.full_like(f1, float(f0))
    freqs = torch.stack([f1, f2, f0t], dim=1)
    w = lvls / lvls.sum(dim=1, keepdim=True)
    return (w * freqs).sum(dim=1).detach()


def _render_chunk(u: torch.Tensor, sr: int, dur: float, f0: float) -> torch.Tensor:
    """u [b, D] -> waveform [b, T]."""
    b = u.shape[0]
    t_len = int(round(sr * dur))
    t_axis = torch.arange(t_len, dtype=u.dtype, device=u.device) / sr

    cont = S.continuous(u)
    cho = S.choices(u)
    nyq = sr / 2.0

    # ---- modulation ------------------------------------------------------
    _, _, pitch_mult_t, tremolo_t = _lfo_blend(u, t_axis, cont, cho)
    tf = _frame_centers(t_len, FILT_HOP, u.dtype, u.device)
    _, cutoff_mult_f, _, _ = _lfo_blend(u, tf, cont, cho)

    # ---- oscillator frequencies -----------------------------------------
    fm_index = cont["fm_amount"][:, None]
    f1 = (f0 * S.octave_multiplier(u, "osc1_oct")[:, None]
          * 2.0 ** (cont["osc1_detune"][:, None] / 1200.0) * pitch_mult_t)
    f2 = (f0 * S.octave_multiplier(u, "osc2_oct")[:, None]
          * 2.0 ** (cont["osc2_detune"][:, None] / 1200.0) * pitch_mult_t)

    def cycles(freq: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(freq, dim=1) / sr * _TWO_PI

    # ---- oscillator 2 (modulator, computed first) ------------------------
    phase2 = cycles(f2)
    a2c = _osc_coeffs(cho["osc2_wave"], cont["osc2_pw"], f2[:, 0],
                      1.0 + 0.6 * cont["fm_amount"])
    osc2 = _harm_sum(phase2, a2c)

    # ---- oscillator 1 (PM from osc2) -------------------------------------
    phase1 = cycles(f1) + fm_index * osc2
    a1c = _osc_coeffs(cho["osc1_wave"], cont["osc1_pw"], f1[:, 0],
                      1.0 + 0.6 * cont["fm_amount"])
    osc1 = _harm_sum(phase1, a1c)

    # ---- sub (sine, one octave below osc1) -------------------------------
    sub = torch.sin(cycles(f1 * 0.5))

    # ---- noise -----------------------------------------------------------
    white, pink = _noise_buffers(t_len, u.dtype, u.device)
    noise = ((1.0 - cont["noise_color"])[:, None] * white[None, :]
             + cont["noise_color"][:, None] * pink[None, :])

    mix = (cont["osc1_level"][:, None] * osc1
           + cont["osc2_level"][:, None] * osc2
           + cont["sub_level"][:, None] * sub
           + cont["noise_level"][:, None] * noise)
    mix = torch.tanh(cont["filter_drive"][:, None] * mix)

    # ---- filter (STFT domain, per-frame cutoff) --------------------------
    fenv_f = _envelope(tf, cont["fenv_attack"][:, None],
                       cont["fenv_decay"][:, None], cont["fenv_sustain"][:, None])
    cutoff_fc = (cont["filter_cutoff"][:, None]
                 * 2.0 ** (cont["filter_env_amount"][:, None] * fenv_f * 5.0)
                 * cutoff_mult_f).clamp(20.0, 0.92 * nyq)
    ftype = cho["filter_type"]
    n_frames = 1 + t_len // FILT_HOP
    wl = ftype[:, 0:1].expand(b, n_frames)
    wh = ftype[:, 1:2].expand(b, n_frames)
    wb = ftype[:, 2:3].expand(b, n_frames)
    slope = cont["filter_slope"][:, None].expand(b, n_frames)
    sig = _spectral_filter(mix, cutoff_fc, cont["filter_res"], wl, wh, wb, slope, sr)

    # ---- VCA (sample-accurate amp envelope) + master ---------------------
    aenv = _envelope(t_axis, cont["aenv_attack"][:, None],
                     cont["aenv_decay"][:, None], cont["aenv_sustain"][:, None])
    y = sig * aenv * tremolo_t
    y = torch.tanh(cont["master_drive"][:, None] * y)

    # ---- RMS normalise ---------------------------------------------------
    rms = torch.sqrt(torch.mean(y * y, dim=1, keepdim=True) + 1e-12)
    return y / rms * TARGET_RMS


def render(u: torch.Tensor, sr: int = SR, dur: float = DUR, f0: float = F0_HZ,
           chunk_b: int | None = 64) -> torch.Tensor:
    """Render a batch of patches u [B, D] to waveforms [B, T]."""
    if u.dim() != 2:
        raise ValueError(f"expected [B, D] patch batch, got {tuple(u.shape)}")
    if chunk_b is None or chunk_b >= u.shape[0]:
        return _render_chunk(u, sr, dur, f0)
    outs = [_render_chunk(u[s:s + chunk_b], sr, dur, f0)
            for s in range(0, u.shape[0], chunk_b)]
    return torch.cat(outs, dim=0)
