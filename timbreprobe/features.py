"""Differentiable DSP feature embedding ("v0 perceptual space").

This is the lower-anchor embedding of E1/E2: a handcrafted spectral + temporal
feature vector computed entirely with torch ops, so the whole
patch -> render -> embedding chain stays differentiable (needed by the true
gradient navigator) and needs no model downloads.

It is deliberately *not* presented as a perceptual model.  It plays two roles:
  1. it lets the navigability experiment (E2) run end-to-end today, and
  2. it is the sanity floor for E1 -- a pretrained embedding (CLAP / MuQ) that
     cannot beat this baseline on human similarity judgements is not useful.

Every feature is smooth (or smooth-a.e.) in the waveform: hard steps such as
argmax / threshold crossings are replaced by soft variants (softmax-based peak
position, sigmoid-based rolloff), otherwise their gradient would be zero
almost everywhere and the true-gradient navigator would silently lose signal.

Output: FRAW = 85 raw dimensions, later z-scored with corpus statistics.
"""

from __future__ import annotations

import math

import torch

SR = 16_000
N_FFT = 512
HOP = 128
N_MELS = 32
FMIN = 40.0
FMAX = 7600.0
_EPS = 1e-8
FRAW = 85

# Canonical grouping of the 85 dimensions, used by reporting and by the
# residual diagnostics (`python -m timbreprobe.cli diagnose`).
FEATURE_BLOCKS: dict[str, slice] = {
    "log-mel mean+std": slice(0, 64),
    "centroid": slice(64, 66),
    "bandwidth": slice(66, 68),
    "rolloff (soft)": slice(68, 70),
    "flatness": slice(70, 72),
    "flux": slice(72, 74),
    "slope": slice(74, 76),
    "crest": slice(76, 77),
    "high/low band ratio": slice(77, 78),
    "envelope stats": slice(78, 82),
    "zero-crossing (smooth)": slice(82, 83),
    "harmonic ratio": slice(83, 84),
    "odd/even ratio": slice(84, 85),
}


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_bank(n_fft: int = N_FFT, sr: int = SR, n_mels: int = N_MELS,
              fmin: float = FMIN, fmax: float = FMAX) -> torch.Tensor:
    """[n_mels, n_fft//2+1] triangular mel filterbank (HTK mel)."""
    n_bins = n_fft // 2 + 1
    f = torch.linspace(0.0, sr / 2.0, n_bins)
    mel_pts = torch.linspace(float(_hz_to_mel(torch.tensor(fmin))),
                             float(_hz_to_mel(torch.tensor(fmax))), n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bank = torch.zeros(n_mels, n_bins)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        left = (f - lo) / max(ctr - lo, 1e-6)
        right = (hi - f) / max(hi - ctr, 1e-6)
        bank[m] = torch.clamp(torch.minimum(left, right), min=0.0)
    return bank


def embed(wave: torch.Tensor, f0_hz: torch.Tensor | None = None,
          sr: int = SR) -> torch.Tensor:
    """waveform [B, T] -> raw feature matrix [B, FRAW]."""
    b, t_len = wave.shape
    device, dtype = wave.device, wave.dtype

    window = torch.hann_window(N_FFT, dtype=dtype, device=device)
    spec = torch.stft(wave, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=window, center=True, return_complex=True)
    mag = spec.abs()                                    # [B, n_bins, F]
    power = mag * mag
    n_bins, n_frames = mag.shape[1], mag.shape[2]
    freqs = torch.linspace(0.0, sr / 2.0, n_bins, dtype=dtype, device=device)

    feats = []

    # ---- log-mel statistics (64) -----------------------------------------
    mel = _mel_bank(sr=sr).to(dtype=dtype, device=device)  # [M, n_bins]
    mel_spec = mel @ power                                 # [B, M, F]
    log_mel = torch.log(mel_spec + _EPS)
    feats.append(log_mel.mean(dim=2))
    feats.append(log_mel.std(dim=2))

    # ---- spectral shape per frame (10) -----------------------------------
    m_sum = mag.sum(dim=1) + _EPS
    centroid = (freqs[None, :, None] * mag).sum(dim=1) / m_sum          # [B,F]
    spread = (freqs[None, :, None] - centroid[:, None, :]).abs()
    bandwidth = (spread * mag).sum(dim=1) / m_sum
    cum = torch.cumsum(mag, dim=1)
    thr = 0.85 * cum[:, -1:, :]
    w_ro = torch.sigmoid((cum - thr) / (0.05 * cum[:, -1:, :] + _EPS))
    rolloff = (freqs[None, :, None] * w_ro).sum(dim=1) / (w_ro.sum(dim=1) + _EPS)
    flatness = torch.exp(torch.log(mag + _EPS).mean(dim=1)) / (mag.mean(dim=1) + _EPS)
    feats.append(centroid.mean(dim=1, keepdim=True))
    feats.append(centroid.std(dim=1, keepdim=True))
    feats.append(bandwidth.mean(dim=1, keepdim=True))
    feats.append(bandwidth.std(dim=1, keepdim=True))
    feats.append(rolloff.mean(dim=1, keepdim=True))
    feats.append(rolloff.std(dim=1, keepdim=True))
    feats.append(flatness.mean(dim=1, keepdim=True))
    feats.append(flatness.std(dim=1, keepdim=True))
    flux = torch.sqrt(((mag[:, :, 1:] - mag[:, :, :-1]) ** 2).sum(dim=1))
    flux = flux / (torch.sqrt((mag[:, :, 1:] ** 2).sum(dim=1)) + _EPS)
    feats.append(flux.mean(dim=1, keepdim=True))
    feats.append(flux.std(dim=1, keepdim=True))

    # ---- spectral slope / crest / band ratio (4) -------------------------
    log_mag = torch.log(mag + _EPS)
    f_mean = freqs.mean()
    f_var = ((freqs - f_mean) ** 2).mean() + _EPS
    centered = log_mag - log_mag.mean(dim=1, keepdim=True)
    slope = ((freqs[None, :, None] - f_mean) * centered).mean(dim=1) / f_var
    feats.append(slope.mean(dim=1, keepdim=True))
    feats.append(slope.std(dim=1, keepdim=True))
    crest = (mag.max(dim=1).values / (mag.mean(dim=1) + _EPS))
    feats.append(crest.mean(dim=1, keepdim=True))
    hi_band = (freqs > 4000.0)[None, :, None] * power
    lo_band = ((freqs > 200.0) & (freqs < 2000.0))[None, :, None] * power
    band_ratio = torch.log((hi_band.sum(dim=(1, 2)) + _EPS) / (lo_band.sum(dim=(1, 2)) + _EPS))
    feats.append(band_ratio[:, None])

    # ---- temporal envelope statistics (4) --------------------------------
    energy = power.sum(dim=1)                                       # [B,F]
    e_sum = energy.sum(dim=1) + _EPS
    t_idx = torch.arange(n_frames, dtype=dtype, device=device)[None, :] / n_frames
    env_centroid = (t_idx * energy).sum(dim=1) / e_sum
    env_spread = (((t_idx - env_centroid[:, None]) ** 2) * energy).sum(dim=1) / e_sum
    frontback = torch.log(
        ((t_idx < 0.5) * energy).sum(dim=1) + _EPS) - torch.log(
        ((t_idx >= 0.5) * energy).sum(dim=1) + _EPS)
    tau = 0.05
    soft_peak = tau * torch.logsumexp(energy / tau, dim=1)          # smooth max
    dyn = 10.0 * torch.log10((soft_peak + _EPS) / (energy.mean(dim=1) + _EPS))
    feats.append(env_centroid[:, None])
    # sqrt(0) has an infinite derivative; the epsilon keeps the backward
    # finite for degenerate envelopes (e.g. all energy in one frame).
    feats.append(torch.sqrt(env_spread + 1e-12)[:, None])
    feats.append(frontback[:, None])
    feats.append(dyn[:, None])

    # ---- zero-crossing rate, smooth proxy (1) ----------------------------
    x1, x2 = wave[:, :-1], wave[:, 1:]
    scale = (wave.pow(2).mean(dim=1, keepdim=True) * 0.05 + _EPS)
    crossings = torch.sigmoid(-(x1 * x2) / scale)
    feats.append((crossings.mean(dim=1) * sr / 2.0)[:, None])

    # ---- harmonic structure, needs the reference f0 (2) ------------------
    if f0_hz is not None:
        f0c = f0_hz[:, None, None].clamp(min=30.0)                  # [b,1,1]
        kmax = int(min(40, math.floor(sr / 2.0 / 30.0)))
        k = torch.arange(1, kmax + 1, dtype=dtype, device=device)[None, :, None]
        dist = (freqs[None, None, :] - k * f0c) / (0.35 * f0c)      # [b,K,nbins]
        kern = torch.exp(-(dist ** 2))
        w_harm = kern.sum(dim=1)                                    # [b, nbins]
        mag_sum_b = mag.sum(dim=1) + _EPS                           # [b, F]
        harm_energy = (w_harm[:, :, None] * mag).sum(dim=1) / mag_sum_b
        feats.append(harm_energy.mean(dim=1, keepdim=True))
        odd_mask = (torch.arange(1, kmax + 1, device=device) % 2 == 1).to(dtype)[None, :, None]
        w_odd = (kern * odd_mask).sum(dim=1)
        w_even = (kern * (1.0 - odd_mask)).sum(dim=1)
        odd = (w_odd[:, :, None] * mag).sum(dim=1)
        even = (w_even[:, :, None] * mag).sum(dim=1)
        odd_even = torch.log((odd + _EPS) / (even + _EPS))
        feats.append(odd_even.mean(dim=1, keepdim=True))
    else:
        feats.append(torch.zeros(b, 1, dtype=dtype, device=device))
        feats.append(torch.zeros(b, 1, dtype=dtype, device=device))

    out = torch.cat([f if f.dim() == 2 else f[:, None] for f in feats], dim=1)
    if out.shape[1] != FRAW:  # pragma: no cover - guards silent feature drift
        raise RuntimeError(f"feature dimension drift: got {out.shape[1]}, expected {FRAW}")
    return out


@torch.no_grad()
def fit_normalizer(raw: torch.Tensor) -> dict:
    """Fit z-score statistics on a corpus feature matrix [N, FRAW]."""
    mean = raw.mean(dim=0)
    std = raw.std(dim=0).clamp(min=1e-6)
    return {"mean": mean, "std": std}


def normalize(raw: torch.Tensor, stats: dict) -> torch.Tensor:
    """Z-score `raw` with stats from `fit_normalizer`.

    Stats may be numpy arrays (as stored on disk) or tensors; the input may
    require grad.  A plain `raw - np_array` would make numpy try to convert the
    *tensor* (and fail for a grad-requiring one), so promote the stats instead.
    """
    mean, std = stats["mean"], stats["std"]
    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, dtype=raw.dtype, device=raw.device)
        std = torch.as_tensor(std, dtype=raw.dtype, device=raw.device)
    return (raw - mean) / std
