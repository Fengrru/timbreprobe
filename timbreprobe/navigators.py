"""E2 navigators: given a target embedding, can we find parameters that reach it?

Implemented strategies
----------------------
* `grad_true`      -- projected Adam with gradients through the real renderer
                      (oracle upper bound; the expensive reference)
* `grad_surrogate` -- projected Adam through the learned surrogate
                      (zero renders during search)
* `grad_reground`  -- surrogate search with periodic re-grounding: every N
                      steps the current patches are rendered for real, appended
                      to a replay buffer and the surrogate copy is fine-tuned,
                      so accuracy is restored exactly where the optimiser
                      actually is (the "learned surrogate accelerates, DSP
                      ground truth validates" loop)
* `sep_cmaes`      -- separable CMA-ES (Ros & Hansen 2008), batched per target,
                      derivative-free baseline on the true objective
* retrieval / random baselines: best-of-N draws from the training corpus

All optimisers work on the full batch of targets simultaneously, which matches
how the renderer is shape-efficient on CPU.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch

from . import features, synth
from .config import Config
from .surrogate import Surrogate


class PerceptualSpace:
    """Patch -> embedding, with render bookkeeping and the corpus normaliser."""

    def __init__(self, stats: dict, cfg: Config):
        self.stats = stats
        self.cfg = cfg
        self.patch_renders = 0
        self.weights = self._metric_weights(cfg)

    @staticmethod
    def _metric_weights(cfg: Config) -> torch.Tensor:
        """Per-dimension weights for the embedding distance.

        `balanced` equalises each feature block's expected contribution:
        after z-scoring every dim has unit variance, so a block of d dims
        contributes d on average; weighting it by 1/sqrt(d) makes blocks
        comparable and stops the 64-dim log-mel block from dominating the
        objective by sheer dimensionality.  Rescaled so the distance scale
        stays comparable to the uniform case.
        """
        w = torch.ones(features.FRAW, dtype=torch.float32)
        if getattr(cfg, "metric", "uniform") == "balanced":
            for _, blk in features.FEATURE_BLOCKS.items():
                w[blk] = 1.0 / math.sqrt(blk.stop - blk.start)
            w = w / torch.linalg.vector_norm(w) * math.sqrt(features.FRAW)
        return w

    def embed(self, u: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        self.patch_renders += int(u.shape[0])
        if no_grad:
            with torch.no_grad():
                w = synth.render(u, sr=self.cfg.sr, dur=self.cfg.dur, chunk_b=self.cfg.chunk_b)
                z = features.embed(w, synth.reference_f0(u, synth.F0_HZ), sr=self.cfg.sr)
                return features.normalize(z, self.stats)
        w = synth.render(u, sr=self.cfg.sr, dur=self.cfg.dur, chunk_b=self.cfg.chunk_b)
        z = features.embed(w, synth.reference_f0(u, synth.F0_HZ), sr=self.cfg.sr)
        return features.normalize(z, self.stats)

    def dist(self, u: torch.Tensor, z_target: torch.Tensor, no_grad: bool = True
             ) -> torch.Tensor:
        z = self.embed(u, no_grad=no_grad)
        return torch.linalg.vector_norm((z - z_target) * self.weights.to(z.device), dim=1)


# ---------------------------------------------------------------------------
# gradient navigation
# ---------------------------------------------------------------------------

def nav_gradient(u0: torch.Tensor, z_target: torch.Tensor, space: PerceptualSpace,
                 steps: int, lr: float, model: Surrogate | None = None,
                 log_every: int = 5, trust_radius: float = 0.0,
                 monotone: bool = True
                 ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Projected Adam with a per-target adaptive step (single backtrack per step).

    Plain fixed-step Adam diverges on this objective: measured on the smoke
    corpus, 30 steps at lr=0.05 pushed the *true* distance from 3.08 up to 5.29
    within five steps and it never recovered.  The space is 31-d in [0,1] with
    log-scaled dimensions (a 0.1 change in the cutoff coordinate is a 2.4x
    change in Hz), so any fixed step is either far too large or uselessly small.

    Here the Adam direction is kept, but the step size is per-target and
    adaptive: a candidate is accepted only if it does not increase the
    objective; the step grows 1.3x on success and halves on failure.  Requires
    one extra objective evaluation per step (no gradient), so with
    `model=None` the search still only uses real renders.

    `trust_radius` > 0 confines the search to an L2 ball around u0 (per target):
    for surrogate runs this separates surrogate error inside the data region
    from drifting off the manifold where the surrogate is arbitrary.

    `monotone=False` drops the accept/reject rule (every candidate is taken)
    while keeping the adaptive step.  This is the ablation separating "the
    landscape is hard" from "monotone search cannot cross valleys": only a
    non-monotone walk can pass through transiently worse states, which is what
    a discrete waveform/octave jump requires.
    """
    u = u0.clone().detach()
    m = torch.zeros_like(u)
    v = torch.zeros_like(u)
    step_size = torch.full((u.shape[0], 1), float(lr), dtype=u.dtype, device=u.device)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def objective(x: torch.Tensor, need_grad: bool) -> torch.Tensor:
        if model is None:
            return space.dist(x, z_target, no_grad=not need_grad)
        return torch.linalg.vector_norm(model(x) - z_target, dim=1)

    with torch.no_grad():
        d_cur = objective(u, need_grad=False)
    traj: list[tuple[int, torch.Tensor]] = [(0, d_cur.clone())]

    for step in range(1, steps + 1):
        x = u.clone().requires_grad_(True)
        d = objective(x, need_grad=True)
        g, = torch.autograd.grad(d.mean(), x)
        with torch.no_grad():
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            direction = (m / (1 - b1 ** step)) / ((v / (1 - b2 ** step)).sqrt() + eps)

            cand = (u - step_size * direction).clamp(0.0, 1.0)
            if trust_radius > 0.0:
                delta = cand - u0
                norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
                cand = u0 + delta * torch.clamp(trust_radius / (norm + 1e-8), max=1.0)

            d_cand = objective(cand, need_grad=False)
            improved = d_cand <= d_cur
            # a non-finite candidate is never acceptable: with the accept gate
            # off, one bad gradient would otherwise poison the whole walk
            take = improved if monotone else torch.ones_like(improved)
            take = take & torch.isfinite(d_cand)
            u = torch.where(take[:, None], cand, u)
            d_cur = torch.where(take, d_cand, d_cur)
            step_size = torch.where(improved[:, None],
                                    (step_size * 1.3).clamp(max=max(4.0 * lr, 1e-4)),
                                    (step_size * 0.5).clamp(min=lr / 64.0))
        if step % log_every == 0 or step == steps:
            traj.append((step, d_cur.clone()))
    return u.detach(), traj


def nav_reground(u0: torch.Tensor, z_target: torch.Tensor, space: PerceptualSpace,
                 model: Surrogate, cfg: Config, u_anchor: torch.Tensor,
                 z_anchor: torch.Tensor, trust_radius: float = 0.0
                 ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]], int]:
    """Surrogate search with periodic re-grounding against the true renderer.

    Returns (u_final, trajectory, patch_renders_used).
    """
    work = copy.deepcopy(model)
    work.train()
    opt = torch.optim.Adam(work.parameters(), lr=cfg.lr * 0.3)

    buf_u = [u_anchor.clone()]
    buf_z = [z_anchor.clone()]
    u = u0.clone().detach()
    traj: list[tuple[int, torch.Tensor]] = []
    renders = 0
    rounds = max(1, cfg.steps_surrogate // cfg.reground_every)
    step = 0
    for _ in range(rounds):
        u, sub = nav_gradient(u, z_target, space, steps=cfg.reground_every,
                              lr=cfg.lr_opt, model=work, log_every=cfg.log_every,
                              trust_radius=trust_radius)
        step += cfg.reground_every
        traj.extend([(step - cfg.reground_every + s, d) for s, d in sub])
        # ---- re-ground on the true renderer at the current operating point --
        with torch.no_grad():
            z_true = space.embed(u)
        renders += int(u.shape[0])
        buf_u.append(u.clone())
        buf_z.append(z_true.clone())
        bu = torch.cat(buf_u, dim=0)[-cfg.reground_buffer:]
        bz = torch.cat(buf_z, dim=0)[-cfg.reground_buffer:]
        work.train()
        for _ in range(cfg.reground_inner):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(work(bu), bz)
            loss.backward()
            opt.step()
    # The trajectory stays in the *surrogate* objective.  Appending the final
    # true-renderer distance here would mix two scales in one curve (it put a
    # spurious 2.5x jump at the end of the plotted trajectory) and would also
    # duplicate the caller's own true-objective evaluation of the result.
    return u.detach(), traj, renders


# ---------------------------------------------------------------------------
# separable CMA-ES (batched over targets)
# ---------------------------------------------------------------------------

def sep_cmaes(u0: torch.Tensor, f_eval: Callable[[torch.Tensor], torch.Tensor],
              gens: int, pop: int, sigma0: float = 0.25,
              log_every: int = 1, seed: int = 0
              ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]], torch.Tensor]:
    """Separable CMA-ES, one independent strategy per row of u0.

    `f_eval` maps [N, D] patches -> [N] costs (lower is better) and must be
    batched; it is called once per generation with M*pop rows.

    Reference: Ros & Hansen, "A Simple Modification in CMA-ES Achieving Linear
    Time and Space Complexity" (PPSN 2008).
    """
    torch.manual_seed(seed)
    M, D = u0.shape
    lam, mu = pop, max(1, pop // 2)
    w = torch.log(torch.tensor(mu + 0.5)) - torch.log(torch.arange(1, mu + 1, dtype=torch.float32))
    w = w / w.sum()
    mueff = 1.0 / float((w ** 2).sum())

    cc = (4.0 + mueff / D) / (D + 4.0 + 2.0 * mueff / D)
    c1 = 2.0 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((D + 2.0) ** 2 + mueff))
    cs = (mueff + 2.0) / (D + mueff + 5.0)
    ds = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (D + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D))

    m = u0.clone()
    cov = torch.ones(M, D, dtype=u0.dtype)
    sigma = torch.full((M,), float(sigma0), dtype=u0.dtype)
    ps = torch.zeros(M, D, dtype=u0.dtype)
    pc = torch.zeros(M, D, dtype=u0.dtype)

    best_x = u0.clone()
    best_f = f_eval(u0.clone())
    traj: list[tuple[int, torch.Tensor]] = [(0, best_f.clone())]

    sqrt_w = math.sqrt(cs * (2.0 - cs) * mueff)
    sqrt_cc = math.sqrt(cc * (2.0 - cc) * mueff)
    for g in range(1, gens + 1):
        z = torch.randn(M, lam, D, dtype=u0.dtype)
        y = z * torch.sqrt(cov)[:, None, :]
        x = (m[:, None, :] + sigma[:, None, None] * y).clamp(0.0, 1.0)
        fit = f_eval(x.reshape(M * lam, D)).reshape(M, lam)

        bi = fit.argmin(dim=1, keepdim=True)
        cand_f = fit.gather(1, bi).squeeze(1)
        cand_x = x.gather(1, bi[:, :, None].expand(M, 1, D)).squeeze(1)
        upd = cand_f < best_f
        best_f = torch.where(upd, cand_f, best_f)
        best_x = torch.where(upd[:, None], cand_x, best_x)

        order = torch.argsort(fit, dim=1)[:, :mu]
        xs = x.gather(1, order[:, :, None].expand(M, mu, D))
        m_new = (w[None, :, None] * xs).sum(dim=1)
        y_w = (m_new - m) / sigma[:, None]
        ys = (xs - m[:, None, :]) / sigma[:, None, None]

        ps = (1.0 - cs) * ps + sqrt_w * y_w
        norm_ps = torch.linalg.vector_norm(ps, dim=1)
        sigma = sigma * torch.exp((cs / ds) * (norm_ps / chi_n - 1.0))
        hsig = (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (g + 1)))
                < (1.4 + 2.0 / (D + 1.0)) * chi_n)
        pc = (1.0 - cc) * pc + hsig[:, None].to(u0.dtype) * sqrt_cc * y_w
        cov = ((1.0 - c1 - cmu) * cov + c1 * (pc * pc)
               + cmu * (w[None, :, None] * ys * ys).sum(dim=1))
        m = m_new

        if g % log_every == 0 or g == gens:
            traj.append((g, best_f.clone()))
    return best_x, traj, best_f


# ---------------------------------------------------------------------------
# retrieval / random baselines
# ---------------------------------------------------------------------------

def retrieval_distances(z_target: torch.Tensor, z_pool: torch.Tensor,
                        draws: int | None = None, seed: int = 0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-pool distance and (optionally) best-of-`draws` random draws."""
    d = torch.cdist(z_target, z_pool)                    # [M, N]
    best = d.min(dim=1).values
    if draws is None:
        return best, best
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, z_pool.shape[0], (z_target.shape[0], draws), generator=g)
    sampled = d.gather(1, idx)
    return best, sampled.min(dim=1).values
