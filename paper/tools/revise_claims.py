"""Revise the paper's claims that the residual evidence does not support.

The diagnostics show the leftover error is spread roughly in proportion to
dimensionality (per-dimension shares 0.85-2.34x uniform), not concentrated in
the discrete-sensitive blocks as the strong valley-crossing story predicts.
Contribution (5) and the diagnostics protocol text said "discrete geometry is
the binding constraint"; that has to go.
"""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENT = "@" + "@"
p = Path("main.tex")
s = p.read_text(encoding="utf-8")


def tex(t: str) -> str:
    return t.replace(SENT, BS)


def rep(a: str, b: str, why: str) -> None:
    global s
    a, b = tex(a), tex(b)
    assert a in s, f"anchor not found: {why}"
    s = s.replace(a, b, 1)


# ---- contribution (5) -----------------------------------------------------
rep("""@@item @@emph{Diagnostics that locate the failure}: a residual decomposition, a
  discrete-jump correlation, and an explicit ablation of the monotone accept
  rule (@@S@@ref{sec:res-diagnostics}), which together identify the parameter
  cube's discrete geometry---not the optimiser---as the binding constraint.""",
    """@@item @@emph{Diagnostics that localise the failure}: a residual
  decomposition, a discrete-jump correlation, an ablation of the monotone
  accept rule, and a block-balanced-metric control
  (@@S@@ref{sec:res-diagnostics}). Together they rule out the accept rule as
  the binding constraint, show that the leftover error is spread across the
  objective roughly in proportion to dimensionality (per-dimension shares
  $0.85$--$2.34@@times$ uniform, i.e.@@ only mildly overweighted in the
  discrete-sensitive blocks), and motivate re-weighting or learning the metric
  rather than a more elaborate search.""", "contribution5")

# ---- diagnostics protocol: soften the pre-registered expectation ----------
rep("""  $z^@@star$. Concentration in the harmonic blocks---the ones that respond to
  discrete choices---would be direct evidence for the discrete-jump reading of
  the plateau; as @@S@@ref{sec:res-diagnostics} reports, the measured
  concentration is mild (2.3@@times{} uniform per dimension for the harmonic
  block, while the 64-dimensional log-mel block absorbs 72@@% of the residual
  simply because of its size).""",
    """  $z^@@star$. Concentration in the harmonic blocks---the ones that respond to
  discrete choices---would be direct evidence for the discrete-jump reading of
  the plateau; as @@S@@ref{sec:res-diagnostics} reports, the measured
  concentration is mild (2.3@@times{} uniform per dimension for the harmonic
  block, while the 64-dimensional log-mel block absorbs 72@@% of the residual
  simply because of its size).""", "protocol")

# ---- discussion 6.1: replace the too-strong reading ----------------------
rep("""The ablation and the residual decomposition in @@S@@ref{sec:res-diagnostics} test
this directly. Note that exploration alone does not rescue the problem: CMA-ES
is non-monotone by construction, and with $3.7@@times$ the render budget it still
fails to improve the median target at all (@@rCMA). The consistent reading is that
@@emph{raw parameter space is the wrong search space}: the geometry that makes a
discrete morph cheap is not the geometry of the parameter cube. This is the
empirical basis for the latent-space phase of the roadmap.""",
    """The ablation and the residual decomposition in @@S@@ref{sec:res-diagnostics} test
this story, and they only partly support it. The free walk does help on a
minority of targets, so the accept rule costs something; but the residual is
\emph{not} concentrated in the discrete-sensitive blocks---it is spread across
the objective roughly in proportion to dimensionality, and the block whose
per-dimension share is largest (the harmonic ratio, $2.3@@times$ uniform) holds
only $2.7@@%$ of the remaining squared distance. The valley-crossing account
therefore explains at most a small part of the plateau.

What the residual does say is that the objective's \emph{weighting} is a first
- 64 of 85 dimensions are log-mel, so envelope mismatches dominate the number
the optimiser is minimising, while the perceptually load-bearing discrete
distinctions live in single dimensions that carry almost no weight. The control
experiment in @@S@@ref{sec:res-diagnostics} tests exactly this by equalising the
blocks' contributions. Together with the CMA-ES result (exploration alone does
not help: $3.7@@times$ the budget, no median improvement), the evidence points
away from "the search algorithm is wrong" and toward "the number being optimised
is wrong"---which is what the embedding-swap and metric-learning phases of the
roadmap address, with the latent-space phase retained for the reachability
problem that the discrete axes still pose.""", "disc-61")

p.write_text(s, encoding="utf-8")
print("claims revised; TBD markers:", s.count("TBD"))
