"""Apply the three claim revisions by line range (robust to escaping).

Find the start line by a unique substring, find the end line by a unique
substring at or after it, replace that inclusive range with new text.
"""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENT = "@" + "@"
p = Path("main.tex")
lines = p.read_text(encoding="utf-8").split("\n")


def tex(t: str) -> str:
    return t.replace(SENT, BS)


def replace_range(start_marker: str, end_marker: str, new_block: str, why: str) -> None:
    global lines
    start_marker, end_marker = tex(start_marker), tex(end_marker)
    i = next((k for k, ln in enumerate(lines) if start_marker in ln), None)
    assert i is not None, f"start marker not found: {why}"
    j = next((k for k in range(i, len(lines)) if end_marker in lines[k]), None)
    assert j is not None, f"end marker not found: {why}"
    block = tex(new_block).split("\n")
    lines = lines[:i] + block + lines[j + 1:]
    print(f"  {why}: replaced lines {i + 1}-{j + 1} with {len(block)}")


# A) contribution (5)
replace_range(
    "@@item @@emph{Diagnostics that locate the failure}",
    "cube's discrete geometry---not the optimiser---as the binding constraint.",
    """@@item @@emph{Diagnostics that localise the failure}: a residual
  decomposition, a discrete-jump correlation, an ablation of the monotone
  accept rule, and a block-balanced-metric control
  (@@S@@ref{sec:res-diagnostics}). Together they rule out the accept rule as
  the binding constraint, show that the leftover error is spread across the
  objective roughly in proportion to dimensionality (per-dimension shares
  $0.85$--$2.34@@times$ uniform, i.e.@@ only mildly overweighted in the
  discrete-sensitive blocks), and motivate re-weighting or learning the metric
  rather than a more elaborate search.""",
    "contribution5")

# B) §4.4 residual-decomposition bullet
replace_range(
    "If the residual concentrates in the harmonic blocks",
    "reading of the plateau.",
    """Concentration in the harmonic blocks---the ones that respond to discrete
  choices---would be direct evidence for the discrete-jump reading of the
  plateau; as @@S@@ref{sec:res-diagnostics} reports, the measured concentration
  is mild ($2.3@@times$ uniform per dimension for the harmonic block, while the
  64-dimensional log-mel block absorbs $72@@%$ of the residual simply because of
  its size).""",
    "protocol-bullet")

# C) §6.1 second paragraph
replace_range(
    "The ablation and the residual decomposition in @@S@@ref{sec:res-diagnostics} test",
    "latent-space phase of the roadmap.",
    """The ablation and the residual decomposition in @@S@@ref{sec:res-diagnostics} test
this story, and they only partly support it. The free walk does help on a
minority of targets, so the accept rule costs something; but the residual is
@@emph{not} concentrated in the discrete-sensitive blocks---it is spread across
the objective roughly in proportion to dimensionality, and the block with the
largest per-dimension share (the harmonic ratio, $2.3@@times$ uniform) holds
only $2.7@@%$ of the remaining squared distance. The valley-crossing account
therefore explains at most a small part of the plateau.
@@medskip

What the residual does say is that the objective's @@emph{weighting} is a first
-order suspect: 64 of the 85 dimensions are log-mel, so envelope mismatches
dominate the number being minimised, while the perceptually load-bearing
discrete distinctions live in single dimensions that carry almost no weight.
The block-balanced control in @@S@@ref{sec:res-diagnostics} tests this directly.
Together with the CMA-ES result (exploration alone does not help: $3.7@@times$
the budget, no median improvement), the evidence points away from "the search
algorithm is wrong" and toward "the number being optimised is wrong"---which is
what the embedding-swap and metric-learning phases of the roadmap address, with
the latent-space phase retained for the reachability problem that the discrete
axes still pose.""",
    "disc-61")

p.write_text("\n".join(lines), encoding="utf-8")
print("claims revised; TBD markers:", p.read_text(encoding="utf-8").count("TBD"))
