"""One-off: fill the final seed-0/seed-1 numbers into paper/main.tex.

Written as a file (not a shell heredoc) because this environment's shell
collapses backslash pairs, which would corrupt every LaTeX escape.
"""

from __future__ import annotations

from pathlib import Path

B = chr(92)          # backslash
p = Path("main.tex")
s = p.read_text(encoding="utf-8")


def rep(a: str, b: str, why: str) -> None:
    """Replace with @@ as a backslash placeholder."""
    global s
    a, b = a.replace("@@", B), b.replace("@@", B)
    assert a in s, f"anchor not found: {why}"
    s = s.replace(a, b, 1)


# ---- macros ---------------------------------------------------------------
for old, new, why in (
    ("{@@nInit}{2.69}", "{@@nInit}{2.74}", "nInit"),
    ("{@@nRand}{4.05}", "{@@nRand}{4.08}", "nRand"),
    ("{@@rRetU}{1.39}", "{@@rRetU}{1.45}", "rRetU"),
    ("{@@rGS}{2.12}", "{@@rGS}{2.59}", "rGS"),
    ("{@@rGST}{1.44}", "{@@rGST}{1.36}", "rGST"),
    ("{@@rGR}{1.69}", "{@@rGR}{1.60}", "rGR"),
    ("{@@rGT}{0.84}", "{@@rGT}{0.875}", "rGT"),
    ("{@@rGTci}{[0.77, 0.94]}", "{@@rGTci}{[0.81, 0.92]}", "rGTci"),
    ("{@@rCMA}{0.96}", "{@@rCMA}{1.00}", "rCMA"),
    ("{@@sOneGT}{TBD}", "{@@sOneGT}{0.877}", "sOneGT"),
    ("{@@sOneCMA}{TBD}", "{@@sOneCMA}{1.000}", "sOneCMA"),
    ("{@@sOneGS}{TBD}", "{@@sOneGS}{2.313}", "sOneGS"),
    ("{@@nCMAfin}{TBD}", "{@@nCMAfin}{2.65}", "nCMAfin"),
    ("{@@nGTfin}{TBD}", "{@@nGTfin}{2.35}", "nGTfin"),
):
    if old in s:
        s = s.replace(old, new, 1)
    else:
        print(f"  (skip, already set: {why})")

# extra macros used by the new text
if "{@@sOneGR}" not in s:
    s = s.replace("@@newcommand{@@sOneGS}{2.313}",
                  "@@newcommand{@@sOneGS}{2.313}\n"
                  "@@newcommand{@@sOneGR}{1.475}\n"
                  "@@newcommand{@@sOneGST}{1.376}\n"
                  "@@newcommand{@@sOneRetU}{1.499}", 1)

# ---- abstract -------------------------------------------------------------
rep("""derivative-free baseline (separable CMA-ES) spends $3.7@@times$ more renders and
ends worse (@@rCMA). All three surrogate-based strategies end @@emph{worse than
not searching} (ratio @@rGST--@@rGS), and---the diagnostic signature---they""",
    """derivative-free baseline (separable CMA-ES) spends $3.7@@times$ more renders and
makes no measurable progress at all (@@rCMA: its median target is unchanged).
All three surrogate-based strategies end @@emph{worse than not searching}
(ratio @@rGST--@@rGS), and---the diagnostic signature---they""", "abstract")

# ---- results (1) ----------------------------------------------------------
rep("""whose interval excludes 1.0: a median reduction of roughly $15@@%$, with
$@@succGT$ of targets more than halving their distance---at the cost of
$@@renGT$ renders. The plateau is visible in Figure~@@ref{fig:e2curves}: the median
falls for about fifteen steps and then stops. Per-target quartiles show that the
improvement is concentrated: the best targets reach ratios well below $0.7$
while the worst barely move.""",
    """whose interval excludes 1.0: a median reduction of $12.5@@%$, with $@@succGT$ of
targets more than halving their distance---at the cost of $@@renGT$ renders. The
plateau is visible in Figure~@@ref{fig:e2curves}: the median falls for about
fifteen steps and then stops. The per-target distribution shows how concentrated
the improvement is: the 10th and 25th percentiles of the ratio are $0.60$ and
$0.76$, the median is $0.875$, and the 75th percentile is $1.00$---a quarter of
the targets are not improved at all by the best available search.""", "results1")

# ---- results (2) ----------------------------------------------------------
rep("""sep-CMA-ES needed $3.7@@times$ the renders ($@@renCMA$ vs.@@ $@@renGT$) and still
scored worse (@@rCMA{} vs.@@ @@rGT). Figure~@@ref{fig:pareto} makes the comparison
budget-matched: at the render budget the oracle actually used, CMA-ES's
best-so-far is still above the oracle's final value, i.e.@@ the oracle reaches the
same quality with fewer renders @@emph{and} goes further. At this dimensionality,""",
    """sep-CMA-ES needed $3.7@@times$ the renders ($@@renCMA$ vs.@@ $@@renGT$), its median
ratio is exactly @@rCMA{}---half its targets are untouched---and its median
distance reaches only @@nCMAfin{} where the oracle reaches @@nGTfin{}
(Figure~@@ref{fig:pareto}); at the oracle's own budget it is still at $2.69$. At
this dimensionality,""", "results2")

# ---- exploitation numbers -------------------------------------------------
rep("""diagnostic---the damage is @@emph{larger} in-distribution than out
(@@rGS{} vs.@@ $1.90$ for the unconstrained variant), which is the opposite of what
a pure generalisation gap predicts and is the classic signature of a learned
objective being optimised adversarially against its own error. A trust region
removes a large part of it in-distribution ($2.63 @@rightarrow 1.61$), which
locates the problem specifically at @@emph{extrapolation}:""",
    """diagnostic---the damage is @@emph{larger} in-distribution than out
($2.83$ vs.@@ $2.50$ for the unconstrained variant), which is the opposite of what
a pure generalisation gap predicts and is the classic signature of a learned
objective being optimised adversarially against its own error. A trust region
removes most of that in-distribution damage ($2.83 @@rightarrow 1.35$: from far
worse than not searching to mildly worse), which locates the problem
specifically at @@emph{extrapolation}:""", "exploit")

# ---- retrieval percentage -------------------------------------------------
rep("""numbers are instructive: random trial-and-error is $57@@%$ worse than retrieval
(@@nRand{} vs.@@ @@nInit), and the best search method improves on retrieval by
$@@sim 15@@%$.""",
    """numbers are instructive: random trial-and-error is $49@@%$ worse than retrieval
(@@nRand{} vs.@@ @@nInit), and the best search method improves on retrieval by
$12.5@@%$.""", "retrieval")

# ---- seed replication -----------------------------------------------------
rep("""@@emph{TBD after the seed-1 run: headline ratios side by side with seed~0
(@@sOneGT{} for the oracle, @@sOneCMA{} for CMA-ES, @@sOneGS{} for the
unconstrained surrogate).}""",
    """Re-running the entire protocol with a different seed---new surrogate
initialisation, new target sample, same corpus---reproduces the picture rather
than the numbers: the oracle's median ratio is $@@rGT$ in seed~0 and $@@sOneGT$
in seed~1 (95@@% CIs @@rGTci{} and $[0.78, 0.96]$), CMA-ES is $@@rCMA$ and
$@@sOneCMA$ (no progress in either), and the surrogate variants stay above 1.0
in both ($@@sOneGS$ vs.@@ $@@rGS$ unconstrained, $@@sOneGST$ vs.@@ $@@rGST$ with a
trust region, $@@sOneGR$ vs.@@ $@@rGR$ re-grounded). The blind-draw baseline is
$@@sOneRetU$ vs.@@ $@@rRetU$. The qualitative ordering is stable across seeds;
the absolute level moves with the difficulty of the sampled targets (median
$d_{@@text{init}}$ $2.74$ vs.@@ $2.93$).""", "replication")

# ---- discussion wording ---------------------------------------------------
rep("""this directly. Note that exploration alone does not rescue the problem: CMA-ES
is non-monotone by construction and still reached only @@rCMA{} within a
$3.7@@times$ larger budget.""",
    """this directly. Note that exploration alone does not rescue the problem: CMA-ES
is non-monotone by construction, and with $3.7@@times$ the render budget it still
fails to improve the median target at all (@@rCMA).""", "disc1")

rep("""better than one that wanders. The trust-region variant is the cheapest possible
version of that idea and it recovers a large part of the damage---but it still
does not beat the start.""",
    """better than one that wanders. The trust-region variant is the cheapest possible
version of that idea and it removes most of the damage
($2.83 @@rightarrow 1.35$)---but it still does not beat the start.""", "disc2")

# ---- Table 2 cells --------------------------------------------------------
for a, b, why in (
    ("& 1.646 & 1.341 & 0.00 & 0", "& 1.616 & 1.352 & 0.00 & 0", "unigram"),
    ("& 2.633 & 1.900 & 0.02 & 48", "& 2.834 & 2.504 & 0.02 & 48", "gs"),
    ("& 1.614 & 1.353 & 0.00 & 48", "& 1.350 & 1.360 & 0.00 & 48", "gst"),
    ("& 0.797 & 0.875 & @@textbf{@@succGT}", "& 0.841 & 0.890 & @@textbf{@@succGT}", "gt"),
    ("& 0.948 & 0.961 & @@succCMA", "& 1.000 & 0.933 & @@succCMA", "cma"),
):
    rep(a, b, why)

p.write_text(s, encoding="utf-8")
print("paper numbers updated. Remaining TBD:", s.count("TBD"))
