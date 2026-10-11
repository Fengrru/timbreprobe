"""Write the full diagnostics section (§5.4) and its controls table + figure."""

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
    sm, em = tex(start_marker), tex(end_marker)
    i = next((k for k, ln in enumerate(lines) if sm in ln), None)
    assert i is not None, f"start marker not found: {why}"
    j = next((k for k in range(i, len(lines)) if em in lines[k]), None)
    assert j is not None, f"end marker not found: {why}"
    block = tex(new_block).split("\n")
    lines = lines[:i] + block + lines[j + 1:]
    print(f"  {why}: replaced lines {i + 1}-{j + 1} with {len(block)}")


NEW_SECTION = """@@subsection{Diagnostics and controls}\\label{sec:res-diagnostics}

Four cheap measurements decide between the competing explanations of the
plateau.  Table~@@ref{tab:diag} collects the two ablations; Figure~@@ref{fig:diag}
shows the residual decomposition and the per-target views.

@@paragraph{Where the residual lives.}  Decomposing the oracle's remaining
squared distance into the embedding's 13 blocks, the 64-dimensional log-mel
block absorbs $71.9@@%$ of it---but that is almost exactly its share of the
dimensions (per-dimension share $0.96@@times$ uniform).  The blocks that are
genuinely overweighted per dimension are the harmonic ratio
($2.34@@times$), the smooth zero-crossing proxy ($1.79@@times$), the spectral
slope ($1.75@@times$) and the envelope statistics ($1.59@@times$), and the
spread across blocks is only a factor of $2.7$ end to end.  The leftover error
is therefore spread roughly in proportion to dimensionality, with a mild excess
in the harmonic (discrete-sensitive) and noisiness blocks---not the localised
discrete bottleneck the valley-crossing account predicted.

@@paragraph{Do discrete jumps predict failure?}  Counting, per target, how many
hardened discrete dimensions differ between the start patch and the target, and
correlating that with the achieved ratio, gives Spearman $@@rho = -0.24$: weak,
and the @@emph{wrong sign} for the hypothesis (targets needing more discrete
changes improved slightly @@emph{more}, if anything).  With 48 targets this is
not significant; the honest statement is that discrete distance does not explain
the outcome.

@@paragraph{Does the accept rule cost anything?}  Re-running the identical search
with every candidate accepted is @@emph{worse}, not better: median ratio $1.000$
versus $0.874$ for the monotone walk, even though the free walk wins on $35@@%$
of individual targets.  An unguarded walk does not cross useful valleys---it
wanders.  The monotone rule is not the binding constraint.

@@paragraph{Is the metric's weighting the constraint?}  The residual result points
at the objective's @@emph{weighting}: 64 of 85 dimensions are log-mel, so
envelope mismatches dominate the number being minimised, while the
discrete-sensitive distinctions sit in single dimensions.  As a control we
re-ran the oracle with @@emph{block-balanced} weights (each block's expected
contribution equalised: per-dimension weight $1/@@sqrt{d_B}$, rescaled so the
distance scale stays comparable).  The median ratio improves from $0.875$ to
@@textbf{$0.809$} and the success rate doubles ($0.06 @@rightarrow 0.12$), with
the same search algorithm, the same targets and the same render budget.  This is
the only intervention in this paper that improves the gate without changing the
search: the bottleneck is at least partly the number being minimised, not the
way it is minimised.

@@begin{table}[t]
@@centering@@small
@@caption{The two controls.  Both change exactly one thing at the same render
budget (5@@,856 renders) on the same targets.  The accept-rule ablation makes
things worse; the metric re-weighting improves them.}
@@label{tab:diag}
@@begin{tabular}{lcc}
@@toprule
variant & median ratio & success@@textsubscript{$<0.5$} @@@@ renders @@@@ \\
@@midrule
oracle, monotone (baseline) & $0.874$ & $0.06$ @@@@ @@renGT \\
oracle, free walk (ablation) & $1.000$ & $0.06$ @@@@ @@renGT \\
oracle, block-balanced metric & @@textbf{$0.809$} & @@textbf{$0.12$} @@@@ @@renGT \\
@@bottomrule
@@end{tabular}
@@end{table}

@@begin{figure}[t]
@@centering
@@includegraphics[width=@@textwidth]{figures/diagnostics.png}
@@caption{(a) Share of the oracle's remaining squared distance by feature block:
the residual is spread in proportion to dimensionality, with a mild per-dimension
excess in the harmonic and noisiness blocks.  (b) Achieved ratio against the
number of discrete dimensions that separate the start patch from the target: the
trend is weak and, if anything, opposite to the valley-crossing prediction.
(c) The accept-rule ablation: points below the diagonal are targets where the
free walk did better.}
@@label{fig:diag}
@@end{figure}"""

replace_range(
    "\\emph{TBD after the diagnostics run.",
    "\\dAblFrac\\% of targets.}",
    NEW_SECTION,
    "section-5.4")

# the diagnostics macros are now unused; drop them
lines = [ln for ln in lines if not (ln.startswith(tex(SENT + "newcommand{" + SENT + "d"))
                                    or ln.startswith("% diagnostics (filled from"))]
p.write_text("\n".join(lines), encoding="utf-8")
out = p.read_text(encoding="utf-8")
print("remaining TBD:", out.count("TBD"), "| remaining d-macro refs:",
      sum(out.count(t) for t in ("\\dResTop", "\\dRho", "\\dAbl")))
