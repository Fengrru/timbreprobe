"""Repair the two LaTeX breakages introduced while filling in the numbers."""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENT = "@" + "@"
p = Path("main.tex")
s = p.read_text(encoding="utf-8")


def tex(t: str) -> str:
    return t.replace(SENT, BS)


# ---- 1) close the math span that the earlier range replacement truncated ---
broken = tex("""with $@@delta = z_{@@text{final}} -
Concentration in the harmonic blocks""")
fixed = tex("""with $@@delta = z_{@@text{final}} - z^@@star$.  Concentration in the
harmonic blocks""")
assert broken in s, "bullet anchor not found"
s = s.replace(broken, fixed, 1)

# ---- 2) rebuild the controls table ---------------------------------------
start = s.index(tex("@@begin{tabular}{lcc}"))
end = s.index(tex("@@end{tabular}"), start)
rows = [
    tex("@@begin{tabular}{lccc}"),
    tex("@@toprule"),
    tex("variant & median ratio & success$_{<0.5}$ & renders @@@@@"),
    tex("@@midrule"),
    tex("oracle, monotone (baseline) & $0.874$ & $0.06$ & @@renGT @@@@@"),
    tex("oracle, free walk (ablation) & $1.000$ & $0.06$ & @@renGT @@@@@"),
    tex("oracle, block-balanced metric & @@textbf{$0.809$} & @@textbf{$0.12$} & "
        "@@renGT @@@@@"),
    tex("@@bottomrule"),
]
s = s[:start] + "\n".join(rows) + "\n" + s[end:]

p.write_text(s, encoding="utf-8")
print("repaired")
