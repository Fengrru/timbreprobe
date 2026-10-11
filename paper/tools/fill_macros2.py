"""Add the two missing macros and fix the success-rate macros."""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENT = "@" + "@"
p = Path("main.tex")
s = p.read_text(encoding="utf-8")


def tex(t: str) -> str:
    return t.replace(SENT, BS)


# 1) success rates were still at the pre-fix values
for name, new, old in (("succGT", "0.06", "0.15"), ("succCMA", "0.06", "0.08")):
    pat = tex("{" + SENT + name + "}{" + old + "}")
    if pat in s:
        s = s.replace(pat, tex("{" + SENT + name + "}{" + new + "}"), 1)
        print(f"  {name}: {old} -> {new}")
    else:
        print(f"  {name}: pattern not found (already {new}?)")

# 2) add the two distance macros the new results text references
if tex(SENT + "nGTfin}") not in s:
    anchor = tex(SENT + "newcommand{" + SENT + "rCMAci}{[0.90, 1.00]}")
    extra = "\n".join((
        tex(SENT + "newcommand{" + SENT + "nGTfin}{2.35}   % median final distance, oracle"),
        tex(SENT + "newcommand{" + SENT + "nCMAfin}{2.65}  % median final distance, CMA-ES"),
    ))
    assert anchor in s, "anchor for new macros not found"
    s = s.replace(anchor, anchor + "\n" + extra, 1)
    print("  added nGTfin, nCMAfin")

p.write_text(s, encoding="utf-8")
print("remaining TBD:", s.count("TBD"))
