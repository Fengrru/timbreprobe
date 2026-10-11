"""Fill the numeric macros in main.tex.

Note for the next reader: `"a" + "b" + x.replace(...)` applies the method call
to the *last literal only*.  Build the placeholder string first, then convert.
"""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENTINEL = "@" + "@"

p = Path("main.tex")
s = p.read_text(encoding="utf-8")

UPDATES = [
    ("nInit", "2.74", "2.69"),
    ("nRand", "4.08", "4.05"),
    ("rRetU", "1.45", "1.39"),
    ("rGS", "2.59", "2.12"),
    ("rGST", "1.36", "1.44"),
    ("rGR", "1.60", "1.69"),
    ("rGT", "0.875", "0.84"),
    ("rGTci", "[0.81, 0.92]", "[0.77, 0.94]"),
    ("rCMA", "1.00", "0.96"),
    ("sOneGT", "0.877", "TBD"),
    ("sOneCMA", "1.000", "TBD"),
    ("sOneGS", "2.313", "TBD"),
    ("nCMAfin", "2.65", "TBD"),
    ("nGTfin", "2.35", "TBD"),
]

done, missing = [], []
for name, new, old in UPDATES:
    raw = "{" + SENTINEL + name + "}{" + old + "}"
    pattern = raw.replace(SENTINEL, BS)
    if pattern in s:
        s = s.replace(pattern, pattern.rsplit("}{", 1)[0] + "}{" + new + "}", 1)
        done.append(name)
    else:
        missing.append(name)

# macros the new seed-replication text needs
if (BS + "sOneGR}") not in s:
    anchor_raw = SENTINEL + "newcommand{" + SENTINEL + "sOneGS}{2.313}"
    anchor = anchor_raw.replace(SENTINEL, BS)
    if anchor in s:
        extra = "\n".join(
            BS + "newcommand{" + BS + name + "}{" + val + "}"
            for name, val in (("sOneGST", "1.376"), ("sOneGR", "1.475"),
                              ("sOneRetU", "1.499"))
        )
        s = s.replace(anchor, anchor + "\n" + extra, 1)
        done.append("sOneGST+sOneGR+sOneRetU")
    else:
        missing.append("extra-macros (anchor missing)")

p.write_text(s, encoding="utf-8")
print("updated:", ", ".join(done) if done else "(none)")
if missing:
    print("NOT FOUND:", ", ".join(missing))
print("remaining TBD:", s.count("TBD"))
