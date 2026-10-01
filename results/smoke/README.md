# Committed smoke-run artefacts

Produced by

```bash
python -m timbreprobe.cli corpus --preset smoke
python -m timbreprobe.cli e2     --preset smoke
python -m timbreprobe.cli plot   --run out/e2_smoke
```

and copied here so the numbers in the top-level README can be checked without
re-running anything.

| file | what it is |
|---|---|
| `e2_results.json` | full result record: config, split sizes, per-method metrics (median ratio, bootstrap CI, success rates, parameter L1, renders, wall time), per-regime breakdown |
| `curves.npz` | search trajectories per method + `d_init` per target, shaped `[logged steps, 48]` |
| `targets.npz` | the 48 sampled targets (`idx`, `holdout` flag, `d_init`, target patch vectors) |
| `surrogate_report.json` | E3 fidelity on val / test / holdout-anchor splits |
| `figures/` | `e2_ratios.png`, `e2_curves.png` |

**Scope and caveats.** These are *smoke* scale: 3 895-patch corpus (16 anchors,
4 of them held out), 48 targets (24 from held-out anchor clusters), single seed,
single machine.  The E2 conclusions ("surrogate is the bottleneck", "gradient
beats population search per render") are robust across the runs we did, but the
absolute numbers should be re-measured with `--preset full` before publishing.
Wall-clock times in `e2_results.json` reflect a 4-core CPU under heavy external
load and are not representative.
