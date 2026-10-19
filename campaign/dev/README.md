# Development scripts of the vmex work (2026-10-07 … 10-10)

These are the one-off benchmarks, checks and helpers written during the speed and accuracy work on the finite-β free
boundary, kept as they were run.
- **How they ran:** they lived in `extra/` of the lhk0-7 tree and ran from a vmex checkout, with `PYTHONPATH=.` and one
  empty GPU.
- **Where the results went:** into vmex itself, PR 27 (vacuum solve on the GPU, grouped pullback, linearize cache:
  free QA step 265 → ~105 s, exact) and the branch commits in `../COMMITS.txt`.

## Speed work on the three-term free boundary

| Script | What it measures |
|---|---|
| `bench_rows.py` | JVP cost of each piece of the three-term interface rows, per 8 state directions |
| `bench_vc.py` | on-surface virtual casing: JVP of B_plasma over source × target chunk sizes |
| `bench_vc_split.py` | merged virtual casing: far-field quadrature vs singular correction, primal and JVP |
| `bench_vc_sweep.py` | sweep variant of `bench_vc_split.py` (its docstring is a copy of that one) |
| `bench_merged_vc.py` | J and B·n in one singular pass, against the package's value, primal and JVP time |
| `bench_factor.py` | block factorization: probe chunk × row batch, time and difference from the default factors |
| `time_linearize.py` | runs the three-term driver with the Jacobian phases timed (tangent solves, interface pushes) |
| `time_refine.py` | phase timings of one certified implicit solve at a moved boundary |
| `time_vacuum.py`, `time_vacuum_cfg.py` | the vacuum-iota row: eager vs jitted, and solver variants against the default (value, gradient, time) |
| `device_test.py` | where the certification factorization runs, inside vs outside the solve callback (found the host-CPU slowdown) |
| `explain_recompile.py` | why the certified solve recompiled: compile and cache-miss log over moved boundaries |

## Correctness checks and debugging

| Script | What it does |
|---|---|
| `fd_rows.py` | finite-difference check of the vacuum-iota-min and axis-\|B\| rows through a finite-β implicit solve |
| `fixed_debug.py` | the fixed-boundary driver with debug output (copy of the driver of that time) |
| `debug_ctor.py` | compares a WOUT's `ctor` with `surface_currents().ctor` and prints ι_I; it showed `ctor` is already μ0·I (fix 3a6ceaf9) |
| `repro_hang.py` | reproduces the tokamak-seed vacuum-iota hang, dumping all thread stacks every 120 s |

## Post-processing and status

| Script | What it does |
|---|---|
| `qs_error.py` | Boozer quasisymmetry error of a WOUT against s |
| `fedge.py` | force residuals and edge force of a WOUT re-solved **with a fixed boundary**, so its fedge is 0 by construction and meaningless for free-boundary WOUTs; a real fedge needs a NESTOR free-boundary solve |
| `vacuum_iota_runs.py` | vacuum iota of saved run boundaries beside their finite-β iota |
| `latest_status.py`, `summ_status.py`, `summ2_status.py` | one-line status of the production runs from their `metrics.jsonl` (the `extra_*.py` helpers of the lhk0-7 tree) |

The campaign's own post-processing and orchestration tools are in `../postprocess/`, `../orchestration/` and
`../checks/`.
