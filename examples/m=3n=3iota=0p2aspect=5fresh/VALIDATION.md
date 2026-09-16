# Published runner validation

A fresh local CPU run completed one accepted step on 2026-09-12 with exit code
0 in 123.75 seconds, using the documented runner with `--device cpu
--target-step 1 --max-wall-hours 0.15` and a new output directory. The runtime
used Python 3.10.19, JAX 0.6.2, Solvax 0.20.0, NumPy 2.2.6, and ESSOS
0.17.dev86+gd9ca5c37d. The VMEX core was commit
`1349135ef90e7bf039a265401ab21c091401397c`.

- Original seed: exact state arrays preserved, float64, all 111 parameters zero.
- Dense adjoints: all four passed; largest relative residual 3.151e-12.
- Changed-coil endpoint: first trial accepted for reducing constraint violation.
- Projected root residual: 1.07532e-6, below 2e-6.
- Largest force/edge residual: 9.89061e-11, below 1e-10.
- Continuous coil-motion bound: 0.622811 mm, below 1 mm.
- Nominal-current change: 1%, at the configured cap.
- Checkpoint, WOUT, coil JSON, and summary were written successfully.
- Checkpoint and WOUT hashes verified; checkpoint arrays finite and float64.
- Independent WOUT iota/aspect/B0 readings agree with the final summary.

All 20 case tests passed, including a fresh-process test proving that runner
initialization preserves double precision when VMEX is imported. The runner
uses `JAX_ENABLE_X64=1` explicitly. The previous spelling `True` temporarily
disabled x64 at VMEX import; its solver import restored x64 before computation.

This is an execution smoke test, not convergence or a completed 200-step
campaign. The first step restores constraints, so QA may increase. The active
GPU campaign and the parent research checkout were not changed by this check.
