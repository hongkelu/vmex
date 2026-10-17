# Finite-β single-stage campaign: QA, QI and QH (Oct 2026)

Coils and plasma optimized together at ⟨β⟩ = 2.5% with a self-consistent bootstrap current: Redl for QA and QH,
DKX Fokker–Planck for QI. Resolution is 8×8 modes, ns = 51, and every step is saved.

- **Free arm** (`examples/optimization/single_stage_free_boundary_optimization_three_term.py`): the free-boundary
  equilibrium of the coils with the three-term boundary condition, B·n = 0, pressure balance and sheet current K = 0
  on the LCFS (`ThreeTermFreeBoundaryProblem`).
- **Fixed arm** (`examples/optimization/single_stage_optimization_coil_constraints.py`): fixed-boundary equilibrium plus
  a coil B·n term.
- **Optimizer:** SciPy SLSQP with hard constraint rows and implicit-differentiation gradients.

The code is this branch's parent commit `e3313c0f` on `qa-mercier-bootstrap-cases`.
[`HANDOFF-2026-10-10.md`](HANDOFF-2026-10-10.md) describes the setup, the server rules and the comparison planned with
Gil et al., PRL 137, 065101 (2026).

## Layout

- `tools/`: helper scripts exactly as run. They lived in `extra/` of the remote tree and run from the tree root.
  - `lcfs_modb_gif.py`: 3D coils and LCFS coloured by |B| over the saved steps.
  - `maxj_scan.py`: maximum-J check, ∂J/∂s < 0 at fixed physical λ over 24 pitches.
  - `eps_eff.py`: effective ripple ε_eff^{3/2}(s) through NEO_JAX. Accepts a WOUT or a compact `.npz` of the Boozer
    fields.
  - `continue_until_converged.py`, `restart_from_latest.py`: continue a run past its step budget, or restart it from its
    latest step.
  - `launch.sh`: one production run, pinned to a GPU by UUID.
  - `checks/fixed_jaccheck.py`: the fixed-arm grouped plasma Jacobian matches the per-row one to 1.2e-14 (116 → 15.5 s).
  - `checks/dkx_map_proto.py`: DKX surfaces in one `lax.map` match the Python loop to 2e-7 (gradient compile
    1016 → 40 s).
- `runs/<run>/`: for each run, `metrics.jsonl` (one row per accepted step), `input.run`, `coils.initial.json`, the latest
  `coils.stepN.json` and `summary.json`. WOUTs are not included (8–31 MB each); they stay on lhk0-7 in
  `/root/autodl-tmp/vmex-qa-cases/prod`.
  - A `-cN` suffix is the N-th continuation (`--restart` from the previous final state).
  - `-m002` restarted with the 0.02 Mercier margin.
- `results/`: post-processing.
  - `qa4-campaign-post/`: baseline QA, free against fixed. Includes the HTML page.
  - `qa4-variants/`: Mercier, bootstrap-weight and vacuum-floor cases. Dense ns = 201 re-solve with `plot_wout`
    figures, `--trace` alpha losses, |B| GIFs, max-J scans and ε_eff.

## Results

The "qa" column of `metrics.jsonl` is the case's target: the QA, QH or QI residual. B·n, PB and K are the RMS
three-term boundary residuals of the free arm.

| Case | Run | Final step | Target | Notes |
|---|---|---|---|---|
| QA, free, Mercier ≥ 0.02 (s ≥ 0.2), vacuum ι ≥ 0.31, ι ≤ 0.495 | `free-qa4-beta-merc-vac0.31` → `-c1` → `-c2` | 999 + 998 + 84 | **QA 1.9e-4** | D_Merc +0.026, ι_I 0.28, A 4.31, B·n / PB / K 1.9 / 2.0 / 2.8e-4 |
| QA, free, Mercier, vacuum ι ≥ 0.27 | lhk3-6 (offline); post-processed at step 267 | | QA 5.5e-4 | alpha loss 0.6%, ε_eff ≤ 0.13%, max-J 63% |
| QA, free, Mercier, vacuum ι ≥ 0.29 | lhk3-6 (offline); post-processed at step 242 | | QA 7.0e-4 | **alpha loss 0%**, ε_eff ≤ 0.14%, max-J 62% |
| QA, fixed, Mercier, vacuum ι ≥ 0.23 | `fixed-qa4-beta-merc-s02-init-m002` → `-c1` → `-c2` | 300 + 300 + 260 | **QA 1.6e-4** | D_Merc +0.020, A 4.5 (at its upper bound) |
| QA, free, Mercier, vacuum ι ≥ 0.23 | `free-qa4-beta-merc-s02-init` | 260 | QA 8.2e-4 | before the margin; alpha loss 3.2% at step 238 |
| QA, free, bootstrap weight w = 0.01 | `free-qa4-beta-jbs` | 300 | QA 3.5e-4 | ι_I 0.31 |
| QA, free, w = 0.1 | `free-qa4-beta-jbs-w0.1` | 300 | QA 1.5e-3 | ι_I 0.21, vacuum ι 0.49 |
| QA, free, w = 1, aspect 3.0–4.5 | `free-qa4-beta-jbs-w1` → `-c1` → `-A3` | 300 + 132 + 1000 | QA 6.7e-3 | ι_I 0.15, A 3.57, total ι 0.61 (past ι = 1/2) |
| QA, fixed, w = 0.01 | `fixed-qa4-beta-jbs` | 110 | QA 3.2e-3 | |
| QI, free, A 6, unscaled steps | `free-qi6-beta-scale1` → `-c1` → `-c1-c1` … `-c1-c5` | 7 × 300 | **QI 1.7e-2** | ι 0.86 at the floor |
| QI, fixed, A 6, grouped Jacobian | `fixed-qi6-beta-grouped` | 260 | QI 2.6e-2 | |
| QI, free, A 8 | `free-qi6-beta-A8` → `-c1` … `-c5` | 5 × 500 + 196 | **QI 3.8e-3** | B·n / PB / K 0.9 / 0.5 / 1.1e-3 |
| QH, free, A 8 | `free-qh4-beta-A8` (+ `-c1`, stopped) | 500 + 6 | QH 7.0e-3 | ι 1.1 at the floor |
| QH, free, A 10 | `free-qh4-beta-A10` → `-c1` … `-c4` | 4 × 500 + 130 | **QH 7.3e-3** | B·n / PB / K 1.5 / 1.5 / 2.2e-4 |

### Notes

- **Snapshot date:** these are the last rows as of 2026-10-16. The runs without a `summary.json` were still running, or
  had been stopped, when copied.
- **Shared tree:** continuations launched after 2026-10-11 ran from the shared lhk0-7 tree, which another session had
  updated past `e3313c0f` (stability rows and a batched field-term pullback with the same gradients). The QA, QI and QH
  rows above use none of the stability rows.
- **lhk3-6 runs:** the run histories of the vacuum-floor 0.27 and 0.29 runs (and the fixed 0.27 arm) are on lhk3-6, which
  has been offline since 2026-10-12. Their post-processing is in `results/qa4-variants/`.
- **QI A10:** never started. Its seed build failed in VMEC: MORE ITERATIONS REQUIRED in the β ramp and in `dkx_current`.
