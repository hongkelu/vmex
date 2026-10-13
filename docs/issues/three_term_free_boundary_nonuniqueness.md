# Three-term free boundary: converged solution depends on the start (angle-origin near-null mode)

## Summary

With fixed coils, `freeboundary_vc` (three-term: B.n, pressure balance, sheet current; virtual casing) converges to
slightly different solutions depending on where it starts. Both solutions pass every convergence test.

| Measure | Difference between the two solutions |
|---|---|
| LCFS surfaces, true geometric distance | 0.14 mm max, 0.05 mm rms |
| Fourier coefficients, compared at the same (θ, ζ) labels | 1.4 mm max |
| Least-squares cost | up to 67 % |
| Two-term QS error of the re-solved equilibrium | 6.6 % |

Most of the 1.4 mm label difference is therefore a relabelling: the poloidal-angle origin of the interior solution shifts.

Consequences in single-stage optimization:

- After a restart, the driver's QS value jumps by −10 % … +90 % for unchanged coils.
- A design-flux re-solve of the saved coils differs from the optimizer value by 5–20 %.

The optimizer value and an independent re-solve should agree to truncation error. They currently do not.

## Reproduction

The coils are `paper/runs/batch/P30A-F12-c5/coils.step1.json` (vacuum Landreman–Paul QA, 12×12, VC 64, NS 51).

- `paper/runs/uniq/seed_check.py` solves the same coils at one boundary twice:
  - **A, B:** seeded from the optimizer's cached state.
  - **C, D:** cold, from the LP deck.
- Results:
  - Seeded cost 5.7e-10 / 6.2e-10 vs cold 1.23e-9 (67 % spread).
  - The cold solve is reproducible (`deterministic=True`).

| Configuration | Cost spread seeded vs cold | State difference A−D |
|---|---|---|
| NS 51, linear edge extrapolation (default) | 67 % | 1.3e-3 |
| NS 51, quadratic edge extrapolation (`VMEX_EDGE_EXTRAP=2`) | 114 % | 1.3e-3 |
| NS 99, linear | 76 % | 1.4e-3 |
| NS 99, quadratic | 40 % | 1.7e-3 |

Neither radial resolution nor the half-mesh→edge extrapolation in `virtual_casing._assemble_surface_field_data` is the cause.

**Breakdown of the converged-state difference** (`seed_check_ns51_brk.json`):

| Field | Difference | Where |
|---|---|---|
| λ | 1.35 % | entirely m = 0, largest on inner surfaces |
| Z | 6e-4 | m = 1 |
| R | 8e-5 | m = 1 |
| Edge | zero | fixed |

This is a shift δ(s, ζ) of the poloidal-angle origin, the near-null m = 1 direction noted in `vmex/core/implicit.py` (refine-tol note). The interior force residual hardly constrains it. The boundary residuals are evaluated at fixed (θ, ζ) labels on a truncated basis, so they change with it.

**Full two-start solves** (`runs/uniq/{lp,ystep0,det_lp,det_y}`):

- Free boundary from the LP deck and from a different plasma (P30A-F12y step 0).
- Both converge.
- Direct Biot–Savart B.n/|B| on the four LCFSs is 1.55–1.73e-5 (area-mean), the same for all of them.
- The surfaces are 0.14 mm apart, but the QS of the re-solved equilibrium is 5.99e-6 vs 6.39e-6.

**This is a boundary solve that stalls, not two exact solutions.**

- **Tighter refinement does not help.** `VMEX_REFINE_TOL=1e-13` (`seed_check_ref1e-13`) gives seeded 6.36e-10 / 6.88e-10 vs cold 1.065e-9. That is the same 67 % spread.
- **The Gauss–Newton model is wrong.** `jac_check_det.json` uses cold, deterministic solves (FTOL 1e-17). The cost is continuous there, but |Δr − εJd| / |εJd| = 1.086 for every ε ≤ 0.03. The implicit Jacobian is the tangent at a fixed labelling. The cold-solve map also drifts along the near-null angle-origin direction as the boundary moves, so J is not its derivative.
- **The solve stops early.** Levenberg–Marquardt gain ratios are 0.2 at every step, and the solve stops on xtol while the cost is still falling.
- **Warm restarts keep lowering the cost.** In `yit_r1..r6`, the cost falls 4.8e-10 → 4.2e-10 while QS wanders between 6.20e-6 and 6.34e-6.

## Comparison with VMEX + NESTOR (same coils, same two starts)

- `paper/runs/uniq/nestor_uniq.py`: mgrid 128×128×64, 12×12, NS 51, FTOL 1e-15.
- Both starts converge (~14,900 iterations).
- QS 1.7462e-4 vs 1.7466e-4 (0.02 %); ι, volume and R0 agree to 1e-6.
- NESTOR is therefore start-independent here.
- Its QS is 30× the three-term value. NESTOR's B.n on the coil field and mgrid-resolution tests are pending (see below), so it is not yet known which surface is closer to the true flux surface.

## Why the earlier benchmark did not show this

In `benchmarks/run_three_term_resolution.py` and `docs/explanation/validation.md`:

- Every method starts from one common boundary (target scaled by 0.97).
- Results are scored by LCFS distance to a known target (three-term: 1.6–2 mm, limited by the target's own error) and by field-line confinement.
- A 0.14 mm start-dependence is far below that benchmark's 1.6 mm resolution, and the benchmark never ran two starts.

The 6 % QS spread only matters when QS is ~1e-6 and is the quantity being optimized.

## Status (2026-10-12 evening)

**Root cause.**
- `residue.f90` zeroes the constrained m=1 Z force (`gcz`) once fsqz < 1e-6.
- The combination it drives, `zcs_int = (rss − zcs)/2` (the antisymmetric `Z_sin(m=1, ±n)` pair), then stays wherever the solver's path left it.
- The implicit linearization also holds it frozen, so the Jacobian is not the derivative of the re-solve map. The boundary Gauss–Newton then stalls.

**Rejected fixes.**

| Fix | Result |
|---|---|
| Tighter Newton refinement | no effect |
| Quadratic edge extrapolation | no effect |
| NS 99 | no effect |
| Pinning `zcs_int` to √s × edge | the pinned profile leaves \|F\| = 9e-3; the anchor stalls |

**Fix in progress: `ImplicitConfig.release_m1` (default on).** The m=1 force stays active in the Newton anchor and the linearization.

Vacuum results:

| Test | Before | After |
|---|---|---|
| Seeded vs cold at one boundary, real coils (12×12, NS 51) | cost 67 % apart, state 1.3e-3 apart | cost 2.7e-9, state 5e-12 |
| Implicit gradient vs cold re-solve FD | 0.7–28 % off | 1e-8 |
| li383 mean ι, implicit vs re-solve FD | 1.0 % / 2.8 % off | 1e-6 / 3e-6 |

**Finite β (LP QA β = 2.5 % bootstrap deck, 5×5, NS 25): not fixed yet.**
- The gradient now matches the re-solve FD (1e-7 – 2e-6, against 1–35 % before).
- A host solve seeded 1e-3 along the combination still converges, under VMEC's convention, to a state 7 % away in λ.
- On the released system, plain Newton from there diverges. The released Jacobian is (nearly) singular along that direction at finite β.
- The fix needs a genuine gauge condition, or a regularized (Levenberg) anchor, rather than the force alone.

## Proposed fixes, in order

1. ~~Tighter Newton refinement~~ (tested at 1e-13: no effect).
2. **Gauge pinning.** Fix the angle origin in the interior solve (constrain λ m = 0 and the R/Z m = 1 modes). The converged interior state is then unique, and the implicit Jacobian becomes the true derivative of the solve map, which restores Gauss–Newton convergence.
3. **Labelling-invariant boundary residuals.** Evaluate B.n, pressure jump and K so that they do not depend on the interior angle origin.
4. **Fallback.** If 1–3 fail, use NESTOR in the optimizer when NESTOR is unique (pending its B.n and mgrid-resolution checks).

Acceptance:

- The optimizer value equals a cold re-solve from the LP deck within 0.1 %.
- No restart jumps.
- The two-start test agrees to 0.1 % in QS.
- Repeat the test at finite β.

Opt-in diagnostics added (uncommitted): `VMEX_EDGE_EXTRAP`, `VMEX_REFINE_TOL`.
