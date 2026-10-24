# Results: Perlmutter session 2026-10-22

> **Status 2026-10-23.** Corrections: (1) no fixture seed and no QH evaluation here passed the strict certificate; the
> "accepted" rows used the opt-in `boundary_normal_tol` 1e-3, which work package 2 removes as an acceptance rule;
> (2) the interior "null" is a near-null direction measured to the probe's precision, not an established exact
> singularity; (3) the mpol attribution of two interior-scan logs is uncertain (section 6).

All runs used one frozen fixture per case (`fixtures/`), on 40/80 GB A100s, with jax 0.12.0.1 and solvax 0.31.0
(README). "C" is `labelling="equal_arc"`; "NL" is no labelling rows (`label_weight 0`). "Gain" is the certificate's
stationarity, the relative cost reduction a Gauss-Newton step with a fresh Jacobian predicts (gate 1e-3).

## 1. Test results by commit

`RUN_FULL=1 pytest tests/test_freeboundary_vc.py tests/test_bootstrap.py -k 'not nestor and not desc'`, split over GPUs.

| code | passed | failed | skipped | failures |
|---|---|---|---|---|
| 095edbec (#121 head, morning) | n/a | 5 | n/a | Run alone, 3 reproduce: `test_rejected_trial_leaves_the_continuation` (radius 123 leaked to the anchor's second start), `test_interior_state_does_not_depend_on_its_start` (rows 1e-3), `test_plasma_map_derivatives_at_nonzero_offsets` (-1.4532 vs FD -1.4799). The other 2 were XLA out-of-memory without preallocation. |
| 96ece254 (#122, first) | 84 | 3 | 10 | the interior and plasma-map tests; the checkpoint fresh-process test (preallocation artifact: passes without it) |
| 86484506 | 86 | 2 | 10 | interior, plasma map |
| afcc6bde (rebased on 3591bd34, polish at POLISH_FTOL as 7d73da7) | **87** | **2** | **10** | interior, plasma map (-1.4532 vs -1.4799, unchanged by the polish target) |

- **Solvax 0.29.0** (the 5090's version, via an overlay): the interior and plasma-map tests still fail.
- **CPU:** the interior test fails too (section 6).

## 2. QH 2.5% seed certification (fixture `qh25`)

| code | C 1e-3 | NL | C 1e-2 |
|---|---|---|---|
| 095edbec | refused, gain 0.992 | refused, 0.996 | |
| 96ece254 (projection, polish on trial states) | refused, 0.982 | refused, 0.986 | |
| 71497038 (tight polish), 5 rounds | refused, 0.040 | refused, 0.080 | refused, 0.0072 (6 rounds) |
| tight polish, 25 rounds | plateau 0.024 | plateau about 0.075 | |
| + `boundary_normal_tol` 1e-3 (opt-in; **not** the strict certificate) | accepted (round 2) | accepted | accepted |

### Where the refused gain sat (round 1, C 1e-3; `analyze.py`)
The cost is |r|^2 = 5.30e-5 over 13926 rows (255 boundary and 53 current columns).

| quantity | value |
|---|---|
| share of the cost in the interface rows | 0.020 |
| share of the cost in the bootstrap rows | 0.980 |
| bootstrap rows after the GN step | 7.5e-17 |
| gain from a current-only correction at fixed boundary | **0.980** |
| reduced boundary gain | 0.006 |
| condition: boundary block | 4.5e7 |
| condition: current block | 1.05 |

The fit solved the current on trial-FTOL (1e-11) states. At the tight state (1e-15), Redl's target differs, and the
weight-100 rows amplify that difference to 50x the fit's cost; the polish on trial states then fell back to the
trial optimum. Fix: `certify` polishes on tight states.

### After the tight polish (round 2, NL)
- **Cost:** 99% in the interface rows; bootstrap rows conditionally stationary (gain 0.0000).
- **Remaining gain 0.236:** 0.18 of it in the 100 weakest directions (s <= 7.5e-3), < 1% in the 10 weakest.
- **Step length:** the 10 weakest directions carry 2/3 of the GN step's length (0.33 of 0.47 scaled).

### Weak directions (`geometry.py`, minor radius 0.286 m)

| certificate | normal share of the weak directions | GN step: normal RMS | GN step: tangential RMS |
|---|---|---|---|
| C 1e-3, round 13 | 3-10% | 4.6e-4 a (0.13 mm) | 1.5e-2 a |
| NL, round 12 | 3.5-11% | 5.9e-4 a (0.16 mm) | 2.1e-2 a |

### Taking the remaining GN step (`[calibrate]`)

| run (accepted at) | step | cost (was) | B.n | K | values max rel change | loss rel change |
|---|---|---|---|---|---|---|
| C 1e-3, tol 1e-3 (round 2) | 0.5 | 1.97e-3 (4.0e-7) | 4.52e-4 | 6.01e-4 | 3.2e-3 | 1.3e-3 |
| | 1.0 | 3.20e-2 | 6.01e-4 | 1.25e-3 | 9.4e-3 | 2.9e-3 |
| NL, tol 1e-3 (round 2) | 0.5 | 5.88e-3 (3.8e-7) | 4.99e-4 | 7.22e-4 | 7.1e-3 | 3.5e-3 |
| | 1.0 | 8.55e-2 | 8.67e-4 | 1.92e-3 | 2.2e-2 | 7.1e-3 |
| C 1e-3, tol 5e-4 (round 3) | 0.5 | 3.36e-3 (3.7e-7) | 4.47e-4 | 5.92e-4 | 7.7e-4 | 6.5e-4 |
| | 1.0 | 4.56e-2 | 6.21e-4 | 1.28e-3 | 2.4e-3 | 1.3e-3 |

The linear model is invalid along this step (the cost rises 1e4-1e5x), so its normal displacement is not a
distance to the solution. `boundary_normal_tol` stays opt-in and is **not recommended** (next table).

### Outputs over polish rounds (`TRAJECTORY=1`; full tables in `raw/qh_trajectories.md`)

| setting | loss, round 2 | loss, round 12/13 | rounds | last rounds |
|---|---|---|---|---|
| grid 48, C 1e-3 | 0.657708 | 0.662719 | 13 | 0.663123, 0.662687, 0.663034, 0.662719 |
| grid 48, NL | 0.658891 | 0.664144 | 13 | 0.663700, 0.663990, 0.664008, 0.664144 |
| grid 64, C 1e-3 | 0.658304 | 0.663374 | 13 | 0.663342, 0.663200, 0.663534, 0.663374 |
| grid 64, NL | 0.659513 | 0.663008 | 13 | 0.663291, 0.663400, 0.662830, 0.663008 |
| max_mode 6, C 1e-3 | 0.663035 | 0.677093 | 13 | rising about 0.2% per round |
| max_mode 6, NL | 0.663215 | 0.670533 | 12 | rising |
| grid 48, C 1e-2 | 0.648216 | 0.694496 | 8 | rising 0.5-1% per round |

- **The fit converges in the loss, but slowly:** about 10 fresh-Jacobian rounds to about 3e-4.
- **An early accept (round 2) is about 0.8% off.**
- **C and NL agree to 0.07% at interface grid 64** (0.2% at 48).
- **max_mode 6** conditions 6x better (8e6 vs 5e7) and does not converge.
- **The constraint values move about 0.3% per round throughout.**

### Seed metrics when accepted at `boundary_normal_tol` (step 0 of the driver)

| run | QH | iota | vacuum iota | DMerc min | B.n | K |
|---|---|---|---|---|---|---|
| C 1e-3, tol 1e-3 | 1.315417 | 1.14704 | 1.0664 | -710.7 | 4.7e-4 | 5.7e-4 |
| C 1e-3, tol 5e-4 | 1.317458 | 1.14402 | 1.0652 | -709.2 | | |
| NL, tol 1e-3 | 1.317781 | 1.14614 | 1.0695 | -720.8 | | |
| NL, tol 5e-4 | 1.318156 | 1.14516 | 1.0678 | -717.7 | | |
| C 1e-2 (either tol) | **1.385622** | 1.14037 | 1.0549 | -767.5 | | |

Model build: 1287 s. No SLSQP step finished within 70 min (5 polish rounds of 20 evaluations).

## 3. Bootstrap layer

- **Edge knot.** (3 v[-1] - v[-2]) / 2 at s = 1; it was 2 v[-1] - v[-2], i.e. s = 1 + h/2. Tests at ns 11/25/51/97:
  an affine I' is exact at the edge and over the last half cell, and v = s integrates to 0.5.
- **Near-axis target.** It now uses `near_axis_redl_j_dot_b` directly with the s_1 coefficients of the identity and
  I(x s_1) = x I(s_1). For flat n, T it is exactly 0, with finite derivatives, linear in the drive (either sign); the
  old ratio was NaN there.
- **G vs G + N I on the QH seed** (`raw/g_convention_qh_seed.json`): N I / G from -2.8e-5 (core) to -1.14e-2
  (edge); Redl <J.B> changes by at most 0.91% of its peak; the Redl-implied CURTOR is -14831 A (simsopt) vs
  -14747 A (isomorphism), -0.57%. The default stays simsopt's G.
- **Picard.** On a budget exit it returns the solved iterate, and the power-series CURTOR is the exact polynomial
  integral. Measuring delta against the fitted polynomial was tried and reverted: on the QH seed ramp it never reached
  tol in 30 iterations (5 with the fixed-point delta). `fit_error` is reported instead.
- **Closure identity in the checkpoint definition** (schema /3): the kinetic profiles, helicity, weight and
  conventions.

## 4. Certification and current solve (library)

- **One `certify`** for the standalone solve and the optimizer.
  - Steps: tight re-solve, fresh Jacobian (with plasma columns), polish rounds on tight states aiming at POLISH_FTOL,
    certificate (gain <= gate, optional xtol and normal_tol), physical-residual bound on the final state, values last.
  - The standalone `converged` now means certified (`certificate` in the result).
- **`solve_boundary`** no longer turns a stall into convergence.
- **Every start of a trial** restarts from the accepted continuation and is certified on its own.
- **Levenberg-Marquardt with the current eliminated.**
  - The boundary steps on the Kaufman reduced model, carrying the current's linear correction.
  - `_project` corrects the current to conditional stationarity (|Q_c^T r|^2 <= 0.1 ftol |r|^2, at most 4 steps).
  - The predicted reduction is that of the evaluated composite step. On a stiff closure test the joint model stalls
    and the reduced one converges.
- **`least_squares_sensitivity`:** exact vs Gauss-Newton dy/dd. Analytic cases give exact 1.25 vs GN 1.0, matching
  re-solved finite differences to 1e-6.
- **Certificate `normal_step` and opt-in `boundary_normal_tol`/`normal_tol`.** The operator matches analytic boundary
  derivatives to 5e-16. Two bugs in it were found by tests before any result used it: a tuple division, and the
  theta derivative taken along the wrong axis.
- **Module constants** `POLISH_MAX_NFEV` (20) and `STALL_STEPS`/`STALL_GAIN` (3, 0.05), defaults unchanged.

## 5. Driver
`profiles.json` holds the kinetic profiles of every bootstrap run; `--restart` and `--seed-run` reuse it instead of
rebuilding n0, T0 from the restored boundary's RBC(0,0) (older runs: their own `input.run`).

## 6. Interior gauge zero mode (core; small LP QA test deck; `raw/interior.txt`)

**Interface rows of the three-term model at one boundary, three interior starts** (`probe.py`, FTOL 1e-13 unless noted):

| setting | rows: near vs cold | rows: perturbed vs cold |
|---|---|---|
| test deck (mpol 2, ns 7, FTOL 1e-10), CPU | 1.38e-3 | 6.83e-4 |
| same, FTOL 1e-14, CPU | 1.37e-3 | 6.83e-4 |
| ns 15, FTOL 1e-13 (scan runs) | 3.8e-4 to 2.6e-3 | 5.8e-4 to 7.9e-3 |
| deck, m = 1 Z force released (VMEC2000) | 1.37e-3 | 6.83e-4 |
| deck, m = 1 Z force kept to convergence | **1.11e-4** | **3.46e-5** |

All solves report fsq <= 1e-13 and `converged`.

**Plain fixed-boundary solves, cold vs perturbed start** (`lam0.py`, FTOL 1e-13):

| run | R_cos | Z_sin | L_sin | iota | note |
|---|---|---|---|---|---|
| ns 7 | 2.3e-5 | 2.3e-5 | 4.0e-4 | 2.1e-5 | |
| ns 15 | 5.2e-5 | 5.0e-5 | 8.7e-4 | 1.3e-5 | |
| ns 31 | 4.8e-5 | 4.5e-5 | 1.3e-3 | 3.4e-5 | |
| ns 51 | 5.5e-5 | 5.1e-5 | 1.8e-3 | 1.8e-5 | |
| other resolution | 8.1e-5 | 7.8e-5 | 2.4e-3 | 3.1e-6 | |
| with `lambda_m0_floor` 1e-3 | unchanged | unchanged | unchanged | unchanged | for both pairs |

- **The mpol labels of these runs are uncertain:** two runs' logs look swapped relative to their names, and the
  ns 15 run reproduces another exactly.
- **What every run shows:** the difference does not shrink with ns, mpol, FTOL or the m = 0 lambda preconditioner,
  and repeat cold solves are bitwise identical (`repeat.py`).
- **The residual along the chord** (`valley.py`) follows fsq(alpha) ~ [alpha(alpha - 1)]^2 to the measured precision:

  | alpha | 0 | 0.25 | 0.5 | 1 | 1.5 | 3 |
  |---|---|---|---|---|---|---|
  | fsqr | 9.6e-14 | 2.2e-11 | 3.8e-11 | 9.9e-14 | 3.4e-10 | 2.2e-8 |

  The linear term is below the measurement's resolution: the force Jacobian is nearly singular along the difference
  (a near-null direction). Eight points do not establish an exact null; the relabelling-generator test of work
  package 1 is the direct check.
- **Decomposition** (`gauge.py`, ns 15): on each surface, 84-97% of |dX|^2 is the m = 0 relabelling theta ->
  theta + f(s, phi) and 95-100% is tangential. The normal part is 1.5-10 um; iota differs by 1.3e-5. With the m = 1
  Z force kept, the difference is 100% tangential, the normal part 0.2-1.3 um, iota 2e-6.
- **Surface distance** (`lam0.py`, ns 15): the curves differ by 39/35/6/2.5 um at js 3/7/11/13 against coefficient
  changes of 2-8e-5. Near the edge the difference is mostly relabelling.
- **Cause.** `residue.f90` (and VMEX) zero the constrained m = 1 Z force once fsqz < 1e-6; implicit.py documents the
  "released m = 1 combination" as an angle gauge.
- **Keeping the force is not a drop-in fix.** With `keepm1plugin`, the interior and plasma-map tests still fail and
  `test_three_term_problem_gradients_match_resolved_differences` fails: the implicit layer freezes the released
  combination.

## 7. Fixtures

| fixture | case | step 0 (seed accepted under the opt-in `boundary_normal_tol` 1e-3, C 1e-3: **not** strictly certified) |
|---|---|---|
| qh25 | qh4-beta-A8-merc, 2.5%, Redl | QH 1.3154, B.n 4.7e-4, K 5.7e-4, condition 8.6e5 at step 0 (5e7 in the fits) |
| qa25 | qa4-beta-merc, 2.5%, Redl, vacuum iota floor 0.31, iota ceiling 0.495 | QA 0.1168, B.n 2.4e-4, K 2.7e-4, Redl max rel 4.3e-5 |
| qi25 | qi6-beta-A8, 2.5%, DKX | QI 0.9327, B.n 2.2e-4, K 2.7e-4 |
| qivac | qi | QI 1.0691, B.n 2.7e-5, condition 4.6e3 |
| qavac | qa3 | QA 0.1586, B.n 4.7e-5, condition 2.1e3 |
| qhvac | qh | QH 1.9163, B.n 1.6e-4, condition 1.8e3 |

Seed equilibrium of qh25: n0 2.808e19, T0 2294.6 eV, CURTOR -14811 A, Redl mismatch 3.7e-6, <beta> 2.4987%,
iota 1.153-1.297. Its coils come from the one coil fit that ran to completion: 2128 L-BFGS-B iterations ending
ABNORMAL, B.n RMS 1.4e-2 to 3.8e-3 (PROVENANCE.md).

## 8. Process notes (what cost time)
- **Overlapping steps share a GPU:** `srun --overlap --gpus=1` steps launched together landed on one GPU and ran out
  of memory → `onnode.sh`.
- **A step cancel did not stop a run:** it kept writing into a reused output folder (the first QH seed). Kill by pid
  and never reuse an output folder.
- **`--modes 10 10` with `--seed-run`** keeps the seed deck's 8 x 8 (those runs were invalid and stopped).

## 9. Open (for the decision on next steps)
1. **A faster solve along the weak boundary directions:** about 10 fresh-Jacobian rounds for 3e-4 in the loss is
   the accuracy and cost blocker. Candidates: more LM steps per Jacobian with a model-based stop instead of the 5%
   stall rule (runs with `STALL_STEPS` off did not reach a certificate within the allocation), Jacobian updates, the
   curvature term on the weak subspace.
2. **The interior gauge:** a designed gauge condition for the m = 0 relabelling with matching implicit derivatives
   (own PR); it decides the two failing tests.
3. **The labelling benchmark** on the new fixtures at interface grid 64 (C vs NL agree there on QH).
4. **G vs G + N I:** keep simsopt's convention, or switch (0.6% CURTOR on QH).
