# Three-term free boundary and single-stage optimization: work ledger

This ledger is the authoritative status. It is linked from `README.md`, `WP0_WP1.md` and `WP2_WP3.md`. One row per
issue.
- **Status:** present / fixed (evidence) / unresolved / superseded / outside the supported model.
- **Cause:** "hypothesis" until demonstrated.
- **Tolerances and iteration budgets:** changes to them are listed separately from algorithmic fixes.

Heads at the start of the 2026-10-22 night program:
- PR #121 (`tt1-robust-fb`): `92b721c8`.
- PR #122 (`fix/pr121-bootstrap-cert`): `6079d1e5`.

Run artifacts are under `paper/runs/{wp0,wp1,wp2,night}` on Perlmutter scratch; compact copies are in `raw/`. GPU budget
for this program: at most 4 nodes / 16 GPUs at once, including running jobs.

## Package A: equation identity and certification

| id | issue | status | evidence / reproduction | next action |
|---|---|---|---|---|
| A0 | Interior start dependence (rows 7e-4 to 1.7e-2 between starts) | **fixed**: cause demonstrated as the frozen m = 1 complement (`constrain_m1` release) | WP0_WP1 items 9-12; swap 1e-13 to 2e-11; K spread 2e-13 to 2e-11; DESC shows no accuracy cost; `test_interior_state_does_not_depend_on_its_start` passes (GPU, 038065f3) | A1 |
| A1 | `M1_COMPLEMENT` is a mutable module global: residual and projector read it at trace time, and configs are canonical (shared), so switching in one process reuses stale executables and solves | **fixed, GPU-verified** (cbc956a7; tests at 5b2f5c38, job 60145562 a0: 4 passed): `ImplicitConfig.m1_complement` in config identity; K -> frozen -> K in one process equals a fresh process; NESTOR pinned frozen; driver `--m1-complement` | CI: the frozen-variant li383 test passed on GPU and failed on CI (closure 0.21) until caches were cleared (5c200092, a test-setup change, not a fix) | make the policy an immutable `ImplicitConfig` field in cfg identity, residual/projector, caches, checkpoint, manifest; test K -> frozen -> K in one process against fresh-process references |
| A2 | The final interior root is not certified in the expanded (K) equations: the status lane uses the pre-refinement host residual, and a stalled refinement can be returned | **implemented**: the refined state's residual is measured independently in the config's equations; status 4 if above `refine_tol` or not finite; `refine_tol=inf` with solved complement refused; **GPU-verified** (`test_status_certifies_the_refined_root_not_the_host_solve`: success 0, moderate stall 4, nonfinite 4) | `_host_solve_and_mask_status` docstring: "That residual is the host solver's own, from before the fixed-point refinement" | measure the final selected residual and complement residual; certified only if the declared root criterion passes |
| A3 | Raw tangent-response acceptance: preconditioned convergence OR `raw_ok`, so a preconditioned pass can override a failed raw residual | **implemented**: acceptance is the raw residual alone (finite, <= raw tolerance); `LinearResponseReport` gains `raw_residual_norm`, `raw_tolerance`, `preconditioned_converged`; the refined column's raw residual replaces the recorded one; tests for the three cases (GPU pending) | `_raw_certified_report`, `_implicit_evolved_tangent_multi_rhs` | govern acceptance by the raw residual (abs + rel); test the four pass/fail combinations, multi-RHS, nonfinite and budget cases; record raw residuals on the production probes |
| A4 | Same equations across trial, finalization, Picard and export: `self_consistent_bootstrap` and export re-solved with the host solver (frozen complement) while the model's tight states are K | **fixed for Picard and export** (demonstrated cause of B1): `optimize.refine_equilibrium` refines a host solve onto the selected root and certifies it; Picard (both lanes) and `equilibrium_from_x` use it with the model's policy; export raises instead of warning when it moves the state. **Open: trial evaluations** (`tight=False`) still use the unrefined host state, i.e. the frozen equations with a history-dependent complement | GPU test pending; trials: decide refine-or-budget |
| A5 | NESTOR path: `_edge_basis` and raw/preconditioned complement treatment versus the K projector | **present** (unreviewed) | `freeboundary_implicit.py` | make the NESTOR path explicitly frozen (legacy) until it is upgraded consistently; test basis completeness |
| A6 | Public API: symmetry and edge-pressure checks at every constructor; `start="best"` ranking; NESTOR seed failure must not erase another seed | **partly implemented**: `_best_three_term_fit` ranks successful fits before cost (test); an optional NESTOR seed's failure no longer aborts `start="best"`; the multigrid three-term result is refined onto the fit's policy and re-reported (raises if the host re-solve moves it). Symmetry/edge-pressure audit of all constructors still open | GPU test of the multigrid path pending |

## Package B: bootstrap current

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| B1 | `test_fixed_boundary_bootstrap_model_matches_picard`: 26/29 entries off, max 3.98% (CI at 5c200092 and 6079d1e5) | **cause demonstrated: different equations (A4)**. Same policy, fresh processes: frozen/frozen model vs Picard 2.9e-5, CURTOR 4e-6, Picard current's closure in the model evaluator 3.2e-5; K model vs frozen Picard 0.95%, closure 1.2% (`raw/wpB/b1_*.json`, job 60144384). The bootstrap current itself depends on the m = 1 treatment at about 1% on this coarse deck (mpol 3) | **fixed**: with A4 (Picard on the model's policy) and the template priming fix (821eab8d) the test passes (CPU run, 6:47) |
| B2 | Finite-weight joint stationarity does not imply exact closure | open (by construction) | the certificate already has a separate closure gate (`CLOSURE_ATOL/RTOL`) | verify closure is reported separately in every status; test a case with stationarity but no closure |
| B3 | Model semantics: prescribed / inner bootstrap / outer design with a mismatch constraint; Redl only for QA/QH; DKX for QI | partly documented | memory rule: Redl only QA/QH; QI fixtures use DKX | document and test each mode |
| B4 | Conventions (G, sign, helicity, nfp, units) on converged fixed-boundary states | **default matches both references**: vmex `g_convention="simsopt"` (prefactor -G e / (psi_a (iota - N))) is DESC's `compute_J_dot_B_Redl` (dbf5fc564, local `refs/DESC-latest`) and simsopt's; DESC's helicity N is nfp-scaled (QH (1, nfp)), as vmex's N = nfp * helicity_n. `"isomorphism"` (G + N I) differs only for QH (QA N = 0; QI uses DKX). Whether QH needs G + N I is **unresolved** (derivation from the Helander/Landreman isomorphism literature not yet done); the size N I / G on the QH fixture is to be measured | derive from the literature; measure N I / G on the qh25 state; no per-case selection |
| B5 | Zero-drive Picard test: 4 iterations against tol 1e-8 (contracts about 1000-fold per step) | **fixed (test setup)**: 8 iterations | GPU pass (job 60144384, t1.log) | none |

## Package C: differentiation

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| C0 | Fixed-boundary implicit derivative vs re-solve FD | **fixed** with K: 9.9e-7 / 2.9e-6 (frozen: 1.0e-2 / 2.8e-2) | `test_li383_mean_iota_matches_resolve_fd_with_the_m1_complement_solved` | |
| C1 | Small-deck design gradient vs converged FD: 0.67%, uniform across the three coordinates (all act through one combination) | **cause demonstrated: the Gauss-Newton fit sensitivity** (`scripts/wp5/exact_sensitivity.py`, `raw/wp5/exact_sens.json`). The exact stationarity derivative (H = J^T J + S_bb and F_d by central differences of g = J^T r at fixed boundaries) matches the re-solved FD to 7.9e-5; the problem's gradient is 0.674% off; GN rebuilt from the same differences equals the problem's gradient to 3e-7 (no other defect). S_bb is 0.27% of J^T J. GN meets the 1e-2 release requirement here and misses the 1e-3 target | decide GN-with-budget vs exact per domain from production measurements (C5); exact reference mode for qualification |
| C2 | The design gradient is Gauss-Newton (documented in `_least_squares_sensitivity`), exact only at zero residual | **decision**: GN stays the optimizer's documented approximate mode; `ThreeTermFreeBoundaryProblem.exact_design_gradient` (GN-preconditioned CG, differenced H products, F_d^T mu along mu) is the qualification reference, **verified**: 8.0e-5 vs the converged re-solved FD (GN 6.7e-3; job 60144384 t11) | measure the GN error per domain (C5) |
| C3 | Raw boundary Jacobian vs tight central differences | **exact at production** (`raw/wp5/jacobian/`, 8 x 8, ns 51, 3 random + weakest direction, h 1e-2/1e-3/1e-4): QI 2.5% 2e-8 to 3e-7 at h 1e-4; QH 2.5% seed and late NL record 5e-7 to 2e-4; QA 2.5% 2e-5 to 1e-3, all converging about as h^2 to the rows' noise floor (QA weakest direction: floor 4.4e-3 against a tiny Jd). **The Redl bootstrap rows are strongly nonlinear**: at h 1e-2 (below a polish round's step, about 0.05) the linear model misses by 4-60%, almost entirely in the bootstrap block | D1: current-following response and step strategy for the bootstrap rows |
| C4 | Parameter ownership: `with_boundary(params_at(P0, d), y)` overwrite; plasma-coordinate detection probed only at x0 | **unreviewed** | | r = w x regressions; explicit design-family metadata |
| C5 | Production design gradients (`--check-gradient`) and boundary Jacobians (`--check-jacobian`) | **QI vacuum: full design gradient verified** (gate 1e-5, one random direction, `raw/wp5/production/`): loss implicit vs FD 4.0e-2 / 3.2e-3 / **9.1e-4** at h 1e-4 / 3e-5 / 1e-5; constraint rows 8.4e-3 / 7.4e-9 / **1.4e-8**. The larger values at bigger h were FD curvature; the GN error is negligible here (tiny vacuum residual). The deterministic K solves are what make h = 1e-5 usable | QH/QA/QI 2.5% (bootstrap) and more directions/design families; GN vs exact at production (`--check-exact-gradient`, next allocation) |

## Package D: boundary coordinates and steps

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| D0 | Slow polish on the 2.5% fixtures | NL plateau **fixed** by K. **Cause located**: Jacobians are exact (C3); the Redl bootstrap rows are strongly nonlinear in the boundary coordinates (the linear model misses by 4-60% at a 1e-2 step). QH 2.5% with the current prescribed certified after 74 rounds; with Redl bootstrap rows it did not after 90. Earlier statement 'the bootstrap coupling is not the obstacle' is **superseded** | Per-row dump (`--jacobian-dump`, night2/dump-qh25): the misprediction is spread over all radii, peaking at the sub-axis knots; the random directions move the 53 current coordinates, and Redl's I' depends on iota, which depends on the current (1/iota feedback). Hypothesis 'the inner current re-solve is inexact per step' **rejected**: proj4 vs proj16 (job 60145562) are identical round for round, with zero unconverged current re-solves; each polish round stops 'stalled' after 14 evaluations (steps with one Jacobian gain under 5%), the signature of the bootstrap rows' curvature | **Inner-current hypothesis rejected** (proj4 vs proj16, job 60145562): identical rounds, zero unconverged projections, 13-14 evaluations per polish round, every round 'stalled' (one useful step per fresh Jacobian). Next: geodesic acceleration (one extra evaluation per step for the rows' second directional derivative), against the measured curvature at the step scale |
| D1 | Rejected without new evidence: Broyden, GN extrapolation, 10 x 10, model-aware stall rule | **superseded** | WP2_WP3 tables | do not revisit |
| D1b | Structured secant (NL2SOL / Dennis-Gay-Welsch) estimate of S across polish rounds | **rejected by offline evidence** (NL-K90, 89 rounds, reduced boundary space): Gauss-Newton mispredicts the round-to-round reduced-gradient change by a median of 3.9x the change itself, and the DGW update diverges, so the curvature along the path is not near-constant | do not implement |
| D1c | Inner current re-solve inexact per step (budget) | **rejected** (proj4 = proj16) | |
| D2 | Current-following linearity (`--jacobian-following`, QA 2.5% seed) | **the strong nonlinearity is the fixed-current path**: with the current re-solved to conditional stationarity (1e-14), the rows' error against the reduced model (I - P_c) J_b d is 1.5e-3 / 1.6e-4 at h 1e-2 (two random directions), against 0.30 / 0.061 with the current fixed; the weakest direction sits at a 2-4e-3 noise floor. **QH 2.5% the same**: 4.2e-3 vs 0.26 at h 1e-2, 3.3e-6 vs 7.7e-5 at h 1e-3 (one 1e-2 point's 1e-14 current solve did not converge in 20 iterations). The fit's projection stops at a gain of 0.1 ftol (1e-5 of the cost in the polish), loose against the per-round cost changes | test a tighter conditional-stationarity tolerance in the polish's current projection |
| D3 | Tighter current-projection gate (`PROJECTION_GATE` 1e-12, 12 corrections; projgate, QH 2.5% NL) | **rejected**: per-round cost equals the baseline (2.860e-7 vs 2.867e-7 at round 6), most rounds end `projection_unconverged` (the 1e-12 gate is below the rows' noise floor), and it is slower per minute | the remaining cause is conditioning: reduced singular values about 12 down to 4e-6 (condition about 3e6); D1 must classify those weak directions (relabelling vs physical vs representation) |
| D1d | Classify the weak directions of the reduced QH 2.5% Jacobian (last records of NL-K90, C-K90, QH-presc80; the record's normal operator) | **predominantly tangential relabelling**: the 10 weakest right singular vectors (singular values about 2-4e-4 of a scaled reduced Jacobian with condition 1.0-1.4e5) move the boundary normally by 0.01-0.08 of a typical direction (the strongest: 44-208). The equal-arc labelling rows at weight 1e-3 do not pin them (C's condition equals NL's) | per D3 rule: develop a geometric chart only with a controlled representation error (8 x 8: 0.36-0.79 measured) - needs higher compatible boundary resolution, or an exact gauge (constraint, not penalty) on the relabelling subspace; a stronger penalty is excluded by the brief |
| D1e | Are the weak directions exactly the boundary relabellings? (`scripts/gauge_offline.py`, last records of NL-K90 and QH-presc80) | **yes**: the 10 weakest reduced right singular vectors lie 97.1-99.9% inside the span of the projected relabelling directions theta -> theta + f (127 of 255 dims; a random unit vector: median 70%). The relabelling basis itself is only partly representable at 8 x 8 (projection error median 11%, max 81%) | finalist 2: an exact gauge (steps orthogonal to the projected relabelling span, plus a deterministic labelling condition), no penalty and no new representation; G1 decides how much a certified result depends on its start |
| D2 | Normal chart: condition 30 vs 2.8e3, but representation error 0.36-0.79 | candidate, not qualified | `chart_offline.py` | D2 qualification only if D1 shows the weak directions are relabelling |

## Package E: dependencies and environment

| id | issue | status | evidence |
|---|---|---|---|
| E0 | One reproducible runtime artifact | **recorded**: `ENVIRONMENT.md`, `raw/env/` (ESSOS 0.21.0, SOLVAX 0.31.0, JAX 0.12.0.1 CUDA 13, DKX 2.11.0; vmex always from a pinned tree) | |
| E1 | ESSOS `DirectCoilField` vs native ESSOS parity (field, JVP/VJP, nfp, symmetry, currents) and its stale rationale | **resolved, adapter kept** (`scripts/wpE/essos_parity.py`, `raw/wpE/`, qh25 nfp 4 and qavac nfp 3, positive and negated currents). Values agree to 4e-16; the adapter's JVP matches central differences to 2e-8; native ESSOS 0.21.0 JVP through `Coils(Curves(dofs), currents)` differs by 1.3-12%, and native coils cannot be re-traced under one `jit` (array pytree metadata). Adapter under `jit`: one compilation, 6 ms warm. Rationale docstring corrected | coil-quadrature convergence and clearance (point-cloud vs continuous) still open |
| E2 | SOLVAX operator certification (dense/QR references, transposes, warm/cold, multi-RHS) | partly covered by A3's raw certificate; dedicated tests **open** | |

## Package F: restart and export

| id | issue | status | evidence |
|---|---|---|---|
| F1 | Atomic checkpoint; build from checkpoint with no seed solve; driver `--checkpoint-every`/`--resume` | **implemented**, GPU test passes (8e2956e6) | `test_checkpoint_restores_the_accepted_equilibrium` |
| F2 | Problem identity is field/plasma values at the saved design only | **implemented** (schema 4): the definition adds coordinate names and scales, value-row count, the m = 1 policy (model options) and a caller-supplied `semantic_identity`; tests refuse other identity and other policy (GPU pending) | |
| F3 | Export (`equilibrium_from_x`) re-solves with the host solver (frozen), a warning only | **present** (same root as A4) | | export the certified state or recertify under the same policy |

## Package G: qualification (strict certificate, K)

| fixture | status | evidence |
|---|---|---|
| qhvac | **certified** (5 rounds) | wp2/fx-qhvac-K |
| qivac | **certified** | wp2/fx-qivac-K |
| qavac | **certified** at 9.7e-5 (24 rounds; reproduced in a fresh allocation) | wp2/fx-qavac-K60, night/qavac-K40 |
| qi25 | **certified** at 8.1e-5 (21 rounds; reproduced) | wp2/fx-qi25-K40, night/qi25-K40 |
| qa25 | 6.2e-3 after 29 rounds; cut by the allocation's end | night/qa25-K40 |
| qh25, prescribed current (fixture deck and current, no `--bootstrap`) | **certified** at 8.9e-5, cost 1.405e-7, after 74 rounds (36 min, 038065f3) | night/QH-presc80 |
| qh25, bootstrap (Redl), NL labelling | **certified** at 5.6e-4 (gate 1e-3) after 91 polish rounds (87 min, 038065f3, job 60144384); **bootstrap closure passed**: relative 1.9e-10, CURTOR applied -14743.890978 A vs Redl -14743.890977 A; cost 1.435e-7; condition 3.6e7 | night/NL-K90 |
| qh25, bootstrap (Redl), equal-arc labelling | at the gate (1.0e-3) after 90 rounds when the allocation ended (cost 1.849e-7) | night/C-K90 |

**Reproducibility across allocations** (fresh processes, different nodes, pins 8e2956e6 vs 038065f3;
`scripts/compare_starts.py`, `raw/g1/`):
- qavac: surfaces 1.2e-13 m, values 8.4e-13;
- qi25: surfaces 1.8e-11 m, values 2.8e-9.

**G1 start dependence:** `--seed-start random EPS SEED | scale EPS` (18fb4ab4); qhvac and qivac from three
perturbed starts are queued (job 60145562).

Tolerance and budget changes (not algorithmic): POLISH_ROUNDS raised per run (fbrecord env), POLISH_FTOL override
(6079d1e5), zero-drive Picard iterations 4 to 8, the frozen li383 test's cache clearing.

## Pre-existing failures (not caused by this work; identical at af30041d)

Under the GPU-default runtime on Perlmutter these tests fail with a CPU/GPU device-placement `ValueError` (CI's
CPU lanes and the two-device lane pass):
- `test_implicit_grad.py::test_three_surface_raw_kernel_matches_global_nonlinear_rows` (x2);
- `test_implicit_grad.py::test_raw_residual_jvp_is_exactly_nearest_neighbor` (x3);
- `test_optimize.py::test_equilibrium_from_x_is_the_state_the_objective_read` (x2);
- `test_optimize.py::test_certified_trial_guards_reject_stale_or_missing_memo`.

`test_implicit_grad_fd.py::test_typed_error_through_pure_callback` fails identically at af30041d: solovev converges
in 3 iterations, so the expected error is never raised. These are environment and test-setup issues, tracked here.
