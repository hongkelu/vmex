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
| C2b | GN vs exact at production (`--check-exact-gradient`, default 1e-3 gate) | **the reference is stable in h** (qivac: 1.88e-2 / 2.07e-2 / 1.89e-2 at h 1e-3 / 1e-4 / 1e-5). Values: qavac 2.0e-3, qivac 2.1e-2, qi25 4.6e-2, qhvac 6.7e-2 (CG not converged in 30). At the 1e-5 gate the GN gradients equal converged FD to 1e-8..1e-6 (C5), so at the 1e-3 gate both formulas linearize a non-stationary fit; neither is the derivative of the re-solved map. **Consequence: derivative qualification requires the 1e-5 gate**; an optimizer run at the 1e-3 gate gets gradients off by up to several percent | decide the production gate (cost vs gradient accuracy); exact check at the 1e-5 gate to close the comparison |
| C3 | Raw boundary Jacobian vs tight central differences | **exact at production** (`raw/wp5/jacobian/`, 8 x 8, ns 51, 3 random + weakest direction, h 1e-2/1e-3/1e-4): QI 2.5% 2e-8 to 3e-7 at h 1e-4; QH 2.5% seed and late NL record 5e-7 to 2e-4; QA 2.5% 2e-5 to 1e-3, all converging about as h^2 to the rows' noise floor (QA weakest direction: floor 4.4e-3 against a tiny Jd). **The Redl bootstrap rows are strongly nonlinear**: at h 1e-2 (below a polish round's step, about 0.05) the linear model misses by 4-60%, almost entirely in the bootstrap block | D1: current-following response and step strategy for the bootstrap rows |
| C4 | Parameter ownership: `with_boundary(params_at(P0, d), y)` overwrite; plasma-coordinate detection probed only at x0 | **unreviewed** | | r = w x regressions; explicit design-family metadata |
| C5 | Production design gradients against central differences of re-solved trials (`--check-gradient`, gate 1e-5, steps 1e-4 / 3e-5 / 1e-5, one random direction; `raw/wp5/production/`) | **verified, every row, three fixtures**: qavac loss 2.6e-8, rows 4e-11 to 5e-10; qivac loss 9.1e-4, rows 2e-11 to 4e-8; **qi25 (DKX current as a design variable) loss 6.1e-6 and 7 of 8 rows 4e-9 to 2e-6, including the DKX mismatch row**. The exception is qi25 **row 6, the coil-plasma clearance**: implicit -0.050 against an FD that does not converge (+0.087, +0.084, +0.129). That was the point-cloud minimum's closest-pair switch; **with the continuous clearance (E3) every qi25 row matches (worst 1.5e-6, clearance 2.6e-8)** | QH 2.5% (bootstrap) still to check (its seed does not certify); more directions and design families |

**C5c CORRECTION (10-23 13:35): the warm gradient checks' small-h matches were circular.** Difference trials start
from the Gauss-Newton boundary prediction. Where it already certifies (nfev 1, accepted 0: qavac every late trial,
qivac and qi25 most), the difference reproduces the GN formula by construction. So the C5 / C5b "matches" (qavac,
qivac, qi25, QA/QI 2.5% prescribed) **do not verify the gradient**, and "QH-specific" is withdrawn. The QH failures
were trials that moved. Fix: `--check-gradient-cold` (4e4c8eb1), and per-trial steps are recorded.
- **First reference independent of differences: qivac, equal arc, polish target 1e-5.** The exact stationarity gradient
  **converged** (CG 26, residual 6.6e-9, certificate 9.8e-6). **GN vs exact: 2.8% (max component 4.4%), cosine
  0.9996.** GN is about 3% off on QI vacuum too.
- **QH 2.5% prescribed, equal arc 1e-3, target 1e-5: the exact gradient converged** (CG 26, residual 4.1e-9). Along
  the check's direction: **exact -2.702**; differences from trials that moved -2.847 (h 3e-5) and -2.929 (h 1e-4),
  within 5-8% of exact; **Gauss-Newton -4.196, 55% off** (over all components 37%, cosine 0.953). The h 1e-5
  "match" (-4.196) was the circular predicted trial.
  - So on QH 2.5% GN is wrong by tens of percent and the converged exact derivative is right.
  - With equal arc it converges in 26 iterations; with NL it had not in 30.
- **First valid cold-difference check: qavac** (equal arc, target 1e-4, at the optimization's step-11 checkpoint; 4e4c8eb1).
  Every trial moved (one accepted step from the seed boundary). The differences are stable in h (-0.0235105 at
  3e-5 and 1e-5). **GN: loss 1.0e-3, quantities 8.9e-3 off.** So GN is approximate on QA vacuum too (0.1-0.9%).
- **qavac exact gradient converged** (CG 16, equal arc, target 1e-4, from a later step of the same optimization):
  **GN vs exact 0.24% (max component 0.38%), cosine 0.999997.** The magnitude matches the cold check. The two
  lanes copied the checkpoint at different steps, so they are not compared along one direction.
- **Cold checks: the second flaw.** A cold trial with **0 accepted steps** never left the seed boundary: the seed
  already passes the gate at the perturbed design, because the leftover is relabelling. Its difference is the
  **fixed-boundary** derivative, with no boundary response.
  - QI 2.5% prescribed: [0, 0] at both h. Not valid.
  - QH 2.5% equal arc: h 3e-5 [3, 1] gives -3.10; h 1e-5 [0, 0] gives -4.33. Exact (converged) is -2.70; GN -4.20.
  - QH 2.5% NL: h 3e-5 [1, 1] gives -4.55; h 1e-5 [0, 0] gives -5.57. GN -5.70.
  - QA 2.5% prescribed: [1, 1] at both h, stable. Loss 0.28% off GN; rows within 0.1-16% except row 5 (0.74 vs 1.37).
  - **Conclusion: at the certificate's gate, the re-solved map is not determined along the weak directions.** A trial
    stops where its start and the gate put it: the GN prediction (giving GN), the seed (giving the fixed-boundary
    derivative), or part way. **Finite differences of re-solved trials are not a reference on these problems; the
    converged exact stationarity derivative is.** The optimizer, too, sees whatever the solver does along those
    directions.
- **QA 2.5% prescribed: exact gradient converged** (CG 25): **GN vs exact 4.1%** (max component 3.5%). Along the
  check direction: GN -0.067266, exact -0.066541, one-step cold trials -0.067458. The partly moved trials sit nearer
  GN than exact, so even "moving" differences are not a reference.
- **QH vacuum NL (target 3e-4): exact gradient converged** (CG 19): **GN vs exact 1.6%** (max component 1.2%).
  - A trustworthy difference check needs trials converged in **step**, not just gain (a boundary step gate). That
    is slow without option 5, which removes the undetermined directions.
- **QA 2.5% prescribed, cold check** (NL, target 1e-5; the trials moved, one step each; differences identical at h 3e-5
  and 1e-5). **Loss: GN 0.28% off.** Rows: iota 0.6%, R 14% (small magnitude: 0.0056 vs 0.0065), B 2.9%, mirror
  0.2%, max iota 0.2%, **vacuum-iota floor 46% (GN 0.738 vs 1.365)**, clearance 1.3%, aspect 4%.
  - The vacuum-iota row (a nested vacuum fixed-boundary solve, min over surfaces) is far outside the others: it is
    a suspect row-specific derivative defect.
  - Running: its exact derivative, `--exact-gradient-row 6`, 4ab973c0.
  - The earlier warm "match" here (1e-8) was the circular check.

**C5b day3 gradient checks** (gate 1e-5, re-solved central differences at h 1e-4 / 3e-5 / 1e-5; `raw/wp5/day3/`):
- qavac, 3 directions: loss matches to 7e-9..8e-8 at h 1e-5. With the **continuous clearance** (22e60e13) its row's
  differences grew as 1/h (f(x+h) - f(x-h) = -7.6e-6 scaled at every h): the parabola vertex jumped when the closest
  coil sample changed. The points control matched to 1e-10. **Fixed in 876542bd** (joint Newton on the interpolated
  coil and the surface; sliding the samples changes the value by < 1e-10, the old one by 5.2e-4 m).
- qi25, 2 directions: loss 6.1e-6 / 2.8e-6, quantities 3.4e-8 / 4.9e-9 at h 1e-5 (larger h: the fit's nonlinearity).
- qivac, 3 directions: loss differences do not settle with h (1.7e-2, 2.0e-3, 1.7e-2 in direction 1) while
  quantities match to 1e-8 in two directions: the loss's finite differences are fit-noise limited at this gate.
  Next: the exact stationarity gradient instead of differences.
- **QH 2.5%, prescribed current** (stationarity 6.1e-6, condition 1.4e5): **every row's implicit gradient is
  12-17% off and the loss 18-23%, at all three h** (differences agree among themselves to about 2%: -4.77, -4.62,
  -4.83 against -5.70). A uniform error points to the Gauss-Newton fit sensitivity (the dropped residual-curvature
  term, which a tighter gate does not remove at a nonzero-residual fit). Next: `--check-exact-gradient` at the same
  seed (queued, G3 allocation); if confirmed, production needs the exact sensitivity on QH 2.5%.
- **QI 2.5%, prescribed current** (G3, 54140c07; seed built in 33 min): **Gauss-Newton matches** at h 1e-5: loss 1.3e-6
  / 2.5e-5, quantities 1.5e-8 / 4.2e-9 (2 directions; larger h is the fit's nonlinearity). So the QH 2.5% failure is
  not general to finite beta. QA 2.5% prescribed, QH vacuum (target 3e-4) and the exact-gradient checks are running.
- **QA 2.5%, prescribed current**: **Gauss-Newton matches** at h 3e-5 and 1e-5: loss 4e-9 to 9e-9, quantities
  2.5e-8 to 3.7e-7. QA and QI at finite beta are fine; QH is the exception.
- **QH vacuum, equal arc 1e-3, target 3e-4**: loss **2.3e-3** and quantities **1.7e-2** off, the same at h 3e-5 and
  1e-5 (systematic, not noise). So the QH error exists in vacuum, about 10x smaller than QH 2.5%'s 12-23%.
- **NL seeds in QA/QI vacuum: 6 of 6 failed to certify in 120-160 polish rounds** (QH vacuum NL seeds do certify:
  3 of 3, point spacing 8.8 against equal arc's 5.45). qivac x3 (gain 1.77e-3, normal step 1.4e-6
  of the minor radius; identical in three processes on two allocations), qavac x2 (7.6e-3, 1.2e-6). The surfaces
  converged; what remains is relabelling (D5). The equal-arc seeds of the same fixtures certified in about 15 min.
- **NL started from the equal-arc fit's certified boundary does not certify either** (qivac, 120 rounds): gain
  9.55e-2, normal step 1.0e-5. That is further from certified than the cold NL start (1.77e-3). Without labelling
  rows the valley's minimum is not where the equal-arc fit sits but where the discretization effects put it, and
  Levenberg-Marquardt drifts slowly toward it. A good start does not cure NL; the relabelling freedom itself is the
  problem (gauge), consistent with D5 and the QH gradient.
- **QH 2.5% prescribed: exact stationarity gradient** (G3 exact-qhpresc, the same deterministic seed: GN . d reproduces
  -5.6988). Along the difference check's direction: **Gauss-Newton -5.70, exact -3.60, re-solved differences -4.62 to
  -4.83**. The exact gradient is **not converged** (CG 30 iterations, residual 1.4e-7; 1657 s). Over all 229
  components: GN vs exact 35%, cosine 0.962.
  - The three disagree by 20-25%, so none is the reference yet.
  - Exact needs H^-1 along the relabelling directions, where H is nearly singular.
  - The differences come from trials that stop about 2e-3 off the stationary point along the same directions (D5).
  - So on QH the stationary point's derivative is ill-conditioned in the relabelling directions.
  - Next: a converged exact gradient (`--exact-gradient-maxiter 90`, from the saved seed, 1b8eb81b), and a physical
    (relabelling-free) sensitivity and certificate (option 5), where GN, exact and differences should agree.
- Speed observation (G3): with no labelling rows (NL), the vacuum seeds' certificates are much slower than with equal
  arc. qivac NL was still polishing after 45 min, while qivac equal arc built in 14 min and is at step 5.

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
| D1f | Does restricting the fit to the complement of the relabelling span fix the conditioning? (`gauge_offline.py`, last records of NL-K90 / QH-presc80 / C-K90) | **no**: in the column-normalized metric the LM uses, condition 2.8e3 / 2.8e3 / 2.2e3 (full) becomes 3.5e3 / 4.7e3 / 2.8e3 (complement), and the complement holds 39-47% of the remaining gain. The 1e5 condition quoted in D1/D1d was in unnormalized scaled coordinates. **The gauge finalist is dropped as a conditioning fix**; the evidence is consistent with Gauss-Newton's linear rate on a nonzero-residual fit (rounds realise their predicted gain, new gain appears through the curvature term S; cf. C1) | the polish is a performance item: QH 2.5% certifies in 74 (prescribed) and 91 (Redl) rounds, under 2 h; a Newton-type step (exact S products, as in `exact_design_gradient`) is the remaining candidate |
| D4 | Geodesic acceleration of LM steps (`GEODESIC`, e5d06267, unit-tested) | **implemented, not evaluated**; deprioritized: it targets curvature along the fixed-current path, which the current-following check (D2) showed is not where the steps lose | |
| D5 | Option 5 offline: how much of a certificate's remaining Gauss-Newton gain is relabelling (`scripts/certificate_split.py`; it recomputes each record's certificate from its stored arrays and refuses a mismatch). `gauge_offline.py`'s `gain_full` is **not** the certificate: it removes non-boundary columns from the rows first and normalizes by the reduced residual | **90-97% relabelling on all 7 records**: qivac 1.2e-4 -> physical 1.0e-5; QH 2.5% Redl NL 5.6e-4 -> 1.5e-5; QH 2.5% prescribed 8.9e-5 -> 8.4e-6 and 6.1e-6 -> 2.7e-7; qhvac (stalled) 2.7e-4 -> 1.5e-5; **qa25 (never certified) 6.3e-3 -> 4.9e-4, under the 1e-3 gate**; qi25 9.5e-6 -> 9.7e-7. The remaining step's physical part is 10-100x shorter than its relabelling part | upper bounds (relabelling basis 6-11% unrepresentable at 8x8). Next: a certificate and a design sensitivity on the physical complement (opt-in), tested on the QH 2.5% gradient |
| D2 | Normal chart: condition 30 vs 2.8e3, but representation error 0.36-0.79 | candidate, not qualified | `chart_offline.py` | D2 qualification only if D1 shows the weak directions are relabelling |

## Package E: dependencies and environment

| id | issue | status | evidence |
|---|---|---|---|
| E0 | One reproducible runtime artifact | **recorded**: `ENVIRONMENT.md`, `raw/env/` (ESSOS 0.21.0, SOLVAX 0.31.0, JAX 0.12.0.1 CUDA 13, DKX 2.11.0; vmex always from a pinned tree) | |
| E1 | ESSOS `DirectCoilField` vs native ESSOS parity (field, JVP/VJP, nfp, symmetry, currents) and its stale rationale | **resolved, adapter kept** (`scripts/wpE/essos_parity.py`, `raw/wpE/`, qh25 nfp 4 and qavac nfp 3, positive and negated currents). Values agree to 4e-16; the adapter's JVP matches central differences to 2e-8; native ESSOS 0.21.0 JVP through `Coils(Curves(dofs), currents)` differs by 1.3-12%, and native coils cannot be re-traced under one `jit` (array pytree metadata). Adapter under `jit`: one compilation, 6 ms warm. Rationale docstring corrected | coil-quadrature convergence and clearance (point-cloud vs continuous) still open |
| E3 | Coil-plasma clearance: point-cloud minimum (`surface_distance`: 192 coil points x 61 x 64 surface grid) | **implemented**: `continuous_surface_distance` (driver default `--clearance continuous`; `points` keeps the old one). Coarse grid search, the 7 best samples refined onto the continuous surface by Newton (Gauss-Newton fails when the distance is comparable to the surface's curvature radius), parabola vertex along the coil, envelope-theorem derivative, runner-up gap reported. Analytic torus/coil (`tests/test_three_term_clearance.py`, CPU, 4 passed): error 9e-7 / 1.7e-7 / 3e-9 at 32 / 64 / 128 samples (cube of the spacing), d/da = -1 to 1e-8, and below the point cloud | **requalified at production**: qi25 with the continuous clearance (22e60e13, day1), all 8 rows at h 1e-5 match the re-solved FD (clearance 2.6e-8; worst row 1.5e-6), loss 6.1e-6 |
| E2 | SOLVAX operator certification (dense/QR references, transposes, warm/cold, multi-RHS) | partly covered by A3's raw certificate; dedicated tests **open** | |

**E4 CTH filament field (benchmark coils, `paper/runs/cth`)**:
- The CTH-like coils (MAKEGRID polygons, `coils.cth_like`) were sampled by arclength at 2048 points per coil. Samples
  straddled the polygon corners, so the midpoint rule converged only linearly: **1.5e-4 of |B|** at the plasma against
  the exact straight-segment (Hanson-Hirshman) field, the size of the interface residuals being compared.
- Fix (`make_coils.py`): whole sub-segments per straight side. The convergence is quadratic (4.1e-5 / 2.1e-6 / 1.3e-7
  at 2048 / 8192 / 32768). The benchmark uses 8192 (2.1e-6).
- DESC (`desc_coils.py`): linear `SplineXYZCoil` through the same polygons, densified to 8180 collinear vertices
  (DESC integrates on the knots: with the 200 original vertices it was 1.6e-4 off). It is 1.0e-7 from exact and
  2.0e-6 from the vmex filaments.
- So all codes see the same field to 2e-6. Every earlier CTH result (10-20) used the 2048 arclength field and is not
  compared.
- The 4x segments made the 12x12 three-term solve ask for a 39.7 GB temporary. `DirectCoilField` now sums over
  chunks of at most 512 points (`3ac4aeda`; benchmark copy identical to 3e-16 in value and gradient).

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

**G1 start dependence** (`--seed-start`, `raw/g1/`; all starts certified at the default 1e-3 gate; compared with the
deck start):

| fixture | starts | surface distance | largest value change |
|---|---|---|---|
| qavac | random 1e-3, scale 1e-3 | 6-7 um | 1.2-1.8e-5 |
| qivac | 2 random, scale | 2-17 um | 0.5-2.6e-5 |
| qhvac | 2 random, scale | 28-65 um | 0.7-1.6e-4 |
| **qi25** | random, scale | **0.25 mm, 1.26 mm** | **0.95%, 2.7%** |

- Identical starts reproduce to 1e-13 m. The vacuum spreads are consistent with the slack a 1e-3 gate leaves.
- **qi25: distinct local minima** (day1, 1e-5 gate, dffc59aa). The deck start and random start 1 both certify
  (stationarity 9.4e-6 and 6.3e-6) and stay **0.27 mm** apart (values 0.22%), as at the 1e-3 gate, so this is not
  fit slack.
  - The costs differ by 1.0% (1.1014e-7 vs 1.0903e-7).
  - The difference lies 81% in the 5 weakest singular directions (97% in 20). The linear model predicts a cost
    change of 25x the cost along it, so each minimum is outside the other's linear regime.
  - So the finite-resolution three-term least squares has several minima along a flat, nonconvex valley (the
    multigrid docstring's "several minima").
  - The scaled start did not reach the 1e-5 gate in 60 rounds (1.3e-4).
  - **Selection rule:** among certified fits from several starts, the lowest cost (A6's `_best_three_term_fit`
    ranking).
  - **Two minima, both reproducible** (day1, 1e-5 gate):
    - two independent random starts (seeds 1 and 2) reach the same lower minimum: 7.4 um apart, values 1.5e-4, cost
      1.0903e-7 both;
    - the deck start reaches the higher one (1.1014e-7);
    - that higher minimum is stable in the virtual-casing grid: 48 -> 64 moves it 3.6 um, values 1.7e-5.

    So the two are genuine minima of the problem at this resolution, not quadrature artefacts. The physical gap
    between them (0.27 mm, 0.2% in outputs) is the size of the qualification uncertainty unless the global one is
    selected.
- The tighter-gate qhvac pair could not certify at 1e-5: QH vacuum's stationarity floor is about 3e-5.

**G2 resolution** (day3, pins 5faa0684 / 1749e663; certified at the default 1e-3 gate, compared with the certified
8 x 8 / VC 48 / ns 51 reference by `scripts/compare_starts.py`; `raw/g2/`):

| fixture | change | surface distance | largest value change |
|---|---|---|---|
| qivac | VC grid 48 -> 64 | 0.6 um | 0.13% |
| qivac | modes 8 x 8 -> 10 x 10 | 32 um | 0.45% |
| qhvac | modes 8 x 8 -> 10 x 10 | 0.53 mm | 0.22% |
| qavac | modes 8 x 8 -> 10 x 10 | 0.12 mm | 0.49% |
| qivac | ns 51 -> 101 | 14 um | 0.80% (min iota 0.5990 -> 0.6038; objective 0.013%) |
| qavac | ns 51 -> 101 | 21 um | 0.79% (min iota 0.4525 -> 0.4561; objective 0.16%) |

- The interface quadrature (VC 48) is converged; the boundary's mode count moves the outputs by 0.2-0.5%, so 8 x 8 is
  the optimization resolution and every final design is validated at 12 x 12 (as planned).
- At the default 5 polish rounds three of the higher-resolution seeds ran out still converging (qivac ns 101:
  stationarity 0.147 -> 0.017; qivac 10 x 10: 0.39 -> 0.076; qhvac VC 64); rerun with 40 rounds. The record then said
  `rejected`; 36976442 names the exhausted budget and fbrecord classifies it as `budget_exhausted`.
- ns 101 moves the boundary by only 14-21 um, but the minimum iota by +0.8% (both cases): ns 51 reads iota low, so
  the iota limit at ns 51 is conservative by about 0.8%; the objective moves 0.01-0.16%.
- Pending: qhvac VC 64 (40 rounds).

Tolerance and budget changes (not algorithmic): POLISH_ROUNDS raised per run (fbrecord env), POLISH_FTOL override
(6079d1e5), zero-drive Picard iterations 4 to 8, the frozen li383 test's cache clearing.

## Test status at the latest commits (GPU)

- `test_freeboundary_vc.py`, full run (night4, 821eab8d): 54 passed, 1 failed, 2 skipped. The failure was
  `test_plasma_map_derivatives_at_nonzero_offsets`, now **rewritten and passing** (dffc59aa, day1): tight re-solved FD,
  exact derivative within 1e-3, GN within its 1e-2 budget. The interior-state, bootstrap-vs-Picard and checkpoint
  tests all pass.
- `test_bootstrap.py`: 49 passed (Picard refines every iterate).
- `test_implicit_grad_fd.py`: all pass except the pre-existing typed-error test (below).
- `test_three_term_clearance.py` (CPU): 4 passed.

## Pre-existing failures (not caused by this work; identical at af30041d)

Under the GPU-default runtime on Perlmutter these tests fail with a CPU/GPU device-placement `ValueError` (CI's
CPU lanes and the two-device lane pass):
- `test_implicit_grad.py::test_three_surface_raw_kernel_matches_global_nonlinear_rows` (x2);
- `test_implicit_grad.py::test_raw_residual_jvp_is_exactly_nearest_neighbor` (x3);
- `test_optimize.py::test_equilibrium_from_x_is_the_state_the_objective_read` (x2);
- `test_optimize.py::test_certified_trial_guards_reject_stale_or_missing_memo`.
- `test_optimize.py::test_from_loss_honors_bound_scalar_method_literally`, `test_least_squares_implicit_jac_solver_block`,
  `test_equilibrium_exterior_field_sizes_its_source_grid` (the same device-placement `ValueError`) and
  `test_reverse_lane_factors_once_per_point` (NaN in the block-tridiagonal fallback): **identical at af30041d and
  54140c07** (day3/G3 baseline lane, `paper/runs/g3/tests-baseline.log`).
- `test_optimize.py::test_from_loss_honors_bound_scalar_method_literally`,
  `test_least_squares_implicit_jac_solver_block`, `test_equilibrium_exterior_field_sizes_its_source_grid` (same
  device-placement `ValueError`; day3 run of the whole file).

`test_optimize.py::test_reverse_lane_factors_once_per_point` returns a NaN gradient ([nan, nan] against
[2.609, 0.867]) identically at af30041d and at 54140c07 (day3, same GPU node, byte-identical output): pre-existing.

Day3 GPU suites (pin 5faa0684): `test_freeboundary_implicit.py` 35 passed, 1 skipped (53 min);
`test_implicit_multi_rhs.py` 20 passed (72 min); `test_optimize.py` 42 passed, 7 failed (all listed above).

`test_implicit_grad_fd.py::test_typed_error_through_pure_callback` fails identically at af30041d: solovev converges
in 3 iterations, so the expected error is never raised. These are environment and test-setup issues, tracked here.
