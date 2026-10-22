# PR #121 working plan (from the 2026-10-21 review)

Source: `review-2026-10-21.md` (same folder). All code work stays on PR #121 (`tt1-robust-fb`); probes and harness on `exp-c-run`. Update the Status column as items land; mirror it in the PR description.

Review claims checked against `tt1-robust-fb` @ 0b7535a (all four confirmed):
- `_linearize` polishes and overwrites `x`/`values` after `fun(x)` returned (L1036-1040): value before and after a gradient request can differ.
- `boundary_max_residual` is checked on the trial (loose) state, before the tight re-solve (L966 vs L980).
- `solve_boundary(refresh_on_stall=True)` turns a fresh-Jacobian `"stalled"` into `converged=True` (L789-790).
- The fit's `fun` uses the trial (loose) interior tolerance, its `jac` the tight one (L754-780).
- Also: the phase row `Im c1/|c1|` is 0 for both signs of real c1 and undefined at c1 = 0 (L165-178).

## Stages

| # | Stage | Items | Status |
|---|---|---|---|
| 1 | Fixtures | QI 2.5% C step 90 (prod snapshot), QH 2.5% shared-seed step 1 (`qhseed`), late QI vacuum (qi-vac-C step 1938); manifest with code revs and hashes (`manifest.json`) | mostly done; add the late-vacuum and QH entries to the manifest; commit harness = `diag/` on exp-c-run |
| 1b | Fixed-model starts | fbbench rebuilds the model from `initial_boundary` (A's anchor changes). Add a mode: one model, different initial iterates | open |
| 2 | Evaluation contract | one `_finalize` per trial: fit → tight state → polish → tight state → certificate → physical-residual check → values. `_linearize` only adds gradients, never moves the boundary. Test: value identical before/after a gradient; residual check on the final state | implemented locally (`_finalize`, `_reject` reasons); tests running on lhk0-7 |
| 2b | Stall ≠ convergence | a stalled fit counts as converged only through the certificate (stationarity) at finalize | with stage 2 |
| 3 | P10 + QH diagnosis | tight replay (trial ftol = deck ftol) of QH step 1 and QI 90; at the QH stuck point: directional derivative check along the step and weak singular vectors, predicted vs actual reduction over step lengths, normal vs tangential share of the step, LM radius/damping log | open (gnstep mode exists; extend) |
| 3a | QH stall remedy | variable projection of the bootstrap knots in the LM steps (9d93346, unit test) | implemented; QH replay + tests running |
| 3b | P10 as target | `boundary_xtol` drives refinement (normal displacement δX_n RMS/max over minor radius), calibrated against tighter solves | open |
| 4 | Labelling decision | C 3e-3/1e-2/3e-2, D 1e-1, NL (lhk8-13 sweep, QI design 2); modal D (a_m = 1/(2m(m+1))) as a new scratch variant; same at QH step 1 and late vacuum; geometric surface distance between starts; iota midpoint shift vs start spread | running (sweep); modal D, QH and vacuum tables open |
| 5 | Structural fallback | constrained spectral gauge (KKT), then scalar chart | conditional on 4 |
| 5' | Reduced-space NL (new candidate, from the sweep) | No labelling rows (NL physics, the most accurate); the tangential near-null directions handled where they matter: minimum-norm steps from the accepted boundary, a geometric relabelling of the accepted surface between outer steps (same surface, new coefficients), and the design gradient on the reduced space (truncated pseudo-inverse of J_b). A hard D constraint is the infinite-weight limit of D, whose bias already grows with weight, so it is unlikely to beat this. | to test: truncated-SVD gradient vs differences at fixed start; relabel-step invariance of values |
| 6 | AD qualification | analytic exact-sensitivity tests (db/dc = 1.25 cases); matrix-free exact stationarity reference; every design family and constraint at start/middle/late, FD windows, JVP/VJP duality | partial (coils ~1%, PHIEDGE 0.7-2.6%, current 1.3e-5); rest open |
| 7 | Restart and regressions | clear `_LAST_REFINED` in the start test + fresh-process seeds incl. production finite beta; checkpoint identity (field/objective config hashes), full validation before replacing the continuation, abs+rel tolerance; driver wiring | design-map hash (field + plasma params at x), transactional restore, `_LAST_REFINED` isolation: implemented; driver wiring, fresh-process seeds open |
| 7b | Supported problem | zero edge pressure enforced in the optimization path after plasma-map updates | implemented (model init + every trial) + test |
| 7c | Phase row | phase = atan2(Im c1, Re c1): one branch; c1 = 0 -> nonfinite -> rejected | implemented + test |
| 8 | Performance | fewer fresh Jacobians per accepted step (~20 at QH), reuse, scaling | after 2-4 |
| 9 | Qualification | six-case campaign, independent starts, physical validation (vacuum: coil-field tracing; finite beta: independent coil+plasma field) | last |
| CI | Coverage gate | changed-line coverage 45/200; run the new tests in the required lane | open |

## Results log (newest first)

- 2026-10-22 late: **work paused at the user's request.** All diagnostic jobs and local waiters/monitor stopped. State: tt1-robust-fb-wip at 7d73da7 (PR #121 at 095edbe, not to be pushed). 7d73da7 tests: radius leak fixed; still failing: plasma-map derivative (gradient -1.4532 vs differences -1.4799, 1.8%; old-vs-new comparison script cases/pmap/pmap.py written, not run) and interior start-independence (real: near-flat interior direction at the small deck). Old-code abc18 campaign runs left running (qi-vac-D on lhk0-7; qi-beta-AfromC90/C/NL on lhk14-19 (qh-beta-D stopped by the user via another session: saturated at QH 8.92e-3, best Mercier-stable save 690, continues as coilforce/runs/qh-maxj); qi-beta-AfromA90/CfromA90, qh-beta-A on lhk8-13).

- 2026-10-22 night: **interior start-dependence is real (small test deck):** with `_LAST_REFINED` cleared, three solves at one fixed boundary (cold, warm from a nearby boundary, perturbed seed) all converge (forces 1e-10 at the deck ftol, and 1e-14 with ftol 1e-14) but differ by the same amounts in both cases: interface rows 1-2e-5, R/Z 3e-5, lambda up to 1% near the axis. The unrefined forward solves differ identically, so it is not the Newton refinement and not the tolerance: the discrete fixed-boundary problem has a near-flat direction at this resolution. `test_interior_state_does_not_depend_on_its_start` fails for this reason (also on 0b7535a, where only the cache hid it). Next: the same comparison at production resolution (QI deck, GPU).
- 2026-10-22 night: test fixes 7d73da7 on tt1-robust-fb-wip (each start of a trial from the accepted continuation; polish rounds aim at POLISH_FTOL, the gate only rejects). Tests running on lhk14-19.

- 2026-10-22 night: **no labelling, converged trial fits (fresh-Jacobian trust regions, ftol 1e-10), QI 2.5% design 1, PR head 095edbe:** the loss is smooth and well defined. Finite differences over h = 1e-3..1e-6 agree to 3-5 digits (current 2.5717e-5 +- 1e-8, coils -1.4807e-3 +- 1e-8); the line scan is smooth (boundary scatter 2e-5). **The GN gradient is off by 2.0% (current) and 0.55% (coils)** -- the Gauss-Newton curvature term (stage 6) is the likely remainder. 13 trials in 30 min (~140 s each, warm). So the earlier erratic differences came from loose fits, not from no labelling. C 1e-2 at production settings rerunning (its 1e-10 trial fits ran out of evaluations).
- 2026-10-22 night: QH replay with 9d93346 traced: the current re-solve takes the fit's cost from 2.0e-4 to 1.0e-5 (20x), the LM still ends "stalled"; the seed is refused with the same 9.44e-1 certificate message as before -- to check (the certificate may read a state other than the fit's).
- 2026-10-22 night: PR head 095edbe test file: 30 passed, **3 failed** (rejected-trial continuation radius 123 leaks; plasma-map derivative 1.8% off; interior start-independence 2e-5) -- to fix on tt1-robust-fb-wip.

- 2026-10-22 night: **no pushes to PR #121 any more (another session works on it); new commits go to `tt1-robust-fb-wip`.** Last PR push: 095edbe (polish rounds with fresh Jacobians until the certificate passes, POLISH_ROUNDS = 5).
- 2026-10-22 night: derivative checks at QI design 1 with the PR head: C 1e-2 seeds pass the certificate (checks running); **no labelling's seed fit stays non-stationary after 5 fixed-Jacobian polish rounds (GN gain 1.5e-2 -> 1.09e-2)**: no-labelling fits need fresh-Jacobian trust regions (50-70 Jacobians) to converge. Rerunning NL with trial fits as tight trust regions (TRIAL_JACOBIAN_FTOL=1e-10).

- 2026-10-22 evening: **labelling benchmarks at all three fixtures (one model per method, 4 starts, LCFS distance between starts):**

  Late QI vacuum (qi-vac-C step 1938; field lines traced in the coil field, 32 lines x 30 transits):

  | method | B.n | field-line drift (mm) | axis iota | LCFS spread (mm) |
  |---|---|---|---|---|
  | NL | 1.12-1.17e-4 | 1.08 | -0.5180..-0.5189 | 0.37 |
  | C 1e-2 | 4.6-4.7e-4 (4x) | 1.07-1.09 | -0.5089..-0.5101 | 0.29 |
  | D 1e-1 | 7.1e-4 (6x) | 1.85 | -0.5101 | 0.20 |
  | modal D 1 | 6.9e-4 (6x) | 1.69 | -0.5145 | 0.045 |
  | NESTOR | 9.9e-4 | 2.47 | -0.5101 | - |

  QI 2.5% design 2 (CfromA90 step 120): **tight NL (ftol 1e-10) from all four starts, NESTOR's included: B.n 6.23-6.24e-4, axis iota -0.8636 and edge iota -0.8608/-0.8609 identical to 1e-4**, LCFS 0.41 mm (loose NL: iota spread 2e-3, 0.99 mm). Modal D 1e-1: B.n 1.3e-3; modal D 1: 2.0e-3, axis iota -0.848..-0.853, spread 1.2 mm. C 3e-2: B.n 2.1e-3, axis iota -0.823..-0.827.

  QI 2.5% design 1 (C step 90): tight NL: three starts at B.n 5.03-5.06e-4, axis iota -0.8437..-0.8443, 0.21-0.27 mm apart; the NESTOR start converges to another stationary point (B.n 6.9e-4, axis iota -0.850, 2.3 mm away): a second branch. Tight NL fits cost 50-70 fresh Jacobians (25-30 min each).

  QH 2.5% step 1: unreliable until the stall is fixed (fits stop after 21-32 evaluations; NL from the random start lands at B.n 2.3e-2; spreads 2.5-10 mm for every method).

  **Reading:** converged NL is start-independent in its physics (design 2: identical to 1e-4 from four starts) and is the most accurate in every table; every penalty labelling biases B.n by 2-6x and the axis iota by up to 4.5%, and is no more accurate in field-line drift (C equals NL in vacuum; D and modal D are worse). NL's open issues are cost (tight fits), the coordinate drift over a campaign (spacing ratio, QI vacuum broke at step 53), derivatives along the flat directions, and branches (design 1's NESTOR start). This supports stage 5' (NL + reduced-space derivatives + geometric relabelling between steps).

- 2026-10-22 QH replay with PR #121 WIP (finalize + projection), C 1e-2, step-1 design: the problem refuses the seed: "boundary fit not stationary: a Gauss-Newton step with its own Jacobian gains 0.944 of the cost". **The old campaign accepted QH 2.5% boundary fits that a GN step would reduce by 94% (stall counted as converged; no certificate before db39cba).** The projection alone did not reach stationarity in the seed fit: tracing the LM steps (LM_TRACE) to see why.

- 2026-10-22 **Fixed-model starts, QI design 1 (one model, 4 initial boundaries, labelling-independent LCFS distance):** NL: B.n 5.5e-4 (NESTOR start 6.8e-4), surfaces 0.3 mm apart (1.1 mm from the NESTOR start), axis iota -0.8466..-0.8494. Modal D 1e-1: B.n 7.7-8.3e-4, 1.5 mm (not pinned). **Modal D 1: B.n 1.03e-3 (+87%), surfaces within 0.075 mm, axis iota -0.8562 (all starts).** So at the default fit tolerance NL does not reach one physical surface either: the flat directions are not only tangential, or the fits stop early. Testing tight NL fits (ftol 1e-10, 300 evaluations) at both QI designs: if the spread collapses, the flatness is a convergence problem (P10), and 5' (NL + reduced-space derivatives) becomes the lead candidate.

- 2026-10-22 **Weight sweep at the second QI 2.5% design (qi-beta-CfromA90 step 120, 4 starts each, rebuilt models):**

  | method | B.n | R00 (4 starts) | axis iota | edge iota |
  |---|---|---|---|---|
  | NL | 5.5-6.0e-4 | 1.0148-1.0159 | -0.863 | -0.861 |
  | C 3e-3 | 6.6e-4 (NESTOR start 9.5e-4) | 1.0145 x3, 1.0124 | -0.852..-0.854 | -0.8626 |
  | C 1e-2 | 1.19-1.27e-3 (2.1x NL) | 1.0086-1.0095 | -0.830..-0.832 | -0.8605..-0.8609 |
  | D 1e-1 | 2.0-2.1e-3 (3.5x NL) | 1.0076-1.0082 | -0.847..-0.852 | -0.861..-0.866 |
  | deck / NESTOR | 5.6e-4 / 2.8e-3 | 1.0153 / 1.0055 | -0.862 / -0.873 | -0.861 / -0.871 |

  At this design C 1e-2 doubles B.n and moves the axis iota by 3.7% from NL; D 1e-1 is worse. **The penalty that pins the labelling changes the physical answer by far more than the review's 10% budget, and the cost depends on the design (+35% at design 1, +110% at design 2).** This is the decision table's row "both smooth only after materially degrading accuracy" -> structural gauge (stage 5). C 3e-2 still running.
- 2026-10-22 QH-A step 400 (spacing ratio 98): C 1e-2 cannot build a free boundary from that design ("seed design has no three-term free boundary").

- 2026-10-22 **Projection test (QH step 1, C 1e-2):** the GN boundary step with the knots held: cost 7.96 at a = 1, all in the bootstrap rows; one GN correction of the 53 knots at the new boundary brings the bootstrap rows to 2e-7 (3e-12 after two). The knots are the stiff, nonlinear part; eliminating them (variable projection) removes the blow-up. At a = 0.5 with one knot correction the cost falls from 2.0e-4 to 9.7e-5 (interface 8.1e-5, net current 1.0e-5); a = 0.25: 1.2e-4. **Implemented in PR #121 WIP (9d93346): `_project` in the LM steps (bootstrap knots re-solved with the step's own Jacobian columns, kept only if lower).** Stale-column variant (the implemented one: the step's own Jacobian columns, no new tangent solves) matches the fresh-column result after two corrections at every a (a = 0.5: 1.4e-4 after one, 9.71e-5 after two; a = 0.1: 1.65e-4). QH replay with the new code queued.
- 2026-10-22 P2 cross-evaluation (QH step 1, every labelling): C 1e-3, NL, D 1e-2, C 1e-1 all stall with the same signature (GN step 2.1, 2.1, 2.6, 3.7; 18-19 evaluations; boundary moves <= 7e-4). **The stall is independent of the labelling.** C 1e-1 costs 7x (1.38e-3 vs 1.9e-4).

- 2026-10-22 Stage 3 contrast, QI 2.5% plateau design (C step 150, DKX current: no bootstrap knots), C 1e-2, all tight: the fit is genuinely stationary (GN step 4.4e-4 scaled, stationarity 1.7e-6, condition 9.5e4); Jacobian matches differences to 1e-6; the three weakest directions are tangential (normal share 1-2%). At C 1e-2 the labelling rows are half the cost here (label 1.36e-6, interface 1.23e-6). **QI fits are healthy; the QH stall is specific to the stiff bootstrap block.**

- 2026-10-22 **Stage 3 stall cause (QH 2.5% step 1, C 1e-2, all residuals tight):**
  - The Jacobian is accurate: central differences agree with J d to 5e-5..1e-3 along the GN step and the three weakest directions (interface rows to 1e-6..1e-5).
  - The three weakest directions are tangential (normal share 3-5%, normal motion <= 1.3e-3 a): labelling-type, as the review predicted; they are not what stops the fit.
  - Along the GN step (2.2 scaled, normal motion RMS 6% of a, max 24%), the interface rows behave as predicted down to a = 0.25 (at a = 0.5: 8.1e-5 actual vs 5.4e-5 predicted, from 1.9e-4). The **other rows (net current, bootstrap self-consistency at weight 100) explode**: predicted ~1e-7, actual 2.2 at a = 1, 0.10 at 0.5, 5.6e-3 at 0.25, 1.4e-4 at 0.1. Only a <= 0.03 gains, and then almost nothing: the LM radius collapses and the fit "stalls".
  - Cause: strong nonlinearity of the stiff closure rows (bootstrap block) under large boundary moves, not noise, not a bad Jacobian, not the labelling. Candidate remedy that leaves the converged problem unchanged: eliminate the current knots inside the boundary step (re-solve them to self-consistency at each trial boundary: variable projection). Testing next.

- 2026-10-22 Stage 3 (QH 2.5% step 1, C's design, C 1e-2): with every residual solved to the deck's tolerance the fit still stalls (19 evaluations, 3 Jacobians, boundary moves 1.2e-4 scaled, cost 2.006e-4 -> 1.996e-4). GN step 4.5 (scaled), condition 3.6e7 (boundary + bootstrap columns), singular values 8537, 1035, 429 ... 2.4e-4. **Loose interior solves are not the cause of the stall.** Directional/line diagnostics rerunning (probe bug fixed).
- 2026-10-22 Labelling benchmark, QI 2.5% C step 90 (fbbench, 4 starts): C 1e-1: B.n 2.45e-3 (4x NL), edge iota -0.94026 (3.6e-3 from NL), R00 1.00786 all starts (pinned). A 1e-2: B.n 5.8-6.7e-4, R00 1.0198 x3 and 1.0094 (NESTOR start): not pinned (each start rebuilds A's anchor: the review's fixed-model point).
- 2026-10-22 P4 KKT, QI 2.5% C step 150: KKT residual 0.78 with 19 near-active constraints, several violated (one 79% above its bound): the plateau is not a constrained stationary point; SLSQP is in an infeasible region.
- 2026-10-22 Campaign (old code, for reference): qi-beta-CfromA90 1.09e-2 at step 144 (A's own best 1.26e-2); qh-beta-D 1.41e-2 at 91; qh-beta-NL 9.15e-3 at 670; qi-vac-D 7.5e-3 at 330.

- 2026-10-21 P9 vacuum field lines, converged qi-vac-C design (coil field traced directly): drift at s = 1.0: C 0.82 mm, NL 1.07 mm, NESTOR 2.46 mm; iota range C 0.517-0.552, NL 0.519-0.552, NESTOR 0.510-0.547.
