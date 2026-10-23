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
| A1 | `M1_COMPLEMENT` is a mutable module global: residual and projector read it at trace time, and configs are canonical (shared), so switching in one process reuses stale executables and solves | **implemented** (cbc956a7): `ImplicitConfig.m1_complement` in config identity; NESTOR pinned frozen; driver `--m1-complement`; GPU test pending (job 60145562) | CI: the frozen-variant li383 test passed on GPU and failed on CI (closure 0.21) until caches were cleared (5c200092, a test-setup change, not a fix) | make the policy an immutable `ImplicitConfig` field in cfg identity, residual/projector, caches, checkpoint, manifest; test K -> frozen -> K in one process against fresh-process references |
| A2 | The final interior root is not certified in the expanded (K) equations: the status lane uses the pre-refinement host residual, and a stalled refinement can be returned | **implemented**: the refined state's residual is measured independently in the config's equations; status 4 if above `refine_tol` or not finite; `refine_tol=inf` with solved complement refused; GPU test pending | `_host_solve_and_mask_status` docstring: "That residual is the host solver's own, from before the fixed-point refinement" | measure the final selected residual and complement residual; certified only if the declared root criterion passes |
| A3 | Raw tangent-response acceptance: preconditioned convergence OR `raw_ok`, so a preconditioned pass can override a failed raw residual | **implemented**: acceptance is the raw residual alone (finite, <= raw tolerance); `LinearResponseReport` gains `raw_residual_norm`, `raw_tolerance`, `preconditioned_converged`; the refined column's raw residual replaces the recorded one; tests for the three cases (GPU pending) | `_raw_certified_report`, `_implicit_evolved_tangent_multi_rhs` | govern acceptance by the raw residual (abs + rel); test the four pass/fail combinations, multi-RHS, nonfinite and budget cases; record raw residuals on the production probes |
| A4 | Same equations across trial, finalization, Picard and export: `self_consistent_bootstrap` and export re-solved with the host solver (frozen complement) while the model's tight states are K | **fixed for Picard and export** (demonstrated cause of B1): `optimize.refine_equilibrium` refines a host solve onto the selected root and certifies it; Picard (both lanes) and `equilibrium_from_x` use it with the model's policy; export raises instead of warning when it moves the state. **Open: trial evaluations** (`tight=False`) still use the unrefined host state, i.e. the frozen equations with a history-dependent complement | GPU test pending; trials: decide refine-or-budget |
| A5 | NESTOR path: `_edge_basis` and raw/preconditioned complement treatment versus the K projector | **present** (unreviewed) | `freeboundary_implicit.py` | make the NESTOR path explicitly frozen (legacy) until it is upgraded consistently; test basis completeness |
| A6 | Public API: symmetry and edge-pressure checks at every constructor; `start="best"` ranking; NESTOR seed failure must not erase another seed | **partly implemented**: `_best_three_term_fit` ranks successful fits before cost (test); an optional NESTOR seed's failure no longer aborts `start="best"`; the multigrid three-term result is refined onto the fit's policy and re-reported (raises if the host re-solve moves it). Symmetry/edge-pressure audit of all constructors still open | GPU test of the multigrid path pending |

## Package B: bootstrap current

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| B1 | `test_fixed_boundary_bootstrap_model_matches_picard`: 26/29 entries off, max 3.98% (CI at 5c200092 and 6079d1e5) | **cause demonstrated: different equations (A4)**. Same policy, fresh processes: frozen/frozen model vs Picard 2.9e-5, CURTOR 4e-6, Picard current's closure in the model evaluator 3.2e-5; K model vs frozen Picard 0.95%, closure 1.2% (`raw/wpB/b1_*.json`, job 60144384). The bootstrap current itself depends on the m = 1 treatment at about 1% on this coarse deck (mpol 3) | fix: Picard on the model's policy (A4); rerun the test |
| B2 | Finite-weight joint stationarity does not imply exact closure | open (by construction) | the certificate already has a separate closure gate (`CLOSURE_ATOL/RTOL`) | verify closure is reported separately in every status; test a case with stationarity but no closure |
| B3 | Model semantics: prescribed / inner bootstrap / outer design with a mismatch constraint; Redl only for QA/QH; DKX for QI | partly documented | memory rule: Redl only QA/QH; QI fixtures use DKX | document and test each mode |
| B4 | Conventions (G, sign, helicity, nfp, units) on converged fixed-boundary states | `G_CONVENTIONS` exist; no paired re-solved comparison | | run on fixed-boundary states now |
| B5 | Zero-drive Picard test: 4 iterations against tol 1e-8 (contracts about 1000-fold per step) | **fixed (test setup)**: 8 iterations | GPU pass (job 60144384, t1.log) | none |

## Package C: differentiation

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| C0 | Fixed-boundary implicit derivative vs re-solve FD | **fixed** with K: 9.9e-7 / 2.9e-6 (frozen: 1.0e-2 / 2.8e-2) | `test_li383_mean_iota_matches_resolve_fd_with_the_m1_complement_solved` | |
| C1 | Small-deck design gradient vs converged FD: 0.67%, uniform across the three coordinates (all act through one combination) | **unresolved**; hypothesis: the Gauss-Newton fit sensitivity drops sum r_i d2 r_i (yy and yd) | `raw/wp5/design_grad.json`: boundary_ftol 1e-3, 1e-6, 1e-9; FD settles at 1e-6 | small exact reference (C3); confirm or reject GN as the cause |
| C2 | The design gradient is Gauss-Newton (documented in `_least_squares_sensitivity`), exact only at zero residual | **present** (by design) | | exact KKT/stationarity derivative, or a qualified GN error budget per domain |
| C3 | The boundary Jacobian (raw rows) is exact on the small vacuum deck: h^2 to 4.5e-7 (mpol 2 and 4, 9 boundary coordinates, 3 directions); the `BETA` option was documented but not implemented | partial | `raw/wp5/jfd_m*.json` | fix the probe (fixture loading); production QH 2.5% raw and current-coupled Jacobians |
| C4 | Parameter ownership: `with_boundary(params_at(P0, d), y)` overwrite; plasma-coordinate detection probed only at x0 | **unreviewed** | | r = w x regressions; explicit design-family metadata |
| C5 | Production design gradients (`--check-gradient`) and boundary Jacobians (`--check-jacobian`, 0f296879, no fit needed) | gc-qivac at gate 1e-7 never certified (1.9e-7 after 30 rounds; QH vacuum floor about 3e-5): FD fits now at 1e-5 (gc-qivac5); Jacobian checks on qh25 (seed and late NL record), qa25, qi25 queued (job 60145562) | | collect |

## Package D: boundary coordinates and steps

| id | issue | status | evidence | next action |
|---|---|---|---|---|
| D0 | Slow polish (about 1% of cost per round) on QH 2.5% and the 2.5% fixtures; NL plateau | NL plateau **fixed** by K; slow descent **present** | WP2_WP3; the reduced linear model mispredicts one round's row change by 34-97%; Jacobian exact (C3) | D1 diagnosis of the weak directions at production |
| D1 | Rejected without new evidence: Broyden, GN extrapolation, 10 x 10, model-aware stall rule | **superseded** | WP2_WP3 tables | do not revisit |
| D2 | Normal chart: condition 30 vs 2.8e3, but representation error 0.36-0.79 | candidate, not qualified | `chart_offline.py` | D2 qualification only if D1 shows the weak directions are relabelling |

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
| qa25 | 6.2e-3 after 29 rounds (40 allowed) | night/qa25-K40 (running) |
| qh25, prescribed current (fixture deck and current, no `--bootstrap`) | **certified** at 8.9e-5, cost 1.405e-7, after 74 rounds (36 min, 038065f3) | night/QH-presc80 |
| qh25, bootstrap (Redl) | not certified: NL 46 min, cost 1.90e-7 and falling, stationarity 1.4e-2; C 47 min, 1.1e-2 | night/NL-K90, C-K90 (running) |

Tolerance and budget changes (not algorithmic): POLISH_ROUNDS raised per run (fbrecord env), POLISH_FTOL override
(6079d1e5), zero-drive Picard iterations 4 to 8, the frozen li383 test's cache clearing.
