# Work packages 0 and 1: status (2026-10-23, Perlmutter)

Brief: the 2026-10-23 work-package plan (PR #121 comments). Code: #122. Raw evidence: `raw/wp0/<case>/`
(manifest.json, config.json, status.json, records_report.txt) and `raw/wp1/`. The record files themselves
(`records/NNNN_<kind>.npz`, tens of MB each) stay on Perlmutter: `paper/runs/wp0/<case>/records/`.

## WP0: reproducible evidence

| item | status |
|---|---|
| 0.1 CI | #122: mypy annotation for `stats`; both PRs list the campaign driver as **untested** (no test runs it). Local: ruff (CI scope), mypy and docs prose pass. The CI reruns are pending on GitHub. |
| 0.2 Self-contained fixtures | `fixtures/<case>/profiles.json` (exact kinetic coefficients) and `fixture.json` (case, driver arguments, current representation, closure model/convention/helicity, hashes), from `scripts/fixture_manifest.py`. The frozen input.run/coils are unchanged (hashes verified). QH: n0 2.8080e19, T0 2294.6 eV, matching the logged provenance. |
| 0.3 Complete records | `ThreeTermFreeBoundaryModel.recorder`: every certificate round and the outcome, with coordinates, interior state, rows, full Jacobian, scales, normal operator, boundary arrays, row/column layout, resolution, tolerances, certificate, status/reason, and values (diagnostic). Saved atomically per run id by `scripts/fbrecord.py`. GPU test `test_certificate_records_are_complete_and_match_the_row_layout`. |
| 0.4 Effective configuration | `manifest.json`: git revision and dirty-diff hash, sha256 of probe/driver/library, versions, overrides, GPUs, Slurm job. `config.json`: effective resolution, labelling, closure identity, layout. The run stops as `configuration_mismatch` if `--modes`/`--ns`/`--vc-grid` did not take effect. Open: record the single GPU the process uses, not only the step's inventory. |
| 0.5 Diagnostic semantics | `model.row_blocks`/`column_blocks` (runtime layout); `scripts/records_report.py` reads the layout and the normal operator from each record (matched state); status is certified, rejected, budget_exhausted, configuration_mismatch or unexpected_error (exit 1). |
| 0.6 Decision documents | `labelling-decision.md`, `plan.md`, RESULTS.md: superseded conclusions marked (C 1e-2 recommendation withdrawn; interior item; near-null wording; no strict certification). |

**Deliverable:** `scripts/reproduce_fixture.sh CASE OUT` (fresh process; fixture checksums verified first).
Run at c155c27d on one 4-node allocation:

| fixture | strict certificate | records | last gain | note |
|---|---|---|---|---|
| qh25 | rejected | 7 | 4.0e-2 | identical, digit for digit, to the 10-22 C 1e-3 trajectory (cost 2.6513e-5 -> 4.0227e-7 -> 3.6729e-7 ..., loss 0.657708, 0.658729, ...) |
| qa25 | rejected | 7 | 9.1e-2 | round 0: 97% current-only (trial-state current); then reduced gain 0.17 -> 0.09 |
| qi25 | **certified** | 7 | <= 1e-4 | DKX current |
| qavac | rejected | 7 | 4.8e-2 | loss stable to 3e-5 per round |
| qhvac | rejected | 7 | 1.7e-3 | gate 1e-3; loss moves 1e-4 per round |
| qivac | **certified** | 7 | 9.3e-5 | |

The slow weak-direction convergence (WP2) affects QA and QH in vacuum too, with far smaller output drift than at
2.5%.

Tests at c155c27d (10 GPUs): **88 passed, 2 failed** (`test_interior_state_does_not_depend_on_its_start`,
`test_plasma_map_derivatives_at_nonzero_offsets`, both from WP1's interior freedom), 10 skipped.

## WP1: interior equilibrium and gauge (findings; the repair is open)

Decks: the small LP QA test deck (mpol 2, ntor 2, ns 7) and the QH 2.5% fixture's interior at production resolution
(mpol 8, ntor 8, ns 51, fixed boundary, prescribed current). Starts: cold; warm (from the deck with its boundary
scaled by 1 + 1e-3); perturbed (cold state times 1 + 1e-3 cos k). FTOL 1e-13 unless noted. Probes:
`scripts/interior/{interior_gauge,chord,gauge,refine_check,gauge_fix_check,canonical_gauge,rows_gauge}.py`.

1. **The frozen m = 1 complement is free in the forward solve.**
   - The host force has no component outside the implicit dof subspace (`|gc - P gc| = 0` at every start).
   - The frozen complement (zcs_int = (rss - zcs)/2 of VMEC's rotated basis) differs between starts: 1.9e-6 / 3.0e-5 (small), 1.2e-5 / 3.4e-5 (QH).
   - It is about 0.0124 at equilibrium (small deck). Setting it to 0 raises the residual from 1e-8 to 27, and refinement cannot recover: zero is not a gauge choice.
2. **Explicit relabelling path** f = eps s(1-s) sin(n Nfp phi):
   - The residual grows linearly in eps (fsq ~ eps^2).
   - Projection error: 2.4e-7 per 1e-4 of eps (small deck); 4e-12 (QH).
   - Force Jacobian along the generator vs random directions of the same norm: 0.10-0.12 vs 8-14 (small), **2.7-2.9 vs about 1.5e4 (QH)**.
   - So the relabelling is a strong near-null, not an exact null.
3. **Continuous families, not branches (chord test).**
   - Re-solves started at alpha = 0.25/0.5/0.75 between the cold and warm states converge to states at exactly alpha along the chord (the distance to cold is alpha times the gap; iota interpolates: QH 9.8e-5 = 0.25 x 3.9e-4).
   - Repeat cold solves are bitwise identical.
4. **Two different freedoms.**

   | | warm vs cold (QH) | perturbed vs cold (QH) |
   |---|---|---|
   | share of the difference that is m = 0 relabelling theta -> theta + f(s, phi) | 4-87% (by surface) | 99.8-100% |
   | any tangential relabelling | 86-97% | 100% |
   | normal (physical) RMS | up to 56 um | 0.4-1.7 um |
   | iota | 3.9e-4 | 1.7e-5 |

5. **The warm (physical) part is a weak physical direction resolved by a tighter solve.**
   - The cold/warm iota gap is 3.9e-4 / 3.7e-5 / 1.8e-5 at FTOL 1e-13 / 1e-15 / 1e-16.
   - The implicit layer's Newton refinement removes it; the refined states are 2.87e-4 apart at every FTOL, which is the gauge part.
   - The model's tight solve refines; trial solves (FTOL 1e-11) do not. This goes into WP2's trial error budget.
6. **Candidate gauge: lambda's m = 0 harmonics constant in s** (f_n(s) = lambda_0n(s) - lambda_0n(edge); f = 0 at the
   fixed boundary, the axis free). It is reached exactly by the nonlinear relabelling (condition 1e-16 after one
   iteration), and every start refines to the 1e-11 (QH) / 1e-15 (small) residual. After gauge + refine:

   | | QH cold-perturbed | QH cold-warm | small cold-perturbed | small cold-warm |
   |---|---|---|---|---|
   | coefficient gap, refined, no gauge | 1.07e-3 | 2.87e-4 | 5.5e-4 | 4.9e-5 |
   | coefficient gap, refined + gauge | **1.4e-5** | 2.4e-4 | 1.5e-4 | 2.7e-5 |
   | iota | 2.0e-6 | 2.5e-6 | 1.6e-6 | 2.3e-7 |
   | surface distance | 0.50 um | 0.66 um | 3.2 um | 2.7 um |

   The physical outputs become reproducible to iota 2e-6 and under 1 um. The remaining coefficient gap is a
   theta-dependent (m >= 1) relabelling.
7. **The interface rows (the three-term test's measure, small deck) improve only partly:**

   | treatment | rows: near vs cold | rows: perturbed vs cold |
   |---|---|---|
   | as is | 1.38e-3 | 6.83e-4 |
   | m = 0 gauge | 1.07e-3 | 6.35e-4 |
   | m = 1 force kept (solver and implicit) | 3.54e-4 | 2.34e-4 |
   | both | 2.50e-4 | 6.7e-5 |

   The test asks for 1e-6. The remaining freedom is m >= 1 relabelling, which spectral condensation pins only weakly.
   The rows are that sensitive because the boundary's B_in comes from the interior surfaces next to the edge.

**WP1 status:**
- **Diagnosed:**
  - a coordinate freedom (the m = 0 relabelling, plus an m >= 1 part weakly held by condensation);
  - a weak physical direction that tighter solves and Newton refinement resolve;
  - no branches.
- **Not yet implemented: the repair.** The candidate (m = 0 lambda gauge) fixes the physical reproducibility but not the
  interface rows at the 1e-6 level, and keeping the m = 1 force breaks the implicit layer's frozen-m = 1 assumption.
- **Next:**
  - a gauge for the full relabelling (m = 0 condition plus an m >= 1 condition consistent with condensation), applied in the forward solve, the refinement, tangents and adjoints (with the derivative of the gauge reference);
  - or a gauge-invariant evaluation of B_in at the boundary.

  Decide by the rows test and the fixed-branch derivative check.
