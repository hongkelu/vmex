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
`test_plasma_map_derivatives_at_nonzero_offsets`, both from WP1's interior freedom), 10 skipped. At af30041d (4 GPUs): **92 passed, the same 2 failed**, 10 skipped.

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

8. **The rows' start dependence does not converge with radial resolution** (small deck, FTOL 1e-13;
   `raw/wp1/rows_scan/`):

   | | ns 7 | ns 15 | ns 31 | ns 51 |
   |---|---|---|---|---|
   | mpol 2, near vs cold | 1.37e-3 | 1.64e-3 | 1.63e-3 | 1.68e-3 |
   | mpol 2, perturbed vs cold | 6.8e-4 | 5.8e-4 | 1.4e-15 | 2.1e-15 |
   | mpol 4, near vs cold | 1.41e-3 | 2.59e-3 | 2.88e-3 | 1.46e-3 |
   | mpol 4, perturbed vs cold | 4.7e-3 | 3.1e-3 | 1.7e-2 | 2.5e-2 |

   - **By block:** B.n, pressure and K all carry it, while the label rows are identical. The m = 0 gauge reduces it by
     at most 3.5x.
   - **The boundary field (B_in) is evaluated from half-mesh contravariant components** B^u, B^v, extrapolated to the
     edge in VMEC angles. In the continuum a relabelling with f = 0 at the boundary leaves the boundary field
     unchanged: X_s changes only by a tangent multiple, so sqrt(g) = X_s . (X_theta x X_zeta) is invariant, and
     lambda_zeta, lambda_theta at s = 1 are unchanged. The half-mesh Jacobian and the edge extrapolation are invariant
     only to O(h).
   - **So gauge-equivalent interiors whose surfaces agree to under 1 um (QH) give B_in rows that differ by 1e-3 to 1e-2.**

9. **An exact relabelling with the boundary fixed: which parts the discrete equilibrium undoes** (small deck,
   `scripts/interior/relabel_rows.py`, `raw/wp1/relabel/`). The relabelling is theta' = theta + eps s(1-s) sin(m theta - n Nfp phi).
   The relabelled state is evaluated as is, and again after the implicit Newton refinement (frozen complement held).
   Rows: maximum relative change, at eps = 1e-3.

   | (m, n) of f | mpol 2, ns 7: as is / refined | ns 15 | ns 31 | mpol 4, ns 15 |
   |---|---|---|---|---|
   | (0, 1) | 2.2e-3 / **6.1e-4** | 2.4e-3 / **6.3e-4** | 2.4e-3 / **6.4e-4** | 2.7e-4 / 4.8e-4 |
   | (1, 0) | 4.4e-3 / 1.4e-8 | 4.7e-3 / 3.5e-9 | 4.8e-3 / 1.0e-9 | 2.2e-3 / 2.8e-5 |
   | (1, 1) | 4.0e-3 / 2.7e-9 | 4.2e-3 / 7.0e-10 | 4.3e-3 / 1.9e-10 | 2.3e-3 / 4.2e-5 |
   | (2, 1) | 5.9e-3 / 1.7e-5 | 6.2e-3 / 1.7e-5 | 6.2e-3 / 1.7e-5 | 5.0e-2 / 2.2e-5 |

   - The relabelled state is not a discrete equilibrium: fsq grows as eps^2.
   - **The m >= 1 relabellings are undone by the refinement.** Their remaining change is the relabelling's
     projection error (2.5e-4 per 1e-3 of eps for (2, 1)) or falls with ns.
   - **The m = 0 relabelling survives the refinement at about 0.6x eps and does not fall with ns.** It is the
     discrete equilibrium's near-null family, and the B_in rows move along it.
   - **This corrects the reading of item 8.** The boundary-field evaluation is not gauge-sensitive at O(1) along
     m >= 1 relabellings; only the m = 0 relabelling is a free direction. So the repair is a gauge for the m = 0
     family that the rows actually see, plus the frozen m = 1 complement (item 10). Option (b) below is not
     supported by this test.
10. **The frozen m = 1 complement is the whole start dependence of the rows** (`scripts/interior/complement_swap.py`,
    `raw/wp1/complement/`). Each start's solved state is refined as is, and again with its complement replaced by
    the cold state's. Rows: maximum relative difference from cold (refined).

    | deck | start | complement gap | rows, as is | rows, cold complement |
    |---|---|---|---|---|
    | mpol 2, ns 7 | near | 3.9e-6 | 1.4e-3 | **1.8e-13** |
    | mpol 2, ns 7 | perturbed | 3.0e-5 | 6.8e-4 | **4.3e-13** |
    | mpol 2, ns 31 | near | 5.0e-6 | 1.6e-3 | **1.7e-12** |
    | mpol 4, ns 15 | near | 4.0e-6 | 2.6e-3 | **1.1e-11** |
    | mpol 4, ns 15 | perturbed | 4.7e-5 | 3.1e-3 | **8.5e-12** |
    | mpol 4, ns 31 | near | 4.8e-6 | 2.9e-3 | **2.3e-11** |
    | mpol 4, ns 31 | perturbed | 3.9e-5 | 1.7e-2 | **1.3e-11** |

    - With the same complement, every start refines to the same discrete equilibrium to round-off. There are no
      other free directions, no branches, and nothing else history-dependent.
    - **The cause:** VMEC's m = 1 rule (`residue.f90` `constrain_m1`: the constrained Z force is zeroed once
      fsqz < 1e-6) freezes the complement wherever the iteration has it when the gate first fires. That depends
      on the start.
    - **What the complement is:** for a near-circular cross-section, zcs_int = (rss - zcs)/2 is the theta-origin
      rotation theta -> theta + c(s) sin(n Nfp phi), i.e. the m = 0 relabelling of item 9. In the continuum it is a
      gauge. The discrete rows respond to it at about 400x the coefficient change.
11. **A prescribed complement profile is deterministic but moves the physics** (`complement_gauge.py`,
    `complement_physics.py`). The complement is set to its edge value times w(s), then refined:
    - w = rho (VMEC's m = 1 initial-guess scaling): the rows' spread across starts is 2e-13 to 2e-11;
    - w = s and w = rho^3 at mpol 4: the jump is too large and the refinement fails (spread 1e-2).
    - The cold complement is about rho for low n but up to 2-3x rho at n = 3-4.

    The rho-gauged root against the cold VMEC root:

    | deck | surfaces | iota | B_min | rows (max rel) |
    |---|---|---|---|---|
    | mpol 2, ns 7 | 0.23 mm | 1.4e-5 | 8.7e-4 T | 2.1e-2 |
    | mpol 2, ns 31 | 0.27 mm | 1.3e-5 | 9.3e-4 T | 2.1e-2 |
    | mpol 4, ns 15 | 0.21 mm | 2.9e-5 | 6.2e-5 T | 3.0e-2 |
    | mpol 4, ns 31 | 0.21 mm | 2.7e-5 | 5.9e-5 T | 3.0e-2 |

    The physical difference between starts is about 1 um, so a prescribed profile is not a pure gauge in the
    discrete problem. **Rejected (no accuracy trade-offs).**
12. **Option K: keep the complement's force and solve for it** (`complement_keep.py`, `raw/wp1/complement/keep/`).
    The constrained m = 1 Z force is never released in the implicit residual, and the refinement's dof projector is
    elementwise, so Newton solves for the complement with the evolved dofs. Every force equation then holds at the
    root.

    | deck | rows' spread across starts (K) | as is | refined residual (K) | K vs cold VMEC: surfaces | iota | rows |
    |---|---|---|---|---|---|---|
    | mpol 2, ns 7 | **2.2e-13** | 1.4e-3 | 1.4e-15 | 26 um | 6.3e-6 | 4.0e-3 |
    | mpol 2, ns 31 | **1.5e-12** | 1.6e-3 | 1.0e-13 | 71 um | 1.1e-5 | 7.6e-3 |
    | mpol 4, ns 15 | **1.2e-11** | 3.1e-3 | 4.4e-14 | 45 um | 5.1e-5 | 1.4e-2 |
    | mpol 4, ns 31 | **2.0e-11** | 1.7e-2 | 3.0e-13 | 42 um | 2.1e-5 | 1.1e-2 |

    - **Deterministic:** the root is a function of the parameters alone, from every start, at round-off.
    - **A complete root:** the frozen VMEC state leaves the complement's own force unbalanced; K satisfies it.
    - **It differs from the cold VMEC root** by tens of um and iota 1e-5 on these coarse decks.
    - **DESC reference** (`desc_reference.py`, `raw/wp1/complement/keep/desc_m4_ns31.json`): same boundary, fixed
      boundary, zero current, DESC L 16 / M 8 / N 8, against the mpol 4, ns 31 roots:

      | | iota max diff | surface max distance |
      |---|---|---|
      | frozen (VMEC) | 3.35e-3 | 2.18 mm |
      | solved (K) | 3.37e-3 | 2.15 mm |

      The mpol 4 truncation error dominates. The K-frozen difference (42 um) is 2% of it, so the choice costs
      nothing measurable in accuracy. K is preferred because it is deterministic and a complete root.
    - **Derivatives (WP5):** with K, the implicit derivative is an independent re-solve FD's (li383_low_res, mean
      iota, h 5e-4; `test_li383_mean_iota_matches_resolve_fd_with_the_m1_complement_solved`):

      | | d/dRBC(1,1) | d/dZBS(-1,1) |
      |---|---|---|
      | frozen: implicit vs re-solve gap | 1.0e-2 | 2.8e-2 (closed only by adding the m = 1 family's drift) |
      | **solved (K)** | **9.9e-7** | **2.9e-6** |

      The frozen test is kept with `M1_COMPLEMENT = "frozen"` (closures 2.3e-6, 1.6e-5).
    - **Implemented in #122:** `implicit.M1_COMPLEMENT = "solved"` (default; `"frozen"` restores VMEC's
      behaviour). It applies to 3D stellarator-symmetric runs; `lasym` is unchanged. The host solver is unchanged:
      the Newton refinement moves the complement from wherever the host left it.

**WP1 status:**
- **Diagnosed:**
  - a coordinate freedom (the m = 0 relabelling, plus an m >= 1 part weakly held by condensation);
  - a weak physical direction that tighter solves and Newton refinement resolve;
  - no branches.
- **Not yet implemented: the repair.** The candidate (m = 0 lambda gauge) fixes the physical reproducibility but not the
  interface rows at the 1e-6 level, and keeping the m = 1 force breaks the implicit layer's frozen-m = 1 assumption.
- **Repair options (decision pending):**
  - (a) a gauge for the full relabelling (m = 0, done as a probe, plus an m >= 1 condition consistent with
    condensation), through the forward solve, the refinement, tangents and adjoints;
  - (b) a gauge-invariant evaluation of B_in at the boundary (boundary-local geometry, the normal part of X_s, and
    lambda at s = 1).

  The scan favours (b): the rows' sensitivity sits in the boundary-field evaluation, which does not converge with ns,
  while the physical state is reproducible. (b) departs from the discretization the force balance uses, so it needs
  an independent reference (DESC, or field-line tracing in vacuum) before it can become the default.
