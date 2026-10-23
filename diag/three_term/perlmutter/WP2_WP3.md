# Work packages 2 and 3: status (2026-10-23, Perlmutter)

## WP2: strict acceptance, conditional current solve, weak-direction convergence

Code (#122):
- **Strict acceptance:** `normal_tol` / `boundary_normal_tol` removed as acceptance conditions; `normal_step` stays a
  certificate diagnostic.
- **Conditional current solve:**
  - `ThreeTermFreeBoundaryModel.current_columns`: a fresh current block at a tight state.
  - `_levenberg_marquardt(current_columns=...)` certifies the finish's conditional stationarity with fresh columns,
    reruns the corrections with them if needed, and reports `projection_unconverged`. `certify`'s tight polish uses it.
  - A rank check of the current block (`CURRENT_BLOCK_RCOND`) at entry.
  - Tests: rotating current columns (stale: true conditional gain 1.2e-4; fresh: 3e-29), unconverged projection,
    rank-deficient block.
- **Options for the comparison** (defaults unchanged):
  - `STALL_RULE = "model"`: stalled only when the gain ratio is under 0.25.
  - `JACOBIAN_UPDATE = "broyden"`: secant updates after accepted steps; certificates always use fresh Jacobians.
    Test: 18 vs 134 evaluations from a stale Jacobian.

Bounded comparison on the QH 2.5% fixture (12 polish rounds, strict gate 1e-3; time from the first certificate
round; `paper/runs/wp2`):

| variant | C 1e-3 | NL |
|---|---|---|
| repaired default (gain stall rule, 20 evaluations per round) | 0.170 @4.7 min, 0.040 @12 min, 0.024 @20 min (12 rounds): rejected | 0.236 @4.7, 0.080 @12.7, 0.065 @23 min: rejected |
| model-aware stall rule, 60 evaluations per round | 0.169 @7.3, 0.082 @12.2, 0.033 @30 min | 0.222 @8.0, 0.115 @13.9, 0.088 @28 min |
| 10 x 10 boundary modes (before K) | 0.315 @6, 0.21 @22, 0.142 @38 min: rejected | 0.297 @6, 0.144 @20, 0.086 @35 min: rejected |
| **m = 1 complement solved (K, 0a59e653)** | 0.221 @4.7, 0.078 @9.8, 0.033 @16, **0.0155 @22 min**: rejected | 0.27 @4.8, 0.074 @10.5, 0.032 @15, **0.028 @19 min**: rejected |
| Broyden updates | 0.274 @4.6, 0.112 @7.9, 0.063 @14 min (stopped at 6 rounds) | 0.277 @4.7, 0.101 @8.2, 0.100 @15 min (stopped) |

- **The repaired default reproduces the 10-22 trajectory exactly.** The fresh-column check confirms the projection;
  the projection was already good there.
- **Solving the m = 1 complement (WP1 item 12) removes the NL plateau.** NL's stationarity now falls
  monotonically, where it bounced between 0.07 and 0.09 for 20 minutes before. C ends 12 rounds at 0.0155 against
  0.0235 before. Both are still above the 1e-3 gate; 40-round runs are in progress (`C-K40`, `NL-K40`).
  - Before K, each tight evaluation's history-dependent complement moved the rows by about 1e-3 relative. That
    is the same order as the per-round gain the GN model could no longer realise.
- **A finer boundary (10 x 10) converges more slowly per minute** than 8 x 8 (more coordinates, more tangential
  freedom; WP3).
- **Broyden updates are not faster per minute** (C 0.063 at 14 min against 0.030 for the default). The
  runs were stopped after 6 rounds each.
- **The model-aware rule is slower per minute:** a fresh Jacobian costs about 1.3 min per round, which is cheap
  compared with the extra evaluations.
- **The plateau is not stationarity:** the GN model keeps predicting a 2-7% gain while about 0.1% is realised per
  round, i.e. slow progress along a curved, ill-conditioned valley.

## WP3: coordinates (first evidence)

Offline chart test on the plateau record (QH 2.5%, C 1e-3, round 12; `scripts/chart_offline.py`). The chart is
X = X_ref + u n_RZ (unit normal in the R-Z plane at fixed phi), with 128 symmetric u modes mapped onto the 255
boundary coordinates by least squares on an oversampled grid:

| | full boundary coordinates (255) | chart, all 128 modes | chart, the 98 modes projected to within 30% | chart, the 33 modes projected to within 10% |
|---|---|---|---|---|
| condition (column-normalized, current eliminated) | 1.08e3 | 30 | 29 | 23 |
| remaining GN gain | 0.0235 | 7.9e-4 | 7.7e-4 | 4.2e-4 |

- **What keeps the QH fit from the strict certificate lies almost entirely in tangential (relabelling) directions.**
  In normal-displacement coordinates the same record is below the 1e-3 gate at about 30x better conditioning.
- **But a normal displacement is poorly representable in the truncated boundary basis:** the relative RMS projection
  error of u n is 0.34-0.41 for m <= 6 and 0.78 at m = 7. At 8 x 8 the representation cannot separate shape from
  labelling. That is why tangential steps still move the outputs (0.1-0.2% loss per round) and why a chart inside the
  same basis is not exact.
- **Next for WP3:** a chart whose surface is given to the equilibrium at a higher boundary resolution than mpol 8 (an
  oversampled boundary), or a measured truncation error budget. Validate by the output and derivative changes after
  recharting, as the brief requires.
