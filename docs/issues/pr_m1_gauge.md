## Problem: equilibria depend on where the solve started

**Who is affected:** anyone using implicit gradients, single-stage optimization or the three-term free boundary.

**The mechanism (VMEC2000 `residue.f90`):**
- VMEC zeroes the force on the constrained m=1 combination `zcs_int = (rss − zcs)/2` once fsqz < 1e-6.
- That combination is the antisymmetric `Z_sin(m=1, ±n)` pair; for `lasym`, also `zcc_int` and `Z_cos(m=1, n=0)`.
- Before that threshold its force is active, so the combination moves. After it, it stays wherever the solver's path left it.

**Effect on solves.** Two solves of the same boundary from different starts both converge (fsq ~1e-17) but to different states:
- 1e-4 to 1e-3 apart in vacuum;
- up to ~6 % apart in λ at finite β.

**Effect on implicit gradients.** The implicit linearization holds that combination fixed, so it is not the derivative of the map a re-solve sees. Relative to a central finite difference of cold re-solves, the errors were:
- 25 % on LP QA;
- 27 % on LP QA at β = 2.5 %;
- 41 % on li383.

**Symptoms in single-stage optimization:**
- restart jumps of −10 … +90 % in QS for unchanged coils;
- optimizer values that disagree with a re-solve by 5–20 %;
- boundary Gauss–Newton steps that stall, with gain ratio around 0.2.

## Fix: fix the gauge (`freeze_m1`, on by default for fixed-boundary solves)

- **The force rule.** The constrained m=1 force is zeroed from the first iteration instead of only after fsqz < 1e-6.
- **The pinned value.** The combination is held at `sqrt(s) * edge`, which is exactly the `profil3d.f` interior guess, so a cold start is already on it.
  - `residuals.pin_m1_gauge` applies the profile at every stage start: seeds, hot restarts, multigrid interpolation.
  - In `implicit._assemble` the profile follows the boundary parameters, so the implicit Jacobian is the derivative of the re-solve map.
- **Where it applies.** `solve`, `solve_multigrid`, `prepare_runtime` and `ImplicitConfig` all take `freeze_m1=True` by default. Every implicit-gradient, optimization and three-term path therefore gets it with no user action.
- **Opting out.** `freeze_m1=False` reproduces VMEC2000's rule bit for bit (for parity tests).
- **NESTOR free boundary keeps VMEC2000's rule.** Its two-start spread is 0.11–0.12 mm with either rule: 12×12 truncation of different labellings, far below its own error, since its surface sits 4–5 mm from the true one (B·n 1.2e-4). Forcing a canonical edge gauge there is not a pure relabelling: it moved the surface 0.26 mm and shifted the aspect ratio.

## Robustness: fixed-boundary interior solve, 6 modes, own multigrid ladder

Each deck is solved cold and from a seed perturbed along the gauge, with λ ×1.05 and R ×1.001. The gradient column compares the implicit gradient with a central FD of cold re-solves.

| Deck | Fix | Seed state difference | Gradient error | Iterations (cold / seeded) |
|---|---|---|---|---|
| LP QA vacuum | off | 5.5e-2 | 25 % | 1364 / 1157 |
| LP QA vacuum | **on** | **5.5e-13** | **1.6e-8** | 1378 / 1165 |
| LP QA β 2.5 % | off | 6.3e-2 | 27 % | 1337 / 1324 |
| LP QA β 2.5 % | **on** | **1.6e-13** | **2.9e-8** | 1319 / 1338 |
| LP QA β 0.5 % | on | 3.9e-13 | 7.1e-8 | 1390 / 818 |
| nfp4 QH, finite β | on | 3.8e-13 | 1.2e-9 | 2348 / 2348 |
| nfp4 QI, finite β | on | 2.9e-13 | 8.0e-9 | 4609 / 1888 |
| CTH-like | on | 4.6e-14 | 1.4e-8 | 475 / 392 |
| Up-down asymmetric tokamak (lasym) | on | 1.2e-12 | 1.9e-9 | 2956 / 1685 |
| li383 | off | 5.7e-2 | 41 % | 659 / 690 |

On the real P30A coils (12×12, NS 51, FTOL 1e-17), seeded vs cold three-term rows at one boundary agree:
- cost spread 2.7e-9 (was 67 %);
- state difference 5e-12 (was 1.3e-3).

## Three-term free boundary: second issue, in progress

**What changed.** With the interior fixed, the boundary trust-region solve now converges cleanly. First-order optimality reaches 2e-10 in 27–47 evaluations; before the fix it stalled.

**What is still wrong.** Three starts (LP deck, another plasma, random) still give three different minima: QS 5.52e-6, 5.95e-6 and 6.06e-6.

**Cause.** The labelling rows in `ThreeTermFreeBoundaryModel._interface_setup` tie the boundary's poloidal labelling to each seed's own surface (`gamma0`). Each start therefore minimizes a slightly different cost.

**Fix being validated: canonical labelling.** It uses VMEC's polar condition zcs_int(edge) = 0 per n. This condition is transversal to the relabelling orbit θ → θ + δ(ζ), so it fixes the labelling, not the shape.
- Two-start and random-start runs are going for vacuum and finite β.
- The temporary `VMEX_LABEL_MODE` switch will be removed before merge.

## Also in this PR

- Removed the `deterministic` option (three-term, driver, evaluator). It was a workaround; hot restarts are exact with the fixed gauge.
- Removed the `VMEX_EDGE_EXTRAP` and `VMEX_REFINE_TOL` diagnostics. Both were rejected hypotheses.
- Tests:
  - `test_li383_fixed_m1_gauge_makes_resolve_fd_exact`: implicit vs re-solve FD within 1e-5, plus seed independence.
  - The gap test for VMEC's rule is kept with `freeze_m1=False`.

## Before merge

- [ ] Three-term canonical labelling: 3+ starts converge to one equilibrium in vacuum and at finite β; then remove the env switch.
- [ ] Remaining robustness cases: li383 with the fix, QHS, DSHAPE, non-symmetric 3-D, nfp1 QI, and 12×12 / NS 99.
- [ ] Full regression suite. VMEC2000 bit-parity tests take an explicit `freeze_m1=False`.
- [ ] Update `docs/explanation/validation.md`.

Performance work (shared compile cache across nodes, dropping the per-column GMRES corrector, one factorization per accepted state) follows in a separate PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
