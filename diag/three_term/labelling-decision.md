# Boundary labelling: evidence for the decision (2026-10-22)

> **Status 2026-10-23: superseded in part.** Read with [perlmutter/RESULTS.md](perlmutter/RESULTS.md).
> - The recommendation below (C at about 1e-2) is **withdrawn**: on QH 2.5% C 1e-2 shifts the seed objective by +5%
>   against C 1e-3/no labelling (RESULTS section 2), and the earlier QI tables already showed B.n 2-6x. C 1e-3 and no
>   labelling agree to 0.07% at interface grid 64 on QH; the labelling choice is open (work package 3).
> - Item 10 below is wrong: the interior start-dependence test fails on the small deck (also on CPU, both solvax
>   versions); see RESULTS section 6.
> - Item 3 (stuck QH fits) is explained in part: the bootstrap current solved on trial-tolerance states (fixed in #122);
>   the rest is slow convergence along weak boundary directions (open).

The three-term fit leaves the boundary's poloidal labelling free: relabelling θ changes neither the surface nor B·n, pressure balance or K. Something must fix it, and how it is fixed decides start-independence, derivative quality, conditioning and accuracy.

## Candidates

| | rows | reference |
|---|---|---|
| A | tangential offset from the start boundary, weight w | the run's start, which goes stale |
| C | equal arc length per cut, plus an m=1 phase row | the current surface |
| D | VMEC's spectral condensation (Hirshman & Breslau), plus the m=1 phase row | the current surface |
| NL | none | — |

## 1. Fixed design: C's QI 2.5% step-90 design, four starts each

The starts are the deck, NESTOR's boundary, a 0.97 scaling and a 2% random perturbation. The current is held at the design's own (from its WOUT). R00 depends on the labelling, so a spread in R00 means the labelling is not pinned. The edge iota is a physical quantity.

| method | B.n | K | iota edge (spread) | R00 groups | condition |
|---|---|---|---|---|---|
| NL | 5.5-6.8e-4 | 7.6-9.0e-4 | -0.93636 .. -0.93679 | 1.011 and 1.021 | 575-816 |
| C 1e-3 | 5.5-6.8e-4 | 7.5-9.0e-4 | -0.93636 .. -0.93684 | 1.011 and 1.020 | 593-776 |
| D 1e-3 | 5.4-6.7e-4 | 7.1-9.0e-4 | -0.93631 .. -0.93669 | 1.011 and 1.021 | 585-783 |
| **C 1e-2** | 7.5-7.6e-4 | 9.1-9.4e-4 | -0.93717 .. -0.93737 | **one (1.0113-1.0116)** | 466-485 |
| D 1e-2 | 6.7-7.3e-4 | 9.9e-4-1.02e-3 | -0.93648 .. -0.93658 | 1.012 and 1.020 | 443-491 |
| **D 1e-1** | 1.03e-3 | 1.37-1.42e-3 | -0.93795 .. -0.93804 | **one (1.0107)** | **142-146** |
| C 1e-1 | running | | | | |
| A 1e-2 | running | | | | |
| deck (fixed boundary) | 6.0e-4 | 8.6e-4 | -0.93594 | 1.0195 | — |
| NESTOR + VMEX | 1.7e-3 | 2.1e-3 | -0.94538 | 1.0113 | — |

What the table shows:
- At w = 1e-3, NL, C and D are the same thing: the labelling is not pinned (two R00 groups from four starts).
- C pins the labelling at 1e-2, at +35% B.n. D needs 1e-1, at +90% B.n, but its fit is four times better conditioned.
- The edge iota shifts with the weight (about 2e-4 at C 1e-2, 1.5e-3 at D 1e-1): the weight is part of the answer, so the smallest weight that pins the labelling is best.

## 2. Why pinning matters: derivatives (QI 2.5%, C's step-90 design)

| labelling | finite differences of the loss along the current profile |
|---|---|
| C 1e-3, fits to ftol 1e-13 | erratic: 8.3e-5, 6.4e-6, 4.2e-5 (the fit wanders along the unpinned direction) |
| C 1e-2 | smooth, agree to 1e-4 across step sizes |
| C 1e-1 | smooth, agree to 1e-4 across step sizes |
| C 1e-1, trials at the deck's ftol (d60268d) | agrees with the Gauss-Newton gradient to 1.3e-5 |

A design gradient is only well defined when the labelling is pinned.

## 3. Campaign evidence (single stage, from step 0)

- **QA and QI vacuum:**
  - C converged on QI vacuum (SLSQP success at step 1938, QI 4.45e-3, 0 failed trials).
  - C had the best or near-best value on every vacuum case. Its point-spacing ratio stays 4-8; A's grows to 9-31.
  - NL drifted until VMEC broke on QI vacuum (ratio 34, step 53).
- **QA 2.5%:** C best (2.54e-4).
- **QI 2.5%:**
  - A was best until the runs were stopped, but C and NL leave their slow region.
  - C restarted from A's design reaches 1.17e-2, better than A's 1.26e-2.
- **QH 2.5%:** the method comparison is confounded, because the runs started from different coils (GPU coil fit). From one shared seed:
  - C reproduces its failure path: B.n 3.4e-3 to 1.2e-2 at step 1, and a spacing spike to 29.8 at step 6.
  - NL holds B.n near 5.5e-3 but stalls in QH.
  - A drifts to a spacing ratio of 98.
- **D:**
  - QI vacuum is rerunning after a crash (bug fixed).
  - QH 2.5%: 5.7e-2 at step 47, spacing ratio 6-7.

## 4. QH 2.5% step 1 (P2), at C's step-1 design

| labelling | GN step, trial fit | GN step, tight fit | tight-fit evaluations, exit | boundary moved | cost |
|---|---|---|---|---|---|
| C 1e-3 | 2.08 | 2.08 | 19, stalled | 2e-4 | 1.92e-4 to 1.91e-4 |
| NL | running | | | | |
| D 1e-2 | running | | | | |
| C 1e-1 | running | | | | |

The fit is stuck far from stationary: the Gauss-Newton step is long, yet Levenberg-Marquardt cannot reduce the cost. That points at the fit itself (ill-conditioning, condition number about 5e5 at QH 2.5%, and a nonconvex cost), not only at the labelling.

## Recommendation (pending the running items) [withdrawn 2026-10-23, see the status note at the top]

1. Choose equal arc (C) at about 1e-2:
   - the smallest tested weight that pins the labelling at the QI design;
   - smooth derivatives;
   - modest accuracy cost;
   - best or near-best on vacuum and QA 2.5% in the campaign.
2. Confirm it:
   - a decade sweep 3e-3 to 3e-2 at fixed designs (one parameter-selection rule);
   - the same table at a QH 2.5% design;
   - field lines and geometric surface distance between starts, rather than the R00 proxy.
3. Keep D at 1e-1 as the robust alternative, in case conditioning (P8 and QH) matters more than the 2x B.n.
4. The scalar chart (pinning by construction) remains the fallback if no penalty weight works on QH.

## Unsolved problems at the core of the free-boundary solve

1. **The labelling is not pinned at weak weights.** The parameterization depends on the start and derivatives are ill-defined. To close: the decision above.
2. **Fit convergence in soft directions (P10).**
   - Trial fits stop with a Gauss-Newton step of 0.13 (scaled) left. ftol 1e-10 still leaves 2-4e-3, after 40-47 evaluations.
   - The stopping rule (relative cost gain, stall exit) cannot see a soft direction.
   - To close: a step-length criterion (`boundary_xtol`) and its cost.
3. **Fits that get stuck** (QH 2.5%, C, step 1): Gauss-Newton step 2.08, yet LM stalls with the boundary unmoved. Ill-conditioning (condition 2e5-8e5 for every method) and a nonconvex cost. To close: diagnose (Jacobian accuracy there, curvature), then globalization or preconditioning.
4. **Conditioning of the boundary coordinates at QH 2.5%** (condition 2e5-8e5). There is no scaling or preconditioning beyond the ESS mode scales. It drives both the stuck fits and the cost.
5. **Cost (P8):** about 20 fresh boundary Jacobians per accepted step at QH 2.5%, from trial fits falling back to full trust regions. To close: a better predictor, rank-one updates, preconditioning.
6. **Trial interior tolerance.** d60268d makes values consistent at the cost of one tight solve per trial. The cheaper alternative, a tolerance-aware predictor, is not studied.
7. **Restart fidelity in the campaign driver.** The API restores exactly (P7), but the driver's `--restart` still re-anchors A and takes PHIEDGE from the WOUT.
8. **Reproducibility of starts.** The GPU coil fit is not deterministic; every comparison needs one saved seed.
9. **References (P9).** NESTOR at this design gives B.n 1.7e-3 and an edge iota 1% off. Field-line tracing (vacuum) is running on the converged C QI-vacuum design; finite-beta references (coil + plasma field) are still to do.
10. **Interior m=1 gauge.** ~~Start-independent at the small test deck (test passes).~~ [superseded 2026-10-23: the test fails; converged interior states from different starts differ by an m = 0 relabelling, RESULTS section 6] Not yet checked at production resolution.
11. **QI 2.5% slow region (P4).** A plateau, not a labelling defect; the KKT check at C's step-150 design is running.
