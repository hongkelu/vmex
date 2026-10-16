# Three-terms free boundary: what it does and where it stops

This page states what the three-terms free boundary does today
(`boundary_condition="three_terms"`, {doc}`/howto/free-boundary`) and where it
stops, with the measurement behind each statement. "LP QA 2.5 %" is the
Landreman-Paul QA design at 2.5 % beta with its fitted coils. Distances are
between last closed flux surfaces: point to curve, on 6 toroidal planes per
half field period.

## Labelling and start independence

The deck's poloidal (θ) labelling is the reference. The unknown is the
displacement of the deck's boundary along the deck equilibrium's radial lines.

- **Start independence.** The answer does not depend on where the solve
  starts. LP QA 2.5 % at 8x8, started from a boundary perturbed by 4 cm, ends
  0.00 mm from the solve started at the deck, with the same QS to 2e-8
  relative (3.5647e-5).
- **Labelling dependence.** The answer depends on the deck's labelling at the
  truncation level, and the dependence falls with resolution. With the same
  physics and coils, a deck relabelled in θ moves the answer by:

  | modes | RMS | max |
  |---|---|---|
  | 8x8 | 0.18 mm | 0.54 mm |
  | 12x12 | 0.10 mm | 0.24 mm |

  For comparison, the resolution change alone moves the answer by:

  | modes | RMS | max |
  |---|---|---|
  | 8 to 12 | 0.88 mm | 2.2 mm |
  | 12 to 14 | 0.34 mm | 0.89 mm |

## Resolution convergence of QS

- **Vacuum.** QS (the quasisymmetry residual) converges by M = 16. On a QA
  vacuum single-stage design with 30 coils:

  | M | 12 | 14 | 16 | 18 | 20 |
  |---|---|---|---|---|---|
  | QS | 4.41e-7 | 5.30e-7 | 5.88e-7 | 5.93e-7 | 5.90e-7 |

  From 16 to 18 it changes by 0.9 %.
- **Finite beta.** QS is not yet converged at M = 14. On LP QA 2.5 %, deck
  start, ns 51:

  | M | 8 | 10 | 12 | 14 |
  |---|---|---|---|---|
  | QS | 3.56e-5 | 1.11e-4 | 2.16e-4 | 2.35e-4 |

  From 12 to 14 it changes by 9.0 %. A re-evaluation at 16 modes and ns 101
  gives the same 9.1 %. Compare finite-beta designs at one stated resolution.

## Local force error

The interior is a VMEC fixed-boundary solve, and it carries VMEC's local force
error. This error comes from the method, not from the free boundary. The
measure used here is ⟨|J×B − ∇p|⟩ on a surface over ⟨|∇p|⟩ in the volume.

- **Same as VMEC2000.** VMEX matches VMEC2000 to under 1 % on every deck
  tested.
- **LP QA 2.5 %, mpol 12, ns 51:**

  | | s = 0.26 | s = 0.5 | s = 0.76 |
  |---|---|---|---|
  | VMEX | 0.031 | 0.031 | 0.046 |
  | VMEC2000 | 0.031 | 0.031 | 0.046 |
  | DESC (L 24, M 12) | 1.4e-4 | 1.2e-4 | 1.5e-4 |

- **More radial surfaces help.** At ns 201 VMEX gives 0.011 / 0.019 / 0.030 at
  the same three surfaces.
- **The first surfaces stay large.** On the first 2 or 3 surfaces off the axis
  the error is 2 to 5 at every resolution tested.

## QH fixed-boundary solves with mpol ≥ 13

The fixed-boundary descent on the QH vacuum deck stops converging once poloidal
modes up to m = 12 are present. The runs use ns 16 / 31 / 51, ftol 1e-17 on the
last stage, and a cap of 40000 iterations.

- **Converges:** mpol 12, in about 3860 iterations, at ntor 12 or 16.
- **Stalls at fsq 3e-12 to 2e-11:** mpol 13, 14 and 16, at any ntor.

What was checked:

- **More radial surfaces:** no fix. At mpol 13 the floor is 5e-12, 1.7e-11 and
  1.4e-12 at ns 51, 101 and 151.
- **The 16/13 rational surface:** not the cause, because mpol 13 has no m = 13
  mode and still stalls.
- **The deck's boundary:** not the cause, because it has only m ≤ 5.
- **The GMRES 2-D preconditioner:** lowers the floor to 2.3e-13 once, then the
  residual stops changing.
- **VMEC2000:** stalls the same way at mpol 16.

QH vacuum at 16 modes is therefore not available at ftol 1e-17. Such a state is
reported as not converged, with its fsq.

## Redl bootstrap current

- **Quasi-axisymmetric (QA) and quasi-helically symmetric (QH) fields only.**
  The Redl model needs the field's quasisymmetry helicity: `helicity_n` must be
  0 or ±1. Other fields, quasi-isodynamic designs included, are refused, because
  they used to take the QA mapping silently (#52).
- **Quasi-symmetric part of |B|.** In the solve, Redl uses the quasi-symmetric
  part of |B|: its Boozer harmonics in θ_B − N ζ_B (#53).
- **Why.** On the solver grid the full |B| of LP QA 2.5 % at 8x8 has 2 to 8
  near-equal local maxima per surface. The gap between them is 2.1e-2 to
  2.5e-4 of B_max − B_min, so B_max, and with it the trapped fraction, jumps as
  the boundary moves. The quasi-symmetric part has one maximum on every
  surface.
- **Effect on the results.** The trapped fraction changes by −0.4 % to −1.7 %,
  and the bootstrap current dI/ds by at most 1.0 % relative. The full-field
  Redl geometry stays available as the parity check against simsopt.

## Unconverged states raise

`opt.solve_equilibrium` raises `VmecConvergenceError` when the final stage of
its ladder stops at the iteration cap short of its tolerance (#50). It used to
return the unconverged state silently. `raise_on_max_iterations=False` is the
explicit opt-out for a caller that checks `result` itself.

## Trust region where Newton stalls

The boundary Newton solve continues in a trust region from the point where
plain Newton stalls (#51). This converges starts that used to stall. On QA in
vacuum, before this change, a start perturbed by 2 cm per metre stalled at
cost 2.4e-3 at M 12 and at 6.6e-3 at M 16.

With the trust region:

| start | M | steps | Jacobians | time |
|---|---|---|---|---|
| 5 mm (max) | 12 | 9 | 6 | 343 s |
| 2 cm per metre | 16 | 27 | 20 | 2069 s |

Far starts converge, but slowly. Starting from a deck close to the answer is
still the fast path.
