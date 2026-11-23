# Solve a free boundary with all three interface conditions

NESTOR satisfies `B . n = 0` and the jump of `|B|^2` across the boundary, but
not the direction of the field: a NESTOR free boundary may carry an edge sheet
current. `three_terms=` (or the deck's `LTHREE_TERMS = T`) solves instead for
the boundary on which `B . n = 0`, pressure balance and continuity of the
tangential field all hold, with the plasma's own field from virtual casing
(`pip install "vmex[freeb]"`). Every trial boundary is a fixed-boundary
equilibrium; a Gauss-Newton least squares with exact implicit Jacobians fits
the boundary Fourier coefficients, from the deck boundary and from this
ladder's NESTOR free boundary, keeping the lower cost:

```python
import vmex as vj

result = vj.solve_free_boundary_multigrid(
    inp, external_field=field,
    three_terms=vj.ThreeTermsConfig(quadrature=(4 * inp.nfp * 48, 96)))
res = result.three_terms.boundary_residual   # RMS B.n, pressure balance and sheet current, relative to |B|
```

The interface conditions leave the boundary's poloidal labelling free;
`labelling` fixes it: `"flat_arc"` (the default: the fit's own labelling, no
label rows), `"equal_arc"` (equal arc length on every toroidal cut of the
current surface, weight `1e-3`) or `"anchored"` (the tangential offset from the
initial boundary, weight `1e-2`). Use a fixed singular quadrature
`(4 nfp N, 2 N)` for an `N x N` interface grid. The edge pressure must vanish.
`vj.solve_free_boundary_three_terms` is one solve, with `previous=` to continue
after a coil change, and `vj.ThreeTermsFreeBoundaryProblem` is the
single-stage problem with its design gradients, for
`vmex.optimize.minimize(problem, method="SLSQP")`. Any function of the
equilibrium is a loss term or a constraint row (`quantities=`): at finite beta,
as in VMEC, the pressure and current are inputs, the current profile is a design
variable (`plasma_from_parameters=`) and its Redl self-consistency a constraint
row (`examples/optimization/single_stage_free_boundary_optimization_three_terms.py`,
QA at 0.5% beta).
`examples/free_boundary_three_terms.py` compares the labellings with NESTOR.

## The label rows and their weight

The label rows only choose the boundary's poloidal labelling; `label_weight`
scales them against the interface rows. `flat_arc` has none by default
(weight 0): the boundary is the interface conditions' own least squares.

- The interface conditions determine the boundary by themselves. On a QA state
  at 8 x 8 modes their boundary Jacobian has condition number 3.4e3, with no
  direction below 1e-4 of the largest, so the fit has one minimum.
- About 40 of its directions are weak (1e-2 to 3e-4 of the largest), and the
  Mercier rows move strongly along them. Label rows at `1e-3` stiffened exactly
  these: a single-stage fit crawled toward the free boundary over 20-150
  fresh-Jacobian rounds and stopped short of it, at a point that depended on
  where it started. That cost hours a step at 12 x 12 modes and made the
  optimizer stall near an optimum.
- Without them, on an X-point design from the same start, B.n fell from 5.1e-4
  to 2.7e-4 and the objective rose 36 %: the rows had held the boundary off
  the free boundary. Steps took one trial and about two minutes.
- For a single solve the weight changes little. W7-X in its 50 non-planar coils
  at 5.4 % beta, 10 x 10 modes: B.n/|B| rms 1.05e-3 and iota 0.842 / 0.855 /
  0.936 (axis / mid / edge) with no label rows, and with `flat_arc` at weight
  `1e-1` and at `1e-3`.
- `equal_arc` keeps its weight `1e-3`: its rows ask for an absolute target,
  equal arc length, and do not vanish at the fit.
