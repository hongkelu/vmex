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
`labelling` fixes it: `"flat_arc"` (the default: the arc-length distribution of
the fit without label rows, held along the fit's flat, tangential directions,
weight `1e-3`), `"equal_arc"` (equal arc length on every toroidal cut of the
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
scales them against the interface rows.

- `label_weight=0` gives a solve with no label rows at all. It works for a
  single solve: the fit then stops somewhere along the nearly flat directions
  the interface conditions leave, a few millimetres from another fit of the
  same case.
- For a single solve the weight changes nothing that matters. W7-X in its 50
  non-planar coils at 5.4 % beta, 10 x 10 modes: B.n/|B| rms 1.05e-3 and iota
  0.842 / 0.855 / 0.936 (axis / mid / edge) with no label rows, with `flat_arc`
  at weight `1e-1` and at `1e-3`.
- In the single-stage problem the weight matters. A trial there is its fit
  under the label rows, which are set again at every accepted fit (a trial that
  was re-fitted without them did not return the accepted fit at the accepted
  design: its Mercier row was off by 10 %, and the optimizer's line search
  stalled). The rows hold directions that are nearly flat, not free, so they
  must be weak: at `1e-1` B.n of a QA run rose to 2.4e-3, at the default `1e-3`
  it stays at 1-4e-4 (8 x 8 modes) and 1-5e-5 (12 x 12).
