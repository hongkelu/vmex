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
    three_terms=vj.ThreeTermsConfig(labelling="equal_arc", quadrature=(4 * inp.nfp * 48, 96)))
res = result.three_terms.boundary_residual   # RMS B.n, pressure balance and sheet current, relative to |B|
```

The interface conditions leave the boundary's poloidal labelling free;
`labelling` fixes it: `"equal_arc"` (equal arc length on every toroidal cut of
the current surface, weight `1e-3`), `"anchored"` (the tangential offset from the
initial boundary, weight `1e-2`) or `"none"`. Use a fixed singular quadrature
`(4 nfp N, 2 N)` for an `N x N` interface grid. The edge pressure must vanish.
`vj.solve_free_boundary_three_terms` is one solve, with `previous=` to continue
after a coil change, and `vj.ThreeTermsFreeBoundaryProblem` is the
single-stage problem with exact design gradients, for
`vmex.optimize.minimize(problem, method="SLSQP")`. Any function of the
equilibrium is a loss term or a constraint row (`quantities=`): at finite beta,
as in VMEC, the pressure and current are inputs, the current profile is a design
variable (`plasma_from_parameters=`) and its Redl self-consistency a constraint
row (`examples/optimization/single_stage_free_boundary_optimization_three_terms.py`,
QA at 0.5% beta).
`examples/free_boundary_three_terms.py` compares the labellings with NESTOR.
