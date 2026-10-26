# Solve a free boundary from the three interface conditions

NESTOR makes the boundary a flux surface of the vacuum field and balances only
`|B|` across it, so a VMEC/NESTOR equilibrium may carry an edge sheet current
`K` (a jump in the direction of the tangential field). The three-term solve
instead finds the boundary on which all three interface conditions hold, with
the plasma's own field from virtual casing (Conlin et al. 2024,
arXiv:2412.05680): `B_out . n = 0`, `|B_out|^2 = |B_in|^2 + 2 mu0 p`, and
`n x (B_out - B_in) = 0`. Each trial boundary is a fixed-boundary equilibrium,
and a least-squares fit with exact implicit derivatives drives the three
residuals to zero ({mod}`vmex.core.freeboundary_three_terms`).

From a deck, with no Python:

```fortran
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  LTHREE_TERMS = T                  ! VMEX extension; VMEC2000 rejects unknown keys
  THREE_TERMS_LABELLING = 'gauge'   ! or 'none'
```

`vmex input.case` then writes the three-term free boundary's `wout_case.nc`.
From Python the keyword sits on the free-boundary ladder, and an explicit
argument wins over the deck (`three_terms=False` turns a deck's request off):

```python
import vmex as vj

result = vj.solve_free_boundary_multigrid(
    inp, external_field=coil_field,
    three_terms=vj.ThreeTermsConfig(labelling="gauge"),     # or "none"
    raise_on_max_iterations=False)
vj.write_wout("wout_case.nc", vj.wout_from_result(inp, result))
print(result.three_terms.boundary_residual)   # RMS of B.n, pressure jump and mu0 K, over |B|
print(result.three_terms.certified, result.three_terms.message)
```

The interface conditions do not fix the boundary's poloidal angle, so the
least squares is flat along relabellings of it. `labelling="gauge"` (the
default) holds VMEC's spectral condensation exactly on the boundary and
Newton-converges the fit on that gauge, so the boundary and its derivatives
are determined; `labelling="none"` is the plain least squares, whose labelling
is wherever the fit ends. `result.converged` means certified and every RMS
interface residual at or below `ThreeTermsConfig.max_residual` (1e-2, a guard
against wrong stationary points); `result.vacuum` is `None`.

The deck boundary is the initial guess, so a fixed-boundary design and the
coils fitted to it are a natural start; `restart_from` (a wout, or a NESTOR
`SolveResult`) starts from another equilibrium's boundary instead. The edge
pressure must vanish (a pressure jump needs a sheet current), the deck must be
stellarator-symmetric, and `virtual-casing-jax` must be installed
(`pip install 'vmex[freeb]'`) with `jax_enable_x64`. Virtual casing dominates
the cost and runs on the default JAX device, so use a GPU; expect several
times the cost of a NESTOR solve. `ThreeTermsConfig` holds the interface grid
(`nphi`, `ntheta`; use the fixed singular quadrature
`quadrature=(4 * nfp * nphi, 2 * ntheta)`), the boundary modes (`max_mode`),
the fit's tolerances and the gauge Newton's options.
{func}`vmex.core.freeboundary_three_terms.solve_free_boundary_three_terms` is the same solve, and its
`previous=` continues from an earlier result (after a coil change, say)
without compiling the model again:

```python
config = vj.ThreeTermsConfig(nphi=32, ntheta=32, quadrature=(4 * inp.nfp * 32, 64))
first = vj.solve_free_boundary_three_terms(inp, external_field=coil_field, config=config)
moved = vj.solve_free_boundary_three_terms(inp, external_field=new_coil_field, config=config,
                                          previous=first)
```

`examples/free_boundary_three_terms.py` solves a coil set's free boundary with
both labellings.

For single-stage optimization,
{class}`vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem` takes objective terms, constraints and
a differentiable `field_from_parameters(x)` (and optionally
`plasma_from_parameters(params, x)`): every trial is a gauge free boundary,
started from the accepted one, and the design gradients are the exact
implicit derivatives of that converged point. Objective terms are upstream's
`(function, target, weight)` tuples; each constraint keeps its bounds beside it:

```python
problem = vj.ThreeTermsFreeBoundaryProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0)], x0,
    constraints=[(opt.mean_iota, 0.40, 0.46), (opt.aspect_ratio, 5.5, 6.5)],
    field_from_parameters=coils_from_x,
    config=vj.ThreeTermsConfig(nphi=32, ntheta=32, quadrature=(4 * inp.nfp * 32, 64)))
result = opt.minimize(problem, method="SLSQP", constraints=[problem.constraints()])
```

Adding a target or a bound is one more tuple. `parameter_constraints` take
`function(state, runtime, x)` (a coil-plasma distance, say); `from_loss` takes a
general scalar loss and the constraint rows without their bounds.

`examples/optimization/single_stage_free_boundary_optimization_three_terms.py`
optimizes coil currents this way. For other uses of many fields and plasma
parameters, {class}`vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel` gives the pieces
directly: `solve_boundary` (cold or warm), `linearize` (the interface Jacobian
and the state responses to the boundary and any plasma parameters), and
`pullback` (reverse-mode gradients of other functions of the equilibrium),
with the coils an argument of the compiled code. To check a NESTOR result,
{func}`vmex.core.freeboundary_three_terms.boundary_residual` evaluates the three conditions on any
equilibrium: VMEC + NESTOR leaves a floor of a few 1e-4 even for an exact
external field, and virtual casing itself is accurate to about 2e-4 on a
48 x 48 grid, so a `sheet_current` well above 1e-3 marks a genuine edge sheet
current.

How close each method gets to a known answer at mpol = ntor = 4 to 13, and at
what cost, is recorded in {doc}`/explanation/validation`
("Three-term free boundary against VMEC + NESTOR and DESC").
