# Run a free-boundary equilibrium

Set `LFREEB = T` and supply the external field — a MAKEGRID `mgrid` file
with `EXTCUR` currents, or ESSOS coils directly — and VMEC finds the plasma
boundary that balances against it: the last closed flux surface is an
output, not an input.

## From an mgrid file

```text
&INDATA
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  EXTCUR = 1.0e5  1.0e5  ...     ! one current per coil group
  NVACSKIP = 6                   ! full NESTOR solve cadence
  ...
```

```console
vmex input.case
```

The console shows the VMEC2000 free-boundary output (`In VACUUM` block,
`VACUUM PRESSURE TURNED ON` banner), and the wout gains the free-boundary
metadata (`nextcur`/`extcur`/`curlabel`/`mgrid_mode`) plus the NESTOR
potential and surface fields (`potsin`/`xmpot`/`xnpot`/`*_sur`). A missing
mgrid file falls back to a fixed-boundary solve with a warning — retained
VMEC2000 behavior, so check the banner if a "free-boundary" run looks
suspiciously fixed. The end-to-end example below reads the CTH-like
`mgrid_cth_like.nc`, which is not stored in git; from a source checkout,
fetch it first with `python tools/fetch_assets.py --bundle reference-nc`:

```{literalinclude} ../../examples/free_boundary_mgrid.py
:language: python
```

## From ESSOS coils (no mgrid file)

VMEX is coil-agnostic: the solver consumes only a magnetic field. Set
`MGRID_FILE = 'DIRECT_COILS'` and pass an ESSOS coils file
(requires `pip install essos`):

```console
vmex input.case --coils coils.json
```

The coils' Biot-Savart field is tabulated once into an in-memory
{class}`~vmex.core.mgrid.MgridField` via
{meth}`~vmex.core.mgrid.MgridField.from_coils` — no temporary mgrid file.
In Python the same call takes an `essos.coils.Coils` object directly
({doc}`use-essos-fields-and-coils`). The underlying
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` adapter also accepts
ESSOS' `B(points)` protocol, SIMSOPT's `set_points(points); B()` protocol, or
any plain Cartesian callable, and the tabulated field is reused across every
radial stage and hot restart.
`examples/free_boundary_essos_coils.py` runs a beta scan against the
Landreman-Paul precise-QA coil set this way:

```{literalinclude} ../../examples/free_boundary_essos_coils.py
:language: python
```

## Mgrid interpolation order

{class}`~vmex.core.mgrid.MgridField` interpolates the tabulated field with
one of two kernels, selected by `order`:

| `order` | Kernel | Smoothness | Use |
|---|---|---|---|
| `1` (default) | trilinear | C0 value, piecewise-constant gradient | VMEC2000 / MAKEGRID parity; reproduces reference wouts |
| `3` | tricubic Catmull-Rom | C1, third-order accurate | accuracy on a given table |

Measured on the Landreman-Paul precise-QA coils tabulated on a
181 x 201 x 64 grid, the error against direct Biot-Savart is rms 1.8e-5 /
max 1.1e-4 (relative) for trilinear and about 8e-8 for tricubic. On the
CTH-like deck one tricubic field evaluation costs 4.2x a trilinear one, and a
full free-boundary solve takes 12% longer (1.63 s to 1.83 s). The trilinear
kink matters on coarse tables: on a 1 cm table two converged tokamak
equilibria differed by 7 mm in axis position; the tricubic kernel or a finer
table removes the discrepancy.

The CLI and `solve_file` always use `order=1`. To solve a deck with the
tricubic kernel, build its field and pass it as `external_field`:

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.case")
field = vj.MgridField.from_input(inp, order=3)   # MGRID_FILE scaled by EXTCUR
result = vj.solve_free_boundary(inp, external_field=field)
```

{meth}`~vmex.core.mgrid.MgridField.from_file` and
{meth}`~vmex.core.mgrid.MgridField.from_mgrid_data` take the same `order`
keyword. {meth}`~vmex.core.mgrid.MgridField.from_coils` and
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` return an
`order=1` field; switch the kernel with
`dataclasses.replace(field, order=3)`.

## Interpolation order of the tabulated field

Every tabulated field ({meth}`~vmex.core.mgrid.MgridField.from_file`,
{meth}`~vmex.core.mgrid.MgridField.from_coils`,
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` and
{meth}`~vmex.core.mgrid.MgridField.from_input`) takes `order=1` (trilinear,
the default) or `order=3` (tricubic, C1). The table itself is the same.
`MgridField.from_input(inp, mgrid_path=None, *, order=1)` builds a deck's
field with the deck's `EXTCUR` (each coil group's table times its current, as
VMEC2000 does in every `mgrid_mode`), so it is the way
to get the tricubic version of the field a deck would use.

Trilinear stays the default because it is the VMEC2000 parity kernel that
every parity test and reference wout is pinned to. Tricubic is the better
choice for coarse tables, coil optimization, and orbit tracing that needs a
continuous grad B:

- Landreman-Paul QA coils, 181x201x64 table, inside the LCFS against
  Biot-Savart: trilinear rms 1.8e-5, max 1.1e-4; tricubic 8e-8.
- Same coils, 96x96x32 table, 64 points 2 cm outside a free-boundary LCFS:
  max |B| error 2.1e-3 (trilinear) vs 1.9e-4 (tricubic); max grad|B| error
  3.1e-2 vs 7.2e-3.
- CTH-like free-boundary deck (ftol 1e-12, CPU): `b_cyl` at 20k points
  1.46 ms vs 6.17 ms (4.2x); full solve 1.63 s vs 1.83 s (+12%), with
  volume and edge R00 agreeing to about 1e-5.

```python
field = vj.MgridField.from_input(inp, order=3)          # deck + mgrid file
coil_field = vj.MgridField.from_coils(coils, order=3)   # ESSOS coils
```

## Three-term free boundary (no sheet current)

NESTOR makes the boundary a flux surface of the vacuum field and balances only
`|B|` across it, so a VMEC/NESTOR equilibrium may carry an edge sheet current
`K` (a jump in the direction of the tangential field). The three-term solve
instead finds the boundary on which all three interface conditions hold, with
the plasma's own field from virtual casing (Conlin et al. 2024,
arXiv:2412.05680): `B_out . n = 0`, `|B_out|^2 = |B_in|^2 + 2 mu0 p`, and
`n x (B_out - B_in) = 0`. Each trial boundary is a fixed-boundary equilibrium,
and a least-squares fit with exact implicit derivatives drives the three
residuals to zero ({mod}`vmex.core.freeboundary_three_term`).

From a deck, with no Python:

```fortran
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  LTHREE_TERM = T                  ! VMEX extension; VMEC2000 rejects unknown keys
  THREE_TERM_LABELLING = 'gauge'   ! or 'none'
```

`vmex input.case` then writes the three-term free boundary's `wout_case.nc`.
From Python the keyword sits on the free-boundary ladder, and an explicit
argument wins over the deck (`three_term=False` turns a deck's request off):

```python
import vmex as vj

result = vj.solve_free_boundary_multigrid(
    inp, external_field=coil_field,
    three_term=vj.ThreeTermConfig(labelling="gauge"),     # or "none"
    raise_on_max_iterations=False)
vj.write_wout("wout_case.nc", vj.wout_from_result(inp, result))
print(result.three_term.boundary_residual)   # RMS of B.n, pressure jump and mu0 K, over |B|
print(result.three_term.certified, result.three_term.message)
```

The interface conditions do not fix the boundary's poloidal angle, so the
least squares is flat along relabellings of it. `labelling="gauge"` (the
default) holds VMEC's spectral condensation exactly on the boundary and
Newton-converges the fit on that gauge, so the boundary and its derivatives
are determined; `labelling="none"` is the plain least squares, whose labelling
is wherever the fit ends. `result.converged` means certified and every RMS
interface residual at or below `ThreeTermConfig.max_residual` (1e-2, a guard
against wrong stationary points); `result.vacuum` is `None`.

The deck boundary is the initial guess, so a fixed-boundary design and the
coils fitted to it are a natural start; `restart_from` (a wout, or a NESTOR
`SolveResult`) starts from another equilibrium's boundary instead. The edge
pressure must vanish (a pressure jump needs a sheet current), the deck must be
stellarator-symmetric, and `virtual-casing-jax` must be installed
(`pip install 'vmex[freeb]'`) with `jax_enable_x64`. Virtual casing dominates
the cost and runs on the default JAX device, so use a GPU; expect several
times the cost of a NESTOR solve. `ThreeTermConfig` holds the interface grid
(`nphi`, `ntheta`; use the fixed singular quadrature
`quadrature=(4 * nfp * nphi, 2 * ntheta)`), the boundary modes (`max_mode`),
the fit's tolerances and the gauge Newton's options.
{func}`vmex.solve_free_boundary_three_term` is the same solve, and its
`previous=` continues from an earlier result (after a coil change, say)
without compiling the model again:

```python
config = vj.ThreeTermConfig(nphi=32, ntheta=32, quadrature=(4 * inp.nfp * 32, 64))
first = vj.solve_free_boundary_three_term(inp, external_field=coil_field, config=config)
moved = vj.solve_free_boundary_three_term(inp, external_field=new_coil_field, config=config,
                                          previous=first)
```

`examples/free_boundary_three_term.py` solves a coil set's free boundary with
both labellings.

For single-stage optimization,
{class}`vmex.ThreeTermFreeBoundaryProblem` takes a loss, constraint rows and
a differentiable `field_from_parameters(x)` (and optionally
`plasma_from_parameters(params, x)`): every trial is a gauge free boundary,
started from the accepted one, and the design gradients are the exact
implicit derivatives of that converged point:

```python
problem = vj.ThreeTermFreeBoundaryProblem.from_loss(
    inp, loss, x0, field_from_parameters=coils_from_x, quantities=(opt.mean_iota,),
    config=vj.ThreeTermConfig(nphi=32, ntheta=32, quadrature=(4 * inp.nfp * 32, 64)))
result = opt.minimize(problem, method="SLSQP", constraints=[problem.nonlinear_constraint(lower, upper)])
```

`examples/optimization/single_stage_free_boundary_optimization_three_term.py`
optimizes coil currents this way. For other uses of many fields and plasma
parameters, {class}`vmex.ThreeTermFreeBoundaryModel` gives the pieces
directly: `solve_boundary` (cold or warm), `linearize` (the interface Jacobian
and the state responses to the boundary and any plasma parameters), and
`pullback` (reverse-mode gradients of other functions of the equilibrium),
with the coils an argument of the compiled code. To check a NESTOR result,
{func}`vmex.boundary_residual` evaluates the three conditions on any
equilibrium: VMEC + NESTOR leaves a floor of a few 1e-4 even for an exact
external field, and virtual casing itself is accurate to about 2e-4 on a
48 x 48 grid, so a `sheet_current` well above 1e-3 marks a genuine edge sheet
current.

How close each method gets to a known answer at mpol = ntor = 4 to 13, and at
what cost, is recorded in {doc}`/explanation/validation`
("Three-term free boundary against VMEC + NESTOR and DESC").

## Key knobs

- `EXTCUR` — coil-group currents scaling the mgrid field.
- `NVACSKIP` — iterations between full NESTOR solves; between them, cheap
  incremental updates reuse the factored potential matrix, and the cadence
  adapts toward convergence ({doc}`/explanation/nestor-vacuum`).
- `--jacobian-retries` — free-boundary recovery after the 75-reset condition
  rebuilds the axis filament and NESTOR structures before continuing
  ({doc}`troubleshoot`).

## Convergence differences from fixed boundary

The vacuum solve activates only once `fsqr + fsqz <= 1e-3`, so early
iterations run effectively fixed-boundary; expect the residual trace to
change character at the `VACUUM PRESSURE TURNED ON` banner. Free-boundary
ladders carry the active-vacuum state and adaptive `NVACSKIP` across
`NS_ARRAY` stages ({doc}`/explanation/iteration`). On GPUs, the dense NESTOR
factor runs on CPU by design ({doc}`run-on-gpu`).

## Differentiability scope

Coil/`extcur` gradients on a specified boundary use the virtual-casing
residual. The coupled NESTOR fixed point is differentiated by
{func}`vmex.core.freeboundary_implicit.solve_free_boundary_implicit`, which
reverse-differentiates the reconverged plasma--vacuum root against plasma
profiles and direct coil variables. This path is experimental; with the
default `device="auto"` it runs on the CPU even on a GPU host.
Its coil examples need ESSOS (`pip install "vmex[coils]"`). Scope is in
{doc}`/reference/capabilities` and the mechanism in
{doc}`/explanation/nestor-vacuum`.
