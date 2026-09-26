# Matched finite-beta single-stage cases with Redl bootstrap current

Prepared from fork main `ce245e7f`. The coil constraints and their geometry
routines are imported from `../coil-constraints-benchmarks`, not copied.
The existing 0.5% zero-current run and its source snapshot are separate.

## Physics

- Shared reference: initial **on-axis beta = 0.03**, measured with VMEX's
  `WOUT.betaxis`, not volume-average beta. Preparation calibrates the common
  fixed-boundary reference, including its converged bootstrap current, to a
  relative beta tolerance of 0.001. It records the actual calibrated value.
- Editable assumed profiles, with normalized toroidal flux `s`:
  `ne = 1e20 (1-s^5) m^-3`, `Te = Ti = T0 (1-s) eV`, `Zeff=1`.
  `T0` is determined during preparation. Pure singly charged ions give
  `ni=ne`, so `p=e ne (Te+Ti)` in Pa. Density and temperature shapes,
  amplitudes, pressure and PHIEDGE are then frozen. No beta penalty or beta
  constraint is added, and no pressure/temperature/flux variable is optimized.
- The same input, profiles, flux and fitted coils initialize both arms.
  The free-boundary initial equilibrium can therefore have slightly different
  beta from the fixed reference; it is measured and reported without retuning
  the profiles to disguise that difference. Both betaxis and betatotal are
  reported at every accepted state.
- No externally driven plasma current. The nonzero plasma current is the
  self-consistent Redl response. Coil currents remain fixed.
- The QA Redl approximation uses helicity N=0. It is a quasisymmetry model,
  not a general 3D kinetic calculation. See
  [Redl et al.](https://doi.org/10.1063/5.0012664) and the stellarator application
  by [Landreman, Buller and Drevlak](https://arxiv.org/abs/2205.02914).

## Constraints and matching

The current derivative uses 13 shifted-Chebyshev coefficients throughout
preparation, fixed/free forward solves, coupled Newton corrections, tangents,
adjoints, and NS201 verification. The native `chebyshev_ip` profile integrates
and evaluates that series without converting to powers of `s`. Saved VMEX
decks retain this representation. It is a VMEX extension; external VMEC2000
readers need an explicit profile conversion with an accuracy check.

Legacy prepared power-series inputs are converted once and the equilibrium
is recertified. `--allow-prepared-code-update` permits only the pinned
additive profile-kernel migration along with the audited finite-beta source
updates. It does not accept arbitrary core changes or inherit old derivative
qualification. Root and linear tolerances are unchanged.

Main supplies three order-16 coils, 256 field quadrature points, and hard SLSQP
inequalities: length <=5 m, peak curvature <=5 /m, mean squared curvature
<=5 /m^2, coil-coil distance >=0.15 m and moving coil-plasma clearance >=0.20 m.
Main's margins, curve regularity and nonintersection guards are retained.
Curvature/length/MSC use 1024 points; distance polygons use 256 points and the
moving plasma surface uses 61x64. Independent geometry verification uses 4096
coil points and 121x128 surface sampling, with main's continuous refinements.
These are filament constraints, not finite winding-pack certification.

Both arms use the same QA/aspect/iota objective and hard iota/radius bounds,
SLSQP, M8/N8/NS51, a 64x64 equilibrium grid, force tolerance 1e-15 and fixed
coil currents. AugLag is absent. The fixed arm varies boundary and coil shape;
the free arm varies coil shape while the boundary follows force balance.
The finite-beta iota floor is **0.28**, with main's additional 0.0005 optimizer
margin. It is set in `bootstrap_settings.py`; vacuum parameters stay unchanged.
The prepared plasma can be reused for this new optimization target without
changing its pressure, profiles, current closure or flux. The original and new
iota targets are recorded in the prepared lineage.
The fixed arm fits both the boundary's normal field and total-pressure balance.
It uses **live total field**, coils plus virtual-casing plasma field, including
its equilibrium and bootstrap dependence in the gradient. NESTOR remains
coupled to the free-boundary force root; neither interface penalty is added to
the free arm. The vacuum reference scripts and shared initial normal-field-only
coil fit are unchanged.

The fixed interface objective is versioned `normal-plus-pressure-v1`:
`normal_cost + pressure_balance_cost`. The added residual is
`d = (|B_out|^2 - |B_in|^2 - 2*mu0*p_edge) / <|B_in|^2 + 2*mu0*p_edge>_area`.
This case has zero edge pressure. The denominator uses the live target field,
not the fitted coil field, and its state and surface-weight derivatives are
retained. One live virtual-casing evaluation supplies both residuals. Pressure
balance uses `0.5*1000*<d^2>` plus a smooth peak-excess penalty of weight `200000`
above `0.008`, matching the form of the normal-field penalty. These starting
weights are configurable in `bootstrap_settings.py`; they are not a hard
optimizer feasibility guarantee. Independent endpoint verification requires
maximum absolute pressure residual <= `0.01` as well as the existing normal,
plasma and geometry checks. Both RMS and peak residuals and the two separate
loss contributions are recorded.

The old 500-step normal-only result is not a result of this revised objective.
Use a new output directory. Authenticated prepared inputs may be reused with
`--allow-prepared-code-update`; the saved contract and lineage identify the
objective change, and old derivative qualification is not inherited. Validate
the revised coupled gradient and a bounded pilot before a new long campaign.

## Current response and numerical method

The root contains active equilibrium coefficients and 13 Chebyshev current
coordinates. It enforces both `F_equilibrium=0` and `q-Redl(state,q)=0`.
The current reconstruction follows VMEX's smooth enclosed-current identity;
a projected closure check and a separate physical `<J.B>` mismatch on interior
surfaces are reported. Gates are relative closure <=1e-8 and interior mismatch
<=2%. Pressure consistency is enforced by construction.

Preparation starts at 20 eV and increases temperature by at most a factor of
two per calibration stage, carrying the preceding equilibrium and current
profile. The initial Picard seed uses ordinary force convergence at 1e-11. It is only
a starting state: the coupled Newton corrector must subsequently pass the
unchanged 1e-15 force and 1e-12 root certificates. The separate high-order
strong-force polishing API is not used by this discrete-root benchmark.

Both finite-beta entry points follow main's `build_problem -> run_optimizer ->
verify_endpoint` layout. Both use the same `optimizer_driver.run_optimizer`
and main's public `optimize.minimize(method="SLSQP")`, with identical options,
constraint construction, and acceptance policy. The two vacuum reference scripts
remain unchanged. Main's `FunctionProblem.with_acceptance` owns promotion and
SLSQP backtracking. Main's parameters, coil inequalities,
and refined sampling are imported, not copied.

The finite-beta physics adapter remains necessary: main's vacuum problem
factories differentiate equilibria with prescribed current. They cannot be
used unchanged for a self-consistent Redl current or the fixed-boundary live
virtual-casing objective. `physics.py` / `linear_root.py` supply that coupled
root and total derivative; the force, NESTOR, Redl, geometry and ordinary
forward solvers come from main. Newton root
polishing must reach 1e-12 before any derivative. Full implicit derivatives
include the current response, plus explicit coil and live virtual-casing terms.
Predictor and adjoint linear residuals must meet 1e-9. A reused LU
preconditions matrix-free solves; a failed residual check triggers one fresh
dense factorization. Both boundary formulations use `device_root.py`:
compiled device FGMRES and LU application, device assembly in batches of 32,
and adjoint RHS batches of two. Immutable LU factors live on the host between
solves; device copies are released after each solve. `DENSE_ASSEMBLY`
can select host assembly/factorization for a memory-constrained experiment;
`DEVICE_SOLVES=False` selects the previous host solver for comparisons.
The saved tape is built
with state and design as dynamic JIT arguments; this avoids recompiling the
Krylov actions at every changed root. Saved Krylov actions are independently
switchable for timing comparisons. Every solution is checked using the actual
current JVP/VJP. Changing either the state or design replaces the saved tape
while reusing its executable; the LU can remain useful as a preconditioner. An LU at exactly
the same point can serve another right-hand side without a Krylov solve.
The measured earlier run timings do not qualify this revised forward adapter.
A new production-resolution derivative check and bounded timing pilot are
needed before claiming performance or numerical qualification for this revision.

Every trial begins from the last accepted equilibrium/current state. Both
arms form a checked full-current tangent and choose a valid low-residual seed.
Free boundary also tests damped tangent seeds. Both arms correct equilibrium and
current together immediately, using an ordinary fixed-current VMEX/NESTOR
solve only after a hard coupled-corrector failure, and retain that fallback
only if it improves the full coupled residual. Slow, strongly damped Newton
progress instead requests a smaller continuation segment for either arm.

Shared continuation first tries the complete proposal, then permits
half or quarter segments and retains certified intermediate roots across retries.
It allows at most six attempts. A proposal requiring smaller segments returns
a recoverable rejection to SLSQP, which can reduce the coil move. Certified
intermediate roots never become optimizer acceptances. The normal 12-step
Newton budget can extend by at most four steps only while the last two
iterations show rapid convergence with modest damping; tolerances never change.
The device backend owns accepted and trial LU factors separately. A rejected
trial discards its recovery factors; an accepted trial may promote them.
Elective refresh uses main's accepted-step rolling-median cost policy, capped
by the remaining accepted-step budget, and checks gradient parity before
promotion. Compilation is excluded from its linear-solve cost samples.
Each speculative continuation segment permits one Newton dense rebuild.
If its preconditioner fails again, the segment is subdivided before another
large matrix is assembled. Initialization and strict tangent/adjoint recovery
retain their checked dense fallback. Two consecutive Newton corrections with
residual ratios above 0.8 and step fractions at most 0.125 request subdivision
early. A single damped correction does not trigger this rule.
Failed segments restore the preceding certified segment's factor state;
successful prefixes retain their recovered factors without promoting an
optimizer acceptance. Measured linear work includes failed segments.
The optional host comparison backend retains the earlier per-call time policy.
Both arms' Newton corrections use Krylov rtol=1e-6 and an actual relative
linear-defect gate of 1e-5, matching main's Newton forcing policy. The final
coupled root still must meet 1e-12 and the force/current certificates are
unchanged. Predictor and adjoint solves retain rtol=1e-10 and the 1e-9 actual
residual gate. Only intermediate fixed-boundary Newton corrections now use
the same inexact forcing policy as free boundary; final certificates are unchanged.
First-use JVP/VJP and device Krylov compilation are timed
separately, so a cold adjoint cannot trigger a rebuild merely for compiling.
Dense-build events separate column assembly from factorization time.

Free boundary disables axis reguessing; fixed boundary retains main's ordinary
internal axis policy but disables coarse-grid and Jacobian-recovery retries.
No cold input seed is substituted. Current is never frozen in the objective
derivative. Trials and intermediate roots are not promoted until SLSQP accepts
that design. Checkpoints, current coordinates, hashes and residuals are retained.
Accepted-state logs separate QA, aspect, iota and normal-field loss components;
the free-boundary normal-field loss is identically zero. No finite-difference
suite runs in production. These recovery changes apply to both finite-beta
solvers; production-resolution speedup must be measured separately.

`benchmarks/finite_beta_free_speedup.py` compares the deployed solver and this
candidate on the same authenticated step-15 to step-16 move, checks full-grid
current/force and gradient parity, independently reconverges finite-difference
endpoints, and runs a separate two-step pilot. It reads the original run and
source only; use a new output directory on a spare GPU under a finite external
timeout. It does not run NS201 verification or claim physical feasibility.

Independent endpoint verification interpolates the accepted state to NS201,
then reconverges equilibrium and current via bounded Picard iteration, the same force/current gates and refined
coil/total-field diagnostics. It avoids an impractical NS201 dense adjoint and
explicitly does **not** claim an NS201 coupled Newton or derivative certificate.
Optimizer status and physical feasibility are reported separately. Exit code 2
means an optimizer budget/failure status or unmet endpoint limits, not convergence.

## Commands

Run from the checkout root with the VMEX/ESSOS environment, including
virtual-casing-jax >=0.0.8. All output directories must be new. Dry-runs do not
solve equilibria or use a GPU:

```sh
python examples/finite-beta-bootstrap-benchmark/prepare_case.py --dry-run
python examples/finite-beta-bootstrap-benchmark/free_boundary_single_stage.py --dry-run
python examples/finite-beta-bootstrap-benchmark/fixed_boundary_single_stage.py --dry-run
```

First generate one shared, authenticated finite-beta reference and fit coils
with the actual finite-beta plasma field and hard coil constraints:

```sh
python examples/finite-beta-bootstrap-benchmark/prepare_case.py \
  --device gpu --output examples/finite-beta-bootstrap-benchmark/runs/prepared-001
```

Preparation fails explicitly if beta/current calibration, coupled-root
certification or the constrained coil fit fails. It does not claim the seed is
independently derivative-qualified. Both arms consume that same bundle:

```sh
python examples/finite-beta-bootstrap-benchmark/free_boundary_single_stage.py \
  --device gpu --prepared examples/finite-beta-bootstrap-benchmark/runs/prepared-001 \
  --accepted-steps 100 --output examples/finite-beta-bootstrap-benchmark/runs/free-001
python examples/finite-beta-bootstrap-benchmark/fixed_boundary_single_stage.py \
  --device gpu --prepared examples/finite-beta-bootstrap-benchmark/runs/prepared-001 \
  --accepted-steps 100 --output examples/finite-beta-bootstrap-benchmark/runs/fixed-001
```

Use a separate output for independent derivatives, before a production pilot:

```sh
python examples/finite-beta-bootstrap-benchmark/verify_gradients.py --arm free \
  --device gpu --prepared examples/finite-beta-bootstrap-benchmark/runs/prepared-001 \
  --output examples/finite-beta-bootstrap-benchmark/runs/free-gradient-001
# Repeat with --arm fixed and a different output directory.
```

This verifier independently reconverges each signed endpoint from the same
accepted root at two step sizes; it checks the scalar objective and physical
constraint rows. Current-shape resolution and Redl quadrature should also be
refined for scientific qualification. The case is programmed; no 3% prepared
reference, GPU pilot, production run or production-resolution gradient result
is implied by passing local tests.

Local validation (2026-09-24): 262 case/profile/manifest checks passed,
including manufactured coupled total derivatives against reconverged endpoints,
real tiny-grid fixed/free force/Redl/NESTOR directional derivatives, dense
fallback, pressure consistency, accepted-state preservation, and endpoint
verification failure handling. The new checks also exercise both actual vacuum
optimizer routines, import isolation, warm forward chart conversion, and
mandatory current correction after an ordinary solve. Free-only acceleration checks
cover cost-based refresh, bounded Newton extension, certified continuation
retries, and the actual 0.28 SLSQP bound. Ruff and both production
dry-run entry points passed. Both reference scripts were verified byte-for-byte
identical to fetched origin/main at ce245e7f.
These checks do not replace the production-resolution independent verifier.

```sh
python -m pytest -q tests/test_finite_beta_bootstrap_case.py \
  tests/test_test_manifest.py::test_collected_suite_has_exact_manifest_ownership
```

### Shared optimizer and trial recovery

Both arms use one SLSQP call, the same iteration budget and objective tolerance,
and the same scaled plasma and coil inequality construction. Both recover a
failed trial through certified continuation from the last accepted equilibrium:
initial fraction 1, minimum fraction 1/4, and at most six correction attempts.
Only the optimizer can promote a complete certified endpoint. Rejected trials
never replace the accepted equilibrium or current profile.

Local validation of the shared policy: 66 finite-beta regressions passed,
including identical accepted/rejected traces for the same manufactured problem
in both arms, followed by a passing manifest ownership check. Both production
dry runs passed. This is local regression evidence; the updated policy has not
yet been qualified in a production-resolution GPU run. Historical jobs retain
their original source snapshots and must not be relabeled as matched-policy runs.

The former fixed-only 1 mm boundary-displacement cap and fixed-only retry ladder
have been removed. A large trial is judged by nonlinear equilibrium recovery,
geometry, and root certification; a recoverable failure triggers the same SLSQP
backtracking in either arm. The fixed and free physical problems still differ
in their design variables, equilibrium equations, and the fixed arm's live
virtual-casing objective. A shared optimizer does not imply identical physical
trajectories or identical inner equilibrium-solver implementations.

Boundary coordinates use the same exponential spectral scaling as main, with
`BOUNDARY_STEP_M=0.01` and `BOUNDARY_SPECTRAL_ALPHA=1.2`. Each prescribed-boundary change is
extended through the accepted interior with regular radial powers. The driver
also forms a full equilibrium/current tangent prediction at the certified
accepted state, using the checked linear solver. It chooses the valid candidate
with the smallest coupled residual, then performs nonlinear Newton polishing.
The radial transport preserves the axis/current seed; the full prediction
includes their coupled response. Neither candidate is accepted without all
original force, geometry, bootstrap and linear-residual gates passing.

`optimizer_driver.py` uses main's public optimizer and
`FunctionProblem.with_acceptance` interface for both arms. Recoverable root
failures become `TrialRejected`; rejected objective/constraint probes do not
terminate the run or become accepted states. Acceptance occurs at the
line-search point's Jacobian request, rather than SLSQP's earlier callback.
Repeated failed probes reuse their failure only until the next accepted state.

For the previously prepared 3% reference, the explicit
`--allow-prepared-code-update` option permits the enumerated finite-beta
driver/initializer source migration, removal of the legacy boundary guard, and
replacement of `CONTINUATION_INITIAL_FREE` by shared `CONTINUATION_INITIAL`.
It also records the newly authenticated vacuum entry points; subsequent
changes to those reference files are rejected. Apart from the exact pinned
Chebyshev profile update described above, core kernels must match. Physical
settings, shared coil constraints, dependency versions and input hashes must match.
The explicit finite-beta iota target and enumerated recovery settings may change.
The historical preparation
manifest is never rewritten; `prepared_lineage.json` records the migration and
does not inherit previous derivative qualification. Run the separate verifier
again before claiming production derivative qualification.

The preceding recovery regression suite passed 22 checks on 2026-09-24, including actual
SLSQP backtracking after an invalid probe, transport invariants, and rejection
of physics/core changes during migration. The measured original failed proposal
is documented in the project-local first-proposal audit. Valid transported
geometry alone is not a converged equilibrium or a feasible optimized design.

### Building on main without changing the vacuum examples

The finite-beta entry points expose `parse_args`, `build_problem`,
`run_optimizer`, `verify_endpoint`, and `main`. Production has no
`--verify-gradients` option; use the separate verifier above. The finite-beta
endpoint verifier is retained because a vacuum endpoint solve cannot certify
bootstrap-current closure.

Reusing main's public optimizer does **not** reuse the vacuum examples'
prescribed-current adjoints. The full coupled implicit derivative and its
checked dense fallback remain in the finite-beta adapter. The device backend
uses main's accepted-step refresh estimator and the same device FGMRES/LU
execution pattern on the augmented equilibrium/current root.
No speedup or new production qualification is implied by this code refactor.
Existing remote runs keep their immutable earlier source snapshots.

### Fresh coil initial condition on the same finite-beta plasma

`fit_initial_coils.py` loads and re-certifies an authenticated 3% reference,
then generates native ESSOS circular coils. It preserves the original physical
coil currents, count, symmetry, plasma profiles and flux. Feasible circles at
minor radii 0.50, 0.60 and 0.65 m are compared on the same virtual-casing target.
The best starts one constrained order-16 shape fit, with at most 200 SLSQP
iterations total. Reaching this budget is not itself a failed fit. Only the
best sampled-feasible candidate is carried to the next stage. Promising
infeasible trials are backtracked along a segment from the best feasible coil,
and the returned candidate is explicitly checked against every inequality.
The original infeasible trial is never exported as a feasible seed; optimizer budget
completion is recorded separately from convergence.

Virtual casing is computed once here because this is a frozen-plasma coil
fit. The subsequent single-stage objective still uses live virtual casing.
Independent dense geometry and finer normal-field checks gate creation of
`prepared.json`; failed checks retain candidate coils and reports without
advertising a ready preparation bundle. No single-stage optimization is
implicitly launched.

```sh
python examples/finite-beta-bootstrap-benchmark/fit_initial_coils.py \
  --device gpu --prepared /path/to/existing/prepared \
  --allow-prepared-code-update --iterations 200 \
  --output /path/to/new/prepared
```

Use the new output as the common `--prepared` input of both finite-beta arms
only after its `coil_fit.json` reports `ready_for_single_stage: true`. This
isolates the coil initialization change; it does not establish that the old
coils caused the coupled linear-solver failure.
