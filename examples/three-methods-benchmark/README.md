# Three-methods benchmark

Three optimization workflows share the NFP=2 vacuum rotating ellipse in
[`input.rotating_ellipse`](input.rotating_ellipse): prescribed-boundary QA,
fixed-boundary single-stage plasma/coil optimization, and free-boundary
single-stage optimization with the coils determining the plasma boundary.

| Workflow | Entry point | Optimizer |
|---|---|---|
| Prescribed-boundary QA | `qa_optimization.py` | Staged least squares |
| Fixed-boundary scalar single stage | `single_stage_optimization_scalar.py` | SLSQP with `--constrained`; BFGS otherwise |
| Free-boundary scalar single stage | `free_boundary_single_stage_optimization_scalar.py` | SLSQP |
| Free-boundary bounded single stage | `free_boundary_single_stage_optimization.py` | L-BFGS-B |

The methods share an input, but the default scripts are not an automatically
matched performance comparison. Match resolution, coils, objective terms,
constraints and budgets explicitly before comparing results. These workflows
are maintained together on `main`; a separate comparison branch is unnecessary.
The superseded custom AugLag drivers remain available in Git history.

## Environment and inputs

Use this checkout's VMEX source with the dependencies in the root
`pyproject.toml`, including the optional coil dependencies (`vmex[coils]`).
Run commands from this directory with that checkout importable. The input
defines zero pressure/current, major radius 1 m, area-equivalent aspect 5,
and PHIEDGE 0.1255044894897643 Wb. Its default resolution is M3/N3/NS31
with a 48x40 angular grid; the free driver supports explicit overrides.

`coils.initial.scalar.json` is a saved initial coil fit, with its generation
metadata beside it. The separately included
`coils_single_stage_scalar_fitted_1789769333906005000.json` is the common
stage-two fit available through `--coils` during qualification. Currents remain fixed.
Neither file is a final plasma/coil optimization checkpoint.

The free-boundary initialization first solves the prescribed ellipse, then
converges the equilibrium supported by the fitted coils. That second state
is optimization step 0, so its boundary and QA can differ from the prescribed
ellipse. Every subsequent candidate and retry starts from the last accepted
equilibrium.

## Two-step SLSQP execution check

Run the two SLSQP entry points with a small optimization budget and ordinary final
verification and plots. Use new output directories for each invocation:

```sh
python -B single_stage_optimization_scalar.py --device gpu --constrained \
  --coils coils_single_stage_scalar_fitted_1789769333906005000.json \
  --ftol 1e-11 --maxiter 2 --output runs/fixed-two-step
python -B free_boundary_single_stage_optimization_scalar.py --device gpu \
  --input input.rotating_ellipse \
  --coils coils_single_stage_scalar_fitted_1789769333906005000.json \
  --accepted-steps 2 --output runs/free-slsqp-two-step
```

The explicit free-boundary input preserves M3/N3/NS31 and the 48x40 grid.
Fixed-boundary SciPy's `--maxiter` limits optimizer iterations. Check the saved
histories for the actual accepted steps. Free-boundary drivers return nonzero at a step-budget
stop even when verification and post-processing complete; inspect
`optimization_summary.json` and any `failure.json` to distinguish that stop
from an execution failure. Two steps do not establish convergence or derivative
qualification.

## Fixed-boundary reference

For a fixed-boundary comparison using the saved common coils:

```sh
python -B single_stage_optimization_scalar.py --constrained --device gpu \
  --coils coils_single_stage_scalar_fitted_1789769333906005000.json \
  --maxiter 100 --output runs/fixed-scalar
```

## Free-boundary production and separate derivative tests

`free_boundary_single_stage_optimization_scalar.py` (formerly
`free_boundary_single_stage_fast.py`) follows the scalar reference directly:
editable parameters, weighted plasma/coil objectives, a public VMEX problem,
`opt.minimize`, and endpoint reporting all appear in the example. It imports
`vmex` and `vmex.optimize`, with ESSOS for coils and surfaces. It does not
import private VMEX modules or a local driver/setup helper.

`opt.FreeBoundaryProblem.from_loss` owns equilibrium correction and the total
scalar gradient, including the moving surface in coil-surface clearance.
`accept_x` promotes only optimizer-accepted states. The example's importable
`build_problem` defines one initialization/objective path for production and
qualification. The superseded private scalar driver and its unused gradient
and resume helpers have been removed; they remain available in Git history.
The fixed-boundary reference still uses `_scalar_constraints.py` and
`_scalar_diagnostics.py`. Saved run outputs are unchanged.

Read the example in four phases: `build_problem` loads input/WOUT, fits or
restores coils and defines the loss; `run_optimizer` chooses physical bounds and
calls `opt.minimize`; `verify_endpoint` independently solves the final coils;
`postprocess` exports accepted-state diagnostics, plots, VTK and the movie.
The library owns optimizer coordinate scaling and accepted-state promotion.
Qualification authentication, checkpoint replay and coil-motion metrics also
use public VMEX APIs. The objective formulas and targets remain in the example.

Inspect settings without a solve or output files:

```sh
python -B free_boundary_single_stage_optimization_scalar.py --dry-run
```

Production runs directly from input/WOUT and coils, with no finite-difference or
matrix-free/dense parity tests at startup:

```sh
python -B free_boundary_single_stage_optimization_scalar.py --output runs/production-new
python -B free_boundary_single_stage_optimization.py --output runs/production-lbfgsb-new
```

Run derivative qualification separately for a configuration. Its passing report
can optionally reuse fitted coils and the certified initial checkpoint:

```sh
python -B verify_free_boundary_single_stage.py --output runs/qualification-new
python -B free_boundary_single_stage_optimization_scalar.py \
  --qualification runs/qualification-new/qualification.json \
  --output runs/production-new
```

For bounded L-BFGS-B, use the same qualification bundle:

```sh
python -B free_boundary_single_stage_optimization.py \
  --qualification runs/qualification-new/qualification.json \
  --output runs/production-lbfgsb-new
```

The L-BFGS-B entry point calls the scalar production workflow with a different
optimizer; it adds no duplicated setup, objective or post-processing code.
Both optimize the same weighted loss. SLSQP enforces the nonlinear iota/radius
inequalities; L-BFGS-B applies bounds `[-5, 5]` in scaled coil coordinates and
uses the existing soft iota penalty. Radius is an endpoint check in L-BFGS-B,
not a loss term. Its optimizer success does not imply physical feasibility.
The shared final checks and exit status require the physical limits to pass.
No augmented Lagrangian is used.

L-BFGS-B promotes only accepted-iteration callbacks, never line-search gradient
probes. If a proposal cannot provide a certified equilibrium/gradient, it stops
and preserves the last accepted state; no replacement gradient is fabricated.
The summary identifies the optimizer and whether nonlinear constraints were
applied. Both entry-point sources are included in the qualification contract
and saved provenance. Changes to the numerical configuration or source invalidate old
qualification evidence; rerun the separate verifier to establish new evidence.

All commands default to GPU. Each output directory must be new. `--dry-run`
prints configuration without creating files or initializing JAX. Production
never runs finite-difference or predictor-parity experiments. Its dense LU
initialization, coupled-root polishing, linear residual checks and derivative
agreement checks during adaptive LU rebuilds remain necessary solver work.
A run without a report records `derivative_qualified: false`; a supplied report
must pass authentication and records `derivative_qualified: true`. No report
is generated automatically during production. The optional report binds input/WOUT contents,
physics and solver controls, source hashes, dependency versions, device kind,
fitted coils and the initial checkpoint. A changed numerical configuration or
source requires fresh qualification. Budgets and plot flags may be changed via
CLI without repeating qualification.

Initialization is defined in the production example:

- With no input option, use the rotating ellipse at `(MPOL, NTOR, NS) =
  (8, 8, 51)` and grid `64 x 64`.
- `--input input.case` requires a vacuum deck and preserves its resolution unless
  `--resolution MPOL NTOR NS` or `--grid NTHETA NZETA` is explicitly supplied.
- `--input input.case --wout wout_case.nc` takes the boundary and restart state
  from WOUT. The matching input deck remains required for pressure/current
  profiles and solver controls. VMEX remaps the restart to the requested grid.
- Without coil input, generate equally spaced coils and fit their geometry on
  the fixed input/WOUT surface with the reference's stage-two L-BFGS-B loss.
  Currents remain fixed. `--initial-coils file.json` fits a supplied initial
  geometry instead. `--coil-fit-maxiter` bounds this fit (default 200).
- `--coils fitted.json` reuses a fitted set, skipping stage two.

The separate verifier calls this shared preparation once. With `--qualification`,
production inherits input and resolution options from the report and restores
the exact checked seed, without stage-two refitting or an initial ordinary solve.
Without a report, production prepares its start directly as described above.
The source input/WOUT and report artifacts must remain available. Restored
states are freshly certified before use.

`verify_free_boundary_single_stage.py` checks the objective and all constraint
rows along one reproducible direction at steps `3e-3` and `1e-3`, using
independent corrections with force tolerance `1e-20` and error gate `1e-3`.
It also checks matrix-free gradient/predictor parity against dense at `1e-6`.
It writes `gradient_check.json`, `matrixfree_check.json`, and a passing
`qualification.json` only after all gates pass. Failed attempts keep their
available diagnostics and a report with `passed: false`; no SLSQP run starts.

Production uses force/edge tolerance `1e-15`, coupled-root polishing to `1e-12`, adjoint full relative residual
`1e-9`, Krylov request `1e-11`, 100 accepted steps and final independent
verification at NS=201, force tolerance `1e-15`. A failed matrix-free adjoint
gets one checked dense retry; its LU replaces the seed only if accepted.
Polishing retains constraint baselines and inactive coordinates, then freshly
recomputes force and vacuum diagnostics. It is bounded to three Newton steps
and eight damping probes per step, with one temporary dense recovery. Failed
refinement rejects the candidate without changing the accepted state.

The public API performs the solver work; the example exposes its accuracy and
optimization parameters near the top. `LU_REFRESH_HORIZON = 10` enables adaptive
refresh: after two warm-up accepted steps, the latest three-step median of
gradient, predictor and polishing costs is compared with the best warm median.
A rebuild occurs when expected savings exceed its measured cost. The horizon
is capped by the remaining step budget; the last accepted step does not trigger
a cost-only rebuild. New derivative rows must agree within `1e-6` before the
new factors replace the accepted seed. Dense recovery remains available.
Each production run saves provenance, solver events, accepted checkpoints,
WOUT/coil outputs and `optimization_summary.json`. Predictor, correction and
gradient and polishing times remain separate. The startup budget is one hour, optimization
twelve hours, final verification thirty minutes. Qualification separately has
a one-hour budget. A budget endpoint or numerical check is not convergence or
physical feasibility. No new full GPU qualification/pilot has been run as part
of this integration. The earlier isolated M8/N8/NS51 campaign supplies the
method's performance evidence, not qualification of this new API integration.
That campaign also exposed curvature peaks between the 64 coil sample points;
the sampled curvature penalty and check in this example do not certify the
continuous curve's maximum. A denser independent engineering check is required
before claiming feasibility. This integration preserves the physics objective
and does not silently change its quadrature or penalty weights.

Post-processing matches the fixed-boundary scalar example: the equilibrium
report includes QA, aspect, mean iota and magnetic well, followed by coil
lengths, curvature, clearance, B.n/B and physical-limit checks. Both initial
and independently verified final surfaces/coils are exported as VTK. The
summary records unmet limits and the best accepted step meeting the plasma
constraints.

After verification, saved accepted checkpoints supply `accepted_steps.csv`,
expanded `accepted_steps.json`, `step_*.npz` coil-motion diagnostics, and
`free_boundary_scalar_objectives.csv` with individual weighted terms. Replay
checks checkpoint hashes and parameter identity, and does not re-solve old
states. This work is outside the optimization timer, with its own thirty-minute
budget. Normal-field history uses the same coil-field diagnostic as the scalar
reference; B.n terms remain excluded from the free-boundary objective.

Figures are enabled by default: `optimization.png`, `objectives.png`,
`optimization.gif`, and the standard WOUT summary, surface, |B|, profile,
stability and 3D figures. The accepted-iterate movie defaults to the saved
equilibrium's total `|B|`, using `MOVIE_SURFACE_COLOR="absB"`; `"B.n/B"` uses
the plasma-vacuum interface API. `--no-movie` skips only the movie;
`--no-plots` skips figures and the movie while retaining reports, CSV/NPZ,
WOUT, coil JSON and VTK exports.


## Historical validation

The previous private-driver experiments and commands are preserved in Git history.
Their timings and qualification results do not qualify these new entry points.
A matched GPU qualification and bounded pilot remain necessary before a long run.

## Optimizer and linear-solver choices

SLSQP and L-BFGS-B share `opt.minimize`; BFGS is available through the same
library adapter. The upstream vacuum and finite-beta free-boundary examples
also use that adapter. `minimize_projected` remains a separate research method
with different target restoration and convergence semantics.

Production uses JAX dense LU to initialize and precondition current-root GMRES.
A failed matrix-free adjoint gets a checked dense retry.
The scalar example enables `problem.enable_root_polishing(tolerance=1e-12)`
before `problem.enable_matrix_free(refresh_horizon=10, refresh_max_steps=...)`.
These policies are opt-in in the library and shared by the SLSQP and L-BFGS-B
example entry points; they do not change the library's default solver behavior.
`problem.solver_info`, solver-event rows and the final summary
identify the policy and actual linear solves.

The accepted-root API now also accepts upstream `boundary_schur`,
`coupled_gcrot` and `edge_response`, plus host dense LU,
through `solver_options["adjoint_solver"]`. Dense methods reuse LU for their
predictor; other choices use `gcrot_tangent`, a forward prediction solver.
The former `reverse_gcrot` adjoint option has been removed. Schur currently
factors each adjoint row separately. These alternatives need matched physical
qualification and timing before replacing the production default; enabling
seed-LU reuse with a non-JAX-dense backend is rejected explicitly.

## Fixed-boundary derivative tests

The constrained fixed-boundary scalar production script also starts without
finite differences. Its production force tolerance is `1e-11`. Run its
constraint derivative audit separately:

```sh
python -B verify_single_stage_constraints.py --output runs/fixed-constraint-check
```

This standalone check uses a separate `1e-22` force tolerance and compares the
constraint adjoint with independent equilibrium re-solves. It records the
input, source/runtime contract, direction seed, step sizes and componentwise
errors in `constraint_gradient_check.json`; it never fits coils or starts an
optimization. A passing constraint test alone does not qualify the full
plasma/coil objective. Upstream documents that changes in frozen coordinates
can create a re-solve/adjoint gap, which this test reports rather than hides.

## Preparing finite-beta support

The shared optimizer is independent of vacuum/finite-beta physics. The current
three-method production builder is deliberately vacuum-only and rejects
nonzero pressure or total plasma current. Its stage-two fit and scalar B.n/B
checks use the coil field alone, which is not the total field at finite beta.

The upstream reference is
`../optimization/single_stage_optimization_finite_beta.py`. It:

1. Sets kinetic density/temperature profiles, calibrates pressure to a seed beta
   target of 0.025, and initializes plasma current with `self_consistent_bootstrap`.
2. Optimizes boundary modes, spline plasma-current shape and total current,
   plus coil geometry/currents. `RedlBootstrapMismatch` joins QA, aspect, iota,
   beta and coil penalties in the objective.
3. Uses `PlasmaVacuumInterface`/virtual casing for the required external field,
   B.n and total-pressure balance, and reports bootstrap and interface diagnostics.

The native `../optimization/single_stage_free_boundary_optimization_finite_beta.py`
already demonstrates a coil-only finite-beta objective with prescribed input
profiles and a bootstrap mismatch term. It does not vary those plasma-current
profiles and is not a qualification of this accepted-state scalar workflow.

Extend the production path in two stages: first prescribed finite-beta profiles,
plasma-aware stage-two fitting and matching total-field/interface diagnostics;
then joint plasma-current variables and their implicit derivatives if bootstrap
self-consistency is required. Preserve the upstream edge-pressure assumptions.
Each stage needs independent reconverged finite-difference checks at finite beta
and a bounded accepted-state pilot. Changing beta or passing a pressure deck
alone does not establish bootstrap self-consistency or numerical qualification.
