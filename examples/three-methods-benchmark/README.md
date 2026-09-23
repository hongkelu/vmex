# Qualified free-boundary production

`free_boundary_single_stage_optimization_scalar.py` (formerly
`free_boundary_single_stage_fast.py`) uses the scalar VMEX API directly:
editable parameters, weighted plasma/coil objectives, a public VMEX problem,
`opt.minimize`, and endpoint reporting all appear in the example. It imports
`vmex` and `vmex.optimize`, with ESSOS for coils and surfaces. It does not
import private VMEX modules or a local driver/setup helper.

`opt.FreeBoundaryProblem.from_loss` owns equilibrium correction and the total
scalar gradient, including the moving surface in coil-surface clearance.
`accept_x` promotes only optimizer-accepted states. The example's importable
`build_problem` defines one initialization/objective path for production and
qualification. No local numerical helper module is used; historical drivers
and frozen run snapshots remain unchanged.

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

Run expensive qualification once for the desired configuration, then reuse its
fitted coils and certified initial checkpoint:

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
and saved provenance. This API/default-tolerance refactor requires fresh qualification.

All commands default to GPU. Each output directory must be new. `--dry-run`
prints configuration without creating files or initializing JAX. Production
requires a passing matching report; it does not rerun finite differences or
dense/matrix-free comparisons. Its own dense LU initialization and all actual
residual checks are still necessary. The report binds input/WOUT contents,
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

Qualification calls this shared preparation once. Production inherits the
input and resolution options from the report and restores the exact qualified
seed; it performs neither stage-two refitting nor an initial ordinary solve.
The source input/WOUT and report artifacts must remain available. Restored
states are freshly certified before use.

`verify_free_boundary_single_stage.py` checks the objective and all constraint
rows along one reproducible direction at steps `3e-3` and `1e-3`, using
independent corrections with force tolerance `1e-20` and error gate `1e-3`.
It also checks matrix-free gradient/predictor parity against dense at `1e-6`.
It writes `gradient_check.json`, `matrixfree_check.json`, and a passing
`qualification.json` only after all gates pass. Failed attempts keep their
available diagnostics and a report with `passed: false`; no SLSQP run starts.

Production uses force/edge tolerance `1e-11`, adjoint full relative residual
`1e-9`, Krylov request `1e-11`, 100 accepted steps and final independent
verification at NS=201, force tolerance `1e-15`. A failed matrix-free adjoint
gets one checked dense retry; its LU replaces the seed only if accepted.
Each production run saves provenance, solver events, accepted checkpoints,
WOUT/coil outputs and `optimization_summary.json`. Predictor, correction and
gradient times remain separate. The startup budget is one hour, optimization
twelve hours, final verification thirty minutes. Qualification separately has
a one-hour budget. A budget endpoint or numerical check is not convergence or
physical feasibility. No new full GPU qualification/pilot has been run as part
of this code refactor.

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


## Comparison workflows

The `free-vs-fixed-single-stage` branch contains the additional prescribed- and
fixed-boundary reference scripts. These production entry points share the same
rotating-ellipse input and qualified free-boundary workflow on both branches.

## Optimizer and linear-solver choices

SLSQP and L-BFGS-B share `opt.minimize`; BFGS is available through the same
library adapter. The upstream vacuum and finite-beta free-boundary examples
also use that adapter. `minimize_projected` remains a separate research method
with different target restoration and convergence semantics.

Production uses JAX dense LU to initialize and precondition current-root GMRES.
A failed matrix-free adjoint gets a checked dense retry. **It does not run
`reverse_gcrot`.** `problem.solver_info`, solver-event rows and the final summary
identify the policy and actual linear solves.

The accepted-root API now also accepts upstream `boundary_schur`,
`coupled_gcrot` and `edge_response`, plus `reverse_gcrot` and host dense LU,
through `solver_options["adjoint_solver"]`. Dense methods reuse LU for their
predictor; other choices use the existing GCROT tangent. Schur currently
factors each adjoint row separately. These alternatives need matched physical
qualification and timing before replacing the production default; enabling
seed-LU reuse with a non-JAX-dense backend is rejected explicitly.

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
