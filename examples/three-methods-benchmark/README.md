# Three-methods benchmark

Three optimization workflows share the NFP=2 vacuum rotating ellipse in
[`input.rotating_ellipse`](input.rotating_ellipse): prescribed-boundary QA,
fixed-boundary single-stage plasma/coil optimization, and free-boundary
single-stage optimization with the coils determining the plasma boundary.

| Workflow | Entry point | Optimizer |
|---|---|---|
| Prescribed-boundary QA | `qa_optimization.py` | Staged least squares |
| Fixed-boundary scalar single stage | `single_stage_optimization_scalar.py` | SLSQP with `--constrained`; BFGS otherwise |
| Free-boundary scalar single stage | `free_boundary_single_stage_optimization_scalar.py` | SLSQP with `--constrained`; BFGS otherwise |

The methods share an input, but the default scripts are not an automatically
matched performance comparison. Match resolution, coils, objective terms,
constraints and budgets explicitly before comparing results. The historical
`single_stage_optimization.py` and `single_stage_free_boundary_optimization.py`
are also included; the former contains a separate AugLag workflow. The scalar
SLSQP commands below do not use AugLag.

## Environment and inputs

Use this branch's VMEX source with the dependencies in the root
`pyproject.toml`, including the optional coil dependencies (`vmex[coils]`).
Run commands from this directory with that checkout importable. The input
defines zero pressure/current, major radius 1 m, area-equivalent aspect 5,
and PHIEDGE 0.1255044894897643 Wb. Its default resolution is M3/N3/NS31
with a 48x40 angular grid; the free driver supports explicit overrides.

`coils.initial.scalar.json` is a saved initial coil fit, with its generation
metadata beside it. The separately included
`coils_single_stage_scalar_fitted_1789769333906005000.json` is the common
stage-two fit used by the scalar free-boundary driver. Currents remain fixed.
Neither file is a final plasma/coil optimization checkpoint.

The free-boundary initialization first solves the prescribed ellipse, then
converges the equilibrium supported by the fitted coils. That second state
is optimization step 0, so its boundary and QA can differ from the prescribed
ellipse. Every subsequent candidate and retry starts from the last accepted
equilibrium.

## Run the scalar examples

Inspect the free driver's options without starting a solve:

```sh
python -B free_boundary_single_stage_optimization_scalar.py --help
```

For a fixed-boundary comparison using the saved common coils:

```sh
python -B single_stage_optimization_scalar.py --constrained --device gpu \
  --coils coils_single_stage_scalar_fitted_1789769333906005000.json \
  --maxiter 100 --output runs/fixed-scalar
```

For a bounded free-boundary matrix-free pilot:

```sh
python -B free_boundary_single_stage_optimization_scalar.py \
  --device gpu --constrained --adjoint matrixfree \
  --resolution 8 8 51 --grid 64 64 \
  --adjoint-max-dofs 20000 --adjoint-batch-size 32 --no-boundary-error \
  --initial-ftol 1e-18 --ftol 1e-11 --fd-ftol 1e-20 \
  --matrixfree-restart 100 --matrixfree-max-cycles 3 --matrixfree-refresh-on-failure \
  --maxiter 10 --accepted-steps 2 --max-trials 40 \
  --wall-seconds 5400 --optimization-seconds 1800 \
  --verification-seconds 1800 --verify-ns 51 \
  --verify-maxiter 12000 --verify-ftol 1e-15 \
  --output runs/free-scalar-matrixfree --no-plots
```

This free configuration is not resolution/objective matched to the fixed
command above. Use a suitable GPU allocation and external finite watchdog
for long runs. The driver creates a new output directory and refuses to
overwrite one. The evaluation budget includes qualification endpoints.

SLSQP uses `ftol=1e-10` and explicit iota/radius inequalities. The physical
limits are minimum half-mesh |iota| >= 0.19 and major radius 0.99–1.01 m;
internal guards are 0.1905 and 0.991–1.009 m. Intermediate SLSQP iterates
can violate those guards. QA, aspect and coil geometry contribute to the
weighted objective. `--no-boundary-error` removes both normal-field objective
terms while retaining their final diagnostics. B0 is diagnostic only.

The native matrix-free adjoint and tangent use a retained LU
preconditioner, current-root JVP/VJP operations and bounded FGMRES. Fresh
finite-difference qualification is required at both h=0.003 and h=0.001,
with objective and constraint relative errors below 0.001. Dense/matrix-free
gradient and tangent parity must be below 1e-8; true linear residual checks
use a 1e-9 gate. Dense recovery is opt-in and factors refresh only after
optimizer acceptance, as described below.

## Validation status

This is research code, not a completed three-method scientific qualification.
The earlier native pilot passed independent M8/N8/NS201 physical checks.
The later recovery/tolerance pilot described below used same-resolution final
verification. Neither establishes a completed 100-step qualification.
An accepted-step budget stop does not establish optimizer convergence.

`QA total` is the sum of squared quasisymmetry residuals, evaluated on ten
surfaces from s=0.1 to 1. It is distinct from the total weighted objective.
The fresh M8/N8/NS51 free run recorded step-0 QA 0.0578603254. Report
best feasible states separately from raw endpoints, and verify final iota
and radius independently at higher resolution before making physical claims.

Local campaign directories, raw checkpoints, machine configuration and
monitoring scripts are excluded from this publication. The active campaign
retains its original local `three-way-benchmark` path so its immutable source
hashes and monitor continue working.


## Optional matrix-free adjoints

Dense adjoints remain the default and reference. To use a retained dense LU
as a preconditioner for current-root Krylov solves, add these options to the
free-boundary command above:

```sh
--adjoint matrixfree --matrixfree-restart 100 --matrixfree-max-cycles 3 \
--matrixfree-rhs-batch-size 3 --matrixfree-rtol 1e-11 \
--adjoint-residual-rtol 1e-9 --predictor-rtol 1e-11 \
--matrixfree-convergence-policy residual --matrixfree-refresh-on-failure \
--ftol 1e-11 --verify-ftol 1e-15
```

The Krylov stopping request and the independently evaluated full adjoint
residual are different checks. The `residual` policy accepts a finite solution
only when the full residual passes; `requested` additionally requires Krylov's
convergence flag. Predictor stopping is independent. Starting-point derivative
qualification is mandatory; its finite-difference equilibrium tolerance stays
at `1e-20` by default and is excluded from optimization timing.

After qualification, a failed matrix-free adjoint can trigger one checked
dense solve at the same certified trial. Its factors replace the retained LU
only if the optimizer accepts that trial. Failed trials and their last accepted
anchors are saved for replay. Recovery cost is included in gradient/step timing.
An explicit `--initial-checkpoint` requires its `--initial-checkpoint-sha256`
and fresh qualification; it starts fresh optimizer history.

The vacuum M8/N8/NS51, 64x64 pilot passed two accepted steps from the formerly
blocked step 10, with a same-resolution final force check at `1e-15`. That
bounded result does not establish long-run convergence, finite-beta accuracy,
or equivalence to the separate projected optimizer on `main`. Keep this path
opt-in until a matched longer run passes numerical and physical checks.
