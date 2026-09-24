# Finite-beta scalar production

`finite_beta_production.py` is the maintained free-boundary production entry.
It follows main's `examples/single-stage-benchmarks`:
`build_problem → configure_solver → opt.minimize → verify_endpoint → postprocess`.
The implementation extends the shared production workflow introduced on main
at `7cd0a1b8`.

## Physics and numerical contract

- Prepared input and fitted coils: `inputs/finite_beta_initial/`. The input and
  coil hashes are checked against their preparation manifest.
- Reference beta 0.5%; pressure and PHIEDGE stay fixed. Actual volume-average
  beta and its relative deviation are diagnostics, with no beta constraint,
  pressure variable or pressure adjoint row.
- `NCURR=1`, `CURTOR=0`, `AC=0`; no bootstrap-current model. Coil currents are
  fixed; the 99 coil-shape variables use the existing 0.05 m coefficient scale.
- M8/N8/NS51, 64×64 equilibrium grid, ordinary force/edge FTOL `1e-15`, coupled
  root polishing to `1e-12`. The prepared coils retain **256** quadrature points.
- The previous finite-beta scalar objective is preserved: QA, aspect, iota-floor,
  total-normal-field and coil penalties. The curvature penalty starts at 6.9/m;
  its reporting limit is 7/m. Coil penalties are soft; feasibility is reported
  separately. SLSQP has hard iota and major-radius constraints; AugLag is disabled.
- NESTOR supplies the coupled free-boundary vacuum response. Virtual casing
  supplies the live plasma contribution to the total-field objective and
  diagnostics. Coil-only normal field is not reported as total normal field.
- Values and adjoints share the polished root. SLSQP acceptance, deferred
  derivatives, checked matrix-free solves and dense fallback use the main API.
  A failed initial tangent geometry gets one retry from the unchanged accepted
  equilibrium. Polishing removes inactive-coordinate drift relative to that
  anchor and checks its active mask and constraint baselines.
- No finite-difference or startup dense/matrix-free gradient-comparison suite
  runs in production. Necessary gradients, linear residual checks and consistency
  checks on a replacement dense factorization remain enabled.
- Production has no elapsed initialization/optimization cutoff, following the
  campaign instruction. Accepted steps, trial count and each nonlinear solve
  remain bounded. Verification and plotting have their own finite phase budgets.

## Commands

From the checkout root, in a current VMEX-compatible environment (including
`virtual-casing-jax>=0.0.8`):

```sh
python examples/three-methods-finite-beta-benchmark/finite_beta_production.py --dry-run
python examples/three-methods-finite-beta-benchmark/finite_beta_production.py \
  --device gpu --accepted-steps 100 \
  --output examples/three-methods-finite-beta-benchmark/runs/production-001
```

Production starts directly without a qualification report or skip-check flag.
The output must be new. `--input`, `--wout`, `--coils`, `--initial-coils`,
`--resolution`, `--grid`, `--no-plots` and `--no-movie` mirror the vacuum
interface. A WOUT requires its input deck; profiles and PHIEDGE come from that
deck. Supplying `--initial-coils` requests a preliminary coil fit using the
fixed equilibrium's virtual-casing plasma field, without derivative comparisons.

Run derivative qualification separately when desired:

```sh
python examples/three-methods-finite-beta-benchmark/verify_finite_beta_production.py \
  --device gpu \
  --output examples/three-methods-finite-beta-benchmark/runs/qualification-001
```

This command runs no optimization. It independently reconverges both endpoints
from the unchanged accepted root with the predictor disabled, at two step sizes,
then compares dense and matrix-free derivatives/tangents. It writes a passing
qualification bundle only if all checks pass. Production can optionally reuse
that bundle with `--qualification .../qualification.json`; it still solves the
fixed reference to select virtual-casing quadrature, but reuses the certified
free seed and fitted coils. An absent report is recorded as
`derivative_qualified=false`, not as a failed or implicitly passed check.

Continue a saved accepted state without an ordinary equilibrium re-solve:

```sh
python examples/three-methods-finite-beta-benchmark/finite_beta_production.py \
  --device gpu --resume-checkpoint runs/production-001/accepted_0005.npz \
  --checkpoint-sha256 SHA256_FROM_CHECKPOINT_0005_JSON \
  --accepted-steps 95 --output runs/production-step5-to100
```

The step budget counts **additional** accepted steps. This example retains the
absolute step numbers 5 through 100 and starts a fresh SLSQP optimizer history.
Use the original input, resolution, grid and stage-two coil chart, not the
optimized coils as a new chart. The checkpoint preserves the exact accepted
state, coil coordinates, masks, baselines and gradient reference. It is
SHA256 authenticated and re-certified before optimization. It does not claim
derivative qualification. `resume.json` records both lineage and the target.
The initial production release is supported by one explicitly checked builder
migration, plus a source-pinned core fix allowing polishing immediately after
restoration at a nonzero step. Polishing setup remains forbidden after any new
accepted step. Other changes to physics, numerical helpers, core or dependencies
reject reuse. Historical campaign NPZ formats remain unsupported.

When relocating a checkpoint to another GPU model, add
`--resume-on-new-hardware`. Only the hardware model comparison is relaxed;
the backend, numerical settings, source migrations and dependencies must still
match. The accepted state is re-certified on the destination. `resume.json`
retains the original and current contracts, including both hardware models;
derivative qualification is not transferred.

Historical campaign checkpoints use a different format and cannot be passed as
public-API qualification bundles. This example includes only the required input
deck and fitted coils, with their SHA256 manifest.

`benchmark.json` preserves the historical comparison settings used by the scalar
loss regression. Its method-status, pressure-variable, and qualification fields
describe that historical setup; they do not configure this production driver.
Production settings are the visible constants in `finite_beta_production.py`.

## Evidence and limits

The focused local regression command is:

```sh
python -m pytest -q tests/test_finite_beta_production.py \
  tests/test_freeboundary_root_polish.py tests/test_freeboundary_fast_example.py \
  tests/test_single_stage_interface.py
```

On 2026-09-23 this passed **86 tests** on the clean main-based integration branch.
An additional API/virtual-casing selection passed **76 tests**, with one
`RUN_FULL=1` case skipped and three long equilibrium/integration cases deselected.
The test-manifest ownership check also passed. Both entry-point dry-runs and
`git diff --check` passed.

The original local overlay had virtual-casing-jax 0.0.7 and failed the existing
far-field error-estimate regression. The new project-local 0.0.8 overlay satisfies
main's requirement and passes that test. Dependency overlays are isolated from
shared environments.

Local regressions exercise the production/test separation, fixed pressure/flux
and currents, unchanged objective and coil gradient against the historical
implementation, live plasma-field normalization, SLSQP acceptance, root
polishing, inactive-coordinate repair, predictor retry and shared vacuum output.
The separate numerical verifier is also tested with independently solved
analytic endpoints. GPU execution evidence is reported separately from these
regressions. A production run without a qualification bundle does not establish
independent finite-difference agreement.

Every production run writes the numerical signature, input/coil hashes,
dependency versions, hardware, solver phase events, accepted checkpoints and
objective history. Independent NS201 endpoint verification writes WOUT, total
normal-field, pressure-jump, beta and engineering diagnostics. A step budget
stop is not convergence, and numerical force convergence does not establish
coil feasibility.
