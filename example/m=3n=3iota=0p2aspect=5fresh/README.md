# M3/N3 QA optimization with practical target tolerances

This case uses M3/N3, NS31, NFP2, the authenticated original four-coil chart,
111 coil/current parameters, and the original QA normalization. The VMEX core
is frozen at main commit `1349135ef90e7bf039a265401ab21c091401397c`.
The numerical case files and authenticated seed reproduce the tested runtime
on this core revision. No VMEX core modifications are required.

## Targets, feasibility and movement limits

| Quantity | Fixed target | Absolute feasibility tolerance (1%) |
|---|---:|---:|
| Mean iota | 0.2 | 0.002 |
| Aspect ratio | 5 | 0.05 |
| Signed on-axis B0 | -0.17506474574437714 T | 0.0017506474574437714 T |

The signed B0 target and QA normalization are inherited from the original
zero-parameter state. A resume never resets them. The 1% feasibility bands are
user-selected design tolerances, not claims of numerical error bars.

Each accepted change has at most 1 mm coil motion and 1% of each original
nominal current. Current parameters are fractional changes from the nominal
currents, so `abs(delta[:3]) <= 0.01`; these are per-step bounds, not cumulative
current bounds. A conservative Fourier harmonic bound covers every coil angle
and its symmetry copies. The sampled displacement is also logged.

## Optimization and measured acceptance

Inside the feasibility bands, project the normalized QA objective gradient
onto the null space of the three equality Jacobian rows in parameter-scaled
coordinates. Size this QA direction using only the geometry/current caps.
Its size is independent of the remaining equality error.

Outside the bands, use a bounded least-squares equality-restoration direction
toward the original targets. Do not enlarge a small restoration correction.
Resume QA descent once all three constraints are feasible.

Try at most six step lengths: 1, 1/2, 1/4, 1/8, 1/16 and 1/32. Each trial
starts from the same accepted state and parameters. All ordinary, edge-force,
projected-root and derivative checks must pass before assessing the actual
endpoint metrics. Candidate states never reanchor the accepted configuration.

- From a feasible state, all three constraints must remain feasible and the
  normalized QA objective must decrease by at least the larger of 1e-10 and
  `1e-4 * alpha * (-objective_directional_derivative)`.
- From an infeasible state, reduce `max(max(abs(error)/tolerance)-1, 0)` by at
  least `1e-4 * alpha` of its old value (with a 1e-12 numerical floor), or enter
  the feasible band. QA may rise during restoration.
- Failed numerical trials and failed endpoint acceptance both trigger a smaller
  changed-coil trial. Programming errors and timeouts do not become retries.

Force and edge tolerances are 1e-10, the projected-root gate is 2e-6, and the
adjoint residual relative gate is 2e-5. Adjoints use `forward_dense_jax`; the
state tangent uses the existing main GCROT path. There is no root polishing,
no independent initialization solve and no same-coil equilibrium retry.

## Stopping and provenance

`converged` requires feasibility and a scaled equality-tangent QA gradient
norm no larger than `max(1e-6, 1e-4 * initial_projected_gradient_norm)`.
This is equality-tangent stationarity, not a full inequality KKT certificate
for all directions inside the tolerance bands. Numerical gradient thresholds
are explicit policy choices; they do not establish physical qualification.

`stagnated` means the finite trial budget found no acceptable step. Tiny coil
motion alone never reports success. `step_budget_reached` is a run-length
limit, not convergence. Numerical failures outside trial evaluation report
`failed`. Every accepted state is checkpointed. Trial candidate files are
explicitly ineligible for resume; only promoted checkpoints may continue.

This separate fresh case starts at absolute step zero from the original
authenticated all-zero 111-parameter seed. It does not resume any optimized
checkpoint. Initial measured values are iota 0.1976945532 and aspect
4.7448345428; the target values are 0.2 and 5. The signed B0 target remains
-0.17506474574437714 T. The constraint tolerances remain at 1% throughout. Acceptance does not imply
exact target equalities or a prescribed final QA value.

The physical run targets absolute step 200 with a four-hour timeout, or stops
earlier with the defined convergence/stagnation condition. The optimizer is
identical to the verified predecessor: ten accepted steps reduced raw QA squared
by 38.2%, with all target, movement and numerical checks passing.

Run with `--target-step 200 --max-wall-hours 4 --output-dir NEW_DIRECTORY`,
without any resume argument. The runner authenticates the original seed and
records source hashes, numerical settings and the selected device. `continuation_plan.json`
records this fresh initialization and the user-confirmed 1% bands.

## Reproduce a fresh run

Use an environment with VMEX from this checkout, ESSOS importable as `essos`,
and a compatible CUDA JAX installation. The validated GPU environment used
Python 3.12, JAX 0.6.2, Solvax 0.20.0 and NumPy 2.2.6. The runner records the
ESSOS coil source hash. GPU hardware was an NVIDIA RTX 5090 (32 GB).

From the repository root, with this checkout on `PYTHONPATH`, run:

```sh
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 \
python -B 'example/m=3n=3iota=0p2aspect=5fresh/run.py' \
  --device cuda:0 --target-step 200 --max-wall-hours 4 \
  --output-dir 'example/m=3n=3iota=0p2aspect=5fresh/runs/fresh200'
```

The output directory must not already exist. To start with a bounded smoke
run, use `--target-step 10 --max-wall-hours 1` and a different output directory.
The default budget is 10 steps / one hour; the command above explicitly selects
200 steps / four hours. Run computation under the allocation rules of your site.

Case tests (CPU, no ESSOS or GPU execution required):

```sh
JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest \
  tests/test_iota02_aspect5_fresh.py -q
```

## Recorded progress snapshot

At 2026-09-12 20:14:50 UTC the fresh campaign had accepted 27 steps and was
still active; this is not a completed 200-step result or a convergence claim.
All three constraints first entered their 1% bands at step 17. Raw QA squared
rose from 0.0989644440 initially to 0.1538852871 during restoration, then fell
to 0.0946083348 at step 27 (38.52% below the first feasible state, 4.40% below
the original seed). The step-27 values were iota 0.1996436654, aspect
4.9575166085, and signed B0 -0.1750651460 T. All accepted numerical gates and
motion bounds passed in that snapshot. Runs, caches and deployment-specific
launch scripts are not part of this reproducible case distribution.

The raw QA metric is the squared norm of the campaign residual, not an
independent Boozer diagnostic. The current run performs no root polishing;
the authenticated legacy seed itself has historical polishing ancestry.
