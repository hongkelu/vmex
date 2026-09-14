# M3/N3 QA optimization on a 48×40 angular grid

This independent case starts from the original zero-parameter coils and spectral
seed. It uses NTHETA=48, NZETA=40, M3/N3, NS31, NFP2, and 111 coil/current
parameters. Its scientific core is published main commit
`d6910b428841766c04767410e9e9cdc39a6cfb19`, the parent of this case branch.

## Targets and acceptance

| Quantity | Target | Absolute tolerance (1%) |
|---|---:|---:|
| Mean iota | 0.2 | 0.002 |
| Aspect ratio | 5 | 0.05 |
| Signed on-axis B0 | −0.17506474574437714 T | 0.0017506474574437714 T |

These bands are design tolerances, not numerical uncertainty estimates. Signed B0
retains the original reference target when refining the initial equilibrium.

- Ordinary force channels and fresh edge residual: **≤1e-11**.
- Projected-root residual norm: **≤2e-6**.
- Adjoint and state-tangent relative residual: **≤2e-5**.
- Adjoint backend: `forward_dense_jax`; the state tangent uses main's GCROT path.
- Maximum coil motion per accepted step: **1 mm**, with a separate **1% of each
  nominal current** step cap. These do not bound cumulative changes.
- No root polish or same-coil equilibrium retries.

When feasible, the equality-tangent QA direction is sized independently of the
equality correction. Acceptance requires actual QA decrease and all constraints
remaining feasible. Otherwise, restoration must reduce normalized constraint
violation; QA may increase. Each proposal has at most six trials with successive
halving. Rejected states never replace the accepted checkpoint.

Convergence requires feasibility and a sufficiently small projected QA gradient.
An exhausted trial budget reports stagnation. A step budget or tiny coil motion
does not establish convergence.

QA error is the squared norm of `QuasisymmetryRatioResidual` at normalized flux
surfaces 0.25, 0.5, 0.75, and 1 with helicity (1, 0). The objective is half that
error divided by the initial QA squared norm. It is not a Boozer-mode error or
an independent field-line confinement certificate.

## Initial state and resume

`inputs/manifest.json` authenticates the bundled input, coils, and spectral seed.
The small seed NPZ contains numerical arrays and a schema label; it is not a
certified 48×40 restart state. Fresh initialization performs one ordinary
refined-grid solve and independently certifies its `result.state`, not the
continuation restart buffer. Ordinary and freshly evaluated edge residuals must
agree with relative tolerance 1e-5 and absolute tolerance 1e-15.

Use a project environment satisfying VMEX's dependencies with compatible ESSOS
`Coils` and `Curves`. The validated campaign used Python 3.12, JAX/JAXLIB 0.6.2,
Solvax 0.20.0, NumPy 2.2.6, SciPy 1.15.3, and netCDF4 1.7.2 on an NVIDIA RTX
4090 D. Its ESSOS `coils.py` SHA256 was
`8988ebb9ffdda232782dabe131d1e1d7cfe02b9e13da23c3ac82bb954fcb987e`.
Runtime manifests record the installed ESSOS source hash rather than assuming
that any release behaves identically.

From the repository root, with an authorized GPU selected:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
case_dir='example/m=3n=3iota=0p2aspect=5angular48x40'
python -B "$case_dir/run.py" --initialize-only --target-step 10 \
  --device cuda:0 --max-wall-hours 1 --output-dir "$case_dir/outputs/initial"
```

Each output directory must be new. After initialization passes, read
`latest_checkpoint.json` for the checkpoint path and SHA256, then supply both:

```sh
python -B "$case_dir/run.py" --resume-checkpoint CHECKPOINT_PATH \
  --checkpoint-sha256 CHECKPOINT_SHA256 --target-step 10 \
  --device cuda:0 --max-wall-hours 24 --output-dir "$case_dir/outputs/validation10"
```

Validate the clean terminal summary, exit, ten accepted checkpoints, gates and
diagnostics before continuing from the authenticated step-10 checkpoint with
`--target-step 3000`. This means **3000 total accepted steps**, not 3000 more.
Each process is limited to 24 hours. Subsequent processes restore accepted states
exactly and recheck certification without a cold equilibrium solve.

Every accepted step saves its checkpoint and records iota, QA error, aspect,
signed B0 and residuals. Paired coil/WOUT exports occur initially, every 20 steps,
and at the final accepted state. Initial compilation is substantial; measure
steady step times using consecutive promotion events within the same segment.

## Monitoring deployment

`campaign.py` and `workflow.py` expect a deployment root containing `source/`,
`external/ESSOS`, `venv/`, authenticated source/test manifests, a validated
checkpoint, `policy.json`, and `registry.json`. They are deployment controllers,
not automatic installation or bootstrap scripts. The controller GPU UUID and
`monitor_both.py` SSH alias/root refer to the original authorized deployment and
must be configured deliberately for another machine.

Despite its legacy filename, `monitor_both.py` checks **only this high-resolution
case**. The hourly watcher invokes it with `--restart`. The locked controller
prevents duplicate processes and resumes eligible kills or timeouts from verified
accepted checkpoints. Scientific/integrity failures, unknown exits, stagnation,
convergence and exhausted budgets require review instead of blind relaunch.

The campaign policy permits 96 cumulative solver hours within seven elapsed days,
at most 64 segments and three consecutive no-progress interruptions, and requires
at least 10 GiB free disk. Per-process limits remain at most 24 hours. Live
deployment metadata, caches, generated WOUTs and campaign results are excluded
from this source publication. Publishing does not modify the active deployment
or install a scheduler for other users.

## Validation

The ten-step GPU validation completed with exit 0 and all numerical gates passed.
Its endpoint remained outside the target bands: numerical validation did not
establish target attainment or QA convergence. The running 3000-step campaign
is not represented as a completed result here.

Run the included lightweight acceptance, checkpoint, diagnostic and controller
tests without launching a GPU campaign:

```sh
python -B -m pytest "$case_dir/tests" "$case_dir/test_campaign.py" \
  "$case_dir/test_workflow.py" -q
```
