# M3/N3, iota near 0.2, main-native strict continuation

Run `run.py` against this VMEX source tree. This example uses main's shared
reverse-GCROT adjoint and ordinary VMEX–NESTOR solver. It imports an accepted
checkpoint without a cold/template equilibrium solve, evaluates fresh force
and projected-root residuals, then takes fixed 1-mm coil steps. No root polish,
independent rebound, or automatic same-coil retries are called. Failed gates
stop the run. The last accepted state supplies checkpoints and WOUT directly.

The 111-vector contains 3 relative current coordinates and 108 order-4 Cartesian
coil coefficients. NS31, NFP2, M3/N3; coils use the authenticated ESSOS chart.
QS is soft; mean iota, Rmajor_p, Aminor_p and signed B0 constrain the local
direction. Nonlinear equality drift and objective improvement remain diagnostic.
The iota label is approximate; targets come from the authenticated anchor.
Campaign QS is distinct from WOUT QS. Numerical acceptance is not qualification.

```
python 'examples/m=3n=3iota=0p2/run.py' \
  --payload /path/payload.json --payload-sha256 HASH \
  --resume-checkpoint /path/checkpoint_step_0010.npz --checkpoint-sha256 HASH \
  --target-step 20 --device cuda:0 --output-dir /path/new-run
```

Target 20 from step 10 requests ten additional promotions. Add `--validate-only`
to certify the imported state without any equilibrium solve. Outputs must be
new directories. `--validate-tangent-only` certifies one linear response to a
1-micron coil coefficient direction at the saved state, without a nonlinear
solve or optimization update. Set `CUDA_VISIBLE_DEVICES` before starting Python; `cuda:0`
then names the first visible physical GPU. All caches/logs/temporary files must
be project-local. `--max-wall-hours` sets a finite runtime bound (default four
hours); deployment should also use an external process timeout.

For a long continuation use `--target-step 2000 --checkpoint-every 20` and
an explicit runtime budget. The target is an absolute accepted-step count.
Checkpoints are written at multiples of 20, at import, and on exit for the
last accepted state. Per-step metrics remain in `progress.jsonl`.

Strict edge convergence refreshes the vacuum matrix and force normalization
on each ordinary iteration, matching the fresh certification definitions.
The default solver retains its existing refresh cadence. A failed gate still
stops the run; there is no retry solve or relaxed acceptance threshold.

The initial tolerance study repeats only linear adjoints at the unchanged root
with 2e-5, 2e-6, and 2e-7 stopping/true-residual tolerances. Predeclared stability
checks require <=0.1% relative scaled row changes and <=1 micron difference in
the resulting 1-mm proposal. These are numerical tolerance-sensitivity checks,
not independent proof of derivative correctness. Failure stops before promotion.
Force acceptance remains 1e-14; main's projected preconditioned root gate is 2e-6.

The legacy reader accepts only the hash-pinned strict step-10 checkpoint.
Its historical evaluator explicitly uses zero constraint baselines; main
reconstructs the active mask and must recertify the unmodified state. New
checkpoints preserve baselines and masks as well as the six state arrays,
parameters, targets, scales, trust radius, counters and source/input provenance.

The small input/coil files are preserved reference inputs. The payload binds
additional external anchor artifacts by hashes; no machine-specific payload or
private checkpoint is embedded in the portable example. ESSOS must supply the
original `Curves`/`Coils` interfaces; its source version is recorded at launch.
