# M3/N3 rotating ellipse with finite pressure and plasma current

This case targets mean iota 0.2, aspect ratio 5 and signed on-axis B0 +5.1 T using prescribed finite-pressure and plasma-current profiles. The VMEX runtime is pinned to main commit `d6910b428841766c04767410e9e9cdc39a6cfb19`; `runtime_manifest.json` verifies the imported VMEX source before solving. This branch contains the inputs, driver and regression tests; it does not claim a converged finite-beta optimization result.

## Physical configuration

Use the original rotating-ellipse coil geometry, M3/N3, NS31, NFP2, four base coils with order-4 Cartesian Fourier curves, and the same 111-parameter chart. No spatial length is rescaled. The initial aspect ratio remains approximately 4.74483; aspect 5 is an optimization target.

| Quantity | New value |
|---|---:|
| Mean-iota target | 0.2 |
| Aspect-ratio target | 5 |
| Signed B0 target | +5.1 T |
| PHIEDGE | +0.728301974551579 Wb |
| Central pressure | 720979.4853000008 Pa |
| Signed total plasma current | -6133652.330056256 A |

All external coil currents and the vacuum toroidal flux are multiplied by `5.1 / (-0.17506474574437714) = -29.132078982063156`. This intentionally reverses their signs. The four nominal base-coil currents become:

| Base coil index | Current (A) |
|---|---:|
| 0 | 1622643.4476277013 |
| 1 | 1684212.155753562 |
| 2 | 1362991.4687456728 |
| 3 | 1769082.753341028 |

The field factor describes the vacuum starting state, not the final finite-beta equilibrium. Internal `L_cos` and `L_sin` must reverse sign with PHIEDGE because VMEX's lambda normalization uses the magnitude of flux; R/Z coefficients remain identical. The transformed warm-start archive has a separate guess schema and is explicitly ineligible for resume. The original vacuum checkpoint is retained separately and unchanged.

Use the **full profiles and amplitudes** from `examples/data/input.nfp2_QA_finite_beta`: the AM polynomial with PRES_SCALE=1 and the cubic_spline_ip AC_AUX_S/AC_AUX_F current profile with NCURR=1. Pressure and plasma current are copied exactly, without multiplying them by the vacuum coil-current scaling factor. CURTOR is the plasma current, distinct from all external coil currents.

Pressure and plasma current remain fixed during optimization. Beta is measured from the finite-beta equilibrium; the reference case's 2.699% beta is not imposed or assumed here. This is a new physical configuration, not a full similarity scaling of the large reference QA equilibrium.

## Initialization and optimization

`run.py` first attempts one ordinary free-boundary solve at the full reference pressure and plasma current, starting from the correctly transformed vacuum guess. It uses no root polish, no axis reguess, and no same-coil Jacobian retry. Failure stops the run and saves an explicitly uncertified diagnostic state. Success must then pass fresh force and projected-root certification before step zero becomes an accepted finite-beta checkpoint.

`--initialize-only` stops after certification and matched WOUT/coil export, without taking optimizer steps. Without that flag, the driver uses the inherited independently capped QA direction, restoration of violated constraints, and finite six-trial backtracking policy. Only checkpoints with this finite-beta schema, input hashes, contract and explicit positive-B0 target can resume.

The frozen targets are [0.2, 5, 5.1 T], with 1% tolerances [0.002, 0.05, 0.051 T]. B0 is never reset to the measured finite-beta starting value. Keep the 1 mm geometric cap and separate 1% nominal-coil-current cap per accepted step. The QA normalization is established at the newly certified finite-beta initial state; raw QA remains available for comparisons.

Force and edge tolerances are 1e-10, the projected-root gate is 2e-6, and the forward_dense_jax adjoint residual gate is 2e-5. The state tangent retains the existing GCROT implementation. No root polish is used. The default optimizer budget is 10 steps with a one-hour wall limit; generated GPU run logs and machine-specific launch records are excluded from this branch. Runtime wall timers and iteration/trial limits bound execution.

## Files and execution

- `inputs/input.json`: full finite-beta VMEX input at the scaled flux.
- `inputs/coils.json`: unchanged geometry with scaled signed currents.
- `inputs/scaling.json` and `inputs/manifest.json`: scaling provenance and input hashes.
- `inputs/original_vacuum_seed.npz`: preserved source checkpoint, not resumable by this case.
- `inputs/vacuum_warm_start.npz`: transformed, uncertified warm-start guess.
- `prepare_inputs.py`: reproducible input generation; refuses to overwrite an existing inputs directory.
- `run.py`: bounded initialization and optimization driver.
- `validation/`: local checks and their scope.

Use a project environment containing JAX, Solvax, ESSOS, NumPy and the plotting/WOUT dependencies. The VMEX source at this branch's base matches the pinned runtime. From the repository root, with ESSOS installed in the environment or its source directory already on PYTHONPATH:

```sh
case_dir="example/m=3n=3iota=0p2aspect=5finitebeta"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export PYTHONDONTWRITEBYTECODE=1
python -B "$case_dir/run.py" --initialize-only --device cuda:0 \
  --output-dir "$case_dir/runs/initial" --max-wall-hours 1
```

For a fresh initialization followed by up to 10 optimizer steps, omit `--initialize-only`, add `--target-step 10`, and choose a new output directory. Output directories must be inside the case and must not already exist. Resume accepts only this case's certified checkpoints and requires their SHA256; a vacuum guess is never resumable.

The checked-in inputs are ready to load. `prepare_inputs.py` reproduces them using the adjacent authenticated vacuum reference inputs and the repository QA reference deck; it intentionally refuses to overwrite an existing `inputs` directory.

Run the nine regression checks from the repository root:

```sh
JAX_PLATFORMS=cpu python -B -m pytest -c /dev/null \
  --confcutdir="$case_dir/tests" --import-mode=importlib \
  "$case_dir/tests/test_finitebeta_case.py" -q
```

The checks cover exact reference profiles, field/flux/lambda sign scaling, fixed targets, checkpoint compatibility and mocked initialization gates. They do not run a finite-beta equilibrium or optimizer campaign.

These checks prepare the experiment; they do not establish that the finite-beta configuration converges or meets the QA/iota/aspect/B0 targets.
