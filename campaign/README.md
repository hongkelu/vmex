# Old campaign: finite-β single-stage QA, QI and QH (lhk0-7 / lhk3-6, 2026-10-08 … 10-13)

This folder (branch `tt1`) holds the exact environment, inputs, run commands and results of the finite-β
(⟨β⟩ = 2.5%) single-stage campaign. Its code is this repository's history: the 10 commits in `COMMITS.txt`, all
ancestors of this branch.

- **Free arm:** three-term free boundary, B·n = 0, pressure balance and K = 0.
- **Fixed arm:** fixed boundary.
- **Bootstrap current:** Redl for QA and QH, DKX for QI. Resolution 8×8 modes, ns = 51.
- **Optimizer:** SLSQP with hard constraint rows.
- **Main results:**
  - QA with the Mercier floor and vacuum ι ≥ 0.27–0.31: QA 1.9e-4 to 7e-4, alpha loss 0–0.6%, ε_eff ≤ 0.14%;
  - QI A8: 3.8e-3;
  - QH A10: 7.3e-3.
  - See `results/` and `reference/runs/`.

## Layout

| Path | What |
|---|---|
| `COMMITS.txt` | the 10 commits the runs launched from, with time and subject; `setup.sh` checks each out as `code/vmex-<commit>/` |
| `env/` | `requirements.txt` (65 pins, sha256-locked), `freeze-campaign.txt` (the production venv frozen 2026-10-10 00:48 UTC+8, essos 0.19.3), and the wheel-against-installed check. The wheels themselves (3.2 GB, Linux x86_64, Python 3.12) are not in git: see below |
| `inputs/seeds/` | `qa4-beta-free`, `qa4-beta-fixed`, `qi6-beta-free`, `qi6-beta-fixed`, `seed-qa4`, `smoke-c2-qa4`: the `input.run` and `coils.initial.json` that runs started from |
| `inputs/restarts/` | the restart states (`input.run`, `coils.json`, `wout.nc`) of the restarted runs |
| `reference/runs/<run>/` | each original run's `metrics.jsonl` (every step), `input.run`, `coils.initial.json`, last `coils.stepN.json` and `summary.json`. A run's own `input.run` and `coils.initial.json` are its exact seed. |
| `reference/watcher-relaunches-lhk0-7.log` | when each `-cN` continuation was relaunched |
| `results/` | post-processing: ns = 201 dense re-solves and `plot_wout` figures, `--trace` alpha losses, coil + LCFS \|B\| GIFs, max-J scans, ε_eff |
| `postprocess/` | `lcfs_modb_gif.py`, `maxj_scan.py`, `eps_eff.py` |
| `orchestration/` | `continue_until_converged.py` (watcher), `restart_from_latest.py`, `launch.sh` (as used on lhk0-7) |
| `checks/` | `fixed_jaccheck.py` (also the driver of `fixed-qi6-beta-grouped`), `dkx_map_proto.py`: exactness checks of the two speed-ups |
| `compare_metrics.py` | step-by-step comparison of a rerun with its reference |
| `RUNS.md` | **every run's commit, essos version, host, launch time and exact command** |
| `setup.sh`, `run.sh` | offline install, and one run with the campaign's environment |

## Use

You need Linux x86_64, Python 3.12, git, an NVIDIA GPU with a CUDA 12 driver, and about 10 GB of disk. No internet.

The 65 wheels are in the full copy of this folder, `env/wheelhouse/`:
- on Perlmutter: `/pscratch/sd/h/hongkelu/freeboundary-single-stage-old/env/wheelhouse`;
- as a tar: `repro-campaign-20261018.tar` in the workspace.

1. Install. This creates `campaign/venv/` and the checkouts `campaign/code/vmex-<commit>/`:

   ```bash
   WHEELHOUSE=/pscratch/sd/h/hongkelu/freeboundary-single-stage-old/env/wheelhouse PYTHON=python3.12 campaign/setup.sh
   ```

2. Run a row of `RUNS.md`, for example the vacuum-floor 0.31 run:

   ```bash
   CUDA_VISIBLE_DEVICES=0 campaign/run.sh 91729930 F qa4-beta-merc --steps 1000 --seed-run seeds/qa4-beta-free --vacuum-iota-floor 0.31 --iota-ceiling 0.495 --output prod/free-qa4-beta-merc-vac0.31
   ```

3. Compare with the original:

   ```bash
   campaign/venv/bin/python campaign/compare_metrics.py campaign/code/vmex-91729930/prod/free-qa4-beta-merc-vac0.31/metrics.jsonl campaign/reference/runs/free-qa4-beta-merc-vac0.31/metrics.jsonl
   ```

**Use the original seed.** A run whose command has no `--seed-run` or `--restart` built its seed on the GPU: a β ramp,
the bootstrap current and, for the free arm, a 3000-iteration coil refit. That build is not bit-for-bit repeatable. Add
`--seed-run <repo>/campaign/reference/runs/<run>` to start from the exact original seed. Exception: the fixed-arm driver has no
`--seed-run` before fa3982a6, but its seed build did reproduce exactly.

On NERSC Perlmutter, run inside a GPU allocation (`salloc -q interactive -A m4656_g -C gpu ...`). Use a Python 3.12
(conda) to create the venv. The wheels bring their own CUDA 12 libraries, so no module is needed for JAX.

## Reproducibility, tested on lhk0-7 (RTX 5090), 2026-10-19

The offline install passed: every file matches the hash manifest, all 65 pins installed with hash checks, and
`pip check` is clean.

| Run (commit) | Steps | Agreement with the original `metrics.jsonl` |
|---|---|---|
| free QA baseline (50b6d527) | 0–3 | QA identical to 10–11 digits |
| free QA, Mercier, vacuum ι ≥ 0.31 (91729930) | 0–3 | 7e-9 by step 3 |
| fixed QA, Mercier (f749d3eb), own seed build | 0–3 | identical to 10 digits |
| free QA w = 1, restart (e3313c0f; originally essos 0.20.0) | 0–2 | identical to 10 digits |
| free QI A8 (0cc38def), own seed build | 0–1 | 5e-6 at step 0: the seed build differs |
| free QI A8 (0cc38def), `--seed-run` the original seed | 0–1 | identical at step 0, 4e-8 at step 1 |

### 12 steps, free boundary, from the original seed or restart

| Run | Through step 12 |
|---|---|
| QA baseline (50b6d527) | QA within ~1e-10 at every step |
| QA w = 1, restart (e3313c0f) | QA identical to 10 digits at every step |
| QA, Mercier, vacuum ι ≥ 0.31 (91729930) | ~1e-8 through step 9, then drifts to 1.5e-5 at step 12 |
| QI A8 (0cc38def) | identical at step 0, 2e-7 at step 1, 2% at step 3, ~20% by step 12 |
| QH A10 (0cc38def) | identical at step 0, 0.4% at step 1, then a different path (step 10: 0.193 vs 0.228) |

GPU arithmetic is not deterministic run to run: two seeded QI A8 reruns on the same GPU model differed at step 1.

- **QA:** it starts near its optimum, so it replays closely.
- **QI and QH:** they start with the target near 1 and take large unscaled SLSQP steps, which turn round-off into a
  different path within a few steps.
- **How to compare:** match long runs, and every QI or QH run, by their endpoints.
- **For bit-for-bit replay:** future runs would need XLA's `--xla_gpu_deterministic_ops=true`, which is untested here.

## Caveats (details in `RUNS.md`)

- **essos version:** two late launches and most `-cN` continuations ran essos 0.20.0. This package pins 0.19.3, the
  version before the upgrade.
- **Late continuations:** six of them, from 2026-10-12 12:20 UTC+8 on, ran newer code from another session's commits.
  Rerun them from e3313c0f instead.
- **Missing lhk3-6 data:** the `-m002` restart states of the lhk3-6 runs (vacuum floor 0.27 and 0.29) and those runs'
  histories are missing, because lhk3-6 is offline. Their post-processing is in `results/qa4-variants/`.
- **Thinned WOUTs on lhk0-7:** only every 5th step's `wout.stepN.nc` is still there. Every step's `metrics.jsonl` row and
  coils are here.
