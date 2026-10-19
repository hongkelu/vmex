# Every production run: code, environment and exact command

The runs did not record their own command line or code version. Everything below is reconstructed from the launch
commands in the session transcript, the watcher logs (`reference/watcher-relaunches-lhk0-7.log`, server time UTC+8)
and the commit times (`code/COMMITS.txt`, UTC−5). A run's code is the last commit before its launch.

## Common to every run

- **Command:** `python <driver> --beta 0.025 --bootstrap --modes 8 8 --ns 51 --save-every 1 <run options> --output <dir>`
  - driver F = `examples/optimization/single_stage_free_boundary_optimization_three_term.py` (free boundary, three-term)
  - driver X = `examples/optimization/single_stage_optimization_coil_constraints.py` (fixed boundary)
- **Environment:**
  `COIL_CASE=<case> CUDA_VISIBLE_DEVICES=<one empty GPU> XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH=<code dir>`
  (`run.sh` sets all of these). The XLA autotune and CUDA-graph flags are set inside the drivers.
- **Hardware:** lhk0-7 is 8× RTX 5090; lhk3-6 is 4× RTX 4090 D. NVIDIA driver 580.76.05, Python 3.12.3, JAX 0.10.2
  with the CUDA 12 plugin.
- **essos:** 0.19.3 for every launch before 2026-10-09 16:48 UTC. Two launches and most continuations came after that
  and ran essos 0.20.0; they are marked ¹ below.

## Launches

Paths such as `prod/…-from-N`, `seeds/…` and `runs/seed-qa4` are in `inputs/seeds/` and `inputs/restarts/` and are linked into each code checkout
by `setup.sh`.

| Run (output dir) | Host | Launched (UTC) | Code | Arm | COIL_CASE | Run options |
|---|---|---|---|---|---|---|
| `prod/free-qa4-beta` | lhk0-7 (vmex-opt-c) | 10-08 07:09 | bf07460b² | F | qa4-beta | `--steps 300 --seed-run runs/smoke-c2-qa4` |
| `prod/free-qi6-beta` | lhk0-7 (vmex-opt-c) | 10-08 07:09 | bf07460b² | F | qi6-beta | `--steps 300` |
| `prod/fixed-qa4-beta` | lhk0-7 (vmex-opt-c) | 10-08 07:09 | bf07460b² | X | qa4-beta | `--steps 300` |
| `prod/fixed-qi6-beta` | lhk0-7 (vmex-opt-c) | 10-08 07:09 | bf07460b² | X | qi6-beta | `--steps 300` (stopped at 50 and continued as `-grouped`) |
| `prod/free-qa4-beta-merc` | lhk0-7 | 10-08 18:16 | 36fa6a27 | F | qa4-beta-merc | `--steps 300 --seed-run runs/seed-qa4` |
| `prod/free-qa4-beta-jbs` | lhk0-7 | 10-08 18:16 | 36fa6a27 | F | qa4-beta-jbs | `--steps 300 --seed-run runs/seed-qa4` |
| `prod/fixed-qa4-beta-merc` | lhk0-7 | 10-08 18:16 | 36fa6a27 | X | qa4-beta-merc | `--steps 300` |
| `prod/fixed-qa4-beta-jbs` | lhk0-7 | 10-08 18:16 | 36fa6a27 | X | qa4-beta-jbs | `--steps 300` |
| `prod/free-qa4-beta-jbs-w0.1` | lhk0-7 | 10-08 20:52 | 89fb7419 | F | qa4-beta-jbs | `--steps 300 --seed-run runs/seed-qa4 --current-iota-weight 0.1` |
| `prod/free-qa4-beta-jbs-w1` | lhk0-7 | 10-08 20:52 | 89fb7419 | F | qa4-beta-jbs | `--steps 300 --seed-run runs/seed-qa4 --current-iota-weight 1.0` |
| `prod/free-qa4-beta-merc-s02` | lhk0-7 | 10-08 21:01 | 0040364e | F | qa4-beta-merc | `--steps 263 --restart prod/free-qa4-beta-merc.s02-restart-from-37` |
| `prod/fixed-qa4-beta-merc-s02` | lhk0-7 | 10-08 21:01 | 0040364e | X | qa4-beta-merc | `--steps 270 --restart prod/fixed-qa4-beta-merc.s02-restart-from-30` |
| `prod/free-qa4-beta-merc-s02-init` | lhk0-7 | 10-08 21:12 | 0040364e | F | qa4-beta-merc | `--steps 300 --seed-run runs/seed-qa4` |
| `prod/fixed-qa4-beta-merc-s02-init` | lhk0-7 | 10-08 21:12 | 0040364e | X | qa4-beta-merc | `--steps 300` |
| `prod/fixed-qi6-beta-grouped` | lhk0-7 | 10-09 00:10 | 3509a571³ | `campaign/checks/fixed_jaccheck.py` as driver | qi6-beta | `--steps 300 --restart prod/fixed-qi6-beta.restart-from-50` |
| `prod/free-qi6-beta-scale1` | lhk0-7 | 10-09 00:10 | 3509a571 | F | qi6-beta | `--steps 300 --design-step-scale 1.0` |
| `prod/free-qi6-beta-scale1-map` | lhk3-6 | 10-09 01:53 | e2c6fdcc | F | qi6-beta | `--steps 300 --design-step-scale 1.0 --seed-run seeds/qi6-beta-free` |
| `prod/free-qa4-beta-merc-vac0.27` | lhk3-6 | 10-09 04:41 | 8210f2b9 | F | qa4-beta-merc | `--steps 1000 --seed-run seeds/qa4-beta-free --vacuum-iota-floor 0.27 --iota-ceiling 0.495` |
| `prod/free-qa4-beta-merc-vac0.29` | lhk3-6 | 10-09 04:41 | 8210f2b9 | F | qa4-beta-merc | same with `--vacuum-iota-floor 0.29` |
| `prod/free-qi6-beta-A8` | lhk0-7 | 10-09 05:15 | 38e45b3e | F | qi6-beta-A8 | `--steps 500 --design-step-scale 1.0` |
| `prod/free-qh4-beta-A8` | lhk0-7 | 10-09 05:15 | 38e45b3e | F | qh4-beta-A8 | `--steps 500 --design-step-scale 1.0` |
| `prod/free-qi6-beta-A10` (seed build crashed) | lhk0-7 | 10-09 05:47 | 38e45b3e | F | qi6-beta-A10 | `--steps 500 --design-step-scale 1.0 --max-iterations 20000` |
| `prod/free-qh4-beta-A10` | lhk0-7 | 10-09 13:08 | 38e45b3e | F | qh4-beta-A10 | `--steps 500 --design-step-scale 1.0 --max-iterations 20000` |
| `prod/fixed-qa4-beta-merc-s02-init-m002` | lhk0-7 | 10-09 13:49 | aa36682e | X | qa4-beta-merc | `--steps 300 --restart prod/fixed-qa4-beta-merc-s02-init.m002-from-215` |
| `prod/free-qa4-beta-merc-vac0.27-m002` | lhk3-6 | 10-09 13:49 | aa36682e | F | qa4-beta-merc | vac0.27 options + `--restart prod/free-qa4-beta-merc-vac0.27.m002-from-225`⁴ |
| `prod/free-qa4-beta-merc-vac0.29-m002` | lhk3-6 | 10-09 13:49 | aa36682e | F | qa4-beta-merc | vac0.29 options + `--restart prod/free-qa4-beta-merc-vac0.29.m002-from-204`⁴ |
| `prod/free-qa4-beta-merc-vac0.31` | lhk0-7 | 10-09 16:19 | aa36682e | F | qa4-beta-merc | `--steps 1000 --seed-run seeds/qa4-beta-free --vacuum-iota-floor 0.31 --iota-ceiling 0.495` |
| `prod/fixed-qa4-beta-merc-vac0.27` ¹ | lhk3-6 | 10-09 19:14 | aa36682e | X | qa4-beta-merc | `--steps 1000 --seed-run seeds/qa4-beta-fixed --vacuum-iota-floor 0.27 --iota-ceiling 0.495` |
| `prod/free-qa4-beta-jbs-w1-A3` ¹ | lhk0-7 | 10-09 19:36 | 9875b6b4 | F | qa4-beta-jbs | `--steps 1000 --seed-run runs/seed-qa4 --current-iota-weight 1.0 --aspect-range 3.0 4.5 --restart prod/free-qa4-beta-jbs-w1-c1.A3-from-132` |

### Footnotes

1. Launched after the essos 0.20.0 upgrade (10-09 16:48 UTC).
2. The `vmex-opt-c` tree was uploaded files, not a checkout. bf07460b is the last commit before the launch.
3. `fixed-qi6-beta-grouped` ran `campaign/checks/fixed_jaccheck.py`, copied to `examples/optimization/fixed_jaccheck_tmp.py`.
   It is the fixed driver of 3509a571 plus a one-off check of the grouped Jacobian against the per-row one at step 0.
   Run it from `code/vmex-3509a571` after that copy. `run.sh` does the copy when given `fixed_jaccheck_tmp`.
4. The `-m002` restart directories of the lhk3-6 runs are not in `inputs/restarts/`: lhk3-6 has been offline since 2026-10-12.
   They are the step-225 and step-204 states of the original vac0.27 and vac0.29 runs, so rerun those first.

## Continuations

The watcher `campaign/orchestration/continue_until_converged.py` relaunched a run that stopped at its step budget. It reused the same
command plus `--restart <previous run dir> --output <run>-cN`.

| Run | Relaunched (server time, UTC+8) | Code | essos |
|---|---|---|---|
| `free-qa4-beta-jbs-w1-c1` | 10-09 17:33 | 38e45b3e | 0.19.3 |
| `free-qi6-beta-scale1-c1` | 10-09 22:53 | aa36682e | 0.19.3 |
| the `-c1`/`-c2` relaunches between 10-10 04:20 and 10-12 03:38 | per the watcher log | 9875b6b4 | 0.20.0 |
| `free-qh4-beta-A10-c3`, `free-qi6-beta-A8-c4`, `free-qa4-beta-jbs-w1-A3-c1`, `free-qa4-beta-merc-vac0.31-c2`, `free-qi6-beta-A8-c5`, `free-qh4-beta-A10-c4` | 10-12 17:10 or later | newer than 9875b6b4⁵ | 0.20.0 |

5. From 10-12 12:20 server time, another session uploaded commits 6b592f79…4acf710b (ballooning and AGNI rows, off unless
   the case has `-ball`/`-agni`) into the same tree. One of them, 35516a32, batches the free arm's field-term pullback with
   the same gradients. These six continuations ran whichever of those was in place. To reproduce them from 9875b6b4,
   restart from the previous continuation's final state.

## What to expect when reproducing

- **Early steps:** starting from the same seed or restart state, with the same code and packages, the target residual
  agreed to 9–11 digits over the first 2–3 steps on lhk0-7 (see the test table in `README.md`). Check it with
  `campaign/compare_metrics.py`.
- **Runs that built their own seed** (no `--seed-run` or `--restart` above): add
  `--seed-run <repo>/campaign/reference/runs/<run>` to start from the exact original seed. Without it, the GPU seed build
  differs slightly; for QI A8 that was 5e-6 at step 0.
  - Exception: the fixed-arm driver has no `--seed-run` before e2c6fdcc, so the fixed runs launched from bf07460b,
    36fa6a27 and 0040364e cannot use it.
  - Their seed build reproduced exactly in the test (fixed Mercier QA, 0040364e).
- **Later steps:** GPU reductions are not bitwise reproducible across GPU models, driver versions or batch sizes. SLSQP
  amplifies round-off, so trajectories drift apart after tens to hundreds of steps. Compare the endpoints (QA/QI/QH,
  constraints), not the step-by-step path.
- **essos:** the runs marked ¹ and the late continuations used essos 0.20.0. This package pins 0.19.3, the version in
  use before the upgrade. Coil fields can differ slightly between the two.
