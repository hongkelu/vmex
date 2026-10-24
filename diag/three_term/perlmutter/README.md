# Perlmutter session 2026-10-22: three-term free boundary follow-up to #121

Everything this session measured, with the means to repeat it. The code changes are in this PR's library commits
(`vmex/core/bootstrap.py`, `vmex/core/freeboundary_vc.py`, the driver); results and their interpretation are in
[RESULTS.md](RESULTS.md); the reports are posted on #121.

## Layout

| path | what |
|---|---|
| `RESULTS.md` | every result, with the numbers and where they come from |
| `fixtures/<case>/` | frozen seeds: `input.run`, `coils.initial.json`, `SHA256SUMS`, `PROVENANCE.md` (qh25, qa25, qi25, qavac, qhvac, qivac) |
| `raw/qh_*.txt` | `[cert]`, `[trajectory]`, `[calibrate]`, `[lm]`, `[step]` lines of every QH probe run |
| `raw/qh_trajectories.md` | per-round tables of those runs (cost, B.n, K, loss, values change, GN gain, condition, normal step) |
| `raw/interior.txt` | interior-gauge probe output |
| `raw/g_convention_qh_seed.json` | G vs G + N I on the QH seed (per-surface data) |
| `raw/fixture_step0.txt` | step-0 metrics of every fixture build |
| `scripts/` | launchers and probes (paths are this session's Perlmutter paths; adjust `P=`/`R=`) |

## Scripts

- `onnode.sh JOB CMDFILE [FIRST_GPU]`, with `NODE=nid...` to pick the node. Runs each line `<log> <command>` of CMDFILE
  on its own GPU (`CUDA_VISIBLE_DEVICES`) inside one `srun --gpus=4` step, with `XLA_PYTHON_CLIENT_PREALLOCATE=true`.
  Do **not** start several `srun --overlap --gpus=1` steps side by side: they can land on the same GPU.
- `launch_round.sh JOB TAG`: the QH calibration runs plus the 8-way test suite at the head of the work tree.
- `gputests.sh`, `qhrun.sh`: the first, simpler launchers (superseded by `onnode.sh`).
- `qhprobe.py OUT <driver args>`: runs the single-stage driver with the probes, all opt-in through env variables:
  - always on: every Levenberg-Marquardt finish verbose (`[lm]`); every certificate dumped to `OUT/cert_NNN.npz`
    (`J`, `r`, `[cert]` line); the latest linearized model coordinates saved to `OUT/last_x.npy`;
  - `TRAJECTORY=1`: the loss, constraint values, B.n and K at every certified polish round (`[trajectory]`);
  - `CALIBRATE=1`: at the certified seed, the outputs after half and the full remaining Gauss-Newton step (`[calibrate]`);
  - `POLISH_ROUNDS`, `POLISH_MAX_NFEV`, `STALL_STEPS`, `NORMAL_TOL`, `MAX_MODE`, `START_X` (start the first boundary
    solve at a saved `last_x.npy`).
- `analyze.py cert.npz`: where a certificate's predicted gain sits (row blocks, current vs boundary, weakest
  singular directions).
- `geometry.py input.run cert.npz`: normal vs tangential share of the weak directions and of the GN step.
- `summarize.py LOG...`: one line per run from `[trajectory]`/`[cert]` lines.
- `gconv.py input.run helicity n0 T0`: G vs G + N I on a solved deck.
- `interior/`: interior-gauge probes on the small LP QA test deck.
  - `probe.py`: the model's interface rows from three starts; env `MPOL`, `NS`, `FTOL`, `KEEP_M1`.
  - `lam0.py`: plain fixed-boundary solves, cold vs perturbed start, with and without `lambda_m0_floor`, plus surface
    distances.
  - `valley.py`: the force residual along the chord between the two converged states.
  - `gauge.py`: the m = 0 relabelling decomposition; `KEEP_M1`.
  - `repeat.py`: GPU determinism of repeat cold solves.
- `keepm1plugin.py`: pytest plugin that never releases the constrained m = 1 Z force (`-p keepm1plugin`).

## Reproducing a QH fixture run

```bash
COIL_CASE=qh4-beta-A8-merc TRAJECTORY=1 POLISH_ROUNDS=12 PYTHONPATH=<tree> python -u scripts/qhprobe.py OUT \
  --modes 8 8 --ns 51 --coil-point-rows --beta 0.025 --bootstrap --design-step-scale 1.0 --max-iterations 20000 \
  --seed-run fixtures/qh25 --steps 0 --labelling equal_arc --label-weight 1e-3 --output OUT-run
```

Run from a tree with the driver (`examples/optimization/single_stage_free_boundary_optimization_three_term.py`) and
`examples/data` (the 2.5% QA/QI cases also need `ESSOS_coils_qa4_beta.json` and `ESSOS_coils_qi6_beta.json`, removed
in 83d2f426: `git show 83d2f426^:examples/data/<file>`). Always start from a fixture (`--seed-run`): the GPU coil fit
is not deterministic.

## Environment

uwplasma-env upgraded 2026-10-22 to the latest releases: jax/jaxlib 0.12.0.1 with CUDA 13, solvax 0.31.0,
booz_xform_jax 0.5.0, numpy 2.5.4, netCDF4 1.7.5, neo-jax 1.1.0, dkx 2.11.0 (previous freeze:
`paper/env-freeze-2026-10-22-before-upgrade.txt`). The 5090 host had jax 0.12.0/CUDA 12 and solvax 0.29.0.
On the 40 GB A100s the larger three-term tests need `XLA_PYTHON_CLIENT_PREALLOCATE=true`. Otherwise they fail with
"All configs failed during profiling" (an out-of-memory error in XLA autotuning). The fresh-process checkpoint test
needs it off, because its child process cannot get GPU memory otherwise.
