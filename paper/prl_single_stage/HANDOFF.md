# Handoff: free-boundary single-stage PRL (state at 2026-10-10 18:45 PDT)

Project root: `/pscratch/sd/h/hongkelu/freeboundary-single-stage/` (everything for this project lives here).
Paper working dir: `paper/`. Code: `vmex/` on branch **`publication`** (pushed to `origin` = hongkelu/vmex).
Long-form notes with every decision and gotcha: `~/.claude/projects/-pscratch-sd-h-hongkelu-freeboundary-single-stage/memory/prl-finite-beta-single-stage.md` (read it first; it is chronological).

## 1. The paper in one paragraph

PRL on free-boundary single-stage stellarator optimization with VMEX (JAX/GPU VMEC): three-term free-boundary condition (B·n = 0, pressure balance, sheet current K = 0 via virtual casing), SLSQP with hard constraints and exact implicit (adjoint) gradients, coils in ESSOS. Claims: (i) at finite β (⟨β⟩ = 2.5 %, self-consistent bootstrap) the free-boundary arm optimizes the equilibrium the coils actually make, while fixed-boundary single stage and stage two do not; (ii) on the vacuum LP QA benchmark (nfp 2, A 6) it traces a better QS-vs-coil-length front than published stage-two sets (Gil et al. PRL 2026 / PRE 114 025202; Wechsung et al. PNAS 2022) under *their* constraints.

Figure plan (user-approved): Fig 1 three methods at 2.5 % β (coils, actual vs target LCFS, B·n/K maps) — blocked on stage-two column; Fig 2 scored vs actual QS per step, fixed vs free arm (`analysis/fig2_pub.py`, done); Fig 3 **Pareto front QS vs coil length** (`analysis/fig_pareto.py`, layout done, data arriving); Fig 4 A4 QA showcase (needs Boozer plot). Table I three methods × {QA, QI}; Table II Gil format (`analysis/table2_gil_format.py`). Style: matplotlib default colours (C0 ours, C1 Gil/fixed arm, C2 Wechsung), publication style, legends off the data.

## 2. Hard operating rules (from the user)

- GPU only through **one** interactive allocation at a time: `salloc -q interactive --constraint=gpu -A m4656_g --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 ...  -t 04:00:00` (16 GPUs, 4 h). Never run GPU work on a login node (`JAX_PLATFORMS=cpu` for any login-node Python).
- Python env: `~/.conda/envs/uwplasma-env/bin/python` (`paper/env.sh` sets `$PY`, x64, JAX cache on scratch). SIMSOPT-for-QFM: `~/.conda/envs/simsopt-env`.
- Every comparison must be fair (same constraints, same evaluator) — the user cares most about coil constraints.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push only to `origin` hongkelu/vmex, branch `publication`.

## 3. How the GPU pipeline runs (automatic)

- `paper/chain_next.sh <jobid> <jobs.txt> <tag>`: waits for the allocation to end, builds restarts (`make_batch_restart.py`, picks the highest `-cN` continuation), requests the next 4-node allocation and runs `run_batch2.sh jobs_<tag>.txt` (one run per GPU via `srun --jobid --exact`). Logs: `runs/chain_<tag>.log`, `runs/<tag>_salloc.log`.
- Extras (scoring, stage two, sweeps) are injected into a running allocation by `extras_batchN.sh` → `launch_jobs.sh <jobid> <jobs.txt>` and `srun --jobid=<id> --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0 ...`.
- **Current**: allocation 7 = job **59651566**, ends **~20:59 PDT**. Armed (setsid, survives session restarts): `chain_next.sh 59651566 jobs_batch7.txt batch8` and `extras_batch8.sh` (fires after the batch-8 launches).
- Before arming any chain: `ps -u $USER -o args= | grep chain_next` and `squeue -u $USER` — **never two chains** (a duplicate once produced a second allocation). Launch helpers with `setsid nohup ... & disown`. Never `scancel --name=python`; cancel single steps with `scancel <jobid>.<step>`. Kill helpers by exact argv via `ps -o pid=,args= | awk`, not `pkill -f` (it matched its own shell).
- Never rewrite a jobs file that a running `run_batch2.sh` is reading; edit the *previous* list before the allocation ends.
- Driver stdout is block-buffered; judge progress from `runs/batch/<run>/metrics.jsonl`.

## 4. The fixed constraint set (decided with the user) — vacuum LP QA Pareto scan

Only coil length varies. Reactor scale (R0 = 10.1266 m); driver works at R0 = 1 m.
| item | value |
|---|---|
| plasma pinned | R0 1.0099 ± 1 mm, A 6.00–6.01, min ι ≥ 0.4152 (target 0.4158) |
| coils | 4 per half period, Fourier order 16, free currents |
| curvature / MSC | ≤ 0.5 /m, ≤ 0.05 /m² (Gil's nominal = Wechsung's 5 /m, 5 /m² at R0 = 1) |
| coil–coil / coil–surface | ≥ 1.0 m / ≥ 1.5 m |
| force | ≤ 0.72 MN/m per turn (loosest published value under one convention; all published 4-coil sets feasible) |
| margins | 0; feasibility checked on final coils |
| evaluator | three-term free-boundary equilibrium of the coils, 12×12 modes, VC grid 64, two-term QS on s = 0.1…1.0 |
| second metric (Table II only) | Gil's QFM pipeline, same code on both sides (`score_run_gil.sh`) |

**Force convention (reproduces Gil's Table 2 to ≤ 2 %)**: SIMSOPT `coil_force`, conductor radius **a = 0.3 m** (his `config_verification.py`, not the 0.15 in his optimizer), his archive field scale (mean |B| on target 5.39 T; Wechsung's archive is at 4.93 T), max over base coils / 200 turns (800/3 for 3 coils). Conversion driver → reactor: F_reactor = 287.3 × F_driver. Driver force row: `FORCE_LIMIT` (N/m at driver scale), `CONDUCTOR_RADIUS = 0.3/10.1266`, validated against SIMSOPT to 1 %.

Pareto runs: `jobs_pareto.txt` (`pareto-L16 … L24`, overrides incl. `FORCE_LIMIT 5.012e5`, `--modes 12 12 --vc-grid 64`), plus `jobs_pareto_extra.txt` (`pareto-L20.09`, `pareto-L23.66` = Gil's exact 203.4 / 239.6 m), started in batch 8.

## 5. Results so far (QS = two-term error of the actual equilibrium)

Published sets at 12×12: Gil L18 1.64e-3, L20 1.88e-4, L24 1.00e-5, 3-coil 9.7e-4; Wechsung L18 1.56e-3, L20 1.69e-4, L24 3.76e-6; target 8.95e-7.

Pareto scan at 18:43 (allocation 7, ~100 steps, driver value at 12×12):
| budget (m, R0 = 1) | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|
| QS | 8.8e-3* | 1.6e-3* | 4.2e-4 | 3.5e-4* | 6.8e-5 | 6.0e-5 | 1.3e-4* | 2.0e-6 |
`*` = still recovering from the length cut (started from 18/20/24 m coils); expect monotone front after batch 8.

Earlier fair twins (same set, own-set thresholds, no force row): L18 4.2e-4, L20 6.7e-5, L24 2.0e-6, 3-coil 1.2e-4 (force 1.16 > limit). Gil-pipeline on fair twins vs published (same code): L18 1.7×, L20 1.5×, L24 ≈1×, 3-coil 6.8× better.

Finite β (A4 QA, 2.5 %, bootstrap): free arm actual 2.5e-4 (scored ≈ actual, LCFS in place, coils feasible, 0/1000 alphas lost in 0.2 s); fixed arm scores 1.2e-4 but actual ≈ 1.08× scored with a 4 mm LCFS offset and K residual 20× larger, and its coils violate curvature by 1–4 %; pure stage one (`qa4-stage1-c4`) 4.2e-4 and still improving → its stage-two column is the missing Fig 1 / Table I number (pipeline `stage_two_fit.py --slsqp` validated: actual/seen 1.04–1.11). A6: stage two on LBD22 boundary 5.5e-3 actual vs our free arm 1.8e-3. QI proof of concept: free 2.3e-2, fixed 7.7e-2 (paused).

## 6. Published data on disk

- `references/gil2026/` — Gil archive (`zenodo_repository/qa_comparison_coils`, scripts), `coils_npz/*.npz` + `summary.json` (numpy rebuild; SIMSOPT 1.11 can't load his JSON).
- `references/wechsung2022/` — full Wechsung archive (Zenodo 5975323, LFS `archive_lfs.zip`): **32 QA sets** = 8 starts × {18, 20, 22, 24} m in `archive/output/*/xmin.txt` (layout: I1..I3, then 4×99 dofs). Converted: `paper/coils/wech_cloud_L<L>_ig<k>.json` + `wech_cloud_index.json`. 12×12 evaluation queued in batch 8.
- `references/hurwitz2024/` — Hurwitz/Landreman/Kaptanoglu 2024 scan (Zenodo 13913510), `output.zip` 17 GB **downloading** (`download.log` prints `hurwitz-done`). 5 coils/hfp, ~25 m/hfp, random thresholds: a force-vs-B·n scan at ~fixed length → use as a filtered cloud near 24–25 m, flagged "5 coils". Kaptanoglu 2025 dipole archive (Zenodo 14934092) returned 404.
- `references/lbd2022/` — Landreman–Buller–Drevlak 2022 QA A6 β 2.5 % (deck scaled: `paper/decks/input.lbd_qa6_R1_8x8`).

## 7. Key scripts (all in `paper/`)

`eval_coilsets_vacuum.py` (fixed coils → three-term equilibrium, `--modes/--vc-grid`), `eval_sweep_batch4.sh` (12×12 sweep), `eval_steps_freeboundary.py` (scored vs actual per step), `score_run_gil.sh` + `gil_postprocess.py` (QFM seeded from own LCFS, marker "own LCFS (v3)") + `gil_coil_metrics.py` (a = 0.3, turns 200 / 800÷3) + `export_simsopt_coils.py` (rescales to 5.39 T), `stage_two_fit.py` (penalty + `--slsqp` hard rows), `convert_coilsets.py`, `convert_wechsung_cloud.py`, `analysis/fig_pareto.py`, `analysis/fig2_pub.py`, `analysis/fig3_pub.py`, `analysis/table2_gil_format.py`, `analysis/benchmark_summary.py`, `analysis/collect_runs.py`.

Driver patches (in `vmex/examples/optimization/`): `lpqa-*` COIL_CASE, `TOTAL_LENGTH_LIMIT`, `COIL_CASE_OVERRIDES` env (two-pass in the fixed-boundary driver), force row (`coil_forces`, `FORCE_LIMIT`, `CONDUCTOR_RADIUS`), `SEED_NITER`; `vmex/core/optimize.py` device fix.

## 7b. Session 2026-10-10 19:00–19:10 (changes since the 18:45 snapshot)

- `vac-3coil-fair-c2` was stuck at step 3 for 90 min (coil slack −0.48: its c1 coils were optimized without the force row and violate 110 MN/m by ~30 %, SLSQP never restored feasibility). Cancelled step `59651566.41`; launched **`vac-3coil-fair2`** at 19:01 (same overrides, 12×12, force row, fresh start = Gil's published 3-coil coils, which sit at the force limit → feasible start). `jobs_batch7.txt` line swapped to `vac-3coil-fair2|…|-` so the batch-8 chain continues it; `jobs_3coil_fair2.txt`, `runs/launch_3coil_fair2.log`.
- `extras_batch8.sh`: Gil-scoring loop globbed `runs/published/*` (picked up the `.gilscore.log` files → `X.gilscore.log.gilscore.log` junk); now `runs/published/*/`; scores `vac-3coil-fair2` instead of `vac-3coil-fair`. The armed helper (pid 1045957) was left running — it was still in its top `until` loop and the edit only changed later lines, so it reads the fixed block when it fires.
- `eval_sweep_batch4.sh`: added the Pareto points (`ours_pareto-L16…L24`, `L20.09`, `L23.66`) and `ours_vac-3coil-fair2` (latest continuation, always re-evaluated) — run it after batch 8 for the filled Fig 3 markers.
- `analysis/fig_pareto.py`: 3-coil point from `vac-3coil-fair2` (falls back to `vac-3coil-fair`); force threshold 0.72 MN/m (Gil convention) in THR and title.
- **Fig 4 draft done**: `analysis/fig4_pub.py` → `analysis/fig4.{pdf,png}` (CPU, `JAX_PLATFORMS=cpu`, ~10 s; `--run`, `--step`, `--alpha` args). Panels: (a) coils + LCFS |B| 3D, (b) |B| in Boozer coordinates at s = 0.5 (booz_xform_jax, mboz 24 / nboz 12), (c) Boozer spectrum vs s, (d) ι and ⟨J·B⟩. Non-symmetric Boozer rms/B00: 1.4e-3 (s 0.25), 6.5e-4 (s 0.5), 2.0e-3 (s 0.98). Re-run on the final free-arm step before submission.
- Hurwitz `output.zip`: 9.6 of 17 GB at 19:07 (~2.5 MB/s → done ~20:00).

## 7c. Session 2026-10-10 19:20–19:55: the showcase must be MHD-stable (user decision)

- Finding: the A4 showcase (`qa4-free-vac029-s1-c5`) is Mercier-unstable on 96 % of surfaces (DMerc −7 → −0.5) and infinite-n ballooning-unstable on s = 0.5–0.9 (λ up to +6e-3); the fixed arm, the pure stage one and even the rotating-ellipse seed are unstable too — **neither paper driver had any stability row**. Certificate script: `analysis/stability_certificate.py <run> [--step N]` (fixed-boundary re-solve of the step, DMerc + ballooning; writes `<run>/stability.stepN.json`). No finite-n code exists in vmex.
- User decision: Fig 4 / Table I showcase = our own A4-like configuration with coil constraints AND Mercier + ballooning stability (not a comparison with other coil papers; the Pareto benchmark stays vacuum QS-vs-length).
- Driver patch (T driver): optional hard rows `MERCIER_FLOOR` (+`MERCIER_MARGIN`, `STABILITY_MIN_S`, smooth min of PHIEDGE²·DMerc, `STABILITY_TEMPERATURE`) and `BALLOONING_LIMIT` (+`BALLOONING_MARGIN`, `BALLOONING_S`, `BALLOONING_ZETA0`, `BALLOONING_LINES`; hard max of λ via `vmex.core.stability.ballooning_growth_rate`); logged as `mercier_min` / `ballooning_max` in metrics.jsonl. Off by default. Tested under jit + grad on the step-59 equilibrium (CPU) before launch.
- Run **`qa4-free-stable`** (launched 19:53 in alloc 7, on the batch-8 list): restart of the free arm step 63 with `MERCIER_FLOOR 0 + margin 0.002`, `BALLOONING_LIMIT 0 − margin 2e-4`, VACUUM_IOTA_FLOOR 0.29, 8×8 modes, step scale 0.2 (`jobs_qa4_stable.txt`). Watch the first steps: the start violates both rows (feasibility restoration). Plan B if it thrashes: fixed-boundary driver with the same rows first (needs the same patch in the F driver — not done yet), then the free arm from that boundary. Then refine at 10×10 modes.
- Alpha losses, free arm step 59: s = 0.25 0/1000, **s = 0.5 13/1000 (1.3 %)**, 0.2 s (`runs/alpha/qa4-free-s05`).
- Coil-shape comparison figure: `analysis/fig_coils_compare.py` (ours vs Gil vs Wechsung at 18/20/24 m; shapes nearly identical, ours sit on the κ/MSC/cc limits).

## 8. Next steps

1. After batch 8 starts (~21:00): confirm 11 chain launches + extras (`runs/extras_batch8.log`, `runs/launch_pareto_extra.log`). Check `vac-3coil-fair-c2` — it sat at step 3 for over an hour in allocation 7 (force row binding hard; inspect its log).
2. When the Pareto points converge: rescore all at 12×12 (add `ours_pareto-L*` to `eval_sweep_batch4.sh`), Gil-pipeline v3 scoring, re-render `analysis/fig_pareto.py` (it already plots the Wechsung cloud and the two exact-length points), send to the user, push to `publication`.
3. When `hurwitz-done`: inspect `listing.txt`, convert 5-coil sets, filter by the fixed thresholds, evaluate the survivors at 12×12, add as a flagged cloud.
4. Stage two on the converged pure stage one → Fig 1 and Table I; Boozer plot for Fig 4.
5. Re-arm the chain each allocation (one chain only) until the front and Fig 1 are done.
