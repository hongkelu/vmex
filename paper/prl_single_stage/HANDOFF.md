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
- 20:00 relaunch with `STABILITY_MIN_S 0.2` (user: Mercier is inaccurate near the axis; vmex's finite-β lane uses 0.2, VMEC2000 drops the first surfaces). Step 0: QS 2.50e-4, DMerc_min(PHIEDGE²) −0.123, λ_max +5.2e-3. Step 1 then sat 30+ min in XLA "very slow compile" of the jit_pullback (adjoint through the stability rows) — the compile is cached on scratch (env.sh JAX cache), so the batch-8 restart should skip it. **Homotopy** for batch 8 (jobs_batch7.txt line): `MERCIER_FLOOR −0.10`, `BALLOONING_LIMIT 0.004` (nearly feasible start) → tighten every allocation (−0.05 / 0.002 → 0 / 0) by editing the previous jobs list before each turnover.
- Hurwitz 2024 scan: 20,444 QA optimizations, 4,628 inside our κ/MSC/cc/cs thresholds (17–36 m/hfp); competitive B·n (≤ 2e-4) only at 22–26 m. Lower envelope (33 sets) → `coils/hurwitz_*.json` + `coils/hurwitz_cloud_index.json` (`convert_hurwitz_cloud.py ../references/hurwitz2024/output/QA`; only the selected dirs have biot_savart.json extracted: `references/hurwitz2024/selected_dirs.txt`). 12×12 evaluation queued in extras_batch8 after the Wechsung cloud; `fig_pareto.py` draws them as grey "+" (5 coils/hfp).
- Interactive-QOS note: the user's `f3rm_phiedge` job (59654806, 4 h from 18:58) uses the second interactive slot; the chain's salloc needs alloc 7 to end first (2 jobs per user).
- Alpha losses, free arm step 59: s = 0.25 0/1000, **s = 0.5 13/1000 (1.3 %)**, 0.2 s (`runs/alpha/qa4-free-s05`).
- Coil-shape comparison figure: `analysis/fig_coils_compare.py` (ours vs Gil vs Wechsung at 18/20/24 m; shapes nearly identical, ours sit on the κ/MSC/cc limits).

## 7d. Force-vs-QS Pareto fronts (user proposal 2026-10-10 22:00, replaces Gil PRL Fig 1 with QS of the actual equilibrium)

- Gil Fig 1 setup (`scripts/auglag_qa_dynamic.py`): LP QA lowres at R0 = 1 m, 5 coils/hfp, order 16 in the archive files, total length 25 m/hfp (also 30 m), κ ≤ 5 and MSC ≤ 5 as L2 penalties (max κ of his sets is 7–190!), cc ≥ 0.1, cs ≥ 0.15, a = 0.05 m, total current 3e5 A (mean |B| on target 0.2271 T), force thresholds 9–14 kN/m. Hurwitz 2024 scan (same target, 5 coils, order 16, length 4.9–5.2 per coil, κ ≤ 12, MSC ≤ 6, cc ≥ 0.083, cs ≥ 0.166): final Pareto front = `output/QA/with-force-penalty/4/pareto` (220 sets, 24.1–26.2 m/hfp, force 7.5–14 kN/m, B·n 4.5e-3 → 4e-5).
- Converted: `coils/hurpar_<uuid8>.json` (220, index `coils/hurwitz_pareto_index.json`), `coils/gilpar_L{25,30}_F<thr>.json` (10, `coils/gil_pareto_index.json`; `convert_gil_pareto.py`). **Validation (`check_pareto_reproduce.py`)**: 8 Hurwitz reference sets reproduce the archive's max force and ⟨|B·n|⟩/⟨B⟩ to ≤ 0.1 %; Gil's 25 m sets give forces at his thresholds (9.7/10.3/10.7/12.0 kN/m for 9/10/10.5/12) and B·n 3.4e-4 → 7.6e-5; `gilpar_L30_F9.5` (κ 1282, F 5e5) and `gilpar_L30_F9` (κ 68, F 2.3e4) are broken archive files → exclude.
- Force convention mapping for our driver at this scale: CONDUCTOR_RADIUS 0.05, F_ours(B ≈ 1.012 T on target) = F_theirs × (1.012/0.2271)² ≈ 19.9 × (e.g. 10 kN/m → FORCE_LIMIT 1.99e5 N/m); evaluate everything at 12×12 / VC 64 (the resolution that performed best in our optimization).
- Plan awaiting user go-ahead: (1) evaluate all 220 + 8 published front sets with the three-term free-boundary evaluator at 12×12 (≈ 45 GPU-h); (2) our front A (plasma pinned: R0 1.0099 ± 1 mm, A 6.00–6.01, ι ≥ 0.4152) and front B (plasma free, QS only) at 25 m/hfp, 5 coils, order 16, κ ≤ 5, MSC ≤ 5, cc ≥ 0.1, cs ≥ 0.15, FORCE_LIMIT scanned over 8/9/10/11/12/14 kN/m, each started from the nearest Gil/Hurwitz front set.

## 7e. Batch 9 arrangement (armed 23:10, 2026-10-10)

- Gil PRE Fig 2 / PRL Fig 1 facts (arXiv 2507.12681 HTML): Hurwitz scan shown under Gil's a-posteriori filter (κ < 12, coil length < 6.5 m, cs > 16.6 cm, cc > 8.3 cm; "8500 optimizations, 41 on the front"), his AL points at 25 m (9/10/10.5/12 kN/m) and 30 m (9–14 kN/m). Our archive has 18,243 optimizations; the front under Gil's filter with B·n < 4e-3 = 108 sets (`references/hurwitz2024/gil_filter_front.json`), stratified cloud sample 71 (`gil_filter_cloud_sample.json`); converted as `coils/hurfront_<uuid8>.json` (350 files incl. the it-4 Pareto set; indices `hurwitz_front_index_<it>.json`).
- `extras_batch9.sh 3` (armed, pid 435223; log `runs/extras_batch9.log`): waits for all batch-9 chain launches, then 3 evaluation lanes over `runs/front_eval_list.txt` (7 Gil + 108 front + 71 cloud, Gil first, then by force) at 12×12/VC 64 → `runs/lpqa_vacuum_eval_12x12/<name>/row.json`; plus stage two on the best saved pure-stage-one step and Gil v3 scoring of `vac-3coil-fair2`. Chain list `jobs_batch8.txt` trimmed to 11 (pareto-L18/L20/L24 flat, dropped) → 11 + 4 = 15 GPUs.
- **23:25 re-prioritized (user: new figure within 8 h)**: `jobs_batch8.txt` now = `qa4-free-stable2` + `ff25A-F{8,9,10,11,12,14}` + `ff25B-F{8,9,10,11,12,14}` (13 chain launches); every other continuation (qa4-fixed, qa4-stage1, vac-3coil-fair2, pareto-L16/17/19/21/22, L20.09, L23.66) is commented `#paused-for-force-fronts-alloc9` — re-enable them in the batch-10 list by removing the prefix (they restart from their latest saved steps). Extras batch 9: 3 evaluation lanes over Gil's 7 + every 3rd front member (36) + every 5th cloud sample (14) first, then the rest; stage two deferred (`STAGE_TWO=1` env to enable).
- Force fronts (full 10-level lists, the 7/13/16/18 levels still to run): `jobs_forcefront_A.txt` (plasma pinned) and `jobs_forcefront_B.txt` (aspect free, upper bound 6.01), 10 force levels each at 25 m/hfp, 5 coils, κ ≤ 5, MSC ≤ 5, cc ≥ 0.1, cs ≥ 0.15, a = 0.05, started from the Hurwitz front member just below each force. Launch with `launch_jobs.sh <jobid> jobs_forcefront_A.txt`; add to the chain list afterwards (src "-").
- Login-node shells became very slow (~1–2 min per command, module init); background helpers unaffected. Self-match gotcha again: `ps | grep "[e]xtras_batch9"` matches the calling shell — check helpers with `awk '$3=="bash" && $4 ~ /^extras_batch9\.sh$/'`.

## 8. Next steps

1. After batch 8 starts (~21:00): confirm 11 chain launches + extras (`runs/extras_batch8.log`, `runs/launch_pareto_extra.log`). Check `vac-3coil-fair-c2` — it sat at step 3 for over an hour in allocation 7 (force row binding hard; inspect its log).
2. When the Pareto points converge: rescore all at 12×12 (add `ours_pareto-L*` to `eval_sweep_batch4.sh`), Gil-pipeline v3 scoring, re-render `analysis/fig_pareto.py` (it already plots the Wechsung cloud and the two exact-length points), send to the user, push to `publication`.
3. When `hurwitz-done`: inspect `listing.txt`, convert 5-coil sets, filter by the fixed thresholds, evaluate the survivors at 12×12, add as a flagged cloud.
4. Stage two on the converged pure stage one → Fig 1 and Table I; Boozer plot for Fig 4.
5. Re-arm the chain each allocation (one chain only) until the front and Fig 1 are done.
