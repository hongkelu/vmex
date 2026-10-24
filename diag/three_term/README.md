# Three-term free boundary (PR #121): hand-off

Everything needed to continue PR #121 from a fresh checkout of this branch. Start with `plan.md`.

## Files

| file | what |
|---|---|
| `plan.md` | The working plan: stages 1-9 from the review, the status of each, and the results log (newest first). Keep it current. |
| `review-2026-10-21.md` | The external review the plan comes from (condensed; every point kept). |
| `labelling-decision.md` | The evidence for the labelling choice as of 2026-10-22 (the PR's decision comment). Superseded in part by the later entries of `plan.md`. |
| `manifest.json` | Code revisions, environment, case arguments and saved-state hashes of the campaign. |
| `../label_probe.py` | The campaign driver with the labelling overridden, plus measurement modes (below). |
| `../fbbench.py` | Fixed-design labelling benchmark: methods x starts at a saved design, interface residuals, iota, field lines (vacuum), LCFS distance between starts. |
| `../fieldlines.py` | Vacuum field-line tracing in the coil field (independent physics check). |
| `../hdiag.py` | Run-log diagnostics. |
| `../abc18.sh`, `../continue.sh` | Launchers of the 18-run labelling campaign (A, C, NL x QA/QH/QI x vacuum/2.5%) and its continuations. |
| `../requirements-v2.txt` | The pinned environment (`venv-tt1-v2`) every result was measured in. |
| `examples/optimization/single_stage_free_boundary_optimization_three_term.py` | The single-stage driver (coils + three-term free boundary, SLSQP). |
| `examples/data/ESSOS_coils_{qa3,qh,qi}.json`, `input.rotating_ellipse_nfp2` | The vacuum campaign's coil seeds and deck. |

## State (2026-10-22, late)

Library commits on this branch since the specification are listed in the PR description. The last three:

- `9d93346` re-solve the bootstrap current after every boundary step (QH stall).
- `095edbe` polish a trial with fresh Jacobians until its certificate passes.
- `7d73da7` every start of a trial from the accepted continuation; polish rounds aim at `POLISH_FTOL`.

Tests at `7d73da7` (`RUN_FULL=1 JAX_ENABLE_X64=1 pytest tests/test_freeboundary_vc.py -k 'not nestor and not desc'`, on a GPU; on CPU it exceeds 4 h): two failures remain.

- `test_plasma_map_derivatives_at_nonzero_offsets`: gradient -1.4532 against differences -1.4799 (1.8%); passed at `0b7535a`. Not yet diagnosed: compare the gradient and differences at h = 1e-2..1e-5 on `0b7535a` and here.
- `test_interior_state_does_not_depend_on_its_start`: a real finding, not a regression. With `_LAST_REFINED` cleared, three converged solves at one boundary (forces 1e-14) differ by 2e-5 in the interface rows (1% in lambda near the axis); unrefined solves differ identically. Check at production resolution next.

Open decision for the user: the labelling (P1). Evidence in `plan.md`: penalty labellings bias the physics by a design-dependent amount (B.n 2-6x); converged no labelling is start-independent in its physics at one QI design and its gradient matched differences to 0.55% (coils) and 2% (current).

## Measurement modes (`label_probe.py W MODE OUT <driver arguments> --restart RUN`)

Run from the tree root. `W` is the label weight (`default` keeps the model's); `LABELLING=start|equal_arc|condensed`.

| mode | measures |
|---|---|
| `cold` | build the problem at the restart design and stop |
| `noise` | gradient against central differences along a seeded direction (`NOISE_FAMILY=coils|phiedge|current|profile|curtor`, `NOISE_H`) |
| `stall` | why a boundary fit stalls: tight replay, Jacobian against differences along the GN step and the weakest directions, predicted against actual cost, normal share of the motion (`STALL_MODE=project|project_stale` re-solve the bootstrap knots after the step) |
| `kkt` | is the restart design a constrained stationary point (NNLS multipliers) |
| `gnstep`, `curvature`, `exactgrad`, `exactdir`, `plasmaparts`, `twostart` | see the docstrings |

Environment switches: `BOUNDARY_FTOL`, `BOUNDARY_MAX_NFEV` (trial fits), `TRIAL_JACOBIAN_FTOL` (trial fits as fresh-Jacobian trust regions to that gain), `LM_TRACE` (verbose Levenberg-Marquardt and projection gains).

`fbbench.py OUT CONFIG`: `CONFIG = prod:RUN:STEP` (a run directory with `input.run`, `coils.stepSTEP.json`, `wout.stepSTEP.nc`); `FBB_METHODS=NL,C:1e-2,D:1e-1` (label:weight); `FBB_FIXED_MODEL=1` (one model, different initial boundaries); `FBB_FTOL`, `FBB_JACOBIAN_FTOL`, `FBB_MAX_NFEV`. Method `M` (VMEC-weighted condensation) needs `labelling="condensed_modal"`, which is only on branch `exp-label-condensed`.

## Servers and data

GPU hosts lhk0-7, lhk8-13, lhk14-19 (RTX 5090), venv `/root/autodl-tmp/venv-tt1-v2`. One job per empty GPU, pinned by UUID, `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`. Campaign runs: `/root/autodl-tmp/abc18/<run>` (each step's input, coils and WOUT; `STOPPED` marks runs stopped by the user). Fixtures: QI 2.5% design 1 `lhk0-7:/root/autodl-tmp/tt1-work/prod-qiC90` (step 90), design 2 `lhk8-13:/root/autodl-tmp/abc18/qi-beta-CfromA90` (step 120), QH 2.5% step 1 `lhk0-7:/root/autodl-tmp/tt1-work/p2x/C-s1`, late QI vacuum `lhk0-7:/root/autodl-tmp/tt1-work/prod-qivacC-final` (step 1938). Production data stay on the servers: this repository is public.
