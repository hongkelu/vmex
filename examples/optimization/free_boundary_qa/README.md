# Free-boundary single-stage QA optimization

This maintained M3/N3 vacuum case optimizes three relative coil currents and
108 Cartesian Fourier coefficients. The LCFS follows the equilibrium solve.
Targets are mean iota 0.2, aspect ratio 5, and signed on-axis B0
-0.17506474574437714 T, each within its original 1% band. The equilibrium grid
is NS31, NTHETA48, NZETA40. Per-step caps are 1 mm coil displacement and 1%
nominal-current change, with at most six half-step trials.

The optimizer combines projected QA descent with bounded adaptive target
restoration. Force/edge, projected-root and derivative gates remain strict.
`case_contract.json`, `optimizer_policy.json` and the input manifest record the
physical and numerical contract; checkpoint migration is explicit.

The public API supports unmodified ESSOS main at
`c9b41222e06aed62c246ca3b35349e427a3ea239`. Runtime data and host-specific pilot
controllers are not part of this example. Passing software tests is not
independent qualification of a new equilibrium or GPU campaign.

### Public problem API

`single_stage_free_boundary_optimization.py` follows
`examples/optimization/QA_optimization.py`: settings, objective tuples,
physical constraints, problem construction, optimization, and reporting.
The design vector contains three relative currents and 108 Cartesian coil
Fourier coefficients. The plasma boundary is determined by equilibrium.

The script calls `opt.FreeBoundaryProblem.from_tuples`, `opt.TargetBand`, and
`opt.minimize_projected`. The problem inherits `FunctionProblem` and provides
`x0`, `scales`, `dof_names`, `residual`, `residual_jac`, `value_and_grad`,
`constraint_values`, `constraint_jac`, `coils_from_x`, and `equilibrium_from_x`.
It owns no command-line arguments or output directory. The scalar objective
and three constraints share the compact derivative path; only an explicit
request computes the full QA residual Jacobian.

`example_support.py` connects the public API to this maintained case's verified
inputs, original coordinate chart, checkpoint schema and diagnostic files.
It enforces the saved targets, tolerances and normalization. The example does
not manipulate anchors, assemble derivatives, or write checkpoint internals.
Changing this case's physical contract requires updating its records. Other
cases can use the public API directly without this persistence adapter.

The example uses the public problem and projected optimizer directly. There is
no case-local numerical problem, optimizer loop, or legacy import adapter.
Checkpoint migration records remain because supported resumes still use them.

For example, from the repository root with the VMEX/ESSOS environment active,
prepare and certify the bundled initial state in a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B \
  'examples/optimization/free_boundary_qa/single_stage_free_boundary_optimization.py' \
  --initialize-only --device cuda:0 --max-wall-hours 1 \
  --output-dir 'examples/optimization/free_boundary_qa/runs/example-initial'
```

To optimize, omit `--initialize-only` and specify `--target-step` (an absolute
accepted-step limit, at most 3000). Resume also requires `--resume-checkpoint`
and `--checkpoint-sha256`; every invocation uses a new output directory.
`--help` and importing the script perform no solves. The new entry point records
the actual checkout revision plus source hashes, and honors the device ordinal.
This code reorganization alone is not numerical qualification of a new run.

From the repository root, run the lightweight tests with a compatible environment:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  'examples/optimization/free_boundary_qa/tests'
```

### Required fast tangent policy

`single_stage_free_boundary_optimization.py` is the case entry point.
Every step uses a dense adjoint and retains its LU
factors for tangent prediction. Backtracking scales a previously checked tangent
at the same accepted root; each response is checked again against the original
matrix-free operator. Promotion closes the old linearization. Missing, stale or
failed linearizations stop or reject the trial; there is no GCROT or
accepted-state-only fallback in this workflow.

The predictor is always `equilibrium_predictor=reused_dense`. The batch size
defaults to `--adjoint-dense-batch-size 32`; set it to `64` to use that batch
instead. Numeric inputs skip the timing comparison. A fresh run still performs
the complete initial equilibrium solve and certification, then computes the
first gradient once. First-use compilation is still required.

Optional `--adjoint-dense-batch-size auto` tunes once per run at the first accepted
root: it warms up batches 32 and 64 and compares three alternating pairs of
complete gradients. Batch 64 must be at least 5% faster in every pair; otherwise
32 wins. Every gradient row must agree with the first batch-32 result to 1e-8
relative (exactly for zero rows), in addition to the original solver checks.
Any numerical failure stops the run. The chosen factors are reused immediately
and the batch is retained for subsequent steps. Tuning uses eight gradient calls,
adds startup/compilation time, and obeys the run's walltime limit. It is not
repeated each step or reused blindly across devices, code versions, or restarts.

The manifest records both the requested policy and actual batch. Optional auto
tuning writes measurements to `adjoint_batch_tuning.json` and the progress log.
Initialization without a gradient does not trigger tuning.
Requests for `--equilibrium-predictor tangent`,
`--equilibrium-predictor accepted` or `--adjoint-dense-batch-size 4` are rejected
explicitly. Automatic batching cannot re-enable the removed slow predictor.

The VMEX 0.9.1 `free_boundary_continuation_state_pullback` API, called with
`return_linearization=True`, returns the gradient, accepted-state identity and
dense factors together. Its `.tangent(accepted, cfg, direction)`
checks root/configuration identity and performs the direct solve; `.close()`
releases the numerical cache. No module monkey-patching is used.

The example follows the fixed-boundary `examples/optimization/QA_optimization.py`
layout: run settings, objective/constraint derivatives, optimization, and output.
`ADJOINT_BATCH_SIZE` is alongside the device and budget settings. Case inputs,
the projected-QA/restoration proposal, acceptance rules, and output remain shared
helpers. Tuning and normal steps call the same native derivative helper; all
linearization lifecycle points share the same cleanup. This is a code-structure
cleanup; it does not adopt the fixed example's optimizer or targets.

Force/edge 1e-11, projected-root 2e-6, linear residual 2e-5, target bands,
optimizer caps, continuation spacing, finite backtracking and checkpoint lineage
are unchanged. No root polishing or augmented Lagrangian is enabled.

The isolated prototype measured 15.8/12.8-second warmed steps on LHK3-6 GPU3,
versus 70.3/67.9 seconds for the matched tangent-GCROT reference. Both accepted
alpha 0.25 with matching gradients. Accepted normalized QA differed by 7.18e-8;
two rejected trials exceeded the extra 1e-7 QA comparison threshold. These are
single-anchor performance results, not identical equilibrium states or a
multi-step qualification. See the workspace reports
`reports/tangent-reuse-20260917/` and
`reports/vacuum-tangent-integration-20260917/` for prototype and integration evidence.

Historical predictor-removal, warm-restart and batch-sweep evidence remains under
`campaigns/step214-to1000-20260914/analysis/`. Archived campaign source is provenance,
not an alternative maintained entry point. This local source update does not
replace, restart or otherwise change an already running remote campaign.
