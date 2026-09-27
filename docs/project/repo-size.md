# Repository size

Measured 2026-09-27 on `main` (3,449 commits, 529 tracked files).

| Clone | `.git` size |
|---|---|
| full `git clone` | 45 MB (pack 43.8 MB, 29,094 objects) |
| `--filter=blob:none` | 8.5 MB |
| `--depth 1` | 5.9 MB |

Blobs in history: 10.7 MB belong to paths still in `HEAD`, **31.8 MB
(about 75%) belong to paths that no longer exist**. The largest dead
paths are `examples/data/mgrid_ncsx_c09r00_small.nc` (3.0 MB), old
`docs/_static/figures/*.png` superseded by `.webp` (about 9 MB total,
largest 1.7 MB) and `benchmarks/polish_recovery_*_state.npz` (about
0.3 MB each). The largest live items are `plan.md` (0.75 MB of history,
318 kB / 3,863 lines now), `README.md` history (0.34 MB) and
one diagnostics summary figure (1.5 MB of history).

Tracked files and working-tree size: `vmex/` 76 files (63.7k lines),
`tests/` 136 (49.0k lines), `docs/` 96 (3.5 MB), `examples/` 124
(1.2 MB), `benchmarks/` 57 (30 JSON, 25 py, 2 md; 0.75 MB).

Users who only need the code today: `git clone --depth 1` or
`--filter=blob:none` (both under 9 MB), or `pip install vmex`.

## Done in this change

Six README/docs `.webp` figures re-encoded (`cwebp -q 82 -m 6`):
1.58 MB to 0.84 MB, provenance manifest refreshed. Only files that
shrank by more than 30% were replaced; the two animated `.webp` files
were left alone.

## Needs owner approval

1. **History rewrite** (saves about 30 MB of the 45 MB clone). After
   open PRs are merged or rebased and collaborators are warned:

   ```sh
   git clone --mirror https://github.com/uwplasma/vmex vmex.git
   cd vmex.git
   git filter-repo --analyze          # confirm the dead-path list
   git ls-tree -r --name-only HEAD > keep.txt
   git filter-repo --paths-from-file keep.txt   # drop every path not in HEAD
   git push --force --mirror
   ```

   Keeping only `HEAD` paths discards the history of renamed files; to
   keep it, use `--invert-paths` with an explicit list of dead large
   paths instead. Risks: every commit SHA changes, so forks, open PRs,
   pinned SHAs in docs/other repos (for example GKX pins) and release
   tags must be re-pointed; GitHub keeps old `refs/pull/*` objects until
   support purges them. All clones must be re-cloned.

2. **Stale branches** (no open PR): merged or squash-merged heads
   `feat/mgrid-tricubic-interpolation`, `fix/mgrid-raw-mode-default-extcur`,
   `perf/interior-flux-coordinates-jit`, `perf/mgrid-sum-groups-on-read`,
   `fix/cache-dir-nesting`, `fix/extender-axis-row-extrapolation`,
   `fix/refinement-restart-from-best`, `fix/solve-file-free-boundary`,
   `rj/force-balance-recovery`, `ds/winding-adjoint-performance`;
   closed without merge: `fix/winding-certified-response`,
   `perf/winding-all-sectors`, `perf/winding-derivatives`,
   `perf/winding-operator-reuse`. Branches with no PR belong to
   collaborators and should be deleted only by their authors.

## Proposed follow-ups (each its own PR)

- **Media.** Keep only figures shown in README/docs; the animated
  single-stage `.webp` files and future movies go to release assets or
  GitHub Pages.
- **benchmarks/.** The 30 JSON records are cited by docs and
  `benchmarks/INDEX.md`; replace dated one-off records
  (`*_2026-09-*`, `review_20260913*`, `*_office`) by one summary table
  per topic and move raw records to release assets.
- **Source.** Merge candidates by line count: `core/_netcdf.py` (28)
  into its only caller, `core/neoclassical.py` (116) and `core/boozer.py`
  (155) into one diagnostics module, `core/step.py` (180) next to the
  iteration loop, `core/qi.py` (185) with the QI objective code.
- **Tests.** Merge files under 120 lines with a same-topic neighbour:
  `test_baseline_benchmark.py` (18), `test_lasym_free_convergence.py`
  and `test_postprocess_currents.py` (56 each), `test_capability_docs.py`
  (58), `test_implicit_solve_kwargs.py` (78) into the `test_implicit_*`
  group, `test_cli_device.py` (90) into `test_cli_*`,
  `test_golden_digests.py` and `test_coverage_margin.py` (about 90) into
  one repository-contract file; the six `test_qi_*` files into two.
  Every test function moves; none is deleted.
- **plan.md.** Replace with a current-state document of at most ~200
  lines (open work, decisions, how to resume). History already lives in
  git and PR descriptions; the old file stays reachable at its last
  commit.
