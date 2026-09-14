# Pilot outcome: initial pressure gate failed

The job finished after 383.56 seconds of single-stage runtime, with zero QA steps and no accepted free-boundary state. GPU 1 was released; only the unrelated GPU 0 process remained at the live status check.

- Five preflight tests passed; runtime and input lineage hashes verified.
- Frozen selected coils/fixed state: fresh NESTOR DEL-BSQ 0.932263856%, passing 1%.
- Ordinary free-boundary solve: converged after 4566 iterations at NS50 and 48×40. FSQR=9.74135e-14, FSQZ=2.62008e-14, FSQL=2.37151e-16, all below the 1e-10 force ceiling. FE​DGE=3.59412e-12 is a separate diagnostic.
- Fresh NESTOR at the relaxed boundary: DEL-BSQ=7.045048292%, failing the unchanged 1% gate.
- Projected root certification and the forward_dense_jax adjoint were not reached. No coil optimization trial or root polish occurred.

The iteration log shows the magnetic-axis R at toroidal angle zero moving from approximately 12.62 m to 17.21 m. This is the axis coordinate, not the major-radius constraint measurement. The frozen pressure pass did not survive boundary relaxation. The final four physical constraints were not evaluated by this run because its pressure gate stopped initialization.

Raw logs and the rejected ordinary state are preserved under `results/pilot_initialization_failure/`; the rejected state remains ineligible for resume. The archive SHA256 is `94da42a68540119223f15fdfd357739a59dfaa61f26d1da54014ebc2dd8cd9bf`.

Remote run: `/root/autodl-tmp/vmex-finitebeta-matched48x40-single-stage-20260914T203732Z`. No automatic retry has been launched.

## Saved-state diagnostic

The read-only audit independently measured all four final physical constraints and confirmed each is outside its 1% band. Frozen-target F_EDGE=8.5073e-5 despite DEL-BSQ<1%; relaxed F_EDGE=3.6054e-12 despite DEL-BSQ=7.045%. The final signed pressure-jump squared Fourier amplitude is 99.8495% outside the retained m<=2, |n|<=3 range. See `analysis/initialization_failure/FINDINGS.md` for definitions and limitations. Diagnostic completed, with no new equilibrium solve or optimization and no task GPU process remaining.
