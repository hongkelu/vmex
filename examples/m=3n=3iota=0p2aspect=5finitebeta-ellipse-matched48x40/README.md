# Finite-beta rotating ellipse at matched 48×40 resolution

Fresh fixed-boundary calibration and stage-two refit. M3/N3, NFP2, NS16→50; angular NTHETA=48/NZETA=40 at every radial stage. Keep Rmajor=11.067033897730942 m, aspect=5, mean iota=0.2, signed B0=5.1 T and full original NCURR=1 pressure/current profiles. Calibrate ellipse shape and PHIEDGE, never prescribe iota.

Old coils are a geometric seed only. Recompute virtual casing from the new WOUT. Require actual frozen-state NESTOR DEL-BSQ ≤1% before an ordinary free-boundary initialization; no root polish. Fixed solve FTOL=1e-13; free acceptance force ceiling=1e-10, projected-root gate=2e-6, adjoint gate=2e-5. Four physical targets retain 1% bands. No optimizer launch is certified by a coil-fit surrogate.
