# Why the pressure gate failed after relaxation

Read-only audit of the exact selected fixed state and rejected ordinary state, using the pinned runtime and unchanged coils. No new equilibrium solve, root polishing, parameter changes, or optimization. Single-GPU diagnostic completed in 79.94 seconds and exited. Both saved states remained bytewise unchanged in memory; input and output hashes are recorded.

## Measured results

| Quantity | Fixed target | Relaxed rejected state |
|---|---:|---:|
| Mean iota | 0.2000004500 | 0.2966739435 |
| Aspect | 5.0000000000 | 5.6844410284 |
| Signed B0 (T) | 5.1000024984 | 3.6871582223 |
| Major radius (m) | 11.0670338977 | 14.6080937797 |
| Magnetic-axis R at phi=0 (m) | 12.6162911856 | 17.2083276307 |
| Fresh DEL-BSQ | 0.932263856% | 7.045048292% |
| Fresh spectral edge-force residual | 8.507262584e-5 | 3.605364799e-12 |
| Full preconditioned-force norm | 4.380667774e-4 | 7.674715106e-8 |

The force norm here is the full evaluate_forces output, not a substitute for the dedicated projected-root certification gate. The ordinary run never reached that gate. Fresh force evaluation independently reproduces the small final residuals, so this is not explained solely by stale printed force values.

## What the implementation does

The ordinary initializer uses fixed coil parameters and evolves the free boundary without the outer optimizer's four target constraints. Those constraints govern trial acceptance later; they are not equations in this initial solve. In free-boundary VMEC the input boundary is an initial guess, as documented at https://simsopt.readthedocs.io/latest/example_vmec.html.

The solver injects NESTOR vacuum pressure into the evolved R/Z boundary rows. With include_edge_in_convergence=False, its ordinary stopping rule uses FSQR/FSQZ/FSQL, generally interior sums after the startup period. DEL-BSQ is a separate fresh angular-grid pressure diagnostic. The final F_EDGE is also small: simply switching that stopping flag would not itself demonstrate removal of the 7% mismatch.

## Fourier audit and limits of the conclusion

MPOL=3 retains poloidal m=0,1,2; NTOR=3 retains toroidal |n|<=3. The 48×40 angular quadrature samples more structure than this retained equilibrium basis can adjust. Increasing angular sampling did not increase MPOL or NTOR.

The signed scalar pressure jump was extended from the 25×40 stellarator-symmetric grid to the full 48×40 grid and decomposed by numpy.fft.fft2. This is an angular Fourier diagnostic of the pressure jump, not the weighted spectral force operator. The fraction of its squared Fourier amplitude outside |m|<=2 and |n|<=3 is 99.0587% initially and 99.8495% finally. Dominant final toroidal frequencies include |n|=8 and 5, beyond NTOR=3. These fractions refer to squared pressure-jump amplitude, not a decomposition of the L1 DEL-BSQ percentage.

The evidence establishes that passing the frozen DEL-BSQ threshold did not make the target a force-balanced free-boundary state, and that small retained-mode force residuals coexist with unresolved spatial pressure structure. Spectral truncation is a supported contributor to the diagnostic mismatch. This audit does not prove that truncation alone caused the full radial displacement, demonstrate physical instability, or rule out other VMEX/NESTOR implementation errors.

Before another optimization pilot, test frozen-target free-boundary edge-force compatibility and equilibrium/vacuum Fourier-resolution convergence, keeping the 48×40 grid and the original 111 coil DOFs. No gates were relaxed and no follow-up solve was launched by this audit.
