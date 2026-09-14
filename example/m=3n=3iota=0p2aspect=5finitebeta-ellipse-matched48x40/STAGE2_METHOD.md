# Stage-two coil fitting at fixed 48×40 equilibrium

The fixed input, WOUT and NS50 spectral state remain frozen. The chart contains 108 Cartesian Fourier coefficients and three relative current variables, with four order-4 independent coils and NFP2 stellarator symmetry. The first current is fixed. Full QA pressure/current profiles and all target values remain unchanged.

The first objective fits normal field and pressure mismatch from coil plus virtual-casing plasma field. The geometry penalty uses a conservative smooth minimum of distances, -tau log(sum(exp(-d/tau))), with tau=0.001 m. This is at most the exact sampled minimum; final clearance checks retain their exact sampled definitions and original bounds. The original derivative mismatch was isolated to coil–coil distance. Length, curvature, magnetic normal field and pressure passed their isolated checks.

Virtual-casing targets use 48 poloidal by 40 toroidal points per field period. The singular integration quadrature uses full-torus 800×144 and 1600×288 nodes, satisfying the package alignment rules. These quadrature sizes do not change the equilibrium angular resolution. Production Biot–Savart uses 75 coil segments; validation also uses 256.

The initial smoothed fit reached its 12000-iteration cap. Its geometry and virtual-casing quadrature checks passed, virtual-casing pressure mismatch was 0.5461652243%, and frozen NESTOR DEL-BSQ was 1.1425474393%. This candidate is preserved but is not selected for a free-boundary trial.

A subsequent frozen-boundary refinement adds angular weighted mean squared relative NESTOR pressure residual, using the same MF4/NF3 potential and 48×40 grid as the independent gate. No plasma solve, root polish, profile change or target change is involved. A joint compiled derivative check exposed discrepancies in the virtual-casing terms at the nonzero seed; isolated forward and reverse derivatives passed. The split version evaluates and differentiates the base objective and NESTOR term separately and sums them. Each component and total must pass finite differences in mixed, current-only and shape-only directions before fitting; the summed gradient must match the sum of the component Jacobian.

Selection requires original geometric limits, virtual-casing convergence, pressure mismatch ≤1%, and a fresh independent frozen-state NESTOR DEL-BSQ ≤1%. An accepted fixed-boundary coil candidate is not itself a qualified free-boundary equilibrium. No free-boundary QA optimization is launched by these scripts.

## Validated NESTOR derivative route

The local-relative-pressure and outer-jitted direct-DEL-BSQ refinements are rejected experimental paths. Explicit starting-value checks showed that the outer-jitted NESTOR derivative calculation changed the primal objective: the direct pressure term was 3841771.91656 instead of the independently expected 130.5414651103. This is a primal-value inconsistency, not permission to relax a gradient or physics gate. The base objective matched its independent expected value.

The retained refinement route is fit_coils_host.py. It differentiates the NESTOR scalar with host-eager jax.value_and_grad, retaining the existing internally jitted NESTOR program. Its direct pressure term reproduced 130.5414651103 and the base term reproduced 18.0379227984. All objective components and the summed gradient passed centered finite differences in mixed, current-only and shape-only directions. The final objective adds 1e6*(actual DEL-BSQ)^2 to the original base coil-fitting objective. Starting values and gradients are checked before fitting.

This change affects only evaluation of the frozen-target coil objective. It does not modify the pinned VMEX core, pressure/current profiles, fixed state, angular resolution, geometric limits, or the 1% acceptance gate. It does not establish a general explanation of earlier ordinary free-boundary convergence failures.
