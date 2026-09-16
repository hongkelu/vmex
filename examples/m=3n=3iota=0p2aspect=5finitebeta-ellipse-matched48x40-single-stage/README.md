# Matched 48×40 finite-beta single-stage pilot

Fresh 111-DOF coil QA optimization from the selected matched48x40 stage-two coils and fixed state. The fixed state is a seed, never an accepted free-boundary root. Full QA pressure and NCURR=1 current profiles and PHIEDGE=91.09295154228212 Wb are preserved.

Targets: mean iota 0.2, aspect 5, signed B0 +5.1 T, major radius 11.067033897730942 m, each within 1%. Angular grid 48×40 throughout; M3/N3, NS50, NFP2; NESTOR MF4/NF3.

Frozen NESTOR recheck must pass 1% before one ordinary free-boundary solve. Requested forward FTOL=1e-13, force ceiling=1e-10, projected root residual ceiling=2e-6, fresh DEL-BSQ ceiling=1%. No root polish and no same-coil retry.

Up to 10 steps in one hour using forward_dense_jax with adjoint residual tolerance 2e-5; independent 1 mm geometry and 1% relative-current caps; up to six halving trials. Feasible QA steps retain all four bands and decrease QA. If a numerically certified state is outside a band, the existing restoration policy reduces violation and reports it as infeasible. Projected QA gradient controls convergence; failed backtracking is stagnation.

Scalar rows and dense adjoint control are host-eager. Centered state-row finite differences and tangent/adjoint duality are consistency checks, not a complete finite-difference validation of nonlinear free-boundary sensitivities. No long campaign is authorized by this pilot.

Pinned VMEX commit d6910b428841766c04767410e9e9cdc39a6cfb19; see input/runtime manifests for exact lineage.
