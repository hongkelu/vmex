# Selected stage-two finite-beta coils

Fixed target: M3/N3, NFP2, NS50, angular 48×40. Mean iota 0.2000004500; aspect 5; signed B0 5.1000024984 T; Rmajor 11.0670338977 m. Full original pressure/current profiles are preserved.

The coil set has four independent order-4 coils and 16 physical coils through stellarator symmetry. The 111-DOF chart varies 108 shape coefficients and three currents; the first base current remains fixed. The legacy JSON field dofs_currents contains physical currents in amperes.

Final checks passed: exact sampled geometric limits at 75 and 256 segments; virtual-casing quadrature convergence; pressure mismatch 0.468856014%; fresh frozen-state NESTOR DEL-BSQ 0.932263856%, below the 1% limit. Normal-field error is 0.450125054% RMS and 1.147344917% maximum; these are diagnostics, not acceptance gates under the agreed policy.

The fit reached its finite 1000-iteration NESTOR-refinement limit and is not claimed to be an optimizer-converged minimum. The selected set is qualified for a gated free-boundary initialization, not a certified free-boundary equilibrium. No free-boundary plasma solve or QA optimization has been run with it. Keep 48×40, NS50 and the recorded MF4/NF3 NESTOR configuration for subsequent checks.

Physical base currents: 17.7470870584, 26.1962687439, 21.5734824252, 16.7140650976 MA.

The matching fixed WOUT, input, state and complete qualification report are included. Source and raw validation evidence are retained in the parent case results/stage2_selected and deployment folders.
