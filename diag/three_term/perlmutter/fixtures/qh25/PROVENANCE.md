# QH 2.5% shared seed fixture (frozen 2026-10-22 08:50 PDT)

- Case: COIL_CASE=qh4-beta-A8-merc, driver examples/optimization/single_stage_free_boundary_optimization_three_term.py
  from origin/exp-c-run-pr121 (b5e85d35); args --modes 8 8 --ns 51 --beta 0.025 --bootstrap --coil-point-rows.
- Seed equilibrium: Redl seed n0 = 2.8080e19 1/m^3, T0 = 2294.6 eV, CURTOR = -14811.2 A, Redl mismatch 3.715e-6,
  <beta> = 2.4987 %, iota 1.1532..1.2968 (identical in the two builds that ran: library 23ac3c04 and 96ece254;
  they differ only in the Picard power-series delta definition, which changed the iteration count, not the seed).
- coils.initial.json: written by the 23ac3c04 build at 08:28:40: [coil fit] 2128 L-BFGS-B iterations (ABNORMAL),
  (B_coils + B_plasma).n/|B| RMS 1.399e-2 -> 3.832e-3 on the fixed-boundary seed. The GPU coil fit is not
  deterministic: every comparison starts from this file (--seed-run), never from a new fit.
- Hosts: Perlmutter A100 (jax 0.12.0.1 cuda13, solvax 0.31.0).
