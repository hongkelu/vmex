# Fixture qi25 (built 2026-10-22, Perlmutter)

- COIL_CASE=qi6-beta-A8; driver from origin/exp-c-run-pr121 (b5e85d35) on library dee17ddb (#122); args: see fixtures/qi25b.log header and fixD.cmds/fixC.cmds.
- Step-0 line: [step 0] QI=9.326806e-01 iota=0.94692 vacuum iota=0.8710 (axis 0.8710) B_axis=0.99996 iota_I=0.4447 DMerc_min=-272.349 R=1.00002 aspect=8.0002 clearance=0.1504 coil_slack=-0.0008 redl=1.22e-04 B.n=2.2
- Use with --seed-run; never refit the coils (the GPU coil fit is not deterministic).
