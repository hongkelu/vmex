# Fixture qhvac (built 2026-10-22, Perlmutter)

- COIL_CASE=qh; driver from origin/exp-c-run-pr121 (b5e85d35) on library dee17ddb (#122); args: see fixtures/qhvac.log header and fixD.cmds/fixC.cmds.
- Step-0 line: [step 0] QH=1.916299e+00 iota=1.01405 vacuum iota=1.0116 (axis 1.0116) B_axis=1.00028 iota_I=0.0000 DMerc_min=3.692 R=1.00018 aspect=6.0015 clearance=0.1577 coil_slack=-0.0537 B.n=1.6e-04 K=2.2e-04 la
- Use with --seed-run; never refit the coils (the GPU coil fit is not deterministic).
