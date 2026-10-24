# Fixture qavac (built 2026-10-22, Perlmutter)

- COIL_CASE=qa3; driver from origin/exp-c-run-pr121 (b5e85d35) on library dee17ddb (#122); args: see fixtures/qavac.log header and fixD.cmds/fixC.cmds.
- Step-0 line: [step 0] QA=1.585934e-01 iota=0.45231 vacuum iota=0.4502 (axis 0.4502) B_axis=1.00034 iota_I=0.0000 DMerc_min=0.104 R=0.99995 aspect=5.9999 clearance=0.1982 coil_slack=-0.0387 B.n=4.7e-05 K=5.7e-05 la
- Use with --seed-run; never refit the coils (the GPU coil fit is not deterministic).
