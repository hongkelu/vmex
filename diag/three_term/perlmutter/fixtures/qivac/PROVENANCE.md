# Fixture qivac (built 2026-10-22, Perlmutter)

- COIL_CASE=qi; driver from origin/exp-c-run-pr121 (b5e85d35) on library dee17ddb (#122); args: see fixtures/qivac.log header and fixD.cmds/fixC.cmds.
- Step-0 line: [step 0] QI=1.069085e+00 iota=0.59897 vacuum iota=0.5945 (axis 0.5945) B_axis=1.00004 iota_I=0.0000 DMerc_min=0.712 R=1.00000 aspect=8.0000 clearance=0.1875 coil_slack=-0.0015 B.n=2.7e-05 K=3.3e-05 la
- Use with --seed-run; never refit the coils (the GPU coil fit is not deterministic).
