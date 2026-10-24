# Fixture qa25 (built 2026-10-22, Perlmutter)

- COIL_CASE=qa4-beta-merc; driver from origin/exp-c-run-pr121 (b5e85d35) on library dee17ddb (#122); args: see fixtures/qa25b.log header and fixD.cmds/fixC.cmds.
- Step-0 line: [step 0] QA=1.167655e-01 iota=0.36007 vacuum iota=0.3099 (axis 0.3099) B_axis=1.00605 iota_I=0.3599 DMerc_min=-0.953 R=1.00004 aspect=4.0001 clearance=0.2260 coil_slack=-0.0027 redl(max rel)=4.3e-05 B
- Use with --seed-run; never refit the coils (the GPU coil fit is not deterministic).
