# Matched 48×40 rebuild

The new guarded pipeline is queued on lhk1-2, supervisor PID 512320. It requires GPU 0 to remain idle for 60 seconds and waits at most 30 minutes. A separate WOUT recomputation occupied GPU 0; no other job was interrupted. The initial launch was rejected before any equilibrium solve by the GPU guard, and that log is preserved.

Resolution/profile preflight passed (1 test). Fixed-boundary NS16→50 keeps angular NTHETA=48, NZETA=40 at both stages. M3/N3 and all original pressure/current profiles are retained. Coil evaluation uses the same 48×40 target nodes; increased singular-integration quadrature verifies virtual-casing accuracy without changing equilibrium resolution.

The finite pipeline runs fresh target calibration (2500 s cap), the 111-DOF coil refit (3300 s cap), and frozen-state NESTOR validation (240 s cap), each only after preceding requirements pass. The fixed target must meet the original physical targets. The coil fit must meet the authorized 1% pressure surrogate plus geometry and quadrature checks. Actual NESTOR DEL-BSQ must then be at most 1% before coils can be selected for a free-boundary trial.

No ordinary free-boundary solve or QA campaign is automatically launched by this rebuild pipeline. There is no new qualified target or coil result yet. Consult the remote pipeline_status.json and phase logs for updated results; this document records launch status only.

Remote root: `/root/autodl-tmp/vmex-finitebeta-ellipse-matched48x40-20260914T171513Z`.

## Relocated to lhk3-6

User requested relocation. The old lhk1-2 waiting supervisor (512320) was stopped before any equilibrium solve. The new pipeline is running on **lhk3-6 GPU 1 (RTX 4090 D, 24 GB)**, supervisor 32639; actual calibration Python PID 32698 has started evaluation 0. Resolution/profile preflight passed (1 test), float64 and finite coil-field checks passed. Core dependency versions match the previous host, and ESSOS coils.py has the same SHA256. The verified virtual-casing package is private to this project directory; the shared environment was not changed.

The first fresh equilibrium uses NS16→50, angular 48×40 throughout, and final FTOL=1e-13. The pipeline advances to the coil refit and frozen NESTOR gate only after prerequisite checks pass. No new equilibrium has yet been qualified at this launch snapshot.

Active remote root: `/root/autodl-tmp/vmex-finitebeta-ellipse-matched48x40-lhk3-6-20260914T174815Z`. See lhk3-6_deployment.json for the immutable source archive hash and previous-host lineage.

## Fixed target passed; coil gradient gate stopped the fit

Calibration completed in 81.1 seconds with four ordinary equilibrium evaluations. Independently verified WOUT: mean iota 0.20000045001254882, B0 5.100002498441042 T, aspect 4.9999999999999964, Rmajor 11.067033897730937 m, beta 0.027034308657760003. All three force residuals are below 1e-13, and full prescribed profiles are unchanged. WOUT and state are saved in fixed/; archive and payload hashes were verified.

The first fit stopped before optimization because the virtual-casing integration grid did not contain the new 48×40 target nodes. Separate qaligned scripts corrected quadrature to 800×144 and 1600×288; package-level checks passed without changing equilibrium resolution. Virtual casing then completed, but the directional derivative gate failed.

A bounded componentwise audit completed in 36.3 seconds. Normal-field and pressure derivatives agree with finite differences; disagreement is isolated to the nonsmooth geometry penalty. At step 1e-6 in the tested full direction, geometry AD=-0.007455083258 and central FD=-0.006776772902; the physical terms differ by less than 6e-9. The derivative threshold was not relaxed. Raw audit results are in results/gradient_audit/summary.json; remote failed attempts are preserved.

The pipeline and audit have ended. Zero coil optimizer iterations, frozen NESTOR checks, free-boundary solves, or QA steps have been executed. Remaining work is to resolve geometric-penalty differentiability and revalidate the gradient before refitting.

## Selected stage-two coils completed

Selected coils are in stage2/coils_selected.json and the matching state/WOUT/input bundle is stage2_selected_bundle.zip. All transferred payload hashes and the selected coil/WOUT lineage were verified locally.

Geometry and virtual-casing accuracy checks passed. Virtual-casing pressure mismatch: 0.468856014%; independent frozen-state NESTOR DEL-BSQ: 0.932263856% (gate 1%). Normal-field RMS: 0.450125054%; maximum: 1.147344917%, retained as diagnostics. The full prescribed profiles, fixed equilibrium and angular 48×40 resolution remain unchanged.

The host-eager NESTOR derivative passed primal consistency and componentwise finite-difference tests. Its bounded refinement reached 1000 iterations; optimizer convergence is not claimed. The selected coils are eligible for a gated free-boundary initialization, but no free-boundary equilibrium or QA campaign has yet been run. All task GPU jobs ended; only unrelated GPU 0 activity remains on lhk3-6. See stage2/qualification.json for the complete results and STAGE2_METHOD.md for rejected intermediate paths.
