# Coil-constraint single-stage benchmarks

The same plasma-and-coil design problem solved two ways, with every physics
and engineering limit a hard SLSQP inequality:

- `free_boundary_single_stage_optimization.py` varies the coil shapes,
  PHIEDGE and, with `--bootstrap`, the plasma current. Every trial is a
  free-boundary equilibrium of those coils
  (`vmex.optimize.FreeBoundaryProblem`), so no B.n term is needed.
- `single_stage_optimization.py` is the fixed-boundary counterpart: SLSQP on
  the boundary and the coils, with the same limits, a B.n/|B| term in the
  objective and a B.n/|B| limit.
- `postprocess.py` reads a finished run of either script. It writes the loss
  and constraint histories, a GIF of the coils and LCFS, and a dense (NS201)
  free-boundary solve of the final coils with the `vmex.plot_wout` figures,
  the like-for-like comparison of both arms. `--poincare 2000` adds a
  Poincare section of the final coils' field, and `--trace` the
  `vmex --trace` alpha losses at ARIES-CS size.
- `fit_coils.py` is the stage-two fit that produced the `coils.<case>.json`
  files.
- `parameters.py` holds the cases, `_coil_constraints.py` the differentiable
  coil inequalities and `_common.py` the helpers the scripts share.

Both arms minimize quasisymmetry (or the constructed QI residual of
`examples/optimization/QI_optimization.py`) subject to: min |iota| above a
floor (and max |iota| below a ceiling for `qi6-beta*`), the aspect ratio in a
band, the major radius within 1 cm of R0 = 1 m, the edge mirror ratio for
the QI cases, per-coil length, curvature and mean squared curvature (MSC),
coil-coil distance (all symmetry copies) and coil-plasma clearance. The
fixed arm adds the B.n/|B| limit and, at finite beta, a band on the edge
R B_phi. With `--bootstrap` both hold the bootstrap mismatch under
`REDL_TOLERANCE`.

## Cases

`COIL_CASE` selects the case (default `ellipse5`); the limits are in
`parameters.py`.

| case | configuration and seed | `--beta` | bootstrap | iota | coils per half period, limits |
|---|---|---|---|---|---|
| `ellipse5` | QA, nfp 2, aspect 4.9-5.1, `input.rotating_ellipse` | volume | Redl | >= 0.41 | 3 x order 16, `coils.initial.json`: 5 m, 5 /m, 5 /m^2, 0.15 m apart, 0.20 m clear |
| `ellipse5-beta7` | as `ellipse5` | on axis | Redl (damped Picard) | >= 0.16 | as `ellipse5` |
| `qa3` | QA, nfp 3, aspect 5.9-6.1, rotating ellipse | volume | Redl | >= 0.41 | 3 x order 8: 3.5 m, 8 /m, 10 /m^2, 0.08 m apart, 0.15 m clear |
| `qh` | QH (1, -1), nfp 4, aspect 5.9-6.1, rotating ellipse | volume | Redl | >= 1.1 | as `qa3` |
| `qi` | QI, nfp 4, aspect 7.9-8.1, mirror <= 0.21 | volume | Redl | >= 0.51 | as `qa3` |
| `qa4-beta` | QA, nfp 2, aspect 3.5-4.5, rotating ellipse | volume | Redl | >= 0.27 | 4 x order 12, limits from the plasma size, 0.10 m apart, 0.20 m clear |
| `qi6-beta` | QI, nfp 4, aspect 5.9-6.1, mirror <= 0.21 | volume | DKX | 0.86-0.98 | 4 x order 12, limits from the plasma size, 0.08 m apart, 0.15 m clear |

`--beta` is <beta> or, for `ellipse5-beta7`, the on-axis beta (WOUT
`betaxis`). A `-tok` suffix (`qa4-beta-tok`, `qi6-beta-tok`) seeds the same
case from a circular tokamak with a 1% helical ripple (`input.minimal_seed_nfp*`),
whose beta ramp starts at a prescribed Ohmic current (`OHMIC_CURRENT`) that is
then blended into the bootstrap current. For `qa4-beta*`
and `qi6-beta*` the coil length, curvature and MSC limits are
(1.8, 2.5, 1.2) times the circumference, curvature and squared curvature of a
circle 0.20 or 0.15 m outside the widest allowed plasma. The iota floors keep
the vacuum field off low-order rationals, where it breaks into islands the
nested-surface equilibrium cannot see: QH between iota = 1 and 8/7, QI above
1/2, and `qi6-beta` in the Stellaris band below the 4/4 islands.

## Field strength and plasma size

Every case has B0 = 1 T at R0 = 1 m. The coil currents are scaled once at the
start so their linked mu0 I / 2 pi (the edge R B_phi) is B0 R0, and then held
fixed; PHIEDGE sets the plasma size. Holding both fixed pins the size. For
`ellipse5` in vacuum the aspect ratio then sits at its 4.9 floor (QA ~0.02),
while a free PHIEDGE reaches aspect 5.1 (QA ~0.004). At finite beta a free
PHIEDGE keeps B0 and beta near their targets. `FREE_PHIEDGE = False` restores
the pinned case in the free arm.

## Finite beta and bootstrap current

`--beta` runs either arm with a fixed p ~ 1 - s pressure and zero net
current, calibrated on the fixed-boundary seed. The free arm keeps the vacuum
loss and constraints. It first refits the seed coils to
(B_coils + B_plasma).n = 0 with virtual casing, and it reports beta, B.n and
pressure balance as diagnostics. The fixed arm limits the total B.n and holds
the edge R B_phi at the coils' mu0 I / 2 pi.

`--bootstrap` (with `--beta`) adds a self-consistent bootstrap current to both
arms. The kinetic profiles are ne ~ 1 - s^5 and Te = Ti ~ 1 - s, with the beta
and collisionality of a Helios-like reactor (R = 8 m, B = 6 T) carried to
R0 = 1 m and B0 = 1 T. A Picard loop makes the seed current Redl's, ramping
beta in steps for the larger values. The current spline values and CURTOR are
then design variables, and the mismatch against the case's bootstrap model is
a hard constraint (sum of squared normalized residuals <= `REDL_TOLERANCE`).
For `qi6-beta*` that model is DKX, the uwplasma drift-kinetic solver, since
Redl assumes quasisymmetry. It needs the optional `dkx` package
(`pip install "vmex[kinetic]"`). The logged
key stays `redl_mismatch` for both models.

    python free_boundary_single_stage_optimization.py --bootstrap --beta 0.01 --steps 5 --output runs/free-redl
    python single_stage_optimization.py --bootstrap --beta 0.01 --steps 5 --output runs/fixed-redl

## Running

The scripts need ESSOS (`pip install "vmex[coils]"`), virtual casing for
`--beta` (`vmex[freeb]`), DKX (`vmex[kinetic]`) for `qi6-beta*` with
`--bootstrap`, and a GPU for the default resolution:

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    python single_stage_optimization.py --steps 5 --output runs/fixed
    python postprocess.py runs/free

    COIL_CASE=qi python free_boundary_single_stage_optimization.py --steps 5 --output runs/free-qi
    COIL_CASE=qi python single_stage_optimization.py --steps 5 --output runs/fixed-qi
    COIL_CASE=qi python postprocess.py runs/free-qi

Each run directory holds `input.run` (the deck as run), `coils.initial.json`,
`metrics.jsonl` (one line per step), `coils.stepN.json` / `wout.stepN.nc`
checkpoints, and the final `coils.json`, `wout.nc` and `summary.json`. The
free arm also writes `diagnostics.jsonl`. `--restart <run>`
continues a finished run.

`_coil_constraints.py` holds the coil chart: the design vector to ESSOS coils,
PHIEDGE and the current profile, and the differentiable field the free arm passes
to `FreeBoundaryProblem` as `field_from_parameters`. It also holds the coil
inequalities and the independent endpoint checks. Coil length, curvature and
curve points come from ESSOS `Curves`. The module adds only what ESSOS lacks as
hard SLSQP rows: mean squared curvature, the minimum coil-coil and nonadjacent
self-segment distances (ESSOS has hinge penalties only) and the coil-plasma
clearance.

## Stage-two coils

`fit_coils.py` fits circular coils to B.n = 0 on the case's fixed-boundary
seed, with the coil limits as penalties (L-BFGS-B, no equilibrium solves).
It then scales the currents to enclose PHIEDGE; the optimization scripts
rescale them to B0 R0 at the start of a run.

    COIL_CASE=qa4-beta python fit_coils.py --output coils.qa4-beta.json

The winding radii were 0.55 m for `qa4-beta*`, 0.46 m for `qi6-beta*` and
0.42-0.5 m for `qa3`, `qh` and `qi` (rms B.n/|B| 3.7e-3, 4.4e-3 and 5e-4).
The `qa3`, `qh` and `qi` coils start up to 5 % over the length and curvature limits, which
SLSQP then enforces. The fit is fast on a GPU and slow on a loaded CPU.

The free arm's derivatives use structured factors by default
(`--factorization structured`): O(ns) memory where the dense LU
(`--factorization dense`) needs O(ns^2). For a finite-beta QA case with a
bootstrap current this halves the peak (15.4 to 7.8 GiB at 8x8 modes and NS 51)
and fits NS 101 (8.9 GiB) and 12x12 modes at NS 51 (10.8 GiB), which the dense
LU cannot fit on a 32 GB GPU. `--ns`, `--modes MPOL NTOR` and
`--max-iterations` (the VMEC cap) set the resolution.

## Three-term free arm

`free_boundary_three_term_single_stage.py` is the free arm with its equilibrium
replaced by the three-term free boundary
(`vmex.core.freeboundary_vc.ThreeTermFreeBoundaryModel`). Every trial is the
boundary that satisfies B.n = 0, pressure balance and no sheet current, not a
VMEC + NESTOR solve. The design variables, loss, constraints and run directory
are the free arm's. Gradients come from the implicit function theorem of the
boundary least squares, and new coils or plasma parameters compile nothing:

    COIL_CASE=qa4-beta python free_boundary_three_term_single_stage.py --beta 0.025 --bootstrap \
        --modes 8 8 --ns 51 --output runs/three-term-qa4

`--seed-run <run>` starts from another run's calibrated deck and fitted coils.
`--check-gradient N` compares the gradients with re-solved central differences,
and `--profile-rows` times and sizes the compiled programs at the seed.
`metrics.jsonl` adds the three interface residuals (`bn`, `pressure_balance`,
`sheet_current`) and each step's forward-solve counts and peak GPU memory.

The tools after a run:

    python three_term_highres.py runs/three-term-qa4 runs/hr12 --modes 12 12 --ns 51   # same coils, finer
    python three_term_desc_check.py runs/hr12/wout_three_term.nc runs/hr12/coils.npz runs/desc12 \
        --compare three_term=runs/hr12/wout_three_term.nc                             # DESC, same filaments
    python three_term_postprocess.py runs/three-term-qa4 --bootstrap                       # dense re-solve, losses
    python evolution_gifs.py "QA, three-term" runs/qa4 runs/three-term-qa4                 # coils and LCFS GIFs

`three_term_highres.py` re-solves the final coils at another resolution and
exports them (`coils.npz`) for `three_term_desc_check.py`, which needs DESC.
`three_term_postprocess.py` re-solves the final boundary as a fixed boundary
at NS 201 for the `vmex.plot_wout` figures and `vmex --trace` alpha losses.
