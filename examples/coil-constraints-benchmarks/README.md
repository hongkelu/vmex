# Coil-constraint single-stage benchmarks

A vacuum rotating ellipse (1 m major radius) with three independent order-16
coils. The case and its limits are in `parameters.py`: minimum |iota| >= 0.41,
aspect ratio in [4.9, 5.1], major radius 1 +/- 0.01 m, coil length <= 5 m,
curvature <= 5 /m, mean squared curvature <= 5 /m^2, coil-coil distance
>= 0.15 m and coil-plasma distance >= 0.20 m.

- `free_boundary_single_stage_optimization.py` varies the coil shapes and
  PHIEDGE, with the coil currents fixed at B0 = 1 T. Every trial is a free-boundary equilibrium of those
  coils, and SLSQP minimizes quasisymmetry with all the limits as hard
  inequalities, through `vmex.optimize.FreeBoundaryProblem`.
- `single_stage_optimization.py` is the fixed-boundary counterpart: SLSQP on
  the boundary and the coils, with the same hard limits and a B.n/|B| term in
  the objective.
- `postprocess.py` reads a finished run of either script and writes the loss
  and constraint histories, a GIF of the coils and LCFS, and a dense (NS201)
  free-boundary solve of the final coils with the `vmex.plot_wout` figures;
  `--trace` adds the `vmex --trace` alpha losses of that equilibrium at
  ARIES-CS size.

The coil currents set the field strength (the edge R B_phi = mu0 I / 2 pi, B0 R0
with B0 = 1 T), and PHIEDGE the plasma size. Holding both fixed pins the size,
and at this iota the aspect ratio then sits at its 4.9 floor (QA ~0.02); a free
PHIEDGE reaches aspect 5.1 (QA ~0.004), and at finite beta keeps B0 and beta
near their targets. `FREE_PHIEDGE = False` and `FLUX_TOLERANCE` restore the
pinned case in the free and fixed scripts.

`--beta` runs either arm at finite beta with a fixed p ~ 1 - s pressure and zero
net current. The free arm keeps the vacuum loss and constraints; it refits the
seed coils to (B_coils + B_plasma).n = 0 with virtual casing and reports beta,
B.n and pressure balance as diagnostics. The fixed arm limits the total B.n and
holds the edge R B_phi at the coils' mu0 I / 2 pi. `COIL_CASE=qa6` selects the
aspect-6 Landreman & Paul (2021) QA case at B0 = 1 T (set through PHIEDGE) with
`--beta` read as on-axis beta, fixed coil currents, iota >= 0.42 and the coil
limits of Jorge et al. (2023) with three order-6 coils and 6.5 m per coil.

`--bootstrap` (with `--beta`) adds a self-consistent Redl bootstrap current to
both arms. The kinetic profiles are ne ~ 1 - s^5 and Te = Ti ~ 1 - s, with the
beta and collisionality of a Helios-like reactor (R = 8 m, B = 6 T) carried to
R0 = 1 m and B0 = 1 T. A Picard loop makes the seed current Redl's. The current
spline values and CURTOR are then design variables, and the Redl mismatch is a
hard constraint (sum of squared relative residuals <= `REDL_TOLERANCE`):

    python free_boundary_single_stage_optimization.py --bootstrap --beta 0.01 --steps 5 --output runs/free-redl
    python single_stage_optimization.py --bootstrap --beta 0.01 --steps 5 --output runs/fixed-redl

The `qa6` commands:

    COIL_CASE=qa6 python free_boundary_single_stage_optimization.py --beta 0.01 --steps 5 --output runs/free
    COIL_CASE=qa6 python single_stage_optimization.py --beta 0.01 --coils runs/free/coils.initial.json --steps 5 --output runs/fixed

`input.rotating_ellipse` is the seed and `coils.initial.json` the coils fitted
to it. `_coil_constraints.py` evaluates the coil inequalities and the
independent endpoint checks. The scripts need ESSOS (`pip install "vmex[coils]"`)
and a GPU for the default resolution:

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    python single_stage_optimization.py --steps 5 --output runs/fixed
    python postprocess.py runs/free
