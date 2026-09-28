# Coil-constraint single-stage benchmarks

A vacuum rotating ellipse (1 m major radius) with three independent order-16
coils. The case and its limits are in `parameters.py`: minimum |iota| >= 0.41,
aspect ratio in [4.9, 5.1], major radius 1 +/- 0.01 m, coil length <= 5 m,
curvature <= 5 /m, mean squared curvature <= 5 /m^2, coil-coil distance
>= 0.15 m and coil-plasma distance >= 0.20 m.

- `free_boundary_single_stage_optimization.py` varies the coil shapes and one
  common current factor. Every trial is a free-boundary equilibrium of those
  coils, and SLSQP minimizes quasisymmetry with all the limits as hard
  inequalities, through `vmex.optimize.FreeBoundaryProblem`.
- `single_stage_optimization.py` is the fixed-boundary counterpart: SLSQP on
  the boundary and the coils, with the same hard limits and a B.n/|B| term in
  the objective.
- `postprocess.py` reads a finished run of either script and writes the loss
  and constraint histories, a GIF of the coils and LCFS, and a dense (NS201)
  free-boundary solve of the final coils with the `vmex.plot_wout` figures.

In vacuum only the flux per ampere sets the plasma size. Holding both the coil
currents and PHIEDGE fixed pins it, and at this iota the aspect ratio then sits
at its 4.9 floor (QA ~0.02); a free current factor reaches aspect 5.1 (QA
~0.004). `SHARED_CURRENT = False` and `FLUX_TOLERANCE` restore the pinned case
in the free and fixed scripts.

`input.rotating_ellipse` is the seed and `coils.initial.json` the coils fitted
to it. `_coil_constraints.py` evaluates the coil inequalities and the
independent endpoint checks. The scripts need ESSOS (`pip install "vmex[coils]"`)
and a GPU for the default resolution:

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    python single_stage_optimization.py --steps 5 --output runs/fixed
    python postprocess.py runs/free
