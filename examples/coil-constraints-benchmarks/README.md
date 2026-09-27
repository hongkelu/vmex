# Coil-constraint single-stage benchmarks

A vacuum rotating ellipse (1 m major radius) with three independent order-16
coils at fixed currents. The case and its limits are in `parameters.py`:
minimum |iota| >= 0.41, aspect ratio in [4.9, 5.1], major radius 1 +/- 0.01 m,
coil length <= 5 m, curvature <= 5 /m, mean squared curvature <= 5 /m^2,
coil-coil distance >= 0.15 m and coil-plasma distance >= 0.20 m.

- `free_boundary_single_stage_optimization.py` varies only the coils. Every
  trial is a free-boundary equilibrium of those coils, and SLSQP minimizes
  quasisymmetry with all the limits as hard inequalities, through
  `vmex.optimize.FreeBoundaryProblem`.
- `single_stage_optimization.py` is the fixed-boundary counterpart. It follows
  `examples/optimization/single_stage_optimization.py`: L-BFGS-B on the
  boundary and coils, with the limits as one-sided quadratic penalties.

`input.rotating_ellipse` is the seed and `coils.initial.json` the coils fitted
to it. `_coil_constraints.py` evaluates the coil inequalities and the
independent endpoint checks. Both scripts need ESSOS (`pip install "vmex[coils]"`)
and a GPU for the default resolution:

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
