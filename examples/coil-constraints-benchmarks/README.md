# Coil constraints benchmarks

Matched fixed- and free-boundary scalar SLSQP examples. Edit **parameters.py**
for shared physics, optimization budgets and coil bounds. Both arms reuse the
same rotating-ellipse input, fitted three-coil seed, fixed currents, order-16
coil coefficients, QA surfaces, plasma targets and MPOL=8 / NTOR=8 / NS=51.
The fixed arm additionally minimizes its normal-field mismatch to the boundary.

## Physical size and bounds

The reference plasma major radius is **1 m**, with physical admissible band
0.99–1.01 m and aspect-ratio target 5. This is the metre-scale normalization
used in the 2022/2023 examples discussed, not reactor scale. Keeping three
independent coils does not reproduce either paper's coil count or total length.
Curvature has units m⁻¹ and MSC has units m⁻²; neither is dimensionless.

| Quantity | Physical requirement |
| --- | --- |
| Each independent coil length | ≤ 5 m |
| Maximum curvature along each coil | **≤ 5 m⁻¹** |
| Mean squared curvature (MSC) of each coil | ≤ 5 m⁻² |
| All distinct coils, including symmetry copies | distance ≥ 0.15 m |
| Coil to current plasma boundary | distance ≥ 0.20 m |
| Nonadjacent segments of a coil | no intersections; 1 μm numerical guard |
| Coil parametrization | nonzero speed; numerical floor 10⁻⁴ m per unit parameter |

MSC = ∫κ² ds / ∫ds. RMS curvature = √MSC; its corresponding bound is √5 m⁻¹.
Length is an upper bound, not an equality. These requirements are explicit
nonlinear constraints, not penalty terms. SLSQP is used in both arms;
L-BFGS-B alone cannot enforce these nonlinear inequalities.

References: [Wechsung et al. 2022, authors' implementation](https://github.com/florianwechsung/CoilsForPreciseQS)
and [Jorge et al. 2023](https://arxiv.org/abs/2302.10622).
The 5 m⁻¹ and 5 m⁻² curvature choices follow these examples. Our length,
clearance and coil count define this controlled benchmark, not an exact
reproduction of their designs. Finite winding packs, conductor strain, forces,
torsion and fabrication tolerances require additional engineering inputs;
filament geometry alone cannot establish manufacturability.

## Resolution and derivatives

Field quadrature is 256 points per coil (checked against 512 on a perturbed seed). Geometry constraints are evaluated
separately: 1024 curvature points, 256 polygon segments for coil separation and
self-intersections, and a 61×64 full-torus moving-surface grid for clearance.
Small interior margins are visible in parameters.py; the curvature constraint
used during optimization is 4.9 m⁻¹ so the final physical gate can remain 5.
Peak/minimum constraints are piecewise differentiable; changing the active
point may slow SLSQP and calls for resolution and derivative checks.

Pure coil rows use direct JAX derivatives. Fixed-boundary clearance also
includes the explicitly varied surface coefficients. Free-boundary clearance
uses `FreeBoundaryProblem.from_loss(coil_quantities=...)`, combining its direct
coil derivative and the equilibrium response in one additional adjoint row.
Pure coil rows do not add equilibrium solves or adjoint rows.

The endpoint is re-solved at NS=201. Independent CPU geometry verification uses
4096/8192-point arc-length quadrature, refined local curvature maxima, a
conservative continuous intercoil distance lower bound and nonadjacent polygon
intersection checks with Fourier chord-error margins. Moving-surface clearance
is refined in continuous Fourier coordinates from multiple dense-grid seeds.
This is a numerical clearance check, not a proof of its global minimum;
`continuous_surface_clearance_certified` remains false. A failed geometry or
resolution check makes the final report infeasible even if SciPy reports success.

## Run explicitly

From this directory, using the project's VMEX/ESSOS environment:

```sh
python single_stage_optimization_scalar.py --dry-run
python free_boundary_single_stage_optimization_scalar.py --dry-run

# Independent derivative qualification; does not run single-stage optimization.
python verify_free_boundary_single_stage.py --device gpu --output runs/free-qualification

# Separate, fresh outputs; no existing job is stopped or resumed.
python single_stage_optimization_scalar.py --device gpu --check-gradients --maxiter 100 --output runs/fixed
python free_boundary_single_stage_optimization_scalar.py --device gpu --accepted-steps 100 --qualification runs/free-qualification/qualification.json --output runs/free
```

The shared fitted seed is infeasible under the tighter bounds: its 512-point
maximum curvature is 5.18588 m⁻¹ and its three lengths are 5.09079, 5.04095
and 5.00535 m. `seed_geometry.json` records the source hash and sampled metrics.
A run must restore feasibility before a lower QA value is a valid constrained
comparison. Free-boundary derivative qualification is separate and includes
all coil inequalities and moving-surface clearance. It requires every row to
meet the unchanged 0.1% relative error gate at two successive perturbation sizes
from `3e-4, 1e-4, 3e-5, 1e-5`. A failed larger perturbation is retained in the
report and refined, since a sampled minimum or maximum can change its active
point. Every endpoint is solved independently from the same accepted seed;
persistent mismatches block production. Fixed-boundary production can run a seed objective/constraint FD check with
`--check-gradients`. Full equilibrium qualification is separate from the CPU
geometry checks below.

```sh
python -B -m pytest test_coil_constraints.py -q
```

The existing baseline examples are in `../single-stage-benchmarks`; their
latest main-branch API updates are retained. Published code does not establish
that the separate GPU benchmarks have converged or passed their final gates.
`source_manifest.json` records the source/input hashes at the time of copying;
run provenance additionally records the modified sources and settings.
The `validation*.json` files record earlier CPU validation snapshots, with their
source hashes and scope. Current-branch checks run through `tools/preflight.py`;
they include the geometry suite and do not launch a GPU optimization.

Order 16 uses 297 shape coefficients for three independent coils. Input order-5
coils are zero-padded on load, preserving their physical curves, currents and
low-mode scaling. The original input JSON files remain unchanged.
