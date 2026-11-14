#!/usr/bin/env python
"""Compare quasi-axisymmetry, piecewise omnigenity and their combinations at low aspect ratio.

One script, one seed, one stage ladder (the one of pwO_optimization_finite_beta.py:
boundary modes in vacuum, then beta ramped to the target with p proportional
to 1-s and no net current); TARGET alone changes the confinement rows:

  "QA"      two-term quasisymmetry on every surface.
  "pwO"     the mapped piecewise-omnigenous field (poloidally closed contours,
            B_max in a band at the period boundary, flattened: zero bootstrap).
  "QA-pwO"  the same map with toroidal helicity: quasi-axisymmetric (toroidally
            omnigenous for a != 0) around the outboard side, and around
            theta = pi a band of straight tilted strips, flat between them
            (Velasco et al., arXiv:2603.20125). The strips can cancel the
            quasi-axisymmetric bootstrap current; their tilt is a free map
            parameter and their amplitude is left to the DKX row.
  "zoned"   quasisymmetry on the surfaces inside ZONE_BOUNDARY, "pwO" outside.

DKX_FINAL_STAGES > 0 also drives the DKX bootstrap <J.B> to zero in the last
stages, for any target; the final design always reports it. START_FILE
continues a saved finite-beta design (for example the "QA" result) at the
target beta instead of climbing the ladder from the seed.
Requires vmex[neoclassical] and dkx.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from dkx.bootstrap import KineticBootstrapMismatch

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import KineticProfiles
from vmex.core.neoclassical import screen_transport
from vmex.core.pwo import FreeMapProblem, MappedPwOResidual
from vmex.core.restart import restart_state

# Confinement target (see above), field periods and aspect ratio:
TARGET = os.environ.get("QA_PWO_TARGET", "QA-pwO")
NFP = 2
ASPECT_TARGET = float(os.environ.get("QA_PWO_ASPECT", 4.0))
ASPECT_WEIGHT = 10.0
print(f"{TARGET} optimization, NFP = {NFP}, aspect {ASPECT_TARGET}")
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Seed: a circular torus at the target aspect ratio, with boundary modes (n, m)
# given a rotating-ellipse perturbation, RBC(n, m) = -a and ZBS(n, m) = +a,
# a being this fraction of the minor radius:
SEED_MODES = {(-1, 1): 0.5}
START_FILE = os.environ.get("QA_PWO_START")  # an input file of this script; None starts from the seed

# Volume-averaged beta of the last stage:
TARGET_BETA = 0.025
BETA_WEIGHT = 1.0e4               # on the relative beta error

# Flux surfaces of the confinement rows, and the Boozer resolution in the loop:
SURFACES = np.linspace(0.1, 0.9, 5)
ZONE_BOUNDARY = 0.4               # "zoned": quasisymmetry inside, pwO outside
BOOZER = dict(mboz=8, nboz=8, oversample=1)
MAP_GRID = dict(ngamma=16, neta=16)

# Initial values of the map (a, b, c, eta_min) and the ones the optimizer may
# change. The toroidal map starts quasi-axisymmetric (a = 0) and also carries
# the tilt of its band strips, fitted with BAND_HARMONICS harmonics:
MAP_PARAMETERS = dict(a=0.3, b=0.0, c=-2.0, eta_min=2.8)
if TARGET == "QA-pwO":
    MAP_PARAMETERS.update(a=0.0, tilt=2.0)
BAND_HARMONICS = 3
FREE_MAP = tuple(filter(None, os.environ.get("QA_PWO_FREE_MAP", "a,b,c,eta_min").split(",")))
PWO_WEIGHT = 1.0e3                # the seed's mapped variance is of order 0.01
BAND_WEIGHT = 1.0e4               # flat B_max band of the poloidal map
STRIP_WEIGHT = float(os.environ.get("QA_PWO_STRIP_WEIGHT", 1.0e3))  # strips of the toroidal band; gains the finite-beta factor
QS_WEIGHT = 10.0                  # the seed's quasisymmetry error is of order 1
BETA_TARGET_FACTOR = 1.0e3        # at finite beta the confinement and limit rows gain this factor

# The last DKX_FINAL_STAGES stages drive the DKX bootstrap <J.B> on DKX_SURFACES
# to zero. The plasma is the Landreman-Buller-Drevlak one at ARIES-CS size; the
# row is in MA T/m^2 at that size (see pwO_optimization.py):
DKX_FINAL_STAGES = int(os.environ.get("QA_PWO_DKX_STAGES", 0))
DKX_SURFACES = (0.25, 0.55, 0.85)
DKX_WEIGHT = float(os.environ.get("QA_PWO_DKX_WEIGHT", 30.0))
DKX_RESOLUTION = dict(Ntheta=15, Nzeta=21, Nxi=48, NL=4, Nx=5)
ARIES_MINOR_RADIUS = 1.7044       # metres
ARIES_DENSITY, ARIES_TEMPERATURE = 2.38e20, 9.45e3  # ne (1 - s^5) in m^-3, Te = Ti (1 - s) in eV

# Stage ladder: highest boundary mode number varied in each stage, its beta as a
# fraction of TARGET_BETA, and the residual evaluations it may spend:
MAX_MODES = [1, 2, 3, 3, 3, 3]
BETA_FRACTIONS = [0.0, 0.0, 0.0, 0.33, 0.67, 1.0]
MAX_NFEV = [15, 15, 15, 15, 20, 30]
NS_ARRAY = np.array([35])
NITER_ARRAY = np.array([20000])
FTOL_ARRAY = np.array([1e-12])
MINIMUM_MPOL = 7
if START_FILE:
    MAX_MODES, BETA_FRACTIONS, MAX_NFEV = [3, 3], [1.0, 1.0], [30, 30]

# Limits, the same for every target:
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
IOTA_WEIGHT = 1.0e4
MIRROR_LIMIT = 0.30               # maximum (Bmax-Bmin)/(Bmax+Bmin); 1/R alone gives 0.24 at s = 0.9
MIRROR_WEIGHT = 1.0e3
AXIS_LIMIT = 0.25                 # maximum RMS helical excursion of the magnetic axis, metres
ELONGATION_LIMIT = 6.0
EDGE_JACOBIAN_FLOOR = 0.25        # minimum min/max Jacobian ratio in each of the three outer cells

# Step control, as in pwO_optimization_finite_beta.py:
PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2

# Verification solve of the optimized boundary:
FINAL_NS = [37, 45, 55, 71]        # restarted in small steps: a large jump flips the Jacobian
FINAL_FTOL = 1.0e-13
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = (f"{TARGET}_nfp{NFP}_A{ASPECT_TARGET:g}" + ("_dkx" if DKX_FINAL_STAGES else "")
               + ("_continued" if START_FILE else ""))

# VMEX_EXAMPLES_CI=1 is the short smoke pass:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, BETA_FRACTIONS, MAX_NFEV = [1, 1], [0.0, 1.0], [3, 3]
    DKX_FINAL_STAGES = min(DKX_FINAL_STAGES, 1)
    FINAL_NS, FINAL_FTOL = [37], 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

if TARGET not in ("QA", "pwO", "QA-pwO", "zoned"):
    raise ValueError(f"unknown TARGET {TARGET!r}")

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(START_FILE or INPUT_FILE)
major_radius = float(inp.rbc[inp.ntor, 0])
minor_radius = major_radius / ASPECT_TARGET
if not START_FILE:
    # VmecInput is frozen, so copy its arrays before shaping the seed boundary.
    mpol = max(inp.mpol, MINIMUM_MPOL)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[:, 1:], zbs[:, 1:] = 0.0, 0.0
    rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = minor_radius
    for (n, m), fraction in SEED_MODES.items():
        rbc[inp.ntor + n, m], zbs[inp.ntor + n, m] = -fraction * minor_radius, fraction * minor_radius
    # The toroidal flux follows the cross-section, so the field stays near 1 T.
    phiedge = inp.phiedge * (minor_radius / float(inp.rbc[inp.ntor, 1])) ** 2
    inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=phiedge, am=np.array([1.0, -1.0]), pres_scale=0.0,
                  ns_array=NS_ARRAY, ftol_array=FTOL_ARRAY, niter_array=NITER_ARRAY,
                  delt=0.5).change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

### Set up the objective ######################################################

inner, outer = SURFACES[SURFACES < ZONE_BOUNDARY], SURFACES[SURFACES >= ZONE_BOUNDARY]
qs_surfaces = {"QA": SURFACES, "zoned": inner}.get(TARGET)
map_surfaces = {"pwO": SURFACES, "QA-pwO": SURFACES, "zoned": outer}.get(TARGET)
helicity = "toroidal" if TARGET == "QA-pwO" else "poloidal"

# Every target reports all three measures on all surfaces.
qs_all = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
pwo_all = MappedPwOResidual(SURFACES, limit_map=True, **BOOZER, **MAP_GRID, a=0.3, b=0.0, c=-2.0, eta_min=2.8)
qa_pwo_all = MappedPwOResidual(SURFACES, limit_map=True, helicity="toroidal", **BOOZER, **MAP_GRID,
                               a=0.0, b=0.0, c=-2.0, eta_min=2.8)
mapped = None if map_surfaces is None else MappedPwOResidual(
    map_surfaces, limit_map=True, helicity=helicity, **BOOZER, **MAP_GRID, **MAP_PARAMETERS,
    band_harmonics=BAND_HARMONICS if helicity == "toroidal" else 0)

size = minor_radius / ARIES_MINOR_RADIUS  # seed minor radius / ARIES-CS
kinetic = KineticBootstrapMismatch(
    KineticProfiles(ARIES_DENSITY / size * np.array([1, 0, 0, 0, 0, -1.]),
                    ARIES_TEMPERATURE * np.array([1, -1.]), ARIES_TEMPERATURE * np.array([1, -1.])),
    DKX_SURFACES, mismatch=False, collision_model="pas+momentum_correction",
    resolution=DKX_RESOLUTION, spectral_angles=True)
kinetic._setup(int(NS_ARRAY[-1]), int(inp.nfp))  # DKX builds its host-side operators eagerly, not under a trace


def dkx_j_dot_B(equilibrium_state, solver_context):
    """DKX <J.B> on DKX_SURFACES in MA T/m^2 at ARIES-CS size."""
    return kinetic.residuals_state(equilibrium_state, solver_context) * size**2


def max_mirror(equilibrium_state, solver_context):
    """Full-surface Boozer mirror ratio on SURFACES."""
    return pwo_all.diagnostics_state(equilibrium_state, solver_context)[3]


def boozer_spectrum(equilibrium_state, solver_context):
    """The field and iota the map reads; FreeMapProblem turns these rows into the mapped residual."""
    return mapped.spectrum_state(equilibrium_state, solver_context)


def stage_terms(beta, with_dkx):
    """The objective rows of one stage; only the confinement rows depend on TARGET."""
    scale = BETA_TARGET_FACTOR if beta > 0 else 1.0
    terms = [] if mapped is None else [(boozer_spectrum, 0.0, 1.0)]  # must stay first: FreeMapProblem reads these rows
    if qs_surfaces is not None:
        terms.append((opt.QuasisymmetryRatioResidual(qs_surfaces, helicity_m=1, helicity_n=0), 0.0, scale * QS_WEIGHT))
    terms += [
        (opt.aspect_ratio, ASPECT_TARGET, scale * ASPECT_WEIGHT),
        (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), scale * IOTA_WEIGHT),
        (max_mirror, opt.at_most(MIRROR_LIMIT), scale * MIRROR_WEIGHT),
        (opt.axis_excursion, opt.at_most(AXIS_LIMIT), scale * 100.0),
        (opt.edge_jacobian_ratio, opt.at_least(EDGE_JACOBIAN_FLOOR), scale * 100.0),
        (opt.max_elongation, opt.at_most(ELONGATION_LIMIT), scale * 10.0),
    ]
    if beta > 0:
        terms.append((opt.volume_average_beta, beta, BETA_WEIGHT / beta**2))
    if with_dkx:
        terms.append((dkx_j_dot_B, 0.0, scale * DKX_WEIGHT))
    return terms


def dmerc(equilibrium_state, solver_context):
    """DMerc from s = 0.2 to inside the edge; positive is stable."""
    values = opt.d_merc_state(equilibrium_state, solver_context)
    s = np.linspace(0.0, 1.0, values.shape[0])
    return values[(s >= 0.2) & (s < 1.0)]


report = opt.EquilibriumReporter(
    ("QA total", qs_all.total, ".4e"), ("pwO total", pwo_all.total, ".4e"),
    ("QA-pwO total", qa_pwo_all.total, ".4e"),
    ("aspect", opt.aspect_ratio, ".4f"), ("beta", opt.volume_average_beta, ".3%"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("max |iota|", opt.max_abs_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"),
    ("min DMerc", lambda state, rt: jnp.min(dmerc(state, rt)), ".3e"),
    ("elongation", opt.max_elongation, ".3f"),
    ("max mirror", lambda state, rt: jnp.max(max_mirror(state, rt)), ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
report("seed", equilibrium)
# Stages that share a boundary resolution, beta and rows share ONE problem:
# every jitted graph is compiled once for the whole group.
n_stage = len(MAX_MODES)
dkx_stage = [i >= n_stage - DKX_FINAL_STAGES for i in range(n_stage)]
group_of = [(max(MAX_MODES[i] + 2, inp.mpol), BETA_FRACTIONS[i], dkx_stage[i]) for i in range(n_stage)]
problem, group = None, None
for i, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    beta = BETA_FRACTIONS[i] * TARGET_BETA
    print(f"\n===== {TARGET} stage {i}, max_mode = {max_mode}, beta = {beta:.2%}, DKX row {dkx_stage[i]} =====")
    if group_of[i] != group:
        group = group_of[i]
        mpol = group[0]
        inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        if beta > 0:  # pressure that gives this beta on the current boundary
            unit = opt.solve_equilibrium(replace(inp, pres_scale=1.0), initial_state=restart_state(equilibrium.wout, inp))
            inp = replace(inp, pres_scale=beta / float(unit.wout.betatotal))
        last = max(j for j in range(n_stage) if group_of[j] == group)
        problem = opt.VmecProblem.from_tuples(
            inp, stage_terms(beta, dkx_stage[i]), max_mode=max(MAX_MODES[i:last + 1]),
            use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium, progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    step = PARAMETER_STEP * stage.scales
    change = MAX_PARAMETER_CHANGE * step
    joint, n_boundary = stage, stage.x0.size
    if mapped is not None:  # the map parameters join the boundary; sigmoid-scaled, a unit step spans their bounds
        factor = BETA_TARGET_FACTOR if beta > 0 else 1.0
        joint = FreeMapProblem(stage, mapped, equilibrium, MAP_PARAMETERS, FREE_MAP, weight=PWO_WEIGHT * factor,
                               band_weight=BAND_WEIGHT if helicity == "poloidal" else STRIP_WEIGHT * factor)
        step = np.concatenate((step, np.ones(len(FREE_MAP))))
        change = np.concatenate((change, np.full(len(FREE_MAP), np.inf)))
    result = least_squares(
        joint.residual, joint.x0, jac=joint.residual_jac,
        x_scale=step, max_nfev=max_nfev, bounds=(joint.x0 - change, joint.x0 + change),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x[:n_boundary])
    if mapped is not None:
        MAP_PARAMETERS = joint.values(result.x)
        mapped.options.update(MAP_PARAMETERS)
        print(f"map = {MAP_PARAMETERS}")
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"stage {i}", equilibrium)
    inp.to_indata(f"input.{OUTPUT_NAME}_stage{i}")
    vj.write_wout(f"wout_{OUTPUT_NAME}_stage{i}.nc", equilibrium.wout)

# The bootstrap current of the last stage, on the grid DKX was set up for.
j_dot_B = np.asarray(dkx_j_dot_B(equilibrium.state, equilibrium.runtime))
print(f"DKX <J.B> at s = {DKX_SURFACES}: {j_dot_B} MA T/m^2 (ARIES-CS size)")

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_equilibrium = equilibrium
for ns in FINAL_NS:
    final_input = replace(inp, ns_array=np.array([ns]), ftol_array=np.array([FINAL_FTOL]),
                          niter_array=np.array([FINAL_NITER]))
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=final_equilibrium.solution,
        verbose=not ci_smoke, raise_on_max_iterations=True)

half_mesh = (np.arange(FINAL_NS[-1] - 1) + 0.5) / (FINAL_NS[-1] - 1)
neo_surfaces = half_mesh[np.abs(half_mesh[:, None] - SURFACES).argmin(axis=0)]
final_Q = screen_transport(final_equilibrium.wout, neo_surfaces)

### Print, plot and save ######################################################

report("final", final_equilibrium)
print(f"epsilon_eff^(3/2) at s = {neo_surfaces}: {final_Q}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
