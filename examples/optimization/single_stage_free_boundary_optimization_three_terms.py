#!/usr/bin/env python
r"""Finite-beta single-stage optimization at the three-term free boundary, with a self-consistent bootstrap current.

The free-boundary design problem of ``single_stage_free_boundary_optimization.py`` (same seed, coils and coil
pre-fit, limits near the seed's) at volume-average beta ``TARGET_BETA`` with a Redl-self-consistent bootstrap current, every trial's
plasma the three-term free boundary of its coils: the boundary on which ``B . n = 0``, pressure balance and
continuity of the tangential field hold together (:class:`vmex.ThreeTermsConfig`), with the plasma field from
virtual casing (``pip install vmex[freeb]``).

Pressure and current are inputs of every free-boundary solve, as in VMEC and in Landreman, Buller & Drevlak (2022).
The kinetic profiles ``n_e = n0 (1 - s^5)``, ``T_e = T_i = T0 (1 - s)`` (hydrogen) give both the deck's pressure
``p = 2 e n_e T_e`` and the Redl bootstrap ``<J.B>``.  The design variables are the coil Fourier coefficients and
the current profile (its first ``N_CURRENT`` AC coefficients and CURTOR), and SLSQP (:func:`vmex.optimize.minimize`)
holds the Redl self-consistency as a constraint row, beside the rotational-transform window and the aspect limit:

    minimize   (1/2) |r_QS|^2
    subject to IOTA_CONSTRAINT <= min |iota|,  max |iota| <= IOTA_CEILING_CONSTRAINT,
               aspect <= ASPECT_CONSTRAINT,  sum_j ((<J.B>_j - <J.B>_Redl,j) / ...)^2 <= REDL_TOLERANCE,
               coil-surface distance >= COIL_SURFACE_DISTANCE_CONSTRAINT,  the coil limits (``essos.coil_design.CoilLimits``)

with the clearance measured to each trial's free boundary.  The coils are ESSOS' (``essos.coil_design``):
:class:`vmex.CoilDesign` puts the current profile in front of ESSOS' coil coordinates, and the coil length,
curvature and separation rows (``CoilLimits``, ``coil_constraint``) need no equilibrium solve.  :class:`vmex.ThreeTermsFreeBoundaryProblem` warm-starts
each trial's boundary fit, accepts it within an interface-residual cap and differentiates the fit implicitly; the coil
field is ESSOS' ``BiotSavart``, a pytree, so new coils never recompile.  The seed's current comes from a
fixed-boundary Picard iteration to Redl's (:func:`vmex.core.bootstrap.self_consistent_bootstrap`).

Run it with ``VMEX_EXAMPLES_CI=1`` for a short smoke pass that reports and exits 0.  Otherwise a run that misses a
target says which and exits 1.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from essos.coil_design import CoilLimits, coil_constraint
from scipy.optimize import Bounds, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im
from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch, self_consistent_bootstrap

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

jax.config.update("jax_enable_x64", True)

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.05, 1.0, 6)

# Seed boundary the coils are fitted to: the rotating ellipse of
# single_stage_optimization.py.
SEED_MINOR_RADIUS = 0.195
SEED_ELLIPSE = 0.10

# Volume-average beta, and the kinetic profiles both the pressure and the Redl current come from:
# n_e = n0 (1 - s^5), T_e = T_i = T0 (1 - s), hydrogen (Z_eff = 1, n_i = n_e), so p = 2 e n_e T_e,
# whose power series in s is PRESSURE_SHAPE times 2 e n0 T0, and the edge pressure vanishes.  n0 is set
# so that the fixed-boundary seed has TARGET_BETA.
TARGET_BETA = 0.005
T0 = 2.0e3                        # eV
PRESSURE_SHAPE = [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0]
BETA_TOLERANCE = 0.1              # relative, for the final check

# The current profile: its first N_CURRENT AC coefficients and CURTOR are design variables (relative
# steps of CURRENT_SCALE), held Redl-consistent by a constraint row, the sum of squared relative <J.B>
# mismatches on REDL_SURFACES, at most REDL_TOLERANCE (REDL_LIMIT in the final check).  The seed's
# current is the fixed-boundary Picard iteration's, PICARD_ITERATIONS at most.
N_CURRENT = 4
CURRENT_SCALE = 0.1
REDL_SURFACES = np.linspace(0.1, 0.9, 9)
REDL_TOLERANCE = 1.0e-3
REDL_LIMIT = 2.0e-3
PICARD_ITERATIONS = 8

# Targets the finished design is checked against (near the seed's values: the demonstration holds the
# constraints while it improves quasisymmetry):
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
IOTA_CEILING = 0.6                # maximum |iota|: the bootstrap current raises iota by ~0.1;
                                  # stay below the 2/3 resonance (NFP = 2)
ASPECT_LIMIT = 6.1                # maximum aspect ratio
COIL_SURFACE_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_LIMIT = 0.17
CURVATURE_LIMIT = 7.0

LENGTH_LIMIT = 5.5                # each coil's length
MEAN_SQUARED_CURVATURE_LIMIT = 25.0

# Values the optimizer is given instead, a little inside the limits above: SLSQP
# holds its constraint rows only to its tolerance (the coil rows' margins: curvature 0.1 1/m,
# coil-coil distance COIL_DISTANCE_MARGIN). The normal-field value is used only by the coil
# pre-fit: on a free boundary B.n = 0 holds to the fit's floor.
IOTA_CONSTRAINT = 0.425
IOTA_CEILING_CONSTRAINT = 0.59
ASPECT_CONSTRAINT = 6.05
NORMAL_FIELD_CONSTRAINT = 0.008
COIL_DISTANCE_MARGIN = 0.02
COIL_SURFACE_DISTANCE_CONSTRAINT = 0.16

# Coils: number of unique shapes, Fourier order, and the circle they start on.
N_COILS = 3
COIL_ORDER = 4
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

# Weight of the coil pre-fit's quadratic penalty on its violated rows (normal field, coil limits,
# clearance to the seed): the pre-fit is unconstrained L-BFGS-B.
CONSTRAINT_WEIGHT = 1.0e3

# Bound on each design variable, in units of its scale (COIL_SCALE m on every coil Fourier coefficient's
# ESSOS degree of freedom, CURRENT_SCALE on the current variables), and the factor on those scales that
# SLSQP steps in: its first step has an identity Hessian, so each scaled variable moves about as far as its
# gradient, and smaller scales keep the steps where the constraints' linearization holds.
COIL_SCALE = 0.05
PARAMETER_BOUND = 3.0
DESIGN_STEP_SCALE = 0.2

# A low-resolution demonstration: the plasma's Fourier resolution (the boundary's modes are
# solved up to MPOL - 1 = NTOR), and the equilibria of every trial: radial surfaces, force
# tolerance and iteration cap; the check at the end re-solves the final coils' free boundary
# on the same grid.
MPOL = NTOR = 3
NS = 11
FTOL = 1.0e-10
NITER = 4000

# The three-term free boundary of every trial: the labelling ("equal_arc", "anchored" or
# "none") and the weight of its rows (None: the labelling's default, 1e-3 for equal
# arc and 1e-2 for anchored), the interface grid over one field period, and its fixed
# singular quadrature (4 nfp N, 2 N).
LABELLING = "equal_arc"
LABEL_WEIGHT = None
N_INTERFACE = 24

# Budgets: accepted SLSQP steps (each trial is one three-term free boundary and,
# when SLSQP asks, one gradient), and the coil-only pre-fit.
MAXITER = 5
COIL_FIT_MAXITER = 200            # coil-only pre-fit, no equilibrium solves

# Surface grid the coil terms use:
NPHI, NTHETA = 37, 32

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    # Only the optimizer budget shrinks: two coil-fit iterations on a coarser
    # grid do not give coils that hold a converged free boundary.
    MAXITER = 1

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()

###############################################################################
### Set up the seed boundary and fit the coils to it ##########################
###############################################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
am = np.zeros(21)
am[:len(PRESSURE_SHAPE)] = PRESSURE_SHAPE
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am, pres_scale=1.0,
              ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 6)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
# ESSOS' design takes unscaled coefficients: COIL_SCALE on each scaled degree of freedom
coil_scales = COIL_SCALE / np.broadcast_to(np.asarray(curves0.scaling), curves0.dofs.shape).ravel()
circular = vj.CoilDesign(coils_circular, scales=coil_scales)
limits = CoilLimits(length=LENGTH_LIMIT, curvature=CURVATURE_LIMIT,
                       mean_squared_curvature=MEAN_SQUARED_CURVATURE_LIMIT, coil_distance=COIL_DISTANCE_LIMIT,
                       distance_margin=COIL_DISTANCE_MARGIN, curvature_points=N_SEGMENTS,
                       distance_points=N_SEGMENTS)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of B.n/|B| on a surface."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    normal_field = jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normal_field**2))


def hinge(constraints):
    """Quadratic penalty for constraints written as c >= 0, normalized by their limits."""
    return 0.5 * CONSTRAINT_WEIGHT * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


def coil_fit_objective(u):
    """The coil half of single_stage_optimization.py with the boundary frozen: the normal field, the coil
    limits and the clearance to the seed as penalties."""
    coils = circular.coils_from_x(jnp.asarray(circular.scales) * u)
    normal_field = 1.0 - normal_field_rms(coils, surface_seed) / NORMAL_FIELD_CONSTRAINT
    clearance = limits.surface_clearance(coils, surface_seed) / COIL_SURFACE_DISTANCE_CONSTRAINT - 1.0
    return hinge(jnp.r_[normal_field, limits.rows(coils), clearance])


print("Running single_stage_free_boundary_optimization_three_terms.py")
coil_fit_value_and_grad = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_value_and_grad(jnp.asarray(u)))),
    np.zeros(circular.size), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
coils0 = circular.coils_from_x(circular.scales * coil_fit.x)
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils0, surface_seed)):.3f}% on the seed")

###############################################################################
### Check that the fitted coils confine the seed ##############################
###############################################################################

# The plasma a free-boundary solve finds depends on the toroidal flux it is
# told to enclose. Take it from the fitted field itself, through the seed's
# phi = 0 cross-section (Gauss-Legendre in radius, uniform in angle), so the
# seed boundary is the one the coils are asked to confine.
rho, rho_weights = np.polynomial.legendre.leggauss(24)
rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
# At phi = 0 every toroidal harmonic has phase m*theta, so sum over n.
m = np.arange(inp.rbc.shape[1])
rbc_phi0, zbs_phi0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
R_edge = np.cos(np.outer(theta, m)) @ rbc_phi0
Z_edge = np.sin(np.outer(theta, m)) @ zbs_phi0
dR_edge = -np.sin(np.outer(theta, m)) @ (m * rbc_phi0)
dZ_edge = np.cos(np.outer(theta, m)) @ (m * zbs_phi0)
R_center = rbc_phi0[0]
R = R_center + rho[:, None] * (R_edge - R_center)
Z = rho[:, None] * Z_edge
area_element = rho[:, None] * ((R_edge - R_center) * dZ_edge - Z_edge * dR_edge)
points = np.stack([R, np.zeros_like(R), Z], axis=-1).reshape(-1, 3)
field0 = BiotSavart(coils0)
B_phi = np.asarray(jax.vmap(field0.B)(jnp.asarray(points)))[:, 1].reshape(R.shape)
toroidal_flux = float(np.sum(rho_weights[:, None] * B_phi * area_element)
                      * (2.0 * np.pi / theta.size))
# The free-boundary solve takes the external field along the same +phi, and
# stops (vacuum.f, ier_flag 7) unless PHIEDGE has the sign of that field.
inp = replace(inp, phiedge=toroidal_flux)
print(f"[flux] PHIEDGE = {inp.phiedge:.6f} Wb, the fitted field's flux through the seed")

# Beta is proportional to the pressure at fixed boundary and flux, so one fixed-boundary solve of the
# seed at PRES_SCALE = 1 gives the scale, 2 e n0 T0, and with it n0.
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)
n0 = inp.pres_scale / (2.0 * ELEMENTARY_CHARGE * T0)
profiles = KineticProfiles(n0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]), T0 * np.array([1.0, -1.0]),
                           T0 * np.array([1.0, -1.0]))
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=2 if ci_smoke else PICARD_ITERATIONS)
inp = replace(picard.input, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array([NS]),
              ftol_array=np.array([FTOL]), niter_array=np.array([NITER]))
print(f"[bootstrap] n0 = {n0:.3e} m^-3 at T0 = {T0:.0f} eV for beta {100 * TARGET_BETA:.2f}%; seed current "
      f"CURTOR = {inp.curtor:.4e} A after {picard.iterations} Picard iterations (converged: {picard.converged})")

###############################################################################
### The three-term single-stage problem #######################################
###############################################################################


# The design: the current variables (the first N_CURRENT AC coefficients and CURTOR, in units of the largest
# of those coefficients and of CURTOR), then the fitted coils' coordinates.
bound_scales = np.r_[np.full(N_CURRENT + 1, CURRENT_SCALE), coil_scales]
chart = vj.CoilDesign(coils0, plasma_current=np.r_[np.asarray(inp.ac)[:N_CURRENT], inp.curtor],
                      scales=DESIGN_STEP_SCALE * bound_scales)
design0 = chart.x0


def input_from_x(x):
    """The deck of the design ``x`` (its current profile), for the verification solve."""
    values = chart.plasma_current + np.asarray(x[:N_CURRENT + 1], dtype=float) * chart.plasma_current_units
    ac = np.asarray(inp.ac, dtype=float).copy()
    ac[:N_CURRENT] = values[:-1]
    return replace(inp, ac=ac, curtor=float(values[-1]))


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
redl = RedlBootstrapMismatch(profiles, 0, surfaces=REDL_SURFACES)


def redl_mismatch(equilibrium_state, runtime):
    return redl.total_state(equilibrium_state, runtime)


def max_abs_iota(equilibrium_state, runtime):
    return jnp.max(opt.abs_iota_profile(equilibrium_state, runtime))


def boundary_surface(equilibrium_state, runtime, nphi=NPHI, ntheta=NTHETA):
    """The free boundary of a solved state, as an ESSOS surface."""
    rmnc, _, _, zmns = im._edge_physical(equilibrium_state, runtime)
    modes = runtime.modes
    ntor = int(np.max(np.abs(modes.n)))
    rows = jnp.asarray(np.asarray(modes.n) + ntor)
    cols = jnp.asarray(np.asarray(modes.m))
    shape = (2 * ntor + 1, int(np.max(modes.m)) + 1)
    rbc = jnp.zeros(shape).at[rows, cols].set(rmnc)
    zbs = jnp.zeros(shape).at[rows, cols].set(zmns)
    return surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=nphi, ntheta=ntheta)


def clearance(rbc, zbs, x):
    """The coils' distance to this trial's free boundary (rows of the fitted boundary)."""
    surface = surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=NPHI, ntheta=NTHETA)
    return limits.surface_clearance(chart.coils_from_x(x), surface)


def loss(equilibrium_state, runtime, x):
    """Quasisymmetry; the coils enter through the constraints."""
    qs_rows = qs.residuals_state(equilibrium_state, runtime)
    return 0.5 * jnp.vdot(qs_rows, qs_rows)


def plasma_values(equilibrium_state, runtime):
    return {"QA total": float(qs.total_state(equilibrium_state, runtime)),
            "aspect": float(opt.aspect_ratio(equilibrium_state, runtime)),
            "mean iota": float(opt.mean_iota(equilibrium_state, runtime)),
            "min |iota|": float(opt.min_abs_iota(equilibrium_state, runtime)),
            "max |iota|": float(max_abs_iota(equilibrium_state, runtime)),
            "beta": float(opt.volume_average_beta(equilibrium_state, runtime)),
            "Redl mismatch": float(redl_mismatch(equilibrium_state, runtime))}


# At this low resolution (MPOL = 3) the interface residuals' floor is ~1e-2 (the seed's sheet current is
# 1.03e-2), above the default ``max_residual`` (1e-2): a free boundary counts as solved below 2e-2 here, for the
# final solve and for every trial (``boundary_max_residual``).
config = vj.ThreeTermsConfig(labelling=LABELLING, label_weight=LABEL_WEIGHT, nphi=N_INTERFACE,
                             ntheta=N_INTERFACE, quadrature=(4 * NFP * N_INTERFACE, 2 * N_INTERFACE), max_residual=2e-2)
problem = vj.ThreeTermsFreeBoundaryProblem.from_loss(
    inp, loss, design0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_from_parameters,
    scales=chart.scales, names=chart.dof_names,
    quantities=(opt.min_abs_iota, max_abs_iota, opt.aspect_ratio, redl_mismatch), boundary_quantities=(clearance,),
    config=config, boundary_max_residual=config.max_residual)
state0, params0 = problem.state(design0)
runtime0 = im.runtime_from_params(params0, problem.model.cfg)
free_seed = boundary_surface(state0, runtime0)
boundary_shift = float(jnp.max(jnp.linalg.norm(free_seed.gamma - surface_seed.gamma, axis=-1)))
seed_values = plasma_values(state0, runtime0)
print("[seed] " + ", ".join(f"{k} = {v:.4g}" for k, v in seed_values.items())
      + f"; free boundary within {boundary_shift:.4f} m of the fitted seed; interface residuals "
      + ", ".join(f"{k} {v:.2e}" for k, v in vars(problem.boundary_residual(design0)).items()))

###############################################################################
### Run the optimization ######################################################
###############################################################################

print(f"Three-term free boundary + ESSOS at beta {100 * TARGET_BETA:.2f}%: {chart.coils.size} coil and "
      f"{N_CURRENT + 1} current variables, SLSQP")
initial_value = problem.fun(design0)
# Each row is scaled by its own limit, so SLSQP weighs a violation of the Redl row (values ~1e-3) like one of
# the iota rows (~0.4).  The coil rows are already scaled by their limits.
constraints = [problem.nonlinear_constraint(
    [IOTA_CONSTRAINT, -np.inf, -np.inf, -np.inf, COIL_SURFACE_DISTANCE_CONSTRAINT],
    [np.inf, IOTA_CEILING_CONSTRAINT, ASPECT_CONSTRAINT, REDL_TOLERANCE, np.inf],
    scales=[IOTA_CONSTRAINT, IOTA_CEILING_CONSTRAINT, 0.1, REDL_TOLERANCE, COIL_SURFACE_DISTANCE_LIMIT]),
    coil_constraint(lambda x: limits.rows(chart.coils_from_x(x)))]


def report(x):
    rows = problem.constraint_values(x)
    print(f"[step {problem.accepted_step}] loss {problem.fun(x):.6e}; min |iota| {rows[0]:.4f}, max |iota| "
          f"{rows[1]:.4f}, aspect {rows[2]:.4f}, Redl mismatch {rows[3]:.3e}, clearance {rows[4]:.4f} m", flush=True)


bounds = Bounds(design0 - PARAMETER_BOUND * bound_scales, design0 + PARAMETER_BOUND * bound_scales)
result = opt.minimize(problem, method="SLSQP", constraints=constraints, bounds=bounds,
                      callback=report,
                      options={"maxiter": MAXITER, "ftol": 1e-12})
x, final_value = problem.accepted.parameters, float(problem.fun(problem.accepted.parameters))
print(f"[solve] {result.accepted_steps} accepted SLSQP steps, {problem.stats['trials']} trials "
      f"({problem.stats['failed']} rejected): {result.message}", flush=True)
optimization_seconds = time.perf_counter() - started
state_x, params_x = problem.state(x)  # the optimizer's own free boundary of the result
runtime_x = im.runtime_from_params(params_x, problem.model.cfg)
optimizer_values = plasma_values(state_x, runtime_x)
print("[optimizer] " + ", ".join(f"{k} = {v:.4g}" for k, v in optimizer_values.items()) + "; interface residuals "
      + ", ".join(f"{k} {v:.2e}" for k, v in vars(problem.boundary_residual(x)).items()))

###############################################################################
### Check the result against the targets ######################################
###############################################################################

# Re-solve the optimized coils' free boundary, with the optimized current profile, independently
# of the optimizer, and check every target there, the interface residuals and the Redl
# self-consistency included.  The fit starts from the optimizer's boundary: at this resolution the
# interface least squares has several minima, and a cold start from the seed lands centimetres away,
# on another free boundary of the same coils.
coils_final = chart.coils_from_x(x)
start = im.input_with_params(inp, params_x)
final_input = replace(input_from_x(x), rbc=start.rbc, zbs=start.zbs)
final = vj.solve_free_boundary_multigrid(final_input, external_field=chart(x),
                                         three_terms=replace(config, start="deck"))
final_deck = final.three_terms.input  # the free boundary as a fixed-boundary deck
final_context = im.runtime_from_params(im.params_from_input(final_deck), im.make_config(final_deck))
final_values = plasma_values(final.state, final_context)
final_residual = final.three_terms.boundary_residual
wout = vj.wout_from_result(final_input, final)
surface_final = SurfaceRZFourier.from_wout_file(
    vj.write_wout("wout_single_stage_free_boundary_three_terms.nc", wout), nphi=61, ntheta=64)
gamma = np.asarray(coils_final.gamma)
points = np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
coil_distance = min(float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
                    for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota, final_aspect = final_values["min |iota|"], final_values["aspect"]

print("[final] " + ", ".join(f"{k} = {v:.4g}" for k, v in final_values.items()))
optimizer_surface = boundary_surface(state_x, runtime_x)
verification_surface = boundary_surface(final.state, final_context)
surface_gap = float(np.max(np.linalg.norm(np.asarray(optimizer_surface.gamma) - np.asarray(verification_surface.gamma),
                                          axis=-1)))
print(f"The verification's free boundary is within {surface_gap:.4f} m of the optimizer's")
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after {result.accepted_steps} SLSQP steps")
print("Interface residuals of the final free boundary: " + ", ".join(
    f"{k} {v:.2e}" for k, v in vars(final_residual).items()))
print(f"Volume-average beta = {100 * final_values['beta']:.4f}% (target {100 * TARGET_BETA:.2f}% "
      f"+- {100 * BETA_TOLERANCE:.0f}%)")
print(f"Redl mismatch = {final_values['Redl mismatch']:.3e} (target <= {REDL_LIMIT:.1e}); CURTOR "
      f"{float(inp.curtor):.4e} -> {float(final_input.curtor):.4e} A")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target <= {ASPECT_LIMIT:.4f})")

checks = (
    ("minimum |iota|", minimum_iota, IOTA_FLOOR, "below"),
    ("maximum |iota|", final_values["max |iota|"], IOTA_CEILING, "above"),
    ("aspect ratio", final_aspect, ASPECT_LIMIT, "above"),
    ("beta", final_values["beta"], TARGET_BETA * (1 - BETA_TOLERANCE), "below"),
    ("beta", final_values["beta"], TARGET_BETA * (1 + BETA_TOLERANCE), "above"),
    ("Redl mismatch", final_values["Redl mismatch"], REDL_LIMIT, "above"),
    ("interface sheet current", final_residual.sheet_current, final.three_terms.config.max_residual, "above"),
    ("minimum coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
    ("minimum coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
    ("maximum curvature", maximum_curvature, CURVATURE_LIMIT, "above"),
)
unmet = [f"{name} {value:.4g} {side} the {limit:.4g} limit"
         for name, value, limit, side in checks
         if (value < limit if side == "below" else value > limit)]
if not final.three_terms.success:
    unmet.append(f"the verification free boundary is not solved: {final.three_terms.message}")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    if ci_smoke:
        print("Smoke mode caps the budget far below what the targets need; exit status 0.")
else:
    print("\nAll stated targets met.")

summary = {
    "example": "single_stage_free_boundary_optimization_three_terms.py", "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1), "coil_fit_iterations": int(coil_fit.nit),
    "accepted_steps": int(result.accepted_steps), "stop_reason": result.stop_reason,
    "trials": int(problem.stats["trials"]), "rejected_trials": int(problem.stats["failed"]),
    "labelling": LABELLING, "label_weight": config.resolved_label_weight, "phiedge": float(inp.phiedge),
    "target beta": TARGET_BETA, "n0": n0, "T0": T0, "picard iterations": int(picard.iterations),
    "seed curtor": float(inp.curtor), "final curtor": float(final_input.curtor),
    "seed boundary shift": boundary_shift, "seed": seed_values, "optimizer": optimizer_values,
    "verification surface gap": surface_gap,
    "final": {**final_values, "interface residuals": vars(final_residual),
              "coil-surface distance": coil_surface_distance,
              "coil-coil distance": coil_distance, "maximum curvature": maximum_curvature},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_free_boundary_three_terms_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
coils_final.to_json("coils_single_stage_free_boundary_three_terms.json")
print("Wrote wout_single_stage_free_boundary_three_terms.nc and coils_single_stage_free_boundary_three_terms.json")
if unmet and not ci_smoke:
    raise SystemExit(1)
