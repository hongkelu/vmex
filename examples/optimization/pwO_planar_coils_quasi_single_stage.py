#!/usr/bin/env python
"""Re-optimize a finite-beta pwO boundary so that planar coils can make its field.

A quasi single stage: no coil is a variable, but every trial boundary is charged for the coil set it
would need. ``essos.planar_coils.planar_coil_set_cost`` puts circular encircling coils around the trial
boundary and a current sheet (the continuous limit of a shaping-coil array) on the surface a fixed
distance outside it, eliminates their currents by one linear solve, and returns the field error left plus
the sheet strength used. That cost is added to the plasma rows (pwO, Mercier floor, GammaC, B_min rising
outward, shear, aspect, beta), so the winding surface moves with the boundary and the plasma gives up
shaping the coils cannot afford. The boundary field stands in for the field the coils must make; at this
beta the plasma currents contribute about 1 % of it. Coils are fitted afterwards with the ESSOS examples.
Requires essos (with planar_coils), virtual-casing-jax and vmex[neoclassical].
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from essos.planar_coils import planar_coil_set_cost
from vmex import optimize as opt
from vmex.core import virtual_casing as vc
from vmex.core.pwo import MappedPwOResidual
from vmex.core.statephysics import _iotas_half

# The design to start from, and the pwO map it was optimized with (held fixed here):
START_FILE = os.environ.get("QSS_START", str(Path(__file__).resolve().parents[1] / "data" / "input.pwO_nfp4_aspect7_finite_beta"))
MAP_PARAMETERS = dict(a=0.10482, b=0.23460, c=-3.93062, eta_min=3.04240)
MAP_PARAMETERS.update({k: float(v) for k, v in (item.split("=") for item in filter(None, os.environ.get("QSS_MAP", "").split(",")))})

# Coil-set cost: encircling coils per half period, distances in minor radii (Helios: 0.67, 0.34, 0.376), the
# weight of the sheet strength inside the cost, and the weight of the cost in the objective:
N_ENCIRCLING = int(os.environ.get("QSS_N_ENCIRCLING", 2))
SHEET_DISTANCE = float(os.environ.get("QSS_SHEET_DISTANCE", 0.67))       # coil-plasma distance (blanket)
ENCIRCLING_CLEARANCE = SHEET_DISTANCE + 0.34 + 0.376
SHEET_REGULARIZATION = float(os.environ.get("QSS_REGULARIZATION", 2.5e-3))
COIL_WEIGHT = float(os.environ.get("QSS_COIL_WEIGHT", 1.0e7))
COIL_GRID = dict(nphi=32, ntheta=32)

# Plasma targets (the rows of the design's own optimization):
ASPECT_TARGET = float(os.environ.get("QSS_ASPECT", 7.0))
TARGET_BETA = float(os.environ.get("QSS_BETA", 0.052))
IOTA_FLOOR = float(os.environ.get("QSS_IOTA_FLOOR", 0.42))
IOTA_RISE = float(os.environ.get("QSS_IOTA_RISE", 0.15))        # |iota| at the edge minus the innermost surface: positive shear
# Floor on DMerc for 0.2 <= s < 1, or "none". The starting design sits on its own floor (+8 at this resolution), where the
# hinge is a kink with a very large one-sided slope that stalls the optimizer: give it a margin or leave it out and check
# Mercier afterwards at converged resolution.
MERCIER_FLOOR = os.environ.get("QSS_MERCIER_FLOOR", "4")
MERCIER_FLOOR = None if MERCIER_FLOOR == "none" else float(MERCIER_FLOOR)
BMIN_SLOPE_FLOOR = 0.01
MIRROR_LIMIT = float(os.environ.get("QSS_MIRROR", 0.40))
ELONGATION_LIMIT = float(os.environ.get("QSS_ELONGATION", 6.0))
GAMMA_C_WEIGHT = float(os.environ.get("QSS_GAMMA_C", 1.0e5))
SURFACES = np.linspace(0.1, 0.9, 5)
BOOZER = dict(mboz=8, nboz=8, oversample=1, ngamma=16, neta=16)
SCALE = 1.0e3                                                    # common factor of the finite-beta rows

MAX_MODE = 4
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
MAX_ITERATIONS = int(os.environ.get("QSS_ITERATIONS", 60))
OUTPUT_NAME = os.environ.get("QSS_NAME", "pwO_planar_coils")

###############################################################################
# End of input parameters.
###############################################################################

inp = vj.VmecInput.from_file(START_FILE)
if os.environ.get("QSS_FTOL"):  # DMerc of this design is not settled at the deck's force tolerance
    inp = replace(inp, ftol_array=np.full(len(inp.ftol_array), float(os.environ["QSS_FTOL"])))
equilibrium = opt.solve_equilibrium(inp)
minor = float(equilibrium.wout.Aminor_p)
pwo = MappedPwOResidual(SURFACES, limit_map=True, **BOOZER, **MAP_PARAMETERS)


def dmerc(state, rt):
    values = opt.d_merc_state(state, rt)
    s = np.linspace(0.0, 1.0, values.shape[0])
    return values[(s >= 0.2) & (s < 1.0)]


def bmin_slope(state, rt):
    low = pwo.diagnostics_state(state, rt)[1]
    return (low[-1] - low[0]) / jnp.mean(low)


def iota_rise(state, rt):
    iota = jnp.abs(_iotas_half(state, rt))
    return iota[-1] - iota[1]


def max_mirror(state, rt):
    return pwo.diagnostics_state(state, rt)[3]


def coil_set(state, rt):
    data = vc.surface_field_data_from_state(inp, state, runtime=rt, **COIL_GRID)
    return planar_coil_set_cost(data.gamma, data.normal, data.B_total, int(inp.nfp), n_encircling=N_ENCIRCLING,
                                sheet_distance=SHEET_DISTANCE * minor, encircling_clearance=ENCIRCLING_CLEARANCE * minor,
                                regularization=SHEET_REGULARIZATION)


terms = [
    (pwo, 0.0, SCALE * 1.0e3),
    (opt.aspect_ratio, ASPECT_TARGET, SCALE * 1.0e3),
    (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), SCALE * 1.0e4),
    (iota_rise, opt.at_least(IOTA_RISE), SCALE * 1.0e3),
    (bmin_slope, opt.at_least(BMIN_SLOPE_FLOOR), SCALE * 1.0e3),
    (max_mirror, opt.at_most(MIRROR_LIMIT), SCALE * 1.0e3),
    (opt.axis_excursion, opt.at_most(0.25), SCALE * 100.0),
    (opt.edge_jacobian_ratio, opt.at_least(0.25), SCALE * 100.0),
    (opt.max_elongation, opt.at_most(ELONGATION_LIMIT), SCALE * 10.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0e4 / TARGET_BETA**2),
    (opt.GammaCSmooth(tuple(SURFACES[1:])), 0.0, SCALE * GAMMA_C_WEIGHT),
]
if MERCIER_FLOOR is not None:
    terms.append((lambda state, rt: jnp.log1p(opt.at_least(MERCIER_FLOOR)(dmerc(state, rt))), 0.0, SCALE * 100.0))
terms.append((lambda state, rt: coil_set(state, rt)["residuals"], 0.0, COIL_WEIGHT))   # sum of squares = the coil-set cost
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE, use_ess=True, ess_alpha=1.2,
                                      restart_from=equilibrium, progress=True)
problem.compile_residual_and_jacobian()


def describe(label, eq):
    out = coil_set(eq.solution, eq.solver_context)
    state, rt = eq.solution, eq.solver_context
    row = dict(coil_cost=float(out["cost"]), field_error=float(out["field_error"]), potential_rms=float(out["potential_rms"]),
               encircling_currents=np.round(np.asarray(out["currents"]), 3).tolist(),
               pwo=float(pwo.total_state(state, rt)), aspect=float(opt.aspect_ratio(state, rt)),
               beta=float(opt.volume_average_beta(state, rt)), iota_min=float(opt.min_abs_iota(state, rt)),
               iota_max=float(opt.max_abs_iota(state, rt)), iota_rise=float(iota_rise(state, rt)),
               min_dmerc=float(jnp.min(dmerc(state, rt))), bmin_slope=float(bmin_slope(state, rt)),
               mirror=float(jnp.max(max_mirror(state, rt))), elongation=float(opt.max_elongation(state, rt)))
    print(f"[{label}] " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in row.items()), flush=True)
    return row


results = dict(start=describe("start", equilibrium), coil_weight=COIL_WEIGHT, n_encircling=N_ENCIRCLING)
if os.environ.get("QSS_CHECK"):  # every term's Jacobian against central differences, along random directions
    rng = np.random.default_rng(0)
    r0, jacobian = problem.residual(problem.x0), problem.residual_jac(problem.x0)
    slices = problem.metadata.get("term_slices") or [("all", 0, r0.size)]
    for trial in range(2):
        direction = rng.standard_normal(problem.x0.size)
        direction /= np.linalg.norm(direction)
        for eps in (1e-2, 1e-3, 1e-4):
            dx = eps * PARAMETER_STEP * problem.scales * direction
            plus, minus = problem.residual(problem.x0 + dx), problem.residual(problem.x0 - dx)
            central, linear, curved = (plus - minus) / 2, jacobian @ dx, (plus + minus) / 2 - r0
            print(f"[check] direction {trial}, step {eps:g} of the parameter step")
            for label, start, stop in slices:
                k = slice(start, stop)
                print(f"[check]   {str(label)[:28]:28s} rows {stop - start:5d} |r| {np.linalg.norm(r0[k]):9.3e} |J dx| {np.linalg.norm(linear[k]):9.3e} "
                      f"|central| {np.linalg.norm(central[k]):9.3e} |central - J dx| {np.linalg.norm(central[k] - linear[k]):9.3e} "
                      f"|even part| {np.linalg.norm(curved[k]):9.3e}", flush=True)
    raise SystemExit
step = PARAMETER_STEP * problem.scales
change = MAX_PARAMETER_CHANGE * step
result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step, max_nfev=MAX_ITERATIONS,
                       bounds=(problem.x0 - change, problem.x0 + change), ftol=1e-6, xtol=1e-10, verbose=2)
print(f"least squares: {result.nfev} evaluations, cost {result.cost:.6e}, {result.message}")
x = np.asarray(result.x)
final_input = problem.input_from_x(x)
final = problem.equilibrium_from_x(x)
results["final"] = describe("final", final)
final_input.to_indata(f"input.{OUTPUT_NAME}")
vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final.wout)
with open(f"{OUTPUT_NAME}.json", "w") as stream:
    json.dump(results, stream, indent=1)
print(f"Wrote input.{OUTPUT_NAME}, wout_{OUTPUT_NAME}.nc and {OUTPUT_NAME}.json")
