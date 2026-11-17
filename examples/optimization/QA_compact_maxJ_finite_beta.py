#!/usr/bin/env python
"""Stage a compact finite-beta QA toward stability, maximum-J and low flux compression.

``QA_optimization_bootstrap.py`` (quasisymmetry with a self-consistent Redl
current) run as a list of stages, each continuing from the previous optimum
and keeping every row added so far. STAGES is a comma-separated list; tokens
joined by + (merc=0.002+ball) share one stage:

    mode=N    highest boundary mode number varied from here on (no stage)
    beta=B    rescale the pressure to this beta and redo the Picard loop (no stage)
    aspect=A  move the aspect-ratio target, rescaling the toroidal flux so the
              field strength stays as the minor radius changes (no stage)
    qa        quasisymmetry, Redl mismatch, aspect ratio, iota floor, beta
    merc=M    add or move the Mercier and resistive-interchange limits to the
              margin M on PHIEDGE**2 DMerc (raise M in small held steps)
    ball      add one row per ideal-ballooning field line
    maxj=S    add or move the maximum-J rows to the surfaces S ... 0.8
    chi=F     cap the flux compression in bad curvature at F times its value
              when the first chi stage starts
    ql=F      cap GKX's quasilinear ion heat flux (mean over the flux tubes
              QL_S) at F times its value when the first ql stage starts
              (needs the turbulence extra)
    shear=D   |iota| rises by at least D from s = 0.1 to 0.9, monotonically
              (negative shear in the tokamak convention)
    vaciota=V floor V on the current-free transform (the shaping part of iota,
              the stand-in for the vacuum transform) at every radius
    lgb=L     floor on the edge magnetic-gradient scale length [m], the
              proxy for how far the coils can sit (blanket space)

Rows are added one at a time because a design pushed to several new limits at
once loses the quasisymmetry that confines its alphas. Exact maximum-J is
incompatible with quasisymmetry near the axis, so its rows start on the outer
surfaces and S is walked inward. Each stage writes its input deck, wout and
one line of metrics; gates are read on a finer re-solve, not here.

Environment: COMPACT_QA_STAGES, COMPACT_QA_ASPECT, COMPACT_QA_BETA,
COMPACT_QA_NFEV, COMPACT_QA_IOTA, COMPACT_QA_ELONGATION, COMPACT_QA_START (input deck of an earlier stage, with its
profiles file beside it) and COMPACT_QA_OUTPUT override the values below.
"""

import faulthandler
import json
import os
import signal
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, self_consistent_bootstrap)
from vmex.core.maxj import MaximumJResidual, common_trapped_pitches_state, maximum_j_profile
from vmex.core.stability import ballooning_lambda
from vmex.core.statephysics import current_free_iota, iota_profile
from vmex.core.turbulence import flux_compression, quasilinear_flux_proxy

faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> prints every thread's stack

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Stage list (see the module docstring), and the evaluations each stage may spend:
STAGES = os.environ.get("COMPACT_QA_STAGES", "mode=1,qa,mode=2,qa,mode=3,qa,merc=0.002")
MAX_NFEV = int(os.environ.get("COMPACT_QA_NFEV", 30))

# Targets. The seed is a circle of minor radius 1 / ASPECT_TARGET at R0 = 1 m
# with B0 = 1 T, plus a rotating ellipse that gives the optimizer a QA basin:
ASPECT_TARGET = float(os.environ.get("COMPACT_QA_ASPECT", 4.5))
TARGET_BETA = float(os.environ.get("COMPACT_QA_BETA", 0.025))
SEED_PERTURBATION = 0.10
IOTA_FLOOR = float(os.environ.get("COMPACT_QA_IOTA", 0.42))  # minimum |iota| over the profile
MIRROR_LIMIT = 0.30
ELONGATION_LIMIT = float(os.environ.get("COMPACT_QA_ELONGATION", 6.0))
IOTA_CEILING = float(os.environ.get("COMPACT_QA_IOTA_MAX", 10.0))  # e.g. 0.48 keeps the profile below 1/2

# Continue from the input deck of an earlier stage instead of the seed:
START = os.environ.get("COMPACT_QA_START", "")

# Flux surfaces the quasisymmetry and bootstrap residuals are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# Term weights. The beta residual is relative because its target is small:
QA_WEIGHT = float(os.environ.get("COMPACT_QA_QA_WEIGHT", 10.0))
BOOTSTRAP_WEIGHT = float(os.environ.get("COMPACT_QA_BOOTSTRAP_WEIGHT", 1.0))
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.0               # a floor: the maximum-J rows (weight 100) overran weight 10
BETA_WEIGHT = 1.0 / TARGET_BETA**2
LIMIT_WEIGHT = 10.0
STABILITY_WEIGHT = 30.0
STABILITY_MIN_S = 0.2
BALLOONING_WEIGHT = float(os.environ.get("COMPACT_QA_BALL_WEIGHT", 30.0))
MAXJ_WEIGHT = float(os.environ.get("COMPACT_QA_MAXJ_WEIGHT", 100.0))
CHI_WEIGHT = 30.0
LGRADB_WEIGHT = 30.0
QL_WEIGHT = 30.0
SHEAR_WEIGHT = 100.0
SHEAR_S = np.linspace(0.1, 0.9, 9)

# Ballooning field lines: a line contributes max(lambda - threshold, 0):
BALLOONING_S = np.linspace(0.1, 0.9, 9)
BALLOONING_LINES = dict(npoints=int(os.environ.get("COMPACT_QA_BALL_POINTS", 97)),
                        nturns=float(os.environ.get("COMPACT_QA_BALL_TURNS", 3.0)),
                        zeta0s=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5))
BALLOONING_THRESHOLD = float(os.environ.get("COMPACT_QA_BALL_THRESHOLD", -5.0e-4))

# Maximum-J: upper bound on (1/J) dJ/ds, the trapped-well depths the common
# pitches are drawn from, and the bounce-action quadrature:
MAXJ_TARGET = -0.01
MAXJ_OUTER_S = float(os.environ.get("COMPACT_QA_MAXJ_OUTER", 0.8))  # outermost surface with matchable wells
MAXJ_SPACING = 0.1
TRAPPING_DEPTHS = (0.2, 0.4, 0.6, 0.8)
ACTION = dict(nalpha=9, points_per_period=64, num_periods=8)
ACTION_MAX_WELLS = 20
ACTION_QUADRATURE = 32
ACTION_MBOZ = 16

# Flux compression: surface and field lines of the proxy:
CHI = dict(s=0.5, nalpha=8, ntheta=96)

# Quasilinear heat flux: flux tubes (alpha = 0) and the gyrokinetic model, as
# QA_optimization_turbulence_quasilinear.py with the drive of the nonlinear check:
QL_S = (0.5, 0.09)
QL = dict(alpha=0.0, ntheta=32, a_over_lt=3.0, a_over_ln=0.0, ly=12.0, n_laguerre=4, n_hermite=8)

# Picard loop that makes the current self-consistent, and the I'(s) spline:
PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-3
N_CURRENT_SPLINE = 8
REDL_N_LAMBDA = 32

# Step control. Boundary coefficients move in metres; the current dofs are
# dimensionless, so they carry their own optimizer scale:
PARAMETER_STEP = 0.02
CURRENT_PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 10.0       # per-stage box guardrail, in scaled step units

# Equilibrium resolution: mode numbers are max_mode + 2 but never below
# MINIMUM_MPOL, on one radial grid:
MINIMUM_MPOL = 5
STAGE_NS = 51
STAGE_FTOL = 1.0e-12
STAGE_NITER = 8000

# Every output file name starts with this:
OUTPUT_NAME = os.environ.get("COMPACT_QA_OUTPUT", "QA_compact_maxJ")

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# Profile shapes of ne and of Te = Ti, polynomials in s (lowest order first):
# Landreman-Buller-Drevlak 1 - s^5 and 1 - s, unless the start deck's profiles
# file gives its own ("ne_shape", "T_shape").
NE_SHAPE, T_SHAPE = np.array([1.0, 0, 0, 0, 0, -1]), np.array([1.0, -1])


def kinetic_profiles(n0, T0):
    """ne = n0 NE_SHAPE(s), Te = Ti = T0 T_SHAPE(s) for the Redl formula."""
    return KineticProfiles(n0 * NE_SHAPE, T0 * T_SHAPE, T0 * T_SHAPE)


def picard(inp, profiles, initial_state=None):
    """Alternate hot-started solves and Redl fits until the current is self-consistent."""
    result = self_consistent_bootstrap(
        inp, profiles, 0, n_iter=PICARD_ITERATIONS, tol=PICARD_TOLERANCE,
        degree=N_CURRENT_SPLINE - 1, s_eval=SURFACES, initial_state=initial_state, verbose=True)
    return result.input, result.equilibrium


if START:
    inp = vj.VmecInput.from_file(START)
    if os.environ.get("COMPACT_QA_NS"):  # radial grid of the restart (ballooning needs ns >= 151)
        inp = replace(inp, ns_array=np.array([33, int(os.environ["COMPACT_QA_NS"])]),
                      ftol_array=np.array([1.0e-10, STAGE_FTOL]),
                      niter_array=np.array([4000, STAGE_NITER]))
    saved = json.loads(Path(START + ".profiles.json").read_text())
    n0, T0, max_mode = saved["n0"], saved["T0"], saved["max_mode"]
    NE_SHAPE = np.asarray(saved.get("ne_shape", NE_SHAPE), dtype=float)
    T_SHAPE = np.asarray(saved.get("T_shape", T_SHAPE), dtype=float)
    equilibrium = opt.solve_equilibrium(inp)
    if os.environ.get("COMPACT_QA_NS"):  # stages hot-start on the final grid alone
        inp = replace(inp, ns_array=inp.ns_array[-1:], ftol_array=inp.ftol_array[-1:],
                      niter_array=inp.niter_array[-1:])
else:
    # VmecInput is frozen, so copy its arrays before shaping the seed boundary.
    inp = vj.VmecInput.from_file(INPUT_FILE)
    minor_radius = 1.0 / ASPECT_TARGET
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = minor_radius
    rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION
    n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
    T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
    am = np.zeros(21)
    am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]     # (1 - s)(1 - s^5)
    ac = np.zeros(21)
    ac[0] = 1.0
    inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, phiedge=np.pi * minor_radius**2,
                  ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
                  niter_array=np.array([STAGE_NITER]), pmass_type="power_series", am=am,
                  pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
                  pcurr_type="power_series", ac=ac, curtor=0.0)
    # One seed solve calibrates the profile amplitude to the requested beta.
    scale = TARGET_BETA / float(opt.solve_equilibrium(inp).wout.betatotal)
    n0, T0 = n0 * scale ** (1 / 3), T0 * scale ** (2 / 3)
    inp = replace(inp, pres_scale=inp.pres_scale * scale)
    inp = replace(inp, tvolume=vj.boundary_volume(inp), lvolume_rfix=True)  # hold the plasma volume
    inp, equilibrium = picard(inp, kinetic_profiles(n0, T0))
    max_mode = 1

### Set up the objective ######################################################

def bootstrap_row():
    return RedlBootstrapMismatch(kinetic_profiles(n0, T0), helicity_n=0, surfaces=SURFACES,
                                 n_lambda=REDL_N_LAMBDA)


def window(profile):
    """Interior surfaces with s >= STABILITY_MIN_S, on any radial grid."""
    s = np.linspace(0.0, 1.0, profile.shape[0])
    return profile[(s >= STABILITY_MIN_S) & (s < s[-1])]


def minimum_dmerc(equilibrium_state, solver_context):
    """Minimum of PHIEDGE**2 DMerc over the stability window (> 0 is stable)."""
    return window(PHIEDGE**2 * opt.d_merc_state(equilibrium_state, solver_context)).min()


def ballooning(equilibrium_state, solver_context):
    """Growth rate of every sampled line, shaped (surface, alpha, zeta0)."""
    return ballooning_lambda(equilibrium_state, solver_context, surfaces=BALLOONING_S,
                             **BALLOONING_LINES)


def ballooning_rows(equilibrium_state, solver_context):
    return jnp.maximum(ballooning(equilibrium_state, solver_context) - BALLOONING_THRESHOLD,
                       0.0).ravel()


def worst_ballooning(equilibrium_state, solver_context):
    return jnp.max(ballooning(equilibrium_state, solver_context))


def chi(equilibrium_state, solver_context):
    return flux_compression(equilibrium_state, solver_context, **CHI)


def iota_samples(equilibrium_state, solver_context):
    """|iota| at SHEAR_S, interpolated on the half mesh."""
    iota = jnp.abs(iota_profile(equilibrium_state, solver_context))
    s_half = (jnp.arange(iota.shape[0]) + 0.5) / iota.shape[0]
    return jnp.interp(SHEAR_S, s_half, iota)


def shaping_iota(equilibrium_state, solver_context):
    """|current-free iota| at SHEAR_S."""
    iota = jnp.abs(current_free_iota(equilibrium_state, solver_context))
    s_half = (jnp.arange(iota.shape[0]) + 0.5) / iota.shape[0]
    return jnp.interp(SHEAR_S, s_half, iota)


def iota_rise(equilibrium_state, solver_context):
    return jnp.diff(iota_samples(equilibrium_state, solver_context))


def quasilinear_flux(equilibrium_state, solver_context):
    return jnp.mean(jnp.stack([quasilinear_flux_proxy(equilibrium_state, solver_context, s=s, **QL)
                               for s in QL_S]))


def maximum_j_term(inner_s, outer_s):
    """Maximum-J residual on inner_s ... outer_s at the pitches trapped on those surfaces."""
    surfaces = np.arange(inner_s, outer_s + 1e-9, MAXJ_SPACING)
    pitch = np.asarray(common_trapped_pitches_state(
        equilibrium.solution, equilibrium.solver_context, surfaces, TRAPPING_DEPTHS, **ACTION))
    return MaximumJResidual(surfaces, pitch, mboz=ACTION_MBOZ, nboz=ACTION_MBOZ,
                            max_wells=ACTION_MAX_WELLS, quadrature_order=ACTION_QUADRATURE,
                            target=MAXJ_TARGET, **ACTION)


def maximum_j_rows(inner_s, outer_s=MAXJ_OUTER_S):
    """Maximum-J rows that stay finite when trapped wells stop matching.

    The residual is NaN for a pitch whose wells cannot be matched between two
    surfaces. Such a row is held at its value in the current equilibrium (zero
    if it has none), so the optimizer gains nothing by breaking a match.
    """
    term = maximum_j_term(inner_s, outer_s)
    start = np.asarray(term.residuals_state(equilibrium.solution, equilibrium.solver_context))
    print(f"maximum-J surfaces {term.surfaces.round(2)}: {int(np.isfinite(start).sum())} of "
          f"{start.size} rows matched, {int((np.nan_to_num(start) > 0).sum())} violated")
    held = jnp.asarray(np.nan_to_num(start))

    def rows(equilibrium_state, solver_context):
        value = term.residuals_state(equilibrium_state, solver_context)
        return jnp.where(jnp.isfinite(value), value, held)

    return rows if np.isfinite(start).any() else None


PHIEDGE = float(inp.phiedge)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)

# Each term is (function, target, weight); stages add to this mapping.
terms = {
    "qa": (qs, 0.0, QA_WEIGHT),
    "bootstrap": (bootstrap_row(), 0.0, BOOTSTRAP_WEIGHT),
    "aspect": (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    "iota": (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), IOTA_WEIGHT),
    "beta": (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    "mirror": (opt.mirror_ratio, opt.at_most(MIRROR_LIMIT), LIMIT_WEIGHT),
    "elongation": (opt.max_elongation, opt.at_most(ELONGATION_LIMIT), LIMIT_WEIGHT),
    "iota_ceiling": (opt.max_abs_iota, opt.at_most(IOTA_CEILING), IOTA_WEIGHT),
}
chi_reference = ql_reference = mercier_margin = None


def metrics(equilibrium):
    """Scalars of one equilibrium, for the stage log."""
    state, context = equilibrium.solution, equilibrium.solver_context
    # Whole volume at the pitches of this field; pairs without matched wells read None.
    maxj = maximum_j_term(0.1, 0.9).compute_state(state, context)
    profile = maximum_j_profile(maxj)
    return {
        "qa": float(qs.total_state(state, context)),
        "bootstrap_mismatch": float(terms["bootstrap"][0].total_state(state, context)),
        "aspect": float(opt.aspect_ratio(state, context)),
        "beta": float(opt.volume_average_beta(state, context)),
        "min_iota": float(opt.min_abs_iota(state, context)),
        "iota_profile": [round(float(v), 4) for v in iota_samples(state, context)],
        "shaping_iota": [round(float(v), 4) for v in shaping_iota(state, context)],
        "mirror": float(opt.mirror_ratio(state, context)),
        "elongation": float(opt.max_elongation(state, context)),
        "min_dmerc": float(minimum_dmerc(state, context)),
        "max_ballooning": float(worst_ballooning(state, context)),
        "chi": float(chi(state, context)),
        "l_grad_b": float(opt.l_grad_b_state(state, context)),
        "maxj_fraction": float(maxj["maximum_j_fraction"]),
        "maxj_excluded_pitch": float(maxj["excluded_pitch_fraction"]),
        "maxj_s": float(profile["s_max_j"]),
        "maxj_profile": [None if np.isnan(f) else round(float(f), 3) for f in profile["fraction"]],
    }


def save(label, equilibrium):
    """Write the deck, its profiles, the wout and the metrics of one stage."""
    deck = inp.to_indata(f"input.{OUTPUT_NAME}_{label}")
    Path(f"{deck}.profiles.json").write_text(
        json.dumps({"n0": n0, "T0": T0, "max_mode": max_mode,
                    "ne_shape": NE_SHAPE.tolist(), "T_shape": T_SHAPE.tolist()}))
    vj.write_wout(f"wout_{OUTPUT_NAME}_{label}.nc", equilibrium.wout)
    row = {"stage": label, **metrics(equilibrium)}
    print("[stage] " + json.dumps(row), flush=True)
    with open(f"{OUTPUT_NAME}_stages.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")


### Run the optimization ######################################################

save("start", equilibrium)
for number, token in enumerate(STAGES.split(","), start=1):
    optimize = False
    for part in token.strip().split("+"):
        name, _, value = part.partition("=")
        if name == "mode":
            max_mode = int(value)
            continue
        if name == "beta":
            scale = float(value) / float(equilibrium.wout.betatotal)
            n0, T0 = n0 * scale ** (1 / 3), T0 * scale ** (2 / 3)
            inp, equilibrium = picard(replace(inp, pres_scale=inp.pres_scale * scale),
                                      kinetic_profiles(n0, T0), equilibrium.solution)
            terms["bootstrap"] = (bootstrap_row(), 0.0, BOOTSTRAP_WEIGHT)
            terms["beta"] = (opt.volume_average_beta, float(value), 1.0 / float(value) ** 2)
            continue
        if name == "aspect":
            now = float(opt.aspect_ratio(equilibrium.solution, equilibrium.solver_context))
            inp = replace(inp, phiedge=inp.phiedge * (now / float(value)) ** 2)
            PHIEDGE = float(inp.phiedge)
            terms["aspect"] = (opt.aspect_ratio, float(value), ASPECT_WEIGHT)
            if mercier_margin is not None:
                terms["mercier"], terms["resistive"] = opt.stability_terms(
                    inp, margin=mercier_margin, weight=STABILITY_WEIGHT, from_s=STABILITY_MIN_S)
            continue
        if name == "merc":
            mercier_margin = float(value)
            terms["mercier"], terms["resistive"] = opt.stability_terms(
                inp, margin=mercier_margin, weight=STABILITY_WEIGHT, from_s=STABILITY_MIN_S)
        elif name == "ball":
            terms["ballooning"] = (ballooning_rows, 0.0, BALLOONING_WEIGHT)
        elif name == "maxj":
            rows = maximum_j_rows(float(value))
            if rows is None:
                raise RuntimeError(f"no matched trapped well for {token}: improve quasisymmetry first")
            terms["maximum_j"] = (rows, 0.0, MAXJ_WEIGHT)
        elif name == "chi":
            if chi_reference is None:
                chi_reference = float(chi(equilibrium.solution, equilibrium.solver_context))
            terms["chi"] = (chi, opt.at_most(float(value) * chi_reference), CHI_WEIGHT)
        elif name == "ql":
            if ql_reference is None:
                ql_reference = float(quasilinear_flux(equilibrium.solution, equilibrium.solver_context))
                print(f"quasilinear heat flux reference = {ql_reference:.4f}")
            terms["quasilinear"] = (quasilinear_flux, opt.at_most(float(value) * ql_reference),
                                    QL_WEIGHT / ql_reference)
        elif name == "vaciota":
            terms["shaping_iota"] = (shaping_iota, opt.at_least(float(value)), IOTA_WEIGHT)
        elif name == "shear":
            terms["shear"] = (iota_rise, opt.at_least(float(value) / (SHEAR_S.size - 1)), SHEAR_WEIGHT)
        elif name == "lgb":
            terms["l_grad_b"] = (opt.l_grad_b_state, opt.at_least(float(value)), LGRADB_WEIGHT)
        elif name != "qa":
            raise ValueError(f"unknown stage {part!r}")
        optimize = True
    if not optimize:
        continue

    print(f"\n===== stage {number}: {token}, max_mode = {max_mode}, rows = {list(terms)} =====")
    mpol = max(inp.mpol, max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    inp = opt.resample_current_profile(inp, N_CURRENT_SPLINE)
    # A RuntimeWarning about uncertified Jacobian columns is expected once the
    # optimizer leaves the seed and needs no action; see examples/README.md.
    problem = opt.VmecProblem.from_tuples(
        inp, list(terms.values()), max_mode=max_mode, current_dofs=N_CURRENT_SPLINE - 1,
        use_ess=True, restart_from=equilibrium, forward_max_iterations=STAGE_NITER, progress=True)
    step = PARAMETER_STEP * problem.scales
    step[-N_CURRENT_SPLINE:] = CURRENT_PARAMETER_STEP  # n-1 spline shapes plus CURTOR
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=MAX_NFEV, ftol=1e-6, xtol=1e-10, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    save(f"s{number:02d}_{token.strip().replace('=', '').replace('+', '_')}", equilibrium)
