#!/usr/bin/env python
r"""Free-boundary single-stage coil optimization on sheet-current-free equilibria.

The free arm of ``single_stage_free_boundary_optimization_coil_constraints.py``
with its equilibrium replaced: every trial is the free boundary that satisfies
all three plasma-vacuum interface conditions (B.n = 0, pressure balance and no
sheet current), not a VMEC + NESTOR solve, which balances only |B| and leaves
the tangential field jump free. The cases (``COIL_CASE``; the case table is in
that script's docstring), the design variables (coil shapes, PHIEDGE and, with
``--bootstrap``, the current spline values and CURTOR), the loss, the
constraint rows and their bounds, and the run directory are that script's --
except with ``BOOTSTRAP_IN_SOLVE`` (Redl): then the current is solved with the
boundary, Redl's on every half-grid surface, and neither its values nor the
bootstrap-mismatch row enter the optimization. The coil currents are design
variables too (``FREE_COIL_CURRENTS``, no bounds), with a row holding the axis
|B| at ``B0``, and where the case sets ``VACUUM_IOTA_FLOOR`` a row floors the
vacuum |iota| of the plasma boundary (``opt.vacuum_iota``).

The problem is :class:`vmex.ThreeTermsFreeBoundaryProblem` (``from_loss(..., config=vj.ThreeTermsConfig(...))``,
the PR's three-term core; this campaign driver stays off main): warm-started
trial boundaries and design gradients by the implicit function theorem of the
boundary fit, without recompiling when the coils or plasma parameters change.

    COIL_CASE=qa4-beta python single_stage_free_boundary_optimization_three_term.py --beta 0.025 --bootstrap \
        --modes 8 8 --ns 51 --output runs/three-term-qa4

``--seed-run <run>`` starts from another run's calibrated deck and fitted coils.
``metrics.jsonl`` adds the three interface residuals (``bn``,
``pressure_balance``, ``sheet_current``), each step's trial and boundary-fit
counts and the peak GPU memory. After a run, ``benchmarks/three_term_highres.py``
re-solves the final coils at another resolution and exports them,
``benchmarks/three_term_postprocess.py`` re-solves the final boundary densely
for the WOUT figures and alpha losses, and
``benchmarks/coil_constraints_evolution_gifs.py`` draws the coils and LCFS of
the saved steps.
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np

# JAX, ESSOS and VMEX are imported inside the functions, so that main() can set JAX_ENABLE_X64 and
# JAX_PLATFORMS before the first JAX import.

# ---- cases -----------------------------------------------------------------------------------------------------------
# COIL_CASE selects the case. The values are ellipse5's; the blocks after them override them per case.
DATA = Path(__file__).resolve().parents[1] / "data"
CASE = os.environ.get("COIL_CASE", "ellipse5")
# A "-merc" or "-jbs" suffix runs the case with a Mercier floor or a bootstrap-current term (MERCIER_FLOOR,
# CURRENT_IOTA_WEIGHT below), e.g. qa4-beta-merc.
VARIANT = next((v for v in ("-merc", "-jbs") if CASE.endswith(v)), "")
CASE = CASE.removesuffix(VARIANT)
# An "-A<aspect>" suffix (e.g. qi6-beta-A8) moves a seeded case to that aspect ratio: seed, aspect band +-0.1 and the
# coil limits that follow it.
ASPECT = float(CASE.rsplit("-A", 1)[1]) if "-A" in CASE else None
CASE = CASE.rsplit("-A", 1)[0] if ASPECT else CASE
RESOLUTION = (8, 8, 51)            # MPOL, NTOR, NS of the optimization solves
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
INPUT_FILE = DATA / "input.rotating_ellipse_nfp2"
COILS_FILE = DATA / "ESSOS_coils_ellipse5.json"
BETA_DEFINITION = "volume"         # --beta is <beta>; "axis": WOUT betaxis
B0 = 1.0                           # T: |B| on the magnetic axis (opt.axis_field_strength), the seed's and a row
FREE_PHIEDGE = True                # free arm: PHIEDGE is a design variable (False: fixed PHIEDGE)
FREE_COIL_CURRENTS = True          # every base-coil current is a design variable, with no bounds
COIL_CURRENT_STEP = 0.06           # coordinate scale of the relative coil-current change
PF_CURRENT_STEP = 0.2              # coordinate scale of a circular coil pair's current change, in 100 kA (--pf-coils)
PF_SHAPE_STEP = 0.02               # coordinate scale of a pair's shape amplitude changes [m] (--pf-shape)
PF_GEOMETRY_STEP = 0.05            # coordinate scale of a pair's radius and height changes [m] (--pf-geometry)

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and CURTOR
# are design variables, and the mismatch sum_j R_j^2 against the BOOTSTRAP_MODEL current (Redl
# or DKX, in Redl's normalized form) is held under REDL_TOLERANCE. The seed's Picard loop is Redl's.
# DKX: the current is a line segment through a value on every half-mesh surface (those values,
# the axis one and CURTOR are design variables), its seed DKX's on the same surfaces. With
# --fixed-current it is no design variable and no row: the restart's current is held fixed.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
PICARD_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAX = 10, 1e-3, 1.0
BOOTSTRAP_BETA_STEP = 0.01        # a larger --beta is ramped in with its bootstrap current, in steps of at most this
BOOTSTRAP_BETA_START = None       # set: the ramp first doubles beta from this, so a seed with little vacuum
                                  # iota stays under its eps iota^2 limit while the bootstrap current grows
OHMIC_CURRENT = None              # set (A): the seed ramp's prescribed current, blended into the bootstrap one
CURRENT_KNOTS, CURRENT_STEP = 8, 0.05   # step relative to the largest knot value and to |CURTOR|
# Redl cases: the current is no design variable but solved with every equilibrium (ThreeTermFreeBoundaryModel
# bootstrap=, both arms), Redl-self-consistent on every half-grid surface, with I'(0) = 0 (a quasisymmetric bootstrap
# current density vanishes on the axis with the trapped fraction, f_t ~ eps^(1/2)).
BOOTSTRAP_IN_SOLVE = False
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None minimizes the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # examples/optimization/QI_optimization.py
MIRROR_LIMIT, MIRROR_MARGIN = None, 0.001  # upper limit on the edge mirror ratio (Bmax - Bmin) / (Bmax + Bmin)
BOOTSTRAP_MODEL = "redl"          # "dkx": the DKX kinetic <j.B> replaces Redl in the self-consistency row
DKX_SURFACES, DKX_COLLISION_OPERATOR = None, 0  # None: every half-mesh surface; 0: momentum-conserving Fokker-Planck

# Physical targets, imposed as hard inequalities.
IOTA_FLOOR, IOTA_MARGIN = 0.41, 0.0005
IOTA_CEILING = None                # upper limit on max |iota|
IOTA_EDGE = False                  # True: floor and ceiling also bound the edge iotaf
VACUUM_IOTA_FLOOR = None           # set: floor on the vacuum |iota| (no pressure or current) of the plasma boundary
MERCIER_FLOOR = None               # set: floor on min DMerc of the surfaces s >= MERCIER_S_MIN (edge excluded)
# DMerc is inaccurate near the magnetic axis; Landreman et al. (2026, arXiv:2606.19523) take its minimum over s > 0.2.
MERCIER_S_MIN = 0.2
MERCIER_MARGIN = 0.02             # SLSQP holds a row only to its tolerance: DMerc stays strictly positive
CURRENT_IOTA_WEIGHT = 0.0          # > 0: the loss adds w/2 iota_I^2 (current_iota), shrinking the bootstrap current
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001
B_AXIS_TOLERANCE, B_AXIS_MARGIN = 1e-3, 1e-4  # T, the band of the axis |B| row about B0

# Coils and their hard engineering limits.
COIL_ORDER, N_SEGMENTS = 16, 256
COIL_STEP = 0.05                   # coordinate scale of the coil Fourier modes
LENGTH_LIMIT = 5.0                 # m, each independent coil
# Set: the limits follow the widest plasma allowed, a = R / aspect_min, and the clearance d:
# LENGTH_LIMIT = c_L 2 pi (a + d), CURVATURE_LIMIT = c_k / (a + d), MSC_LIMIT = c_m / (a + d)^2.
COIL_LIMIT_FACTORS = None          # (c_L, c_k, c_m)
COIL_FIT_MAXITER = 200             # L-BFGS-B iterations of the free arm's finite-beta coil refit
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, to the current plasma boundary
# Interior margins of the sampled constraints; endpoint checks use the limits.
CURVATURE_MARGIN, MSC_MARGIN, LENGTH_MARGIN, DISTANCE_MARGIN = 0.10, 0.02, 1e-5, 0.001

if CASE == "ellipse5-beta7":
    # Helios-like: a low iota floor, --beta on axis, the bootstrap current supplying the rest of the transform.
    BETA_DEFINITION = "axis"
    IOTA_FLOOR = 0.16
    PICARD_ITERATIONS, PICARD_RELAX = 30, 0.5  # damped: the current dominates iota (bootstrap.self_consistent_bootstrap)
elif CASE in ("qa3", "qh", "qi"):
    COILS_FILE = DATA / f"ESSOS_coils_{CASE.replace('-', '_')}.json"
    COIL_ORDER = 8
    LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT = 3.5, 8.0, 10.0
    COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
    if CASE == "qa3":
        SEED, IOTA_FLOOR, ASPECT_RANGE = (3, 6.0, 0.5), 0.41, (5.9, 6.1)
    elif CASE == "qh":
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1)
        IOTA_FLOOR = 1.1           # between the iota = 1 and 8/7 resonances
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 8.0, 0.5), None, "QI", (7.9, 8.1), 0.21
        IOTA_FLOOR = 0.51          # above the iota = 1/2 resonance
elif CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta", "qh4-beta"):
    # Finite-beta, self-consistent bootstrap cases (Redl for the QA and QH, DKX for the QI), 4 order-12 coils per
    # half period (qh4-beta: the 3 QH coils).
    COILS_FILE = DATA / f"ESSOS_coils_{CASE.replace('-', '_') if CASE != 'qh4-beta' else 'qh'}.json"
    # (1.8, 2.5, 1.2): mid-range of Wechsung et al. (2022), Jorge et al. (2023) and Wiedman et al. (2024)
    COIL_ORDER, COIL_LIMIT_FACTORS = 12, (1.8, 2.5, 1.2)
    COIL_FIT_MAXITER = 3000  # at 200 the qa4-beta refit left B.n/|B| ~3e-3 and the first free solve could fail
    PICARD_ITERATIONS, PICARD_RELAX, BOOTSTRAP_BETA_STEP = 30, 0.5, 0.005
    # The total iota floor bounds the half-mesh surfaces and the edge; R0 is held to 1 mm.
    IOTA_EDGE, RADIUS_TOLERANCE, RADIUS_MARGIN = True, 1e-3, 1e-4
    if CASE.startswith("qa4-beta"):
        SEED, IOTA_FLOOR, ASPECT_RANGE = (2, 4.0, 0.5), 0.27, (3.5, 4.5)
        # The bootstrap current can carry the whole transform of an axisymmetric boundary, so its vacuum iota is
        # floored too, at LBD22's optimized QA (0.23): the boundary keeps its stellarator shaping.
        VACUUM_IOTA_FLOOR = 0.23
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.10, 0.20
        BOOTSTRAP_IN_SOLVE = True
    elif CASE.startswith("qh4-beta"):
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1)
        IOTA_FLOOR = 1.1           # between the iota = 1 and 8/7 resonances, as qh
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
        BOOTSTRAP_IN_SOLVE = True
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 6.0, 0.7), None, "QI", (5.9, 6.1), 0.21
        # Stellaris (Lion et al. 2025): iota 0.86 on axis to 0.98 at the edge, below the 4/4 islands
        IOTA_FLOOR, IOTA_CEILING, BOOTSTRAP_MODEL = 0.86, 0.98, "dkx"
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
elif CASE != "ellipse5":
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
if CASE.endswith("-tok"):
    # A circular tokamak with a 1% helical ripple b / a (TOK_RIPPLE), as vmex examples/data/input.minimal_seed_nfp*.
    # It has no vacuum transform: an Ohmic current of about the final bootstrap current carries the beta ramp.
    SEED = (SEED[0], SEED[1], float(os.environ.get("TOK_RIPPLE", "0.01")))
    # Same sign as the bootstrap current it hands over to (negative for the (1, -1) QH), so the blend
    # never passes through zero current, where the seed has no transform.
    OHMIC_CURRENT = 1.0e5 if CASE.startswith("qa4-beta") else -6.0e4 if CASE.startswith("qh4-beta") else 6.0e4
if VARIANT == "-merc":
    MERCIER_FLOOR = 0.0
elif VARIANT == "-jbs":
    CURRENT_IOTA_WEIGHT = 1e-2     # w/2 iota_I^2 ~ 5e-4 at the seed's iota_I ~ 0.3, the size of the QA term late on
REDL_HELICITY = 0 if HELICITY is None else HELICITY[1]  # Redl's quasisymmetry N (simsopt convention)
# SLSQP's first step has an identity Hessian, so it grows with the target residual, which starts near 1 for
# a QH or QI seed (|q| ~ 1.6) against ~0.3 for a QA one; their design coordinates are scaled down, as the
# fixed arm scales its seeded boundary steps (0.1 -> 0.02).
DESIGN_STEP_SCALE = 1.0 if TARGET_NAME == "QA" else 0.2
if ASPECT:
    SEED, ASPECT_RANGE = (SEED[0], ASPECT, SEED[2]), (ASPECT - 0.1, ASPECT + 0.1)
if COIL_LIMIT_FACTORS is not None:
    _radius = RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT
    LENGTH_LIMIT = COIL_LIMIT_FACTORS[0] * 2 * math.pi * _radius
    CURVATURE_LIMIT, MSC_LIMIT = COIL_LIMIT_FACTORS[1] / _radius, COIL_LIMIT_FACTORS[2] / _radius**2

OPTIMIZER_FTOL = 1e-10
PHIEDGE_STEP = 0.05  # coordinate scale of the relative PHIEDGE change
# Finite-beta seed coil refit: B.n/|B| unit and penalty weight of the scaled coil rows.
COIL_FIT_NORMAL_SCALE, COIL_FIT_WEIGHT = 1.0e-3, 1.0e3


# ---- seeds, targets and restarts -------------------------------------------------------------------------------------
def seed_input():
    """The case's fixed-boundary seed at the optimization resolution.

    With ``SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T;
    otherwise it is ``INPUT_FILE``'s. ``finite_beta_input`` then calibrates PHIEDGE.
    """
    import vmex as vj

    mpol, ntor, ns = RESOLUTION
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if SEED is not None:
        inp = replace(inp, nfp=SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=GRID[0], nzeta=GRID[1])
    if SEED is not None:
        _, aspect, ratio = SEED
        minor = RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([EQUILIBRIUM_FTOL]), lfreeb=False)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``HELICITY``, or constructed QI."""
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if HELICITY is None:
        return ConstructedQIResidual(np.asarray(QI_SURFACES), **QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), *HELICITY)


def finite_beta_input(inp, beta, device, am=(1.0, -1.0)):
    """Give ``inp`` a pressure p ~ ``am`` (power series in s, default 1 - s) whose fixed-boundary seed has ``beta``.

    ``BETA_DEFINITION`` "volume": ``beta`` is <beta>; "axis": WOUT ``betaxis``.
    The pressure is ramped in with hot restarts, then three corrections rescale
    PHIEDGE until the axis |B| (``opt.axis_field_strength``) is ``B0``, and the
    pressure to ``beta``. Returns the input and the last fixed-boundary solve.
    """
    from vmex import optimize as opt

    axis = BETA_DEFINITION == "axis"
    b0 = B0
    pressure = beta * b0**2 / (8e-7 * np.pi) if axis else beta / (4e-7 * np.pi)
    shape = np.zeros_like(np.asarray(inp.am, dtype=float))
    shape[: len(am)] = am
    inp = replace(inp, am=shape)
    fixed = None
    ramp = (0.25, 0.5, 0.75, 1.0) if beta > 0 else (1.0,)
    for index in range(len(ramp) + 3):
        if index < len(ramp):
            inp = replace(inp, pres_scale=ramp[index] * pressure)
        else:
            # beta ~ p / PHIEDGE^2 at fixed shape, so a flux rescale carries its pressure along
            measured = float(fixed.wout.betaxis if axis else fixed.wout.betatotal)
            flux = b0 / float(opt.axis_field_strength(fixed.state, fixed.runtime))
            inp = replace(inp, phiedge=float(inp.phiedge) * flux,
                          pres_scale=inp.pres_scale * flux**2 * (beta / measured if beta > 0 else 1.0))
        fixed = opt.solve_equilibrium(inp, initial_state=None if fixed is None else fixed.state, device=device,
                                      raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"PRES_SCALE = {inp.pres_scale:.6e} Pa, PHIEDGE = {float(inp.phiedge):.6f} Wb: betaxis = "
          f"{float(w.betaxis):.4%}, <beta> = {float(w.betatotal):.4%}, axis |B| = "
          f"{float(opt.axis_field_strength(fixed.state, fixed.runtime)):.5f} T")
    return inp, fixed


def redl_surfaces():
    """Surfaces of the Redl self-consistency check: ``REDL_SURFACES``, or by default every VMEC
    half-grid surface s = (j - 1/2) / (ns - 1) at the run's ns, as simsopt's ``RedlGeomVmec``."""

    if REDL_SURFACES is not None:
        return np.asarray(REDL_SURFACES)
    ns = int(RESOLUTION[2])
    return (np.arange(1, ns) - 0.5) / (ns - 1)


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0, b0 = float(inp.rbc[inp.ntor, 0]), B0
    t0 = REACTOR_T0 * (b0 / REACTOR_B0) ** (2 / 3) * (r0 / REACTOR_R0) ** (1 / 3)
    n0 = REACTOR_N0 * (b0 / REACTOR_B0) ** (4 / 3) * (REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, REDL_HELICITY, redl_surfaces(), n_lambda=REDL_N_LAMBDA)


def bootstrap_input(inp, beta, device):
    """``finite_beta_input`` with reactor-like kinetic profiles and their self-consistent Redl current.

    ne = n0 (1 - s^5) and Te = Ti = T0 (1 - s), so p = 2 e ne Te ~ (1 - s)(1 - s^5). n0 and T0
    start at the Helios-like reactor's collisionality nu* ~ n R / T^2 and beta ~ n T / B^2 moved
    to this R0 and B0, and follow the beta calibration as n ~ p^(2/3), T ~ p^(1/3), which keeps
    nu*. A Picard loop then makes the current Redl's, and it is resampled onto
    ``CURRENT_KNOTS`` spline knots (DKX: instead DKX's on every half-mesh
    surface, ``dkx_current``); with ``BOOTSTRAP_IN_SOLVE`` (Redl) it is instead solved
    in the run's own representation, Redl's on every half-mesh surface
    (``bootstrap.HalfMeshCurrent``), so the first step starts self-consistent.
    Above ``BOOTSTRAP_BETA_STEP`` beta is ramped in steps of at most that size, each step's
    pressure ramp carrying the previous step's bootstrap current, so its transform holds the
    equilibrium together as beta rises; with ``BOOTSTRAP_BETA_START`` the ramp first doubles
    beta from that value. With
    ``OHMIC_CURRENT`` (a near-axisymmetric seed) beta is instead ramped at that prescribed
    current and the current then blended into Redl's.
    Returns the input, the equilibrium and the Redl mismatch.
    """
    from vmex import optimize as opt
    from vmex.core.bootstrap import self_consistent_bootstrap
    from vmex.core.freeboundary_three_terms import ThreeTermsModel

    ac = np.zeros_like(np.asarray(inp.ac, dtype=float))
    ac[0] = 1.0
    inp = replace(inp, ncurr=1, pcurr_type="power_series", ac=ac, curtor=0.0)
    steps = max(1, math.ceil(round(beta / BOOTSTRAP_BETA_STEP, 9)))
    betas = [beta * k / steps for k in range(1, steps + 1)]
    start = BOOTSTRAP_BETA_START
    while start is not None and start < betas[-steps]:  # double up to the first linear stage
        betas.insert(len(betas) - steps, start)
        start *= 2
    stages = len(betas)

    def picard_at(inp, n_iter):
        return self_consistent_bootstrap(inp, redl_profiles(inp)[0], REDL_HELICITY, n_iter=n_iter,
                                         tol=PICARD_TOLERANCE,
                                         relax=PICARD_RELAX, degree=CURRENT_KNOTS - 1,
                                         s_eval=redl_surfaces(), solve_kwargs=dict(device=device))

    if OHMIC_CURRENT:
        # A near-axisymmetric seed has no vacuum transform and, at low beta, little bootstrap current: an
        # Ohmic current I' = 2 I (1 - s) holds it together while beta rises, then is blended into the Redl
        # current at full beta, so the seed ends with the bootstrap current alone.
        ohmic = np.zeros_like(ac)
        ohmic[:2] = 2.0 * OHMIC_CURRENT * np.array([1.0, -1.0])
        inp = replace(inp, ac=ohmic, curtor=OHMIC_CURRENT)
        for stage_beta in betas:
            inp, _ = finite_beta_input(inp, stage_beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
        for weight in (0.25, 0.5, 0.75):
            boot = picard_at(inp, 1).input  # one Picard step: the Redl current of the current equilibrium
            ac_boot = np.zeros(max(boot.ac.size, ohmic.size))
            ac_boot[: boot.ac.size] = boot.ac
            ac_mix = (1 - weight) * np.pad(ohmic, (0, ac_boot.size - ohmic.size)) + weight * ac_boot
            inp = replace(boot, ac=ac_mix, curtor=(1 - weight) * OHMIC_CURRENT + weight * float(boot.curtor))
            print(f"Ohmic-to-bootstrap blend {weight:.2f}: CURTOR = {float(inp.curtor):.1f} A", flush=True)
        betas = []
        picard = picard_at(inp, PICARD_ITERATIONS)
        profiles, redl = redl_profiles(inp)
        inp = picard.input
    for stage, stage_beta in enumerate(betas, 1):
        inp, _ = finite_beta_input(inp, stage_beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
        profiles, redl = redl_profiles(inp)
        picard = picard_at(inp, PICARD_ITERATIONS)
        inp = picard.input
        if stages > 1:
            print(f"bootstrap ramp {stage}/{stages}: beta {stage_beta:.4f}, CURTOR = "
                  f"{float(inp.curtor):.1f} A, Picard {picard.iterations} iterations (converged {picard.converged})",
                  flush=True)
    n0, t0 = float(profiles.ne_coeffs[0]), float(profiles.Te_coeffs[0])
    if BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl":
        model = ThreeTermsModel(inp, bootstrap=profiles, bootstrap_helicity=REDL_HELICITY, fixed_boundary=True)
        current = model.solve_boundary(model.params0, None, ftol=1e-10, max_nfev=30)
        state, _, params, _ = current["aux"]
        print(f"Redl current on its {current['x'].size} knots: max relative mismatch "
              f"{model.bootstrap_residual(state, params):.1e}", flush=True)
        inp = model.bootstrap.deck_with(model.fixed, current["x"])
    elif BOOTSTRAP_MODEL == "dkx":
        inp = dkx_current(inp, picard.equilibrium, device)
    else:
        inp = opt.resample_current_profile(inp, CURRENT_KNOTS)
    fixed = opt.solve_equilibrium(inp, initial_state=picard.equilibrium.state, device=device,
                                  raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"Redl seed: n0 = {n0:.4e} 1/m^3, T0 = {t0:.1f} eV, Picard {picard.iterations} iterations "
          f"(converged {picard.converged}), CURTOR = {float(inp.curtor):.1f} A, mismatch = "
          f"{float(redl.total(w)):.3e}; <beta> = {float(w.betatotal):.4%}, iota = "
          f"{float(min_abs_iota(fixed.state, fixed.runtime)):.4f}..{float(max_abs_iota(fixed.state, fixed.runtime)):.4f}"
          f", edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed, redl


def min_abs_iota(state, runtime):
    """|iota| floor row: the half-mesh |iota| (``opt.abs_iota_profile``), with ``IOTA_EDGE`` also the edge iotaf."""
    import jax.numpy as jnp
    from vmex import optimize as opt

    return jnp.min(opt.abs_iota_profile(state, runtime, include_edge=IOTA_EDGE))


def max_abs_iota(state, runtime):
    """|iota| ceiling row, the counterpart of ``min_abs_iota``."""
    import jax.numpy as jnp
    from vmex import optimize as opt

    return jnp.max(opt.abs_iota_profile(state, runtime, include_edge=IOTA_EDGE))


def min_d_merc(state, runtime):
    """Mercier floor row: the smallest ``DMerc`` of VMEC's interior surfaces [2:-1] with ``MERCIER_S_MIN <= s``."""
    import jax.numpy as jnp
    from vmex import optimize as opt

    s = jnp.asarray(runtime.setup.s_full)
    index = jnp.arange(s.size)
    return jnp.min(jnp.where((index >= 2) & (index < s.size - 1) & (s >= MERCIER_S_MIN),
                             opt.d_merc_state(state, runtime), jnp.inf))


def current_iota(state, runtime):
    """``mu0 |I| R0 / (2 pi a^2 B_axis)``: the edge iota the net toroidal current ``I`` gives a cylinder of the
    plasma's major radius, minor radius ``R0 / aspect`` and axis field -- the current's share of iota."""
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core.fields import surface_currents
    from vmex.core.statephysics import _field_chain  # private: the covariant fields behind ctor

    fields = _field_chain(state, runtime)[3]
    mu0_current = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=runtime.trig,
                                   s=jnp.asarray(runtime.setup.s_full), signgs=runtime.setup.signgs).ctor
    radius = opt.major_radius(state, runtime)
    minor = radius / opt.aspect_ratio(state, runtime)
    return jnp.abs(mu0_current) * radius / (2 * jnp.pi * minor**2 * opt.axis_field_strength(state, runtime))


def bootstrap_mismatch(inp, redl, device):
    """(state, runtime) -> the bootstrap self-consistency mismatch held under ``REDL_TOLERANCE``.

    Redl's, or with ``BOOTSTRAP_MODEL = "dkx"`` the equilibrium's <j.B> against DKX's
    drift-kinetic one on ``DKX_SURFACES``, in Redl's normalized form (Redl assumes quasisymmetry).
    """
    if BOOTSTRAP_MODEL == "redl":
        return redl.total_state
    mismatch = dkx_mismatch(inp)
    # DKX builds its per-surface operators and Boozer plan on the host at the first
    # call and caches them; make that call on a concrete equilibrium, so the
    # optimizer's traced calls only find the cache.
    from vmex import optimize as opt

    seed = opt.solve_equilibrium(replace(inp, lfreeb=False), device=device, raise_on_max_iterations=True,
                                 polish_force_balance=False)
    print(f"DKX bootstrap mismatch of the fixed-boundary seed: {float(mismatch(seed.state, seed.solver_context)):.3e}",
          flush=True)
    return mismatch


def dkx_kinetic(inp):
    """DKX's bootstrap ``<j.B>`` for ``inp``'s kinetic profiles on the ``DKX_SURFACES`` rows (needs ``dkx``).

    DKX (>= 2.8, vmex's floor) compiles the rows as one batched program with a reverse-mode JVP; E_r = 0.
    """
    from dkx.bootstrap import KineticBootstrapMismatch

    surfaces = redl_surfaces() if DKX_SURFACES is None else DKX_SURFACES
    return KineticBootstrapMismatch(redl_profiles(inp)[0], surfaces=surfaces, collision_operator=DKX_COLLISION_OPERATOR,
                                    mboz=QI_OPTIONS["mboz"], nboz=QI_OPTIONS["nboz"])


def on_grid(runtime):
    """``runtime`` with a concrete radial grid: DKX reads it on the host, and under jit it is a tracer
    (always linspace(0, 1, ns))."""
    grid = np.linspace(0.0, 1.0, runtime.setup.s_full.shape[0])
    return replace(runtime, setup=replace(runtime.setup, s_full=grid))


def dkx_mismatch(inp):
    """(state, runtime) -> DKX's bootstrap mismatch for ``inp``'s kinetic profiles (needs the ``dkx`` package)."""
    kinetic = dkx_kinetic(inp)
    return lambda state, runtime: kinetic.total(state, on_grid(runtime))


def dkx_current(inp, fixed, device):
    """``inp`` with DKX's self-consistent bootstrap current, a value on every half-mesh surface.

    Picard steps I' <- I'_DKX from the equilibrium ``fixed``: DKX's ``<j.B>`` on its rows,
    interpolated onto every half-mesh surface and inverted for dI/ds there
    (``bootstrap.current_derivative``), under-relaxed by ``PICARD_RELAX``. The profile is
    ``line_segment_ip`` through those values, the axis and edge ones extrapolated linearly,
    and CURTOR its integral: no spline fit.
    """
    import jax
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core.bootstrap import current_derivative
    from vmex.core.profiles import current

    kinetic, ns = dkx_kinetic(inp), int(np.asarray(inp.ns_array)[-1])
    s = (np.arange(1, ns) - 0.5) / (ns - 1)
    rows = kinetic.rows(ns)[1]
    knots = np.r_[0.0, s, 1.0]
    enclosed = lambda z: current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, z, bloat=inp.bloat)  # noqa: E731
    v = float(inp.curtor) / float(enclosed(1.0)) * np.asarray(jax.vmap(jax.grad(enclosed))(jnp.asarray(s)))
    kinetic.total(fixed.state, on_grid(fixed.runtime))  # DKX builds its operators on the host at a concrete call
    kinetic_j = jax.jit(lambda state, runtime: kinetic.current_profiles(state, on_grid(runtime))[2])
    derivative = jax.jit(current_derivative)
    for iteration in range(PICARD_ITERATIONS):
        values = np.r_[1.5 * v[0] - 0.5 * v[1], v, 1.5 * v[-1] - 0.5 * v[-2]]
        inp = replace(inp, pcurr_type="line_segment_ip", ac_aux_s=knots, ac_aux_f=values,
                      curtor=float(current("line_segment_ip", inp.ac, knots, values, 1.0)))
        fixed = opt.solve_equilibrium(inp, initial_state=fixed.state, device=device, raise_on_max_iterations=True,
                                      polish_force_balance=False)
        jk = np.interp(s, rows, np.asarray(kinetic_j(fixed.state, fixed.runtime)))
        target = np.asarray(derivative(jnp.asarray(jk), fixed.state, fixed.runtime)[1])
        delta = float(np.max(np.abs(target - v)) / np.max(np.abs(target)))
        print(f"DKX Picard {iteration}: CURTOR = {float(inp.curtor):.1f} A, max|I' - I'_DKX| / max|I'_DKX| = "
              f"{delta:.2e}", flush=True)
        if delta <= PICARD_TOLERANCE:
            break
        v = (1.0 - PICARD_RELAX) * v + PICARD_RELAX * target
    return inp


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current profile (``--restart``).

    A run that held its current fixed (``--fixed-current`` in its ``run.json``) keeps the current of its
    ``input.run``: the WOUT stores the computed current, which is not exactly the held one.
    """
    import vmex as vj

    final = vj.VmecInput.from_wout(run / "wout.nc")
    record = run / "run.json"
    held = record.exists() and "--fixed-current" in json.loads(record.read_text())["argv"]
    names = ("rbc", "zbs", "phiedge") + (() if held else ("pcurr_type", "ac", "ac_aux_s", "ac_aux_f", "curtor"))
    return replace(vj.VmecInput.from_file(run / "input.run"), lfreeb=False,
                   **{name: getattr(final, name) for name in names})


# ---- coils and their field -------------------------------------------------------------------------------------------
def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes (ESSOS ``Curves.order``) without changing the curves or currents."""
    from essos.coils import Coils

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    curves = coils.curves.copy()
    curves.order, curves.n_segments = order, n_segments
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


def scale_coil_currents(coils, rbtor):
    """Coils with every current scaled so the linked mu0 I / 2 pi is ``rbtor``.

    mu0 I / 2 pi is the mean of R B_phi on the loop R = R0, Z = 0, with R0 the
    case's ``RADIUS_TARGET`` (the seed's RBC(0,0)). The loop used to sit at a
    hard-coded R = 1 m, which is R0 for every case.
    """
    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.fields import BiotSavart

    r0, phi = RADIUS_TARGET, np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = jnp.asarray(r0 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    field = np.asarray(jax.vmap(BiotSavart(coils).B)(points))
    linked = abs(r0 * float(np.mean(-np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1])))
    return Coils(coils.curves, coils.dofs_currents_raw * (rbtor / linked), currents_scale=coils.currents_scale)


def coil_field(coils):
    """points (..., 3) -> the coils' Biot-Savart field B (..., 3)."""
    import jax
    from essos.fields import BiotSavart

    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


def weighted_rms(weights, values):
    """sqrt(sum weights values^2): the RMS for area weights that sum to one."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(weights * values**2))


def total_normal_field(interface, field):
    """(B_coils + B_plasma).n/|B| on a ``vmex.PlasmaVacuumInterface``, B_coils = ``field``."""
    import jax.numpy as jnp

    return interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)


# ---- coil limits ----------------------------------------------------------------------------------------------------
CURVATURE_POINTS = max(512, 64*COIL_ORDER)
COIL_POINT_ROWS = False  # --coil-point-rows
DISTANCE_POINTS = max(128, 16*COIL_ORDER)
SURFACE_GRID = (61, 64)


def coil_limits():
    """The case's coil limits and margins as ``vj.CoilLimits`` (self clearance 1e-6 m, minimum speed 1e-4)."""
    import vmex as vj

    return vj.CoilLimits(length=LENGTH_LIMIT, curvature=CURVATURE_LIMIT, mean_squared_curvature=MSC_LIMIT,
                         coil_distance=COIL_DISTANCE_LIMIT, length_margin=LENGTH_MARGIN,
                         curvature_margin=CURVATURE_MARGIN, mean_squared_curvature_margin=MSC_MARGIN,
                         distance_margin=DISTANCE_MARGIN, curvature_points=CURVATURE_POINTS,
                         distance_points=DISTANCE_POINTS)


# ---- the optimization ------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--current-iota-weight", type=float, help="w of the loss term w/2 iota_I^2 "
                        "(default: CURRENT_IOTA_WEIGHT, 1e-2 for the -jbs cases)")
    parser.add_argument("--conductor-weight", type=float, default=0.0, metavar="W",
                        help="the loss adds W x the coils' total |current| x length in MA m (base coils with their "
                             "symmetric copies and the --pf-coils): the tape a wound coil set needs")
    parser.add_argument("--vacuum-iota-floor", type=float, help="floor on the boundary's vacuum |iota| "
                        "(default: VACUUM_IOTA_FLOOR, 0.23 for qa4-beta)")
    parser.add_argument("--iota-floor", type=float, help="floor on min |iota| (default: IOTA_FLOOR)")
    parser.add_argument("--iota-ceiling", type=float, help="ceiling on max |iota| (default: IOTA_CEILING)")
    parser.add_argument("--check-gradient-fit", type=float, nargs=2, metavar=("FTOL", "NFEV"),
                        help="with --check-gradient: converge the differenced trials' boundary fits to FTOL")
    parser.add_argument("--gradient-report", action="store_true",
                        help="print the gradient of the loss and of every constraint row by group of design "
                             "coordinates (plasma, coils, circular pairs) at the start point, and stop")
    parser.add_argument("--edge-smooth-reach", type=float, metavar="RHO",
                        help="--edge-wall: the line's largest distance to the boundary as a smooth maximum over its "
                             "nodes, max + log(mean(exp(RHO (d - max)))) / RHO [RHO in 1/m; below the largest by at "
                             "most log(nodes) / RHO]: the hard maximum has a kink wherever the farthest node changes")
    parser.add_argument("--gradient-scan", type=float, nargs="+", default=(), metavar="U",
                        help="with --gradient-report: move the design against the gradient left by the rows at a "
                             "bound, the largest scaled coordinate by each U, and compare the loss with first order")
    parser.add_argument("--check-gradient", type=int, default=0, metavar="N",
                        help="compare the seed's design gradients with central differences along N random "
                        "directions (gradient_check.json), then stop")
    parser.add_argument("--coil-point-rows", action="store_true", help="one coil-limit row per sample point "
                        "(curvature, coil-coil and self distance) instead of each coil's extreme")
    parser.add_argument("--extension", type=Path, help="Python file of extra constraint ROWS and objective TERMS "
                        "on the equilibrium (see load_extension)")
    parser.add_argument("--aspect-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="aspect-ratio bounds (default: ASPECT_RANGE); the coil limits keep the case's")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--coils", type=Path, default=COILS_FILE)
    parser.add_argument("--coil-limits", type=float, nargs=3, metavar=("LENGTH", "CURVATURE", "MSC"),
                        help="per-coil limits (m, 1/m, 1/m^2) in place of the case's (e.g. Gil et al.'s)")
    parser.add_argument("--no-coil-fit", action="store_true", help="use --coils as given, without the seed refit")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--ns", type=int, help="radial resolution (default: RESOLUTION)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"), help="default: RESOLUTION")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--current-constraint", action="store_true",
                        help="the current is an input, as in VMEC (CURRENT_KNOTS values and CURTOR are design "
                        "variables) held Redl-self-consistent by a constraint row, instead of solved with every "
                        "equilibrium (BOOTSTRAP_IN_SOLVE, the qa4-beta and qh4-beta default)")
    parser.add_argument("--fixed-current", action="store_true",
                        help="hold the current (the restart's) fixed: no design variable and no bootstrap row")
    parser.add_argument("--restart", type=Path, help="continue a run from its input.run, coils.json and wout.nc")
    parser.add_argument("--seed-run", type=Path, help="start from another run's input.run and coils.initial.json "
                        "(its calibrated deck and fitted coils), skipping the bootstrap ramp and the coil fit")
    parser.add_argument("--vc-grid", type=int, default=48, help="virtual-casing grid per field period")
    parser.add_argument("--chunk", type=int, default=8, help="Jacobian columns per batch (memory)")
    parser.add_argument("--quadrature", type=int, nargs=2, metavar=("NT", "NP"),
                        help="virtual-casing singular quadrature (default 4 nfp grid x 2 grid: 384 x 96 for nfp 2; "
                        "grid x grid leaves a ~3e-4 tangential plasma-field error; see ThreeTermFreeBoundaryModel)")
    parser.add_argument("--max-iterations", type=int, help="VMEC iteration cap of every solve (default: the deck's)")
    parser.add_argument("--design-step-scale", type=float, default=DESIGN_STEP_SCALE,
                        help="factor on the design-coordinate scales (default: DESIGN_STEP_SCALE)")
    parser.add_argument("--max-step", type=float, metavar="U",
                        help="trust box: a trial moving any scaled design coordinate more than U from the accepted "
                             "point is rejected unsolved, so SLSQP's line search shortens the step, a tenth at a time: "
                             "a cap below the usual step slows the run tenfold")
    parser.add_argument("--loss-scale", type=float, default=1.0, metavar="S",
                        help="the optimizer minimizes S times the loss: SLSQP starts each run, and each restart "
                             "after a rejected trial, from an identity Hessian, so its first steps are the size of "
                             "the scaled gradient; 1e3 tripled the progress per hour on a slow tail")
    parser.add_argument("--step-box", type=float, metavar="U",
                        help="bound every step to U in each scaled design coordinate inside SLSQP's subproblem "
                             "(SLSQP restarted at each accepted point, from an identity Hessian)")
    parser.add_argument("--trial-ftol", type=float, default=1e-3,
                        help="relative cost change at which a trial's boundary steps stop")
    parser.add_argument("--trial-target-cost", type=float, default=4.0, metavar="F",
                        help="a trial's boundary fit on the accepted point's Jacobian is kept if its cost ends below F "
                             "times the accepted point's, and refitted with fresh Jacobians otherwise (0: every trial "
                             "is refitted, so neighbouring trials compare smoothly; slower)")
    parser.add_argument("--max-boundary-residual", type=float, default=0.05,
                        help="largest interface RMS (B.n, pressure balance or K, over |B|) of a solved trial")
    parser.add_argument("--trial-forward-ftol", type=float, default=1e-11,
                        help="force residual of the trials' equilibria (each accepted point is re-solved to the "
                        "deck's tolerance before its gradient)")
    parser.add_argument("--labelling", choices=("anchored", "equal_arc", "minimal", "flat_arc", "none"), default="anchored",
                        help="the boundary's poloidal labelling rows: anchored to the start boundary (anchored), "
                        "equal arc length of the current surface (equal_arc), the seed fit's flat tangential "
                        "directions pinned (minimal) or held by equal arc (flat_arc), or none; see vmex.ThreeTermsConfig")
    parser.add_argument("--label-weight", type=float,
                        help="weight of the labelling rows (default: the labelling's, 1e-2 anchored, 1e-3 equal arc; "
                        "0 is the no-label model)")
    parser.add_argument("--gauge-tau", type=float,
                        help="minimal and flat_arc labellings: pin the flat tangential directions with singular values below this "
                        "fraction of the largest (default: vmex.ThreeTermsConfig's)")
    parser.add_argument("--gauge-refresh", type=int, default=0, metavar="K",
                        help="minimal and flat_arc labellings: set the gauge again at the accepted fit every K accepted "
                        "steps (default 0: once, at the start)")
    parser.add_argument("--redl-rows", action="store_true",
                        help="--current-constraint with a row per Redl surface, |R_j| <= sqrt(REDL_TOLERANCE / n), "
                        "instead of the one row sum_j R_j^2 <= REDL_TOLERANCE (which the rows imply)")
    parser.add_argument("--current-knots", type=int, default=CURRENT_KNOTS, metavar="N",
                        help="number of spline knots of the current with --current-knots-at-ends")
    parser.add_argument("--current-knots-at-ends", action="store_true",
                        help="the current's CURRENT_KNOTS spline knots at s = sin(pi t / 2)^2, t uniform (closer at the axis and edge)")
    parser.add_argument("--redl-s-min", type=float, default=0.0, metavar="S",
                        help="the Redl surfaces start at this normalized flux (default: all half-grid surfaces)")
    parser.add_argument("--gauge-release", action="store_true",
                        help="flat_arc labelling: each gauge refresh first re-fits the accepted boundary without the gauge rows")
    parser.add_argument("--gauge-max-share", type=float, metavar="S",
                        help="minimal and flat_arc labellings: set the gauge again at an accepted fit where its rows' "
                        "share of the fit's cost exceeds S")
    parser.add_argument("--gradient-rounds", type=int, default=0, metavar="N",
                        help="the exact (Newton) design gradient: at most N block Krylov rounds, a direction per row "
                        "each (default 0: the Gauss-Newton gradient)")
    parser.add_argument("--gradient-rtol", type=float, default=1e-3, metavar="TOL",
                        help="--gradient-rounds: stop at this relative residual of the rows' adjoint equations")
    parser.add_argument("--gradient-fit-xtol", type=float, metavar="TOL",
                        help="converge each gradient's fit (Gauss-Newton steps) until a step is below TOL")
    parser.add_argument("--certify", type=float, metavar="F",
                        help="reject a trial whose fit step or Newton row residual ends above F times "
                             "--gradient-fit-xtol / --gradient-rtol (needs both and --gradient-rounds)")
    parser.add_argument("--promote-trials", action="store_true",
                        help="evaluate every trial as its acceptance leaves it (gauge refresh, converged gradient)")
    parser.add_argument("--pf-coils", type=float, nargs=3, action="append", metavar=("R", "Z", "I"),
                        help="add a pair of circular coils of radius R [m] at +-Z [m], each with current I [A] "
                             "(repeatable); the currents are design variables (vmex.core.coil_design.PoloidalCoilDesign). "
                             "Default with --restart: the run's pf.json")
    parser.add_argument("--pf-fixed", action="store_true", help="hold the --pf-coils currents (no design variables)")
    parser.add_argument("--pf-geometry", action="store_true",
                        help="the --pf-coils radii and heights are design variables too")
    parser.add_argument("--pf-shape", type=int, metavar="K",
                        help="bend the --pf-coils with the field period: K cosine amplitudes of each pair's radius and "
                             "K sine amplitudes of its height are design variables (stellarator symmetric)")
    parser.add_argument("--pf-curvature", type=float, default=2.0, metavar="MAX",
                        help="with --pf-shape: the pairs' largest curvature [1/m] (one more coil-limit row)")
    parser.add_argument("--pf-length-max", type=float, metavar="L",
                        help="largest length of a circular pair coil [m] (one more coil-limit row; with "
                             "--pf-geometry it pulls a longer pair inwards)")
    parser.add_argument("--pf-current-max", type=float, metavar="A",
                        help="keep each pair's |current| below A (one more coil-limit row)")
    parser.add_argument("--pf-coil-distance", type=float, metavar="MIN",
                        help="keep the --pf-coils at least MIN [m] from the other coils (one more coil-limit row)")
    parser.add_argument("--edge-line", type=Path, metavar="JSON",
                        help="a closed field line outside the plasma (benchmarks/edge_fixed_points.py's fixed_points.json, "
                             "entry --edge-line-index of its total field; default with --restart: the run's edge_line.json): its trace, gap to the boundary and distance "
                             "to the coils are logged every step (vmex.core.periodic_fieldlines)")
    parser.add_argument("--edge-line-index", type=int, default=0)
    parser.add_argument("--edge-gap", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="constrain the line's smallest distance to the plasma boundary [m]")
    parser.add_argument("--edge-coil-distance", type=float, metavar="MIN",
                        help="constrain the line's smallest distance to any coil [m]")
    parser.add_argument("--edge-wall", type=float, nargs="+", metavar="D CLEARANCE [COILS]",
                        help="a wall D [m] off the plasma boundary between the line and the coils: the line's largest "
                             "distance to the boundary stays below D - CLEARANCE and the coils' smallest above "
                             "D + COILS (default CLEARANCE), so the line's legs end on the wall before they reach a coil")
    parser.add_argument("--edge-height-spread", type=float, metavar="MAX",
                        help="constrain the standard deviation of the line's height Z around the torus [m]")
    parser.add_argument("--edge-legs", type=float, metavar="MAX",
                        help="constrain |sine| of the angle at which the line's two legs towards Z = 0 cross that plane "
                             "(vmex.core.periodic_fieldlines.manifold_crossing; 0: they join the mirror line's legs "
                             "into one separatrix)")
    parser.add_argument("--edge-leg-maps", type=int, nargs=2, default=(2, 2), metavar=("UNSTABLE", "STABLE"),
                        help="return maps that bring each leg to Z = 0 (benchmarks/edge_legs.py prints them)")
    parser.add_argument("--edge-outer-legs", type=float, metavar="MIN",
                        help="constrain how far from the plasma boundary [m] each of the line's two legs away from "
                             "Z = 0 gets (vmex.core.periodic_fieldlines.manifold_outer_leg): above the --edge-wall's D, "
                             "both divertor legs reach the wall")
    parser.add_argument("--edge-outer-leg-maps", type=int, default=4, help="return maps the outer legs are followed")
    parser.add_argument("--edge-leg-coils", type=float, metavar="MIN",
                        help="constrain the distance [m] from each outer leg to the coils and the circular pairs: "
                             "the closest of its field lines over --edge-outer-leg-maps return maps, from next to "
                             "the line out to the --edge-wall's D, sampled 128 times per map (16 missed the closest "
                             "approach to a coil by centimetres)")
    parser.add_argument("--vessel", type=Path, metavar="NPZ",
                        help="a vacuum vessel: 'outline' (P, m, 2), its (R, Z) outlines on P planes over one field "
                             "period (results/.../vessel.py); with it --edge-leg-coils counts the legs inside it")
    parser.add_argument("--vessel-gap", type=float, metavar="MIN",
                        help="constrain the plasma boundary and the edge line to at least MIN [m] inside the vessel")
    parser.add_argument("--vessel-coils", type=float, metavar="MIN",
                        help="constrain the coils and the circular pairs to at least MIN [m] outside the vessel")
    parser.add_argument("--vessel-legs", type=float, metavar="MIN",
                        help="constrain each outer leg to reach at least MIN [m] beyond the vessel (its targets)")
    parser.add_argument("--edge-trace", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="constrain the trace of the line's tangent map (|Tr M| > 2: X-point)")
    parser.add_argument("--edge-vc-grid", type=int, default=64, help="virtual-casing grid per field period of the "
                        "field outside the plasma (direct quadrature: the line stays a few grid spacings off the boundary)")
    args = parser.parse_args(argv)
    if args.edge_wall and len(args.edge_wall) not in (2, 3):
        parser.error("--edge-wall takes D CLEARANCE [COILS]")
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
        if args.pf_coils is None and (args.restart / "pf.json").exists():
            saved = json.loads((args.restart / "pf.json").read_text())
            args.pf_coils = [list(v) for v in zip(saved["radius_m"], saved["height_m"], saved["current_a"])]
            args.pf_saved_shape = saved.get("shape_m")
    if args.restart is not None and args.edge_line is None and (args.restart / "edge_line.json").exists():
        args.edge_line = args.restart / "edge_line.json"  # the run's own line at its last save
    if (args.edge_gap or args.edge_coil_distance is not None or args.edge_trace or args.edge_wall
            or args.edge_height_spread is not None or args.edge_legs is not None
            or args.edge_outer_legs is not None or args.edge_leg_coils is not None or args.vessel_gap is not None
            or args.vessel_legs is not None) and args.edge_line is None:
        parser.error("the edge constraints need --edge-line")
    if (args.vessel_gap, args.vessel_coils, args.vessel_legs) != (None,) * 3 and args.vessel is None:
        parser.error("the vessel constraints need --vessel")
    if args.pf_coil_distance is not None and not args.pf_coils:
        parser.error("--pf-coil-distance needs --pf-coils")
    if (args.pf_geometry or args.pf_shape or args.pf_current_max is not None
            or args.pf_length_max is not None) and not args.pf_coils:
        parser.error("--pf-geometry, --pf-shape, --pf-current-max and --pf-length-max need --pf-coils")
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    if args.fixed_current and not args.bootstrap:
        parser.error("--fixed-current needs --bootstrap")
    return args


def fit_coils_to_plasma(coils, wout, inp):
    """Coils whose field with the plasma's own is tangent to a fixed-boundary finite-beta seed."""
    import jax
    import jax.numpy as jnp
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj

    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=37, ntheta=32)
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])
    x0 = jnp.asarray(coils.curves.dofs).ravel()
    limits = coil_limits()

    def coils_from_u(u):
        return coils.with_dofs(jnp.concatenate((x0 + COIL_STEP * u, coils.dofs_currents)))

    def normal_field_rms(new):
        return weighted_rms(interface.weights, total_normal_field(interface, coil_field(new)))

    def objective(u):
        new = coils_from_u(u)
        rows = jnp.concatenate([limits.rows(new), jnp.atleast_1d(
            (limits.surface_clearance(new, surface) - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN)
            / COIL_SURFACE_DISTANCE_LIMIT)])
        return (0.5 * (normal_field_rms(new) / COIL_FIT_NORMAL_SCALE)**2
                + 0.5 * COIL_FIT_WEIGHT * jnp.sum(jnp.minimum(rows, 0.0)**2))

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    before = float(normal_field_rms(coils))
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))), np.zeros(x0.size), jac=True,
                   method="L-BFGS-B", bounds=[(-5.0, 5.0)] * x0.size,
                   options=dict(maxiter=COIL_FIT_MAXITER, maxcor=20, ftol=1e-15, gtol=1e-12))
    fitted = coils_from_u(jnp.asarray(fit.x))
    print(f"[coil fit] {fit.nit} L-BFGS-B iterations ({fit.message}): (B_coils + B_plasma).n/|B| RMS "
          f"{before:.3e} -> {float(normal_field_rms(fitted)):.3e} on the fixed-boundary seed", flush=True)
    return fitted


def check_gradient(problem, directions, out, steps=(1e-2, 1e-3, 1e-4), fit=None):
    """Implicit design gradients of the loss and every quantity at the seed against central differences of
    re-solved trials, along ``directions`` random unit directions in the optimizer's coordinates, at each step in
    ``steps``; the relative difference ``|implicit - difference| / |difference|`` is written to
    ``gradient_check.json`` (the loss on its own, the quantities as one vector).  With ``fit = (ftol, max_nfev)``
    the differenced trials' boundary fits run to that tolerance, so each lands on its design's free boundary
    rather than near the first-order prediction it starts from."""
    x = np.asarray(problem.accepted.parameters, dtype=float)
    _, grad = problem.value_and_grad(x)
    jac = problem.constraint_jac(x)
    if fit is not None:
        problem.boundary_ftol, problem.boundary_max_nfev = float(fit[0]), int(fit[1])
    rng = np.random.default_rng(0)
    records = []
    for k in range(directions):
        d = rng.standard_normal(x.size)
        d /= np.linalg.norm(d)
        implicit = np.r_[grad @ d, jac @ d]
        # the loss of trials next to the accepted point: a trial is solved from the accepted point's fit, so these
        # differ from the accepted loss by the trials' own scatter, with no design change to speak of
        near = [problem.fun(x + s * 1e-9 * d) for s in (1, -1, 2, -2)]
        print(f"[gradient check] direction {k}: accepted loss {problem.fun(x):.9e}, trials at +-1e-9 and +-2e-9 "
              + " ".join(f"{v:.9e}" for v in near), flush=True)
        for h in steps:
            values = [np.r_[problem.fun(x + s * h * d), problem.constraint_values(x + s * h * d)] for s in (1, -1)]
            fd = (values[0] - values[1]) / (2 * h)
            loss = abs(implicit[0] - fd[0]) / max(abs(fd[0]), 1e-300)
            rows = np.linalg.norm(implicit[1:] - fd[1:]) / max(np.linalg.norm(fd[1:]), 1e-300)
            records.append(dict(direction=k, h=h, loss_implicit=float(implicit[0]), loss_difference=float(fd[0]),
                                loss_relative=float(loss), rows_relative=float(rows),
                                rows_implicit=implicit[1:].tolist(), rows_difference=fd[1:].tolist()))
            print(f"[gradient check] direction {k} h={h:.0e}: loss {implicit[0]:.6e} vs {fd[0]:.6e} "
                  f"(relative {loss:.1e}), quantities relative {rows:.1e}", flush=True)
    (out / "gradient_check.json").write_text(json.dumps(records, indent=1) + "\n")


def gradient_report(problem, names, lower, upper, row_scales, out, coil_rows=None, scan=()):
    """Where the design gradient sits: the norm, over each group of design coordinates, of the gradient of the loss
    and of every constraint row (in the row's unit), in the optimizer's scaled coordinates (the first-order change
    for a unit step of the group).  Printed and written to ``gradient_report.json``, with the rows' values and bounds
    and the gradient's entries on the circular pairs' coordinates.

    Then the first-order test of a constrained minimum: nonnegative least-squares multipliers of the rows within
    1e-3 of a bound (``coil_rows`` too), the part of the loss gradient they leave, and for each row the relative
    gain of the loss if its bound were relaxed by 1% of the row's unit.  With ``scan`` (sizes, the largest move of a
    scaled coordinate) the design is then moved against the part of the gradient left, and the loss's change compared
    with its first-order prediction, with the smallest slack of the rows and of ``coil_rows`` there."""
    from vmex.core.errors import TrialRejected

    x = np.asarray(problem.accepted.parameters, dtype=float)
    scales = np.asarray(problem.scales, dtype=float)
    loss, grad = problem.value_and_grad(x)
    values = np.asarray(problem.constraint_values(x))
    rows = np.vstack((grad, problem.constraint_jac(x) / np.asarray(row_scales)[:, None])) * scales
    labels = ["loss"] + [f"row {i}" for i in range(len(values))]
    kinds = dict(plasma=("phiedge", "plasma_current", "curtor"), coil_currents=("current[",), coil_shapes=("coil[",),
                 pair_currents=("pf_current",), pair_geometry=("pf_radius[", "pf_height["),
                 pair_shapes=("pf_radius_cos", "pf_height_sin"))
    groups = {kind: np.array([n.startswith(prefixes) for n in names]) for kind, prefixes in kinds.items()}
    groups = {kind: mask for kind, mask in groups.items() if mask.any()}
    print("[gradient report] " + " ".join(f"{k}={int(m.sum())}" for k, m in groups.items()) + " coordinates", flush=True)
    record = dict(loss=float(loss), rows=[])
    pairs = [i for i, n in enumerate(names) if n.startswith("pf_")]
    for i, (label, row) in enumerate(zip(labels, rows)):
        norms = {kind: float(np.linalg.norm(row[mask])) for kind, mask in groups.items()}
        bound = "" if i == 0 else f" value {values[i - 1]:+.4e} in [{lower[i - 1]:.4g}, {upper[i - 1]:.4g}]"
        print(f"[gradient report] {label}{bound}: " + " ".join(f"{k}={v:.2e}" for k, v in norms.items()), flush=True)
        record["rows"].append(dict(label=label, norms=norms, pairs={names[j]: float(row[j]) for j in pairs},
                                   **({} if i == 0 else dict(value=float(values[i - 1]), lower=float(lower[i - 1]),
                                                             upper=float(upper[i - 1]), unit=float(row_scales[i - 1])))))
    from scipy.optimize import nnls

    slack = np.minimum(values - np.asarray(lower), np.asarray(upper) - values) / np.asarray(row_scales)
    tight = np.flatnonzero(slack < 1e-3)
    side = np.where(values[tight] - np.asarray(lower)[tight] < np.asarray(upper)[tight] - values[tight], 1.0, -1.0)
    active, tags = rows[1:][tight] * side[:, None], [f"row {i}" for i in tight]  # feasible side: active @ step >= 0
    if coil_rows is not None:
        at = np.flatnonzero(np.asarray(coil_rows.fun(x)) < 1e-3)
        active = np.vstack((active, np.asarray(coil_rows.jac(x))[at] * scales))
        tags += [f"coil row {i}" for i in at]
    multipliers = nnls(active.T, rows[0])[0] if len(tags) else np.zeros(0)  # a minimum needs them >= 0
    left = float(np.linalg.norm(rows[0] - multipliers @ active) / np.linalg.norm(rows[0]))
    print(f"[gradient report] {len(tags)} rows at a bound leave {left:.3f} of the loss gradient "
          f"(0: a constrained minimum to first order)", flush=True)
    record.update(gradient_left=left, multipliers=[])
    for i in np.argsort(-multipliers)[:np.count_nonzero(multipliers)]:
        gain = 0.01 * float(multipliers[i]) / float(loss)
        print(f"[gradient report] {tags[i]}: relaxing by 1% of its unit gains {gain:+.2%} of the loss", flush=True)
        record["multipliers"].append(dict(row=tags[i], multiplier=float(multipliers[i]), gain_per_percent=gain))
    direction = -(rows[0] - multipliers @ active) if len(tags) else -rows[0]
    direction, record["scan"] = direction / np.max(np.abs(direction)), []
    for size in scan:
        trial = x + size * direction * scales
        try:
            change = float(problem.fun(trial)) - float(loss)
            at = np.asarray(problem.constraint_values(trial))
            worst = float(np.min(np.minimum(at - np.asarray(lower), np.asarray(upper) - at) / np.asarray(row_scales)))
            coil = float(np.min(coil_rows.fun(trial))) if coil_rows is not None else np.nan
        except TrialRejected as error:
            print(f"[gradient report] scan {size:g}: rejected ({error})", flush=True)
            continue
        predicted = size * float(rows[0] @ direction)
        print(f"[gradient report] scan {size:g}: loss {change:+.3e} (first order {predicted:+.3e}), "
              f"smallest slack rows {worst:+.2e} (was {float(np.min(slack)):+.2e}) coils {coil:+.2e}", flush=True)
        moved = (at - values) / np.asarray(row_scales)  # each row's change in its unit, against first order
        first = size * (rows[1:] @ direction)
        for i in np.argsort(-np.abs(moved - first))[:4]:
            print(f"[gradient report] scan {size:g}: row {i} moved {moved[i]:+.2e} (first order {first[i]:+.2e}), "
                  f"slack {slack[i]:+.2e}", flush=True)
        record["scan"].append(dict(size=size, change=change, predicted=predicted, row_slack=worst, coil_slack=coil,
                                   rows_moved=moved.tolist(), rows_first_order=first.tolist()))
    (out / "gradient_report.json").write_text(json.dumps(record, indent=1) + "\n")


def load_extension(path):
    """``ROWS`` and ``TERMS`` of an ``--extension`` file, empty without one.

    ``ROWS = {name: (f, lower, upper, scale)}``: hard constraint rows ``lower <= f(state, runtime) <= upper`` (a
    scalar or an array, every element a row; ``scale`` is the row's unit).  ``TERMS = {name: (f, weight)}``: the
    objective adds ``weight / 2 |f(state, runtime)|^2``.  ``f`` is a JAX function of the equilibrium state and
    its runtime, like ``vmex.optimize.aspect_ratio``; its gradient comes from the same implicit adjoint as the
    built-in rows.  Their values are logged as ``extension_<name>``.
    """
    if path is None:
        return {}, {}
    import importlib.util

    spec = importlib.util.spec_from_file_location("single_stage_extension", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "ROWS", {})), dict(getattr(module, "TERMS", {}))


def write_run_record(out, args):
    """``run.json``: what reproduces this run (command, environment, code, packages, device and input hashes)."""
    import hashlib
    import importlib.metadata
    import platform
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]

    def digest(paths):
        h = hashlib.sha256()
        for path in sorted(paths):
            h.update(str(path.relative_to(root) if path.is_relative_to(root) else path).encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def gpus():  # from nvidia-smi: importing JAX here would fix its config before main sets JAX_ENABLE_X64
        listing = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        rows = [dict(zip(("uuid", "name", "driver"), line.split(", "))) for line in listing.strip().splitlines()]
        return [r for r in rows if visible is None or r["uuid"] in visible.split(",")]

    inputs = {}
    for name in ("seed_run", "restart", "coils"):
        path = getattr(args, name, None)
        if path is not None and Path(path).exists():
            files = [p for p in Path(path).rglob("*") if p.is_file()] if Path(path).is_dir() else [Path(path)]
            inputs[name] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    record = dict(
        argv=sys.argv, cwd=os.getcwd(), driver=str(Path(__file__).resolve()),
        environment={k: v for k, v in os.environ.items() if k.startswith(("COIL_CASE", "CUDA", "XLA", "JAX"))},
        git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        code_sha256=digest([Path(__file__).resolve(), *(root / "vmex").rglob("*.py")]),
        python=platform.python_version(), gpus=gpus(),
        packages={n: version(n) for n in ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "scipy", "essos", "dkx",
                                          "solvax", "booz_xform_jax", "virtual-casing-jax", "neo-jax")},
        inputs=inputs)
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")


def main(argv=None):
    global RESOLUTION, CURRENT_IOTA_WEIGHT, VACUUM_IOTA_FLOOR, IOTA_FLOOR, IOTA_CEILING, ASPECT_RANGE, COIL_POINT_ROWS
    global BOOTSTRAP_IN_SOLVE, LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT
    args = parse_args(argv)
    if args.coil_limits is not None:
        LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT = args.coil_limits
    if args.redl_rows and BOOTSTRAP_MODEL != "redl":
        raise SystemExit("--redl-rows needs BOOTSTRAP_MODEL = 'redl'")
    args.current_constraint = args.current_constraint or args.redl_rows
    if args.redl_s_min:
        global REDL_SURFACES
        REDL_SURFACES = [float(s) for s in redl_surfaces() if s >= args.redl_s_min]
    if args.current_constraint or args.fixed_current:
        BOOTSTRAP_IN_SOLVE = False
    if args.current_iota_weight is not None:
        CURRENT_IOTA_WEIGHT = args.current_iota_weight
    if args.vacuum_iota_floor is not None:
        VACUUM_IOTA_FLOOR = args.vacuum_iota_floor
    if args.iota_floor is not None:
        IOTA_FLOOR = args.iota_floor
    if args.iota_ceiling is not None:
        IOTA_CEILING = args.iota_ceiling
    if args.aspect_range is not None:
        ASPECT_RANGE = tuple(args.aspect_range)
    COIL_POINT_ROWS = bool(args.coil_point_rows)
    RESOLUTION = (*(args.modes or RESOLUTION[:2]), args.ns or RESOLUTION[2])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_run_record(out, args)
    os.environ["JAX_ENABLE_X64"] = "1"
    # The vacuum-iota solve runs inside a jax.pure_callback; a GPU program XLA compiles there (its Krylov
    # certification fallback) deadlocks in the autotuner, which waits on the device the callback holds.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im

    started = time.monotonic()
    inp, redl = seed_input(), None
    if args.max_iterations:  # before the seed is built: its beta ramp and current blend solve with the same cap
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.seed_run is not None:
        inp = replace(vj.VmecInput.from_file(str(args.seed_run / "input.run")), lfreeb=False)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
        args.coils = args.seed_run / "coils.initial.json"
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, "gpu")
    else:
        inp, fixed = finite_beta_input(inp, args.beta, "gpu")
    if args.modes and (args.restart is not None or args.seed_run is not None):  # a saved run at another resolution
        inp = inp.change_resolution(mpol=args.modes[0], ntor=args.modes[1])
    fold = args.bootstrap and BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl"
    if args.current_knots_at_ends:
        knots = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, args.current_knots)) ** 2
        knots[0], knots[-1] = 0.0, 1.0
        inp = opt.resample_current_profile(inp, args.current_knots, knots=knots)
    profiles = redl_profiles(inp)[0] if fold else None
    redl = None if fold or args.fixed_current else redl
    if args.max_iterations:
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    inp.to_indata(out / "input.run")

    coils = resize_coils(Coils.from_json(str(args.coils)), COIL_ORDER, N_SEGMENTS)
    if args.restart is None and args.seed_run is None:
        coils = scale_coil_currents(coils, abs(float(fixed.wout.rbtor)))
        if args.beta > 0 and not args.no_coil_fit:
            coils = fit_coils_to_plasma(coils, fixed.wout, inp)
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    current = np.r_[np.asarray(inp.ac_aux_f)[:-1], inp.curtor] if args.bootstrap and not fold and not args.fixed_current else None
    current_dofs = tuple(np.flatnonzero(np.asarray(coils0.dofs_currents_raw))) if FREE_COIL_CURRENTS else ()
    scales = args.design_step_scale * np.r_[
        [PHIEDGE_STEP] * FREE_PHIEDGE, [CURRENT_STEP] * (0 if current is None else current.size),
        [COIL_CURRENT_STEP] * len(current_dofs),
        COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = vj.CoilDesign(coils0, current_dofs=current_dofs, scales=scales,
                      phiedge=float(inp.phiedge) if FREE_PHIEDGE else None,
                      plasma_current=current, plasma_current_spline=True)
    if args.pf_coils:
        from vmex.core.coil_design import PoloidalCoilDesign

        # the restarted run's shape, padded with zeros to --pf-shape modes
        saved_shape = getattr(args, "pf_saved_shape", None)
        modes = max(args.pf_shape or 0, 0 if saved_shape is None else np.shape(saved_shape)[-1])
        pf_shape = np.zeros((len(args.pf_coils), 2, modes)) if modes else None
        if saved_shape is not None:
            pf_shape[:, :, :np.shape(saved_shape)[-1]] = saved_shape
        chart = PoloidalCoilDesign(chart, *np.asarray(args.pf_coils, dtype=float).T, shape=pf_shape, nfp=int(inp.nfp),
                                   shape_scale=args.design_step_scale * PF_SHAPE_STEP if args.pf_shape else None,
                                   current_scale=None if args.pf_fixed else args.design_step_scale * PF_CURRENT_STEP,
                                   geometry_scale=args.design_step_scale * PF_GEOMETRY_STEP if args.pf_geometry else None)
    qs = target_residual()
    mirror = (opt.mirror_ratio,) if MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if IOTA_CEILING else ()
    # rows of the trial's fitted boundary (rbc, zbs), differentiated through the boundary fit
    vacuum = opt.vacuum_iota(inp)
    if VACUUM_IOTA_FLOOR:  # a concrete solve before the jitted rows trace it: its host solve compiles off the GPU's back
        vacuum(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs))
    vacuum_floor = ((lambda rbc, zbs, x: jnp.min(vacuum(rbc, zbs))),) if VACUUM_IOTA_FLOOR else ()
    limits = coil_limits()

    def clearance(rbc, zbs, x):
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])
        return limits.surface_clearance(chart.coils_from_x(x), surface)

    ext_rows, ext_terms = load_extension(args.extension)

    # A closed field line outside the plasma, in the coils' plus the plasma's virtual-casing field: (trace of its
    # tangent map, gap to the boundary, distance to the coils, largest distance to the boundary, spread of its height
    # [, the two legs' sines][, the two outer legs' reach][, the two outer legs' distance to the coils]), NaN where
    # Newton from the line's last accepted nodes does not converge.
    edge_bounds = [] if args.edge_line is None else [
        tuple(args.edge_trace or (-np.inf, np.inf)) + (1.0,), tuple(args.edge_gap or (-np.inf, np.inf)) + (0.01,),
        (-np.inf if args.edge_coil_distance is None else args.edge_coil_distance, np.inf, 0.01),
        (-np.inf, np.inf if args.edge_wall is None else args.edge_wall[0] - args.edge_wall[1], 0.01),
        (-np.inf, np.inf if args.edge_height_spread is None else args.edge_height_spread, 0.01),
        *([(-args.edge_legs, args.edge_legs, 0.1)] * 2 if args.edge_legs is not None else []),
        *([(args.edge_outer_legs, np.inf, 0.01)] * 2 if args.edge_outer_legs is not None else []),
        *([(args.edge_leg_coils, np.inf, 0.01)] * 2 if args.edge_leg_coils is not None else []),
        *([(args.vessel_gap, np.inf, 0.01)] if args.vessel_gap is not None else []),
        *([(args.vessel_legs, np.inf, 0.01)] * 2 if args.vessel_legs is not None else [])]
    vessel = None if args.vessel is None else np.asarray(np.load(args.vessel)["outline"], dtype=float)
    # with a wall, the coils stay outside it
    coil_clearance = max(COIL_SURFACE_DISTANCE_LIMIT, args.edge_wall[0] + args.edge_wall[-1] if args.edge_wall else 0.0)
    edge_constrained = any(np.isfinite(b[:2]).any() for b in edge_bounds)
    if args.edge_line is not None:
        from vmex.core import periodic_fieldlines as pf
        from vmex.core import virtual_casing as vc

        saved = json.loads(args.edge_line.read_text())
        entry = saved["total"][args.edge_line_index]
        edge_guess = dict(R=np.asarray(entry["R_nodes"]), Z=np.asarray(entry["Z_nodes"]), periods=int(saved["periods"]))

        def edge_values(state, runtime, x, R, Z):
            data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=args.edge_vc_grid,
                                                    ntheta=args.edge_vc_grid)
            parameters_of, field = pf.exterior_field(chart)
            parameters = parameters_of(data, x)
            line = pf.periodic_fieldline(field, parameters, R, Z, nfp=int(inp.nfp), periods=edge_guess["periods"],
                                         max_iterations=30)
            M = pf.tangent_map(field, parameters, line)
            trace = jnp.trace(M)
            wires = [chart.coils_from_x(x).gamma.reshape(-1, 3)] + (
                [chart.poloidal_coils_from_x(x).gamma.reshape(-1, 3)] if args.pf_coils else [])
            to_coils = jnp.sqrt(jnp.min(jnp.sum((line.xyz[:, None] - jnp.concatenate(wires)[None]) ** 2, axis=-1)))
            to_plasma = pf.surface_distances(line.xyz, data.gamma, int(inp.nfp))
            reach = jnp.max(to_plasma)
            if args.edge_smooth_reach:  # nodes within ~1/RHO of the farthest all count: no kink where they swap
                rho = args.edge_smooth_reach
                reach = reach + jnp.log(jnp.mean(jnp.exp(rho * (to_plasma - reach)))) / rho
            values = jnp.stack((trace, jnp.min(to_plasma), to_coils, reach, jnp.std(line.Z)))
            found = line.step < 1e-8
            if args.edge_legs is not None:
                legs = [pf.manifold_crossing(field, parameters, line, M, nfp=int(inp.nfp), periods=edge_guess["periods"],
                                             iterations=n, stable=stable)
                        for n, stable in zip(args.edge_leg_maps, (False, True))]
                values = jnp.concatenate((values, jnp.stack([leg[1] for leg in legs])))
                found = found & (jnp.max(jnp.stack([leg[2] for leg in legs])) < 1e-6)
            frozen = jax.tree.map(jax.lax.stop_gradient, (parameters, line, M))
            options = dict(nfp=int(inp.nfp), periods=edge_guess["periods"], iterations=args.edge_outer_leg_maps)
            if args.edge_outer_legs is not None:  # the farthest each outer leg gets from the plasma
                cos, sin = jnp.cos(line.phi[0]), jnp.sin(line.phi[0])

                def off_plasma(leg_r, leg_z):
                    points = jnp.stack((leg_r * cos, leg_r * sin, leg_z), axis=-1).reshape(-1, 3)
                    return pf.surface_distances(points, data.gamma, int(inp.nfp))

                for stable in (False, True):  # the farthest point is found without gradients, then traced with them
                    far = jnp.argmax(off_plasma(*pf.manifold_outer_leg(field, *frozen, stable=stable, **options)))
                    reach = off_plasma(*pf.manifold_outer_leg(field, parameters, line, M, stable=stable, index=far, **options))
                    values = jnp.concatenate((values, reach))
            def chunked(f, points, size=1024):  # f of (size, 3) points, a chunk at a time: bounded memory
                n = points.shape[0]
                padded = jnp.pad(points, ((0, -n % size), (0, 0)), mode="edge").reshape(-1, size, 3)
                return jax.lax.map(f, padded).ravel()[:n]

            def in_vessel(p):  # signed distance to the vessel, negative inside
                return pf.vessel_distances(p, vessel, int(inp.nfp))

            if args.edge_leg_coils is not None:  # the closest each outer leg gets to a coil, inside the wall
                coils = jnp.concatenate(wires)
                wall = np.inf if args.edge_wall is None else args.edge_wall[0]

                def to_coils_of(R, Z, phi):  # (angles, ...) traces -> distance of every point to the coils
                    points = jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1).reshape(-1, 3)
                    near = lambda p: jnp.sqrt(jnp.min(jnp.sum((p[:, None] - coils[None]) ** 2, axis=-1), axis=1))
                    return points, chunked(near, points)

                for stable in (False, True):  # the closest point is found without gradients, then traced with them
                    R, Z, phi = pf.manifold_outer_leg(field, *frozen, stable=stable, path=True, per_map=128,
                                                      **options)
                    points, distance = to_coils_of(R, Z, jnp.broadcast_to(phi[:, None], R.shape))
                    inside = (chunked(in_vessel, points) <= 0.0 if vessel is not None else
                              chunked(lambda p: pf.surface_distances(p, data.gamma, int(inp.nfp)), points) <= wall)
                    angle, sample = jnp.divmod(jnp.argmin(jnp.where(inside, distance, jnp.inf)), R.shape[1])
                    R, Z, phi = pf.manifold_outer_leg(field, parameters, line, M, stable=stable, index=sample,
                                                      path=True, per_map=128, **options)
                    point = jnp.stack((R[angle] * jnp.cos(phi[angle]), R[angle] * jnp.sin(phi[angle]), Z[angle]))
                    values = jnp.concatenate((values, jnp.sqrt(jnp.min(jnp.sum((point - coils) ** 2, axis=-1)))[None]))
            if args.vessel_gap is not None:  # how far inside the vessel the boundary and the line stay
                boundary = jnp.moveaxis(data.gamma, 0, -1).reshape(-1, 3)
                inside = -chunked(in_vessel, jnp.concatenate((boundary, line.xyz)))
                values = jnp.concatenate((values, jnp.min(inside)[None]))
            if args.vessel_legs is not None:  # how far beyond the vessel each outer leg gets
                def beyond(leg_r, leg_z):
                    phi0 = line.phi[0]
                    points = jnp.stack((leg_r * jnp.cos(phi0), leg_r * jnp.sin(phi0), leg_z), axis=-1).reshape(-1, 3)
                    return in_vessel(points)

                for stable in (False, True):  # the farthest point is found without gradients, then traced with them
                    far = jnp.argmax(beyond(*pf.manifold_outer_leg(field, *frozen, stable=stable, **options)))
                    values = jnp.concatenate((values, beyond(*pf.manifold_outer_leg(
                        field, parameters, line, M, stable=stable, index=far, **options))))
            return jnp.where(found, values, jnp.nan), line

        def edge(state, runtime, x):
            # The guess is read when the row runs, not when it is compiled: it follows the accepted designs (log_step),
            # so Newton's method starts next to the line however far the line has moved since the start.
            node = jax.ShapeDtypeStruct(edge_guess["R"].shape, jnp.float64)
            R, Z = jax.pure_callback(lambda _: (np.asarray(edge_guess["R"], dtype=float), np.asarray(edge_guess["Z"], dtype=float)),
                                     (node, node), jax.lax.stop_gradient(x[0]), vmap_method="sequential")
            return edge_values(state, runtime, x, R, Z)[0]

    def extension_cost(state, runtime):
        return sum((0.5 * w * jnp.sum(jnp.ravel(f(state, runtime)) ** 2) for f, w in ext_terms.values()), 0.0)

    def conductor(x):  # the coils' total |current| x length [MA m]
        from vmex.core.coil_design import ampere_metres

        sets = [chart.coils_from_x(x)] + ([chart.poloidal_coils_from_x(x)] if args.pf_coils else [])
        return 1e-6 * ampere_metres(*sets)

    def loss(state, runtime, x):
        residuals = qs.residuals_state(state, runtime)
        current = 0.5 * CURRENT_IOTA_WEIGHT * current_iota(state, runtime) ** 2 if CURRENT_IOTA_WEIGHT else 0.0
        tape = args.conductor_weight * conductor(x) if args.conductor_weight else 0.0
        loss = 0.5 * jnp.vdot(residuals, residuals) + current + tape + extension_cost(state, runtime)
        return args.loss_scale * loss

    def aspect(state, runtime, x):
        return opt.aspect_ratio(state, runtime)

    problem = vj.ThreeTermsFreeBoundaryProblem.from_loss(
        inp, loss, chart.x0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_params_at,
        scales=chart.scales, names=chart.dof_names,
        quantities=(min_abs_iota, opt.major_radius, opt.axis_field_strength, *mirror, *ceiling,
                    *([redl.residuals_state if args.redl_rows else bootstrap_mismatch(inp, redl, "gpu")]
                      if redl is not None else []),
                    *([min_d_merc] if MERCIER_FLOOR is not None else []), *([current_iota] if CURRENT_IOTA_WEIGHT else []),
                    *(f for f, *_ in ext_rows.values())),
        parameter_quantities=(*([edge] if edge_constrained else []), aspect),
        boundary_quantities=(*vacuum_floor, clearance),
        config=vj.ThreeTermsConfig(labelling=args.labelling, label_weight=args.label_weight, nphi=args.vc_grid,
                                   ntheta=args.vc_grid, chunk=args.chunk,
                                   quadrature=args.quadrature or (4 * int(inp.nfp) * args.vc_grid, 2 * args.vc_grid),
                                   trial_ftol=args.trial_forward_ftol, bootstrap=profiles,
                                   bootstrap_helicity=REDL_HELICITY,
                                   **({} if args.gauge_tau is None else {"gauge_tau": args.gauge_tau})),
        boundary_ftol=args.trial_ftol, boundary_target_cost=args.trial_target_cost,
        boundary_max_residual=args.max_boundary_residual,
        boundary_start=None if args.restart is None else args.restart / "wout.nc",  # its interior state too
        gauge_refresh=args.gauge_refresh, gauge_max_share=args.gauge_max_share, gauge_release=args.gauge_release,
        gradient_rounds=args.gradient_rounds, gradient_rtol=args.gradient_rtol,
        gradient_fit_xtol=args.gradient_fit_xtol, gradient_certify=args.certify,
        promote_trials=args.promote_trials)
    model = problem.model
    print(f"[model] {model.n_boundary} boundary and {model.x0.size - model.n_boundary} bootstrap-current coordinates, "
          f"{chart.size} design coordinates; built in {time.monotonic() - started:.0f} s"
          + (f"; minimal gauge on {problem.stats['gauge_directions']} directions" if args.labelling in ("minimal", "flat_arc")
             else ""), flush=True)

    aspect_lower, aspect_upper = ASPECT_RANGE
    width, b_width = RADIUS_TOLERANCE - RADIUS_MARGIN, B_AXIS_TOLERANCE - B_AXIS_MARGIN
    nredl = 0 if redl is None else redl_surfaces().size if args.redl_rows else 1
    redl_row = np.sqrt(REDL_TOLERANCE / nredl) if args.redl_rows else None  # all rows met: f_boot <= REDL_TOLERANCE
    state0, params0 = problem.state(problem.x0)
    runtime0 = im.runtime_from_params(params0, model.cfg)
    extension_bounds = [(lo, hi, scale) for f, lo, hi, scale in ext_rows.values()
                        for _ in range(int(np.size(f(state0, runtime0))))]
    lower, upper, row_scales = zip(
        (IOTA_FLOOR + IOTA_MARGIN, np.inf, IOTA_FLOOR),
        (RADIUS_TARGET - width, RADIUS_TARGET + width, RADIUS_TOLERANCE),
        (B0 - b_width, B0 + b_width, B_AXIS_TOLERANCE),
        *([(-np.inf, MIRROR_LIMIT - MIRROR_MARGIN, MIRROR_LIMIT)] if mirror else []),
        *([(-np.inf, IOTA_CEILING - IOTA_MARGIN, IOTA_FLOOR)] if ceiling else []),
        *([(-redl_row, redl_row, redl_row)] * nredl if args.redl_rows else [(-np.inf, REDL_TOLERANCE, REDL_TOLERANCE)] * nredl),
        *([(MERCIER_FLOOR + MERCIER_MARGIN, np.inf, 1.0)] if MERCIER_FLOOR is not None else []),
        *([(-np.inf, np.inf, 1.0)] if CURRENT_IOTA_WEIGHT else []),  # logged; the loss carries it
        *extension_bounds,
        *(edge_bounds if edge_constrained else []),
        (aspect_lower, aspect_upper, 0.5 * (aspect_upper - aspect_lower)),
        *([(VACUUM_IOTA_FLOOR + IOTA_MARGIN, np.inf, VACUUM_IOTA_FLOOR)] if vacuum_floor else []),
        (coil_clearance + DISTANCE_MARGIN, np.inf, COIL_SURFACE_DISTANCE_LIMIT))
    # thousands of point rows: their Jacobian in forward mode, 32 directions at a time
    def pf_distance(x):  # smallest distance between the circular pairs and the other coils [m]
        pairs, others = (c.gamma.reshape(-1, 3) for c in (chart.poloidal_coils_from_x(x), chart.coils_from_x(x)))
        return jnp.sqrt(jnp.min(jnp.sum((pairs[:, None] - others[None]) ** 2, axis=-1)))

    def coils_off_vessel(x):  # smallest distance [m] of the coils and the pairs outside the vessel
        from vmex.core.periodic_fieldlines import vessel_distances

        points = jnp.concatenate([chart.coils_from_x(x).gamma.reshape(-1, 3)] + (
            [chart.poloidal_coils_from_x(x).gamma.reshape(-1, 3)] if args.pf_coils else []))
        return jnp.min(vessel_distances(points, vessel, int(inp.nfp)))

    def all_coil_rows(x):
        rows = limits.rows(chart.coils_from_x(x), per_point=COIL_POINT_ROWS)
        more = []  # the pairs' own limits, each feasible at >= 0
        if args.pf_coil_distance is not None:
            more.append(pf_distance(x) / args.pf_coil_distance - 1.0)
        if args.pf_shape:
            more.append(1.0 - jnp.max(chart.poloidal_coils_from_x(x).curvature) / args.pf_curvature)
        if args.pf_length_max is not None:
            gamma = chart.poloidal_coils_from_x(x).gamma
            lengths = jnp.sum(jnp.linalg.norm(jnp.roll(gamma, -1, axis=1) - gamma, axis=-1), axis=1)
            more.append(1.0 - jnp.max(lengths) / args.pf_length_max)
        if args.pf_current_max is not None:
            more.append(1.0 - jnp.max(jnp.abs(chart.pairs_at(x)[2])) / args.pf_current_max)
        if args.vessel_coils is not None:  # every coil point MIN or more outside the vessel
            more.append(coils_off_vessel(x) / args.vessel_coils - 1.0)
        return jnp.concatenate((jnp.ravel(rows), jnp.stack(more))) if more else rows

    coil_rows = vj.coil_constraint(all_coil_rows, forward_chunk=32 if COIL_POINT_ROWS else None)
    constraints = [problem.nonlinear_constraint(list(lower), list(upper), scales=list(row_scales)), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        chart.coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        if edge_latest:  # the accepted design's line, in the layout of benchmarks/edge_fixed_points.py
            (out / f"edge_line{tag}.json").write_text(json.dumps(dict(periods=edge_guess["periods"], total=[edge_latest])))
        if args.pf_coils:
            (out / f"pf{tag}.json").write_text(json.dumps(chart.pairs_dict(jnp.asarray(x))))
        vj.write_wout(str(out / f"wout{tag}.nc"), problem.equilibrium_from_x(x).wout)

    last, edge_latest = dict(time=time.monotonic(), x=None), {}
    diagnostics = jax.jit(lambda state, params: (current_iota(state, im.runtime_from_params(params, model.cfg)),
                                                 min_d_merc(state, im.runtime_from_params(params, model.cfg))))
    extension_values = jax.jit(lambda state, params: (
        {name: f(state, im.runtime_from_params(params, model.cfg)) for name, (f, *_) in {**ext_rows, **ext_terms}.items()},
        extension_cost(state, im.runtime_from_params(params, model.cfg))))

    def label_speed_ratio(params):
        """Largest over smallest |d(R, Z)/d theta| on the LCFS, worst of 9 toroidal cuts (1: equal arc length)."""
        rbc, zbs = np.asarray(params.rbc), np.asarray(params.zbs)
        ntor = (rbc.shape[0] - 1) // 2
        m = np.arange(rbc.shape[1])[None, :]
        n = (np.arange(rbc.shape[0]) - ntor)[:, None] * int(inp.nfp)
        theta, worst = np.linspace(0, 2 * np.pi, 361, endpoint=False), 1.0
        for phi in np.linspace(0, 2 * np.pi / int(inp.nfp), 9, endpoint=False):
            angle = m[None] * theta[:, None, None] - n[None] * phi
            dR = -np.sum(rbc[None] * m[None] * np.sin(angle), axis=(1, 2))
            dZ = np.sum(zbs[None] * m[None] * np.cos(angle), axis=(1, 2))
            speed = np.hypot(dR, dZ)
            worst = max(worst, float(speed.max() / speed.min()))
        return worst

    def boundary_jacobian_condition():
        """Condition number of the last boundary Jacobian (columns scaled to unit norm), None before one."""
        J_b = problem._anchor.get("J_b") if isinstance(problem._anchor, dict) else None
        if J_b is None:
            return None
        J_b = np.asarray(J_b)
        sv = np.linalg.svd(J_b / np.maximum(np.linalg.norm(J_b, axis=0), 1e-300), compute_uv=False)
        return float(sv[0] / sv[-1])

    edge_at = (jax.jit(lambda state, params, x, R, Z: edge_values(state, im.runtime_from_params(params, model.cfg), x, R, Z))
               if args.edge_line is not None else None)

    def log_step():
        x = problem.accepted.parameters
        h = problem.constraint_values(x)
        redl_h = np.asarray(h[3 + len(mirror) + len(ceiling):][:nredl])
        res = problem.boundary_residual(x)
        now = time.monotonic()
        state_x, params_x = problem.state(x)
        runtime_x = im.runtime_from_params(params_x, model.cfg)
        edge_row = {}
        outer = 5 + 2 * (args.edge_legs is not None)  # where the outer legs' values start
        leg_coils = outer + 2 * (args.edge_outer_legs is not None)
        vessel_gap = leg_coils + 2 * (args.edge_leg_coils is not None)
        vessel_legs = vessel_gap + (args.vessel_gap is not None)
        if edge_at is not None:  # the line follows the accepted designs: each solve starts from the last one's nodes
            values, line = edge_at(state_x, params_x, jnp.asarray(x), jnp.asarray(edge_guess["R"]),
                                   jnp.asarray(edge_guess["Z"]))
            values = np.asarray(values)
            if np.all(np.isfinite(values)):
                edge_guess.update(R=np.asarray(line.R), Z=np.asarray(line.Z))
            edge_row = dict(edge_trace=float(values[0]), edge_gap_m=float(values[1]), edge_coil_distance_m=float(values[2]),
                            edge_reach_m=float(values[3]), edge_height_spread_m=float(values[4]),
                            **({"edge_leg_sines": values[5:7].tolist()} if args.edge_legs is not None else {}),
                            **({"edge_outer_legs_m": values[outer:outer + 2].tolist()}
                               if args.edge_outer_legs is not None else {}),
                            **({"edge_leg_coils_m": values[leg_coils:leg_coils + 2].tolist()}
                               if args.edge_leg_coils is not None else {}),
                            **({"vessel_gap_m": float(values[vessel_gap])} if args.vessel_gap is not None else {}),
                            **({"vessel_legs_m": values[vessel_legs:vessel_legs + 2].tolist()}
                               if args.vessel_legs is not None else {}),
                            edge_R_m=float(line.R[0]), edge_Z_m=float(line.Z[0]), edge_newton_step_m=float(line.step))
            if np.all(np.isfinite(values)):
                edge_latest.update(R=float(line.R[0]), Z=float(line.Z[0]), trace=float(values[0]),
                                   R_nodes=np.asarray(line.R).tolist(), Z_nodes=np.asarray(line.Z).tolist())
            if args.pf_coils:
                edge_row.update({f"pf_{k}": v for k, v in chart.pairs_dict(jnp.asarray(x)).items() if k != "nfp"})
        try:  # a diagnostic: its own fixed-boundary solve may fail where the accepted point is fine
            vac = np.asarray(vacuum(params_x.rbc, params_x.zbs))
        except RuntimeError as error:
            vac = np.full(3, np.nan)
            print(f"[step {problem.accepted_step}] vacuum iota not available: {str(error).splitlines()[0][:200]}",
                  flush=True)
        iota_i, d_merc = map(float, diagnostics(state_x, params_x))
        ext_values, ext_cost = extension_values(state_x, params_x)
        tape = float(conductor(jnp.asarray(x)))
        row = dict(step=problem.accepted_step,
                   qa=2 * (problem.fun(x) / args.loss_scale - args.conductor_weight * tape)
                   - CURRENT_IOTA_WEIGHT * iota_i**2 - 2 * float(ext_cost), conductor_ma_m=tape,
                   **{f"extension_{name}": np.asarray(v).tolist() for name, v in ext_values.items()},
                   current_iota=iota_i, min_d_merc=d_merc, min_abs_iota=float(h[0]),
                   geometric_axis_iota=float(abs(opt.axis_iota(state_x, runtime_x))),
                   vacuum_iota_axis=float(vac[0]), vacuum_iota_min=float(vac.min()), vacuum_iota_edge=float(vac[-1]),
                   major_radius_m=float(h[1]), b_axis_t=float(h[2]),
                   beta=float(opt.volume_average_beta(state_x, runtime_x)),
                   coil_currents_a=np.asarray(chart.base_currents_at(jnp.asarray(x))).tolist(),
                   **({"redl_mismatch": float(np.sum(redl_h**2) if args.redl_rows else redl_h[0]),  # f_boot either way
                       "redl_max_row": float(np.max(np.abs(redl_h)))} if nredl else {}),
                   **({"redl_max_relative": model.bootstrap_residual(state_x, params_x)} if fold else {}),
                   coil_surface_distance_m=float(h[-1]), aspect=float(h[-2 - len(vacuum_floor)]),
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   **({"vessel_coils_m": float(coils_off_vessel(jnp.asarray(x)))} if vessel is not None else {}),
                   phiedge_factor=float(chart.phiedge_at(jnp.asarray(x)) / chart.phiedge) if FREE_PHIEDGE else 1.0,
                   bn=res.normal, pressure_balance=res.pressure, sheet_current=res.sheet_current,
                   label_speed_ratio=label_speed_ratio(params_x), boundary_jacobian_cond=boundary_jacobian_condition(),
                   step_box=None if last["x"] is None else float(np.max(np.abs(x - last["x"]) / problem.scales)),
                   **edge_row,
                   **({"pf_coil_distance_m": float(pf_distance(jnp.asarray(x)))} if args.pf_coils else {}),
                   step_seconds=now - last["time"], elapsed_seconds=now - started, **problem.stats,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        if model._gauge is not None:  # the gauge rows' share of the accepted fit's cost (0 where it was set)
            M, reference = model._gauge
            g = np.asarray(M @ ((model._arc_label(params_x.rbc, params_x.zbs) if model.labelling == "flat_arc"
                                 else model._boundary_of(params_x)) - reference))
            r = np.asarray(model._rows(state_x, params_x, problem.field(jnp.asarray(x))))
            row["gauge_share"] = float(g @ g / (r @ r))
        last.update(time=now, x=np.array(x, dtype=float))
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"vacuum iota={row['vacuum_iota_min']:.4f} (axis {row['vacuum_iota_axis']:.4f}) "
              f"B_axis={row['b_axis_t']:.5f} iota_I={row['current_iota']:.4f} DMerc_min={row['min_d_merc']:.3f} "
              f"R={row['major_radius_m']:.5f} aspect={row['aspect']:.4f} "
              f"clearance={row['coil_surface_distance_m']:.4f} "
              f"coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              + (f"redl={row['redl_mismatch']:.2e} " if nredl else "")
              + (f"redl(max rel)={row['redl_max_relative']:.1e} " if fold else "")
              + f"B.n={res.normal:.1e} K={res.sheet_current:.1e} label_ratio={row['label_speed_ratio']:.2f} "
              + (f"cond={row['boundary_jacobian_cond']:.0f} " if row["boundary_jacobian_cond"] else "")
              + (f"gauge={row['gauge_share']:.1e}/{row['gauge_directions']} " if "gauge_share" in row else "")
              + (f"newton={row['newton_products']}/{row['newton_residual']:.0e} " if "newton_residual" in row else "")
              + (f"fit={row['fit_steps']}/{row['fit_step']:.0e} " if "fit_step" in row else "")
              + (f"edge: Tr M={row['edge_trace']:.3f} gap={row['edge_gap_m']:.4f} reach={row['edge_reach_m']:.4f} dZ={row['edge_height_spread_m']:.4f} "
                 + (f"legs={row['edge_leg_sines'][0]:+.4f},{row['edge_leg_sines'][1]:+.4f} " if "edge_leg_sines" in row else "")
                 + (f"outer={row['edge_outer_legs_m'][0]:.4f},{row['edge_outer_legs_m'][1]:.4f} " if "edge_outer_legs_m" in row else "")
                 + (f"leg-coils={row['edge_leg_coils_m'][0]:.4f},{row['edge_leg_coils_m'][1]:.4f} "
                    if "edge_leg_coils_m" in row else "")
                 + (f"vessel-gap={row['vessel_gap_m']:.4f} " if "vessel_gap_m" in row else "")
                 + (f"vessel-coils={row['vessel_coils_m']:.4f} " if "vessel_coils_m" in row else "")
                 + (f"vessel-legs={row['vessel_legs_m'][0]:.4f},{row['vessel_legs_m'][1]:.4f} "
                    if "vessel_legs_m" in row else "")
                 + f"coils={row['edge_coil_distance_m']:.4f} "
                 f"(R, Z)=({row['edge_R_m']:.4f}, {row['edge_Z_m']:.4f}) " if edge_row else "")
              + f"conductor={tape:.3f} "
              + (f"step={row['step_box']:.2f} " if row["step_box"] is not None else "")
              + f"{row['step_seconds']:.1f}s "
              f"(trials {row['trials']}, boundary evaluations {row['boundary_evaluations']}, "
              f"peak {row['peak_gpu_gib']:.1f} GiB)", flush=True)
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    if args.gradient_report:
        gradient_report(problem, chart.dof_names, lower, upper, row_scales, out, coil_rows, scan=args.gradient_scan)
        return
    if args.check_gradient:
        check_gradient(problem, args.check_gradient, out, fit=args.check_gradient_fit)
        return
    log_step()
    result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                          callback=lambda x: log_step(),
                          options=dict(maxiter=args.steps - problem.accepted_step, ftol=OPTIMIZER_FTOL,
                                       restart_on_rejection=True, max_step=args.max_step,
                                       step_box=args.step_box))
    save("")
    summary = dict(accepted_steps=problem.accepted_step, success=bool(result.success), message=str(result.message),
                   stop_reason=getattr(result, "stop_reason", None),
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   rejection_reasons=problem.stats["rejection_reasons"],
                   boundary_residual={k: float(v) for k, v in
                                      vars(problem.boundary_residual(problem.accepted.parameters)).items()},
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
