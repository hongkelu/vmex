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

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and CURTOR
# are design variables, and the mismatch sum_j R_j^2 against the BOOTSTRAP_MODEL current (Redl
# or DKX, in Redl's normalized form) is held under REDL_TOLERANCE. The seed's Picard loop is Redl's.
# DKX: the current is a line segment through a value on every half-mesh surface (those values,
# the axis one and CURTOR are design variables), its seed DKX's on the same surfaces.
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
    # A circular tokamak with a 1% helical ripple b / a, as vmex examples/data/input.minimal_seed_nfp*.
    # It has no vacuum transform: an Ohmic current of about the final bootstrap current carries the beta ramp.
    SEED = (SEED[0], SEED[1], 0.01)
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

    The surfaces run through one ``lax.map`` instead of DKX's Python loop, which traces a Boozer transform, kinetic
    operator and solve per surface: at 50 surfaces the gradient compiled in 1016 s, mapped in 40 s, at the same run
    time and memory and the same values to the linear solve's tolerance (2e-7).
    """
    import dataclasses

    import jax
    import jax.numpy as jnp
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from dkx.bootstrap import PARALLEL_CURRENT, KineticBootstrapMismatch
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve
    from dkx.workflows.geometry_adapters import PSI_A_HAT_LEAVES, kinetic_operator_on_boozer_surface
    from vmex.core.boozer_tables import boozer_input_tables

    class Mapped(KineticBootstrapMismatch):
        def _kinetic(self, state, rt):
            setup = rt.setup
            s_full = np.asarray(setup.s_full)
            nfp = int(rt.resolution.nfp)
            rows, s_rows, templates = self._setup(int(s_full.size))
            psi_a = jnp.abs((s_full[1] - s_full[0]) * jnp.sum(jnp.asarray(setup.phipf)[1:]))
            tables = boozer_input_tables(state, rt, int(rows[0]))
            xm, xn = np.asarray(tables["xm"]), np.asarray(tables["xn"])
            plan = self._plan(xm, xn, nfp)

            def surface(args):  # dkx.bootstrap.KineticBootstrapMismatch._kinetic's loop body
                row, template = args
                tables = boozer_input_tables(state, rt, row)
                booz = booz_xform_jax_impl(
                    **{k: jnp.asarray(tables[k])[None] for k in (
                        "rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")},
                    xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm), xn_nyq=jnp.asarray(xn),
                    constants=plan.constants, grids=plan.grids, plan=plan)
                operator = kinetic_operator_on_boozer_surface(
                    dataclasses.replace(template, **{leaf: getattr(template, leaf) / psi_a
                                                     for leaf in PSI_A_HAT_LEAVES}),
                    bmnc_b=booz["bmnc_b"][0], ixm_b=np.asarray(plan.grids.xm_b), ixn_b=np.asarray(plan.grids.xn_b),
                    nfp=nfp, iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0],
                    signgs=int(np.sign(setup.signgs)))
                solved = solve(operator, operator.rhs(), method="auto", tol=self.tol, differentiable=True,
                               tier1_keep_lowest=operator.n_xi, emit=None)
                moments = profile_moments_from_operator(operator, solved.x.reshape(-1))
                return moments["FSABjHat"].reshape(()) * PARALLEL_CURRENT

            stacked = jax.tree.map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *templates)
            return s_rows, jax.lax.map(surface, (jnp.asarray(rows), stacked))

    surfaces = redl_surfaces() if DKX_SURFACES is None else DKX_SURFACES
    return Mapped(redl_profiles(inp)[0], surfaces=surfaces, collision_operator=DKX_COLLISION_OPERATOR,
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
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current profile (``--restart``)."""
    import vmex as vj

    final = vj.VmecInput.from_wout(run / "wout.nc")
    return replace(vj.VmecInput.from_file(run / "input.run"), lfreeb=False, **{
        name: getattr(final, name) for name in ("rbc", "zbs", "phiedge", "pcurr_type", "ac", "ac_aux_s", "ac_aux_f",
                                                "curtor")})


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
    parser.add_argument("--vacuum-iota-floor", type=float, help="floor on the boundary's vacuum |iota| "
                        "(default: VACUUM_IOTA_FLOOR, 0.23 for qa4-beta)")
    parser.add_argument("--iota-ceiling", type=float, help="ceiling on max |iota| (default: IOTA_CEILING)")
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
    parser.add_argument("--trial-ftol", type=float, default=1e-3,
                        help="relative cost change at which a trial's boundary steps stop")
    parser.add_argument("--max-boundary-residual", type=float, default=0.05,
                        help="largest interface RMS (B.n, pressure balance or K, over |B|) of a solved trial")
    parser.add_argument("--trial-forward-ftol", type=float, default=1e-11,
                        help="force residual of the trials' equilibria (each accepted point is re-solved to the "
                        "deck's tolerance before its gradient)")
    parser.add_argument("--labelling", choices=("anchored", "equal_arc", "none"), default="anchored",
                        help="the boundary's poloidal labelling rows: anchored to the start boundary (anchored), "
                        "equal arc length of the current surface (equal_arc), or none; see vmex.ThreeTermsConfig")
    parser.add_argument("--label-weight", type=float,
                        help="weight of the labelling rows (default: the labelling's, 1e-2 anchored, 1e-3 equal arc; "
                        "0 is the no-label model)")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
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


def check_gradient(problem, directions, out, steps=(1e-2, 1e-3, 1e-4)):
    """Implicit design gradients of the loss and every quantity at the seed against central differences of
    re-solved trials, along ``directions`` random unit directions in the optimizer's coordinates, at each step in
    ``steps``; the relative difference ``|implicit - difference| / |difference|`` is written to
    ``gradient_check.json`` (the loss on its own, the quantities as one vector)."""
    x = np.asarray(problem.accepted.parameters, dtype=float)
    _, grad = problem.value_and_grad(x)
    jac = problem.constraint_jac(x)
    rng = np.random.default_rng(0)
    records = []
    for k in range(directions):
        d = rng.standard_normal(x.size)
        d /= np.linalg.norm(d)
        implicit = np.r_[grad @ d, jac @ d]
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
    global RESOLUTION, CURRENT_IOTA_WEIGHT, VACUUM_IOTA_FLOOR, IOTA_CEILING, ASPECT_RANGE, COIL_POINT_ROWS
    global BOOTSTRAP_IN_SOLVE
    args = parse_args(argv)
    if args.current_constraint:
        BOOTSTRAP_IN_SOLVE = False
    if args.current_iota_weight is not None:
        CURRENT_IOTA_WEIGHT = args.current_iota_weight
    if args.vacuum_iota_floor is not None:
        VACUUM_IOTA_FLOOR = args.vacuum_iota_floor
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
    fold = args.bootstrap and BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl"
    profiles = redl_profiles(inp)[0] if fold else None
    redl = None if fold else redl
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
    current = np.r_[np.asarray(inp.ac_aux_f)[:-1], inp.curtor] if args.bootstrap and not fold else None
    current_dofs = tuple(np.flatnonzero(np.asarray(coils0.dofs_currents_raw))) if FREE_COIL_CURRENTS else ()
    scales = args.design_step_scale * np.r_[
        [PHIEDGE_STEP] * FREE_PHIEDGE, [CURRENT_STEP] * (0 if current is None else current.size),
        [COIL_CURRENT_STEP] * len(current_dofs),
        COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = vj.CoilDesign(coils0, current_dofs=current_dofs, scales=scales,
                      phiedge=float(inp.phiedge) if FREE_PHIEDGE else None,
                      plasma_current=current, plasma_current_spline=True)
    qs = target_residual()
    mirror = (opt.mirror_ratio,) if MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if IOTA_CEILING else ()
    # rows of the trial's fitted boundary (rbc, zbs), differentiated through the boundary fit
    vacuum = opt.vacuum_iota(inp)
    vacuum_floor = ((lambda rbc, zbs, x: jnp.min(vacuum(rbc, zbs))),) if VACUUM_IOTA_FLOOR else ()
    limits = coil_limits()

    def clearance(rbc, zbs, x):
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])
        return limits.surface_clearance(chart.coils_from_x(x), surface)

    ext_rows, ext_terms = load_extension(args.extension)

    def extension_cost(state, runtime):
        return sum((0.5 * w * jnp.sum(jnp.ravel(f(state, runtime)) ** 2) for f, w in ext_terms.values()), 0.0)

    def loss(state, runtime, x):
        residuals = qs.residuals_state(state, runtime)
        current = 0.5 * CURRENT_IOTA_WEIGHT * current_iota(state, runtime) ** 2 if CURRENT_IOTA_WEIGHT else 0.0
        return 0.5 * jnp.vdot(residuals, residuals) + current + extension_cost(state, runtime)

    def aspect(state, runtime, x):
        return opt.aspect_ratio(state, runtime)

    problem = vj.ThreeTermsFreeBoundaryProblem.from_loss(
        inp, loss, chart.x0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_params_at,
        scales=chart.scales, names=chart.dof_names,
        quantities=(min_abs_iota, opt.major_radius, opt.axis_field_strength, *mirror, *ceiling,
                    *([bootstrap_mismatch(inp, redl, "gpu")] if redl is not None else []),
                    *([min_d_merc] if MERCIER_FLOOR is not None else []), *([current_iota] if CURRENT_IOTA_WEIGHT else []),
                    *(f for f, *_ in ext_rows.values())),
        parameter_quantities=(aspect,), boundary_quantities=(*vacuum_floor, clearance),
        config=vj.ThreeTermsConfig(labelling=args.labelling, label_weight=args.label_weight, nphi=args.vc_grid,
                                   ntheta=args.vc_grid, chunk=args.chunk,
                                   quadrature=args.quadrature or (4 * int(inp.nfp) * args.vc_grid, 2 * args.vc_grid),
                                   trial_ftol=args.trial_forward_ftol, bootstrap=profiles,
                                   bootstrap_helicity=REDL_HELICITY),
        boundary_ftol=args.trial_ftol, boundary_max_residual=args.max_boundary_residual)
    model = problem.model
    print(f"[model] {model.n_boundary} boundary and {model.x0.size - model.n_boundary} bootstrap-current coordinates, "
          f"{chart.size} design coordinates; built in {time.monotonic() - started:.0f} s", flush=True)

    aspect_lower, aspect_upper = ASPECT_RANGE
    width, b_width = RADIUS_TOLERANCE - RADIUS_MARGIN, B_AXIS_TOLERANCE - B_AXIS_MARGIN
    nredl = int(redl is not None)
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
        *[(-np.inf, REDL_TOLERANCE, REDL_TOLERANCE)] * nredl,
        *([(MERCIER_FLOOR + MERCIER_MARGIN, np.inf, 1.0)] if MERCIER_FLOOR is not None else []),
        *([(-np.inf, np.inf, 1.0)] if CURRENT_IOTA_WEIGHT else []),  # logged; the loss carries it
        *extension_bounds,
        (aspect_lower, aspect_upper, 0.5 * (aspect_upper - aspect_lower)),
        *([(VACUUM_IOTA_FLOOR + IOTA_MARGIN, np.inf, VACUUM_IOTA_FLOOR)] if vacuum_floor else []),
        (COIL_SURFACE_DISTANCE_LIMIT + DISTANCE_MARGIN, np.inf, COIL_SURFACE_DISTANCE_LIMIT))
    # thousands of point rows: their Jacobian in forward mode, 32 directions at a time
    coil_rows = vj.coil_constraint(lambda x: limits.rows(chart.coils_from_x(x), per_point=COIL_POINT_ROWS),
                                   forward_chunk=32 if COIL_POINT_ROWS else None)
    constraints = [problem.nonlinear_constraint(list(lower), list(upper), scales=list(row_scales)), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        chart.coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), problem.equilibrium_from_x(x).wout)

    last = dict(time=time.monotonic())
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

    def log_step():
        x = problem.accepted.parameters
        h = problem.constraint_values(x)
        res = problem.boundary_residual(x)
        now = time.monotonic()
        state_x, params_x = problem.state(x)
        runtime_x = im.runtime_from_params(params_x, model.cfg)
        try:  # a diagnostic: its own fixed-boundary solve may fail where the accepted point is fine
            vac = np.asarray(vacuum(params_x.rbc, params_x.zbs))
        except RuntimeError as error:
            vac = np.full(3, np.nan)
            print(f"[step {problem.accepted_step}] vacuum iota not available: {str(error).splitlines()[0][:200]}",
                  flush=True)
        iota_i, d_merc = map(float, diagnostics(state_x, params_x))
        ext_values, ext_cost = extension_values(state_x, params_x)
        row = dict(step=problem.accepted_step,
                   qa=2 * problem.fun(x) - CURRENT_IOTA_WEIGHT * iota_i**2 - 2 * float(ext_cost),
                   **{f"extension_{name}": np.asarray(v).tolist() for name, v in ext_values.items()},
                   current_iota=iota_i, min_d_merc=d_merc, min_abs_iota=float(h[0]),
                   geometric_axis_iota=float(abs(opt.axis_iota(state_x, runtime_x))),
                   vacuum_iota_axis=float(vac[0]), vacuum_iota_min=float(vac.min()), vacuum_iota_edge=float(vac[-1]),
                   major_radius_m=float(h[1]), b_axis_t=float(h[2]),
                   beta=float(opt.volume_average_beta(state_x, runtime_x)),
                   coil_currents_a=np.asarray(chart.base_currents_at(jnp.asarray(x))).tolist(),
                   **({"redl_mismatch": float(h[3 + len(mirror) + len(ceiling)])} if nredl else {}),
                   **({"redl_max_relative": model.bootstrap_residual(state_x, params_x)} if fold else {}),
                   coil_surface_distance_m=float(h[-1]), aspect=float(h[-2 - len(vacuum_floor)]),
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   phiedge_factor=float(chart.phiedge_at(jnp.asarray(x)) / chart.phiedge) if FREE_PHIEDGE else 1.0,
                   bn=res.normal, pressure_balance=res.pressure, sheet_current=res.sheet_current,
                   label_speed_ratio=label_speed_ratio(params_x), boundary_jacobian_cond=boundary_jacobian_condition(),
                   step_seconds=now - last["time"], elapsed_seconds=now - started, **problem.stats,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        last.update(time=now)
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
              + f"{row['step_seconds']:.1f}s "
              f"(trials {row['trials']}, boundary evaluations {row['boundary_evaluations']}, "
              f"peak {row['peak_gpu_gib']:.1f} GiB)", flush=True)
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    if args.check_gradient:
        check_gradient(problem, args.check_gradient, out)
        return
    log_step()
    result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                          callback=lambda x: log_step(),
                          options=dict(maxiter=args.steps - problem.accepted_step, ftol=OPTIMIZER_FTOL,
                                       restart_on_rejection=True))
    save("")
    summary = dict(accepted_steps=problem.accepted_step, success=bool(result.success), message=str(result.message),
                   stop_reason=getattr(result, "stop_reason", None),
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   boundary_residual={k: float(v) for k, v in
                                      vars(problem.boundary_residual(problem.accepted.parameters)).items()},
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
