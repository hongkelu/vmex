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
vacuum |iota| of the plasma boundary (``vacuum_iota``).

The problem is ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
(:class:`vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem`): warm-started
trial boundaries and design gradients by the implicit function theorem of the
boundary fit, without recompiling when the coils or plasma parameters change.

    COIL_CASE=qa4-beta python single_stage_free_boundary_optimization_three_term.py --beta 0.025 --bootstrap \
        --modes 8 8 --ns 51 --output runs/three-term-qa4

``--seed-run <run>`` starts from another run's calibrated deck and fitted coils.
``--resume <dir>/checkpoint.npz`` continues a run from its accepted point; with
the ``slsqp_state.npz`` written beside every checkpoint (SciPy SLSQP's whole
state at that point: quasi-Newton Hessian, multipliers, coordinates;
``opt.save_slsqp_state``) SLSQP continues with its Hessian instead of the
identity, so a run chained over 2 h segments takes the uninterrupted run's
steps. A state of another point is refused (an identity Hessian, logged).
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
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any

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


#: The kinetic profiles of a run this one continues (``--restart``/``--seed-run``), read from its profiles.json: a
#: restart keeps the physical problem instead of rebuilding n and T from the restored boundary's RBC(0,0).
PINNED_PROFILES = None


def save_profiles(profiles, path):
    """Write the kinetic profiles' coefficients (profiles.json)."""
    path.write_text(json.dumps({k: np.atleast_1d(np.asarray(getattr(profiles, k), dtype=float)).tolist()
                                for k in ("ne_coeffs", "Te_coeffs", "Ti_coeffs", "Zeff_coeffs")}, indent=1) + "\n")


def pin_profiles(run):
    """Use the kinetic profiles of the run folder ``run``: its profiles.json, or, for a run from before it was
    written, those of its own input.run (the deck it was launched with, not a restored boundary)."""
    global PINNED_PROFILES
    from vmex.core.bootstrap import KineticProfiles
    import vmex as vj

    if (run / "profiles.json").exists():
        data = json.loads((run / "profiles.json").read_text())
        PINNED_PROFILES = KineticProfiles(**{k: np.asarray(v, dtype=float) for k, v in data.items()})
        return
    print(f"warning: {run} has no profiles.json: kinetic profiles rebuilt from its input.run", flush=True)
    PINNED_PROFILES = redl_profiles(vj.VmecInput.from_file(str(run / "input.run")))[0]


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch
    (the pinned profiles of a continued run, :func:`pin_profiles`, when there are any)."""
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    if PINNED_PROFILES is not None:
        return PINNED_PROFILES, RedlBootstrapMismatch(PINNED_PROFILES, REDL_HELICITY, redl_surfaces(),
                                                      n_lambda=REDL_N_LAMBDA)
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
    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel

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
        model = ThreeTermFreeBoundaryModel(inp, bootstrap=profiles, bootstrap_helicity=REDL_HELICITY,
                                           fixed_boundary=True)
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


def abs_iota(state, runtime):
    """|iota| bounded by the floor and ceiling rows.

    The half-mesh surfaces (axis slot excluded), as ``opt.min_abs_iota``; with
    ``IOTA_EDGE`` also VMEC's edge iotaf[-1] = 1.5 iotas[-1] - 0.5 iotas[-2].
    The axis is left out: there a bootstrap current's steep part of iota
    (~ s^(1/4)) makes VMEC's extrapolated iotaf[0] drift with ns, and the
    vacuum iota row bounds the shaping's share (``vacuum_iota``).
    """
    import jax.numpy as jnp
    from vmex.core.statephysics import _iotas_half  # private: opt exposes only the half-mesh minimum

    half = _iotas_half(state, runtime)[1:]
    if IOTA_EDGE:
        half = jnp.concatenate([half, 1.5 * half[-1:] - 0.5 * half[-2:-1]])
    return jnp.abs(half)


def vacuum_iota(inp):
    """(state, runtime) -> the vacuum |iota| of the state's plasma boundary: axis, half-mesh surfaces and edge.

    The fixed-boundary equilibrium of that boundary without pressure or current
    (``inp`` otherwise, hot-restarted), differentiable through its implicit
    adjoint; the axis value is ``opt.axis_iota``, the edge VMEC's iotaf[-1].
    """
    import jax
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core import implicit as im
    from vmex.core.statephysics import _iotas_half

    vacuum = replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                     ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None, lfreeb=False)
    device = jax.devices()[0]  # as ThreeTermFreeBoundaryModel: unpinned, the certification runs on the host CPU
    cfg = im.make_config(vacuum, multigrid=True, hot_restart=True, device=device)
    base = im.params_from_input(vacuum, device=device)
    im.runtime_from_params(base, cfg)  # its setup, built here: run_setup cannot be traced inside a jit

    def iota(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)  # private: no public traced LCFS of a state
        rmnc, zmns = jax.device_put((rmnc, zmns), device)  # a fixed-boundary state may live on the host
        rows, cols = np.asarray(runtime.modes.n) + int(inp.ntor), np.asarray(runtime.modes.m)
        params = replace(base, rbc=jnp.zeros_like(base.rbc).at[rows, cols].set(rmnc),
                         zbs=jnp.zeros_like(base.zbs).at[rows, cols].set(zmns))
        vstate = im.solve_implicit(params, cfg)
        vrt = im.runtime_from_params(params, cfg)
        half = _iotas_half(vstate, vrt)[1:]
        return jnp.abs(jnp.concatenate([opt.axis_iota(vstate, vrt)[None], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]]))

    return iota


def current_iota(state, runtime):
    """``mu0 |I| R0 / (2 pi a^2 B_axis)``: the edge iota the net toroidal current ``I`` gives a cylinder of the
    plasma's major radius ``R0``, minor radius ``a = R0 / aspect`` and axis field -- the current's share of iota.

    ``mu0 I`` is VMEX's ``ctor`` (``fields.surface_currents``, in T m), the bootstrap current where it is solved
    with the equilibrium.
    """
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


def min_d_merc(state, runtime):
    """Smallest Mercier ``DMerc`` (positive is stable) of the full-mesh surfaces ``MERCIER_S_MIN <= s < 1``."""
    import jax.numpy as jnp
    from vmex import optimize as opt

    d_merc = opt.d_merc_state(state, runtime)
    s = jnp.asarray(runtime.setup.s_full)
    return jnp.min(jnp.where((s >= MERCIER_S_MIN) & (s < 1.0), d_merc, jnp.inf))


def min_abs_iota(state, runtime):
    """Smallest |iota| of ``abs_iota``: ``opt.min_abs_iota``, with ``IOTA_EDGE`` including the edge."""
    import jax.numpy as jnp

    return jnp.min(abs_iota(state, runtime))


def max_abs_iota(state, runtime):
    """Largest |iota| of ``abs_iota``, the counterpart of ``min_abs_iota``."""
    import jax.numpy as jnp

    return jnp.max(abs_iota(state, runtime))


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


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""

    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    return current_from_wout(replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)


def current_from_wout(inp, w):
    """``inp`` with ``w``'s current profile -- its type, knots or coefficients -- and CURTOR, when ``inp``
    prescribes the current (a run that solved the current changes its type and knots: line_segment_ip)."""

    if int(inp.ncurr) == 1:
        kind = str(w.pcurr_type).strip()
        if "spline" in kind or "line_segment" in kind:  # VmecInput trims both to the knots before the -1 padding
            inp = replace(inp, pcurr_type=kind, ac_aux_s=np.asarray(w.ac_aux_s, dtype=float),
                          ac_aux_f=np.asarray(w.ac_aux_f, dtype=float), curtor=float(w.ctor))
        else:
            inp = replace(inp, pcurr_type=kind, ac=np.asarray(w.ac, dtype=float)[: np.size(inp.ac)],
                          curtor=float(w.ctor))
    return inp


# ---- coils and their field -------------------------------------------------------------------------------------------
def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
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


# ---- coil chart and coil inequalities -------------------------------------------------------------------------------
CURVATURE_POINTS = max(512, 64*COIL_ORDER)
COIL_POINT_ROWS = False  # --coil-point-rows
DISTANCE_POINTS = max(128, 16*COIL_ORDER)
SURFACE_GRID = (61, 64)
SELF_CLEARANCE = 1e-6  # numerical nonintersection guard, metres


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Biot-Savart field of filament coils, as a differentiable pytree.

    ESSOS 0.21.0's ``BiotSavart`` is a pytree, but its ``Coils`` carries arrays
    in the pytree metadata (its current normalization), so a jitted function of
    new coil values fails JAX's cache-key comparison: a second geometry under one
    ``jit`` raises, and its JVP through ``Coils(Curves(dofs), currents)`` differs
    from central differences by 1-12% (``diag/three_term/perlmutter/scripts/wpE/
    essos_parity.py``).  This pytree carries ``gamma``, ``gamma_dash`` and the
    currents as leaves: one compiled program serves every trial, its JVP matches
    central differences to 2e-8, and its values equal ESSOS' to 4e-16 (the same
    integrand: the mean over each coil's quadrature points of the filament
    Biot-Savart term).
    ``gamma``/``gamma_dash`` are ESSOS' points and tangents, shape
    ``(coils, points, 3)`` in metres; ``currents`` are in amperes.
    """

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r, phi, z):
        """Evaluate the filament field in cylindrical coordinates, in tesla.

        The sum runs over chunks of at most 512 quadrature points per coil (a scan whose body is recomputed in
        reverse mode), so memory is points x coils x chunk, not points x coils x quadrature points: 8192-point
        filaments on a 12 x 12 interface grid needed a 39.7 GB temporary.  Up to 512 points it is one chunk,
        the plain sum."""
        import jax
        import jax.numpy as jnp

        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        gamma, gamma_dash = jnp.asarray(self.gamma), jnp.asarray(self.gamma_dash)
        currents = jnp.asarray(self.currents)
        ncoil, nseg = gamma.shape[0], gamma.shape[1]
        chunk = nseg if nseg <= 512 else next(c for c in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if nseg % c == 0)

        def split(a):
            return jnp.swapaxes(a.reshape(ncoil, nseg // chunk, chunk, 3), 0, 1)

        @jax.checkpoint
        def body(total, pair):
            g, dg = pair
            displacement = xyz[..., None, None, :] - g
            radius2 = jnp.sum(displacement * displacement, axis=-1)
            inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
            differential = jnp.cross(dg, displacement) * inv_radius3[..., None]
            weighted = differential * jnp.reshape(currents, (1,) * (xyz.ndim - 1) + (-1, 1, 1))
            return total + jnp.sum(weighted, axis=(-3, -2)), None

        total, _ = jax.lax.scan(body, jnp.zeros(xyz.shape, dtype=xyz.dtype), (split(gamma), split(gamma_dash)))
        bxyz = 1.0e-7 * total / nseg
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


class CoilChart:
    """Design vector -> ESSOS coils and plasma parameters: the free arm's ``field_from_parameters``.

    ``x`` holds, in order: with ``phiedge`` (the deck's PHIEDGE in Wb) a relative
    PHIEDGE change, PHIEDGE = phiedge * (1 + x[0]); with ``plasma_current`` (the
    deck's leading ``k`` AC coefficients, or AC_AUX_F spline values when
    ``plasma_current_spline``, then CURTOR in A; ``ncurr = 1``) ``k + 1``
    additive changes of those values in units of the largest nominal shape value
    and of ``abs(CURTOR)``; relative changes of the selected base-coil currents
    (``current[i] = nominal[i] * (1 + x[k])``); and additive changes of every
    base coil's Cartesian Fourier coefficients in metres. ``x0 = 0`` is the
    nominal state. Calling the chart gives the :class:`DirectCoilField` of
    ``coils_from_x(x)``; ``plasma_from_parameters`` (``None`` without plasma
    coordinates) is ``FreeBoundaryProblem``'s map of the plasma parameters.
    ``scales`` default to 0.06 per relative current, 0.002 m per constant Fourier
    coefficient, ``0.002 / k**2`` m per coefficient of order ``k``, and
    ``phiedge_scale`` / ``plasma_current_scale`` for the plasma coordinates.
    """

    def __init__(self, coils, *, current_dofs, scales=None, phiedge=None, phiedge_scale=0.05, plasma_current=None,
                 plasma_current_spline=False, plasma_current_scale=0.05):
        import jax

        try:  # the field is a pytree of its arrays; registered here, after main() has configured JAX
            jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"],
                                             meta_fields=[])
        except ValueError:  # registered by an earlier chart
            pass
        self.coefficients = np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :]
        self.currents = np.asarray(coils.dofs_currents_raw, dtype=float)
        self.current_dofs = tuple(int(i) for i in current_dofs)
        if len(set(self.current_dofs)) != len(self.current_dofs) or any(
                not 0 <= i < len(self.currents) or self.currents[i] == 0 for i in self.current_dofs):
            raise ValueError("current_dofs must be unique indices of base coils with nonzero currents")
        self.nfp, self.stellsym, self.n_segments = int(coils.nfp), bool(coils.stellsym), int(coils.n_segments)
        if phiedge is not None and not (np.isfinite(phiedge) and phiedge != 0 and phiedge_scale > 0):
            raise ValueError("phiedge must be finite and nonzero with a positive phiedge_scale")
        self.phiedge = None if phiedge is None else float(phiedge)
        self.nphiedge = int(phiedge is not None)
        self.plasma_current = None if plasma_current is None else np.asarray(plasma_current, dtype=float)
        self.plasma_current_spline = bool(plasma_current_spline)
        self.nplasma = 0 if self.plasma_current is None else self.plasma_current.size
        if self.plasma_current is not None:
            nominal = self.plasma_current
            if nominal.ndim != 1 or nominal.size < 2 or not plasma_current_scale > 0:
                raise ValueError("plasma_current is [k >= 1 shape values..., CURTOR], with a positive scale")
            self.plasma_current_units = np.r_[np.full(nominal.size - 1, np.max(np.abs(nominal[:-1]))),
                                              abs(nominal[-1])]
            if not np.all(np.isfinite(self.plasma_current_units)) or np.any(self.plasma_current_units == 0):
                raise ValueError("plasma_current needs finite values, a nonzero shape value and a nonzero CURTOR")
        self._coil0 = self.nphiedge + self.nplasma
        self.size = self._coil0 + len(self.current_dofs) + self.coefficients.size
        self.x0 = np.zeros(self.size)
        order = (self.coefficients.shape[2] - 1) // 2
        modes = ["constant"] + [f"{kind}({k})" for k in range(1, order + 1) for kind in ("sin", "cos")]
        self.dof_names = tuple(["phiedge/nominal"] * self.nphiedge
                               + [f"plasma_current[{j}]/unit" for j in range(self.nplasma - 1)]
                               + ["curtor/nominal"] * bool(self.nplasma)
                               + [f"current[{i}]/nominal" for i in self.current_dofs]
                               + [f"coil[{i}].{axis}.{mode}" for i in range(len(self.currents))
                                  for axis in "xyz" for mode in modes])
        if scales is None:
            mode_scales = [0.002] + [0.002 / k**2 for k in range(1, order + 1) for _ in range(2)]
            scales = np.r_[np.full(self.nphiedge, phiedge_scale), np.full(self.nplasma, plasma_current_scale),
                           np.full(len(self.current_dofs), 0.06), np.tile(mode_scales, 3 * len(self.currents))]
        self.scales = np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("one positive finite scale per coordinate required")
        self.plasma_from_parameters = self.plasma_params_at if self._coil0 else None

    def phiedge_at(self, x):
        """PHIEDGE [Wb] at ``x``; only for a chart built with ``phiedge``."""
        import jax.numpy as jnp

        if self.phiedge is None:
            raise ValueError("this chart does not vary PHIEDGE")
        return self.phiedge * (1.0 + jnp.asarray(x)[0])

    def plasma_params_at(self, params, x):
        """``params`` (``ImplicitParams``) with the PHIEDGE and current profile of ``x``."""
        import jax.numpy as jnp

        x = jnp.asarray(x)
        if self.phiedge is not None:
            params = replace(params, phiedge=self.phiedge_at(x))
        if self.plasma_current is not None:
            values = jnp.asarray(self.plasma_current) + x[self.nphiedge:self._coil0] * self.plasma_current_units
            name = "ac_aux_f" if self.plasma_current_spline else "ac"
            profile = getattr(params, name).at[: self.nplasma - 1].set(values[:-1])
            params = replace(params, curtor=values[-1], **{name: profile})
        return params

    def base_currents_at(self, x):
        """Physical base-coil currents in amperes."""
        import jax.numpy as jnp

        x = jnp.asarray(x)
        currents = jnp.asarray(self.currents)
        for local, base in enumerate(self.current_dofs):
            currents = currents.at[base].add(x[self._coil0 + local] * self.currents[base])
        return currents

    def coils_from_x(self, x):
        """ESSOS coils at ``x``, without changing the nominal coils."""
        import jax.numpy as jnp
        from essos.coils import Coils, Curves

        x = jnp.asarray(x)
        raw = jnp.asarray(self.coefficients) + x[self._coil0 + len(self.current_dofs):].reshape(
            self.coefficients.shape)
        return Coils(Curves(raw, self.n_segments, self.nfp, self.stellsym), self.base_currents_at(x))

    def __call__(self, x):
        """The differentiable filament field of ``coils_from_x(x)``."""
        import jax.numpy as jnp

        coils = self.coils_from_x(x)
        return DirectCoilField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def resampled(coils, points, *, symmetric=True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; base curves only unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


def segment_distances(a, b):
    """All distances between two closed polygons' segments, including interiors."""
    import jax.numpy as jnp

    u, v = jnp.roll(a, -1, axis=0)-a, jnp.roll(b, -1, axis=0)-b
    w = a[:, None]-b[None, :]
    aa, bb = jnp.sum(u*u, axis=-1)[:, None], jnp.sum(v*v, axis=-1)[None, :]
    uv = jnp.einsum('ik,jk->ij', u, v)
    uw, vw = jnp.sum(u[:, None]*w, axis=-1), jnp.sum(v[None, :]*w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa*bb-uv*uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv*vw-bb*uw)/safe, (aa*vw-uv*uw)/safe
    def distance(s, t):
        q = w+s[..., None]*u[:, None]-t[..., None]*v[None, :]
        return jnp.sum(q*q, axis=-1)
    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw/bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw+uv)/bb, 0, 1)),
                  distance(jnp.clip(-uw/aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv-uw)/aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1), distance(s,t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0)+1e-30)


def separations(points):
    """Minimum intercoil and nonadjacent self-segment distance, all symmetry copies."""
    import jax
    import jax.numpy as jnp

    pairs = jnp.asarray([(i,j) for i in range(len(points)) for j in range(i+1,len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(segment_distances(points[ij[0]], points[ij[1]])), pairs)
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:,None]-jnp.arange(n)[None,:])
    nonadjacent = jnp.minimum(diff, n-diff)>1
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, segment_distances(p,p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils, *, curvature_points=CURVATURE_POINTS, distance_points=DISTANCE_POINTS):
    """ESSOS length, peak curvature and speed of each base coil, plus what ESSOS lacks."""
    import jax.numpy as jnp

    base = resampled(coils, curvature_points, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    cc, own = separations(resampled(coils, distance_points).gamma)
    return dict(length=base.length, peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2*speed, axis=1)/jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed, axis=1))


def coil_inequalities(coils):
    import jax.numpy as jnp

    m = coil_metrics(coils)
    return jnp.concatenate(((LENGTH_LIMIT-LENGTH_MARGIN-m['length'])/LENGTH_LIMIT,
        (CURVATURE_LIMIT-CURVATURE_MARGIN-m['peak'])/CURVATURE_LIMIT,
        (MSC_LIMIT-MSC_MARGIN-m['msc'])/MSC_LIMIT,
        jnp.atleast_1d((m['coil_distance']-COIL_DISTANCE_LIMIT-DISTANCE_MARGIN)/COIL_DISTANCE_LIMIT),
        jnp.atleast_1d((m['self_distance']-SELF_CLEARANCE)/COIL_DISTANCE_LIMIT),
        (m['min_speed']-1e-4)/LENGTH_LIMIT))


def coil_point_inequalities(coils):
    """:func:`coil_inequalities` with one row per sample point instead of each coil's extreme: the peak curvature,
    coil-coil and self distance become every base-coil point's (its curvature; its distance to the other coils, all
    symmetry copies; to its own non-adjacent segments).  Each row is smooth where the extreme switches points, and
    the smallest of them equals the extreme row."""
    import jax.numpy as jnp

    base = resampled(coils, CURVATURE_POINTS, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    nbase = curvature.shape[0]
    points = resampled(coils, DISTANCE_POINTS).gamma  # base coils first, then their symmetry copies
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    nonadjacent = jnp.minimum(diff, n - diff) > 1

    def per_point(b):
        others = jnp.stack([jnp.min(segment_distances(points[b], points[j]), axis=1)
                            for j in range(points.shape[0]) if j != b])
        own = jnp.min(jnp.where(nonadjacent, segment_distances(points[b], points[b]), jnp.inf), axis=1)
        return jnp.min(others, axis=0), own

    coil_distance, self_distance = map(jnp.concatenate, zip(*[per_point(b) for b in range(nbase)]))
    return jnp.concatenate(((LENGTH_LIMIT - LENGTH_MARGIN - base.length) / LENGTH_LIMIT,
                            jnp.ravel(CURVATURE_LIMIT - CURVATURE_MARGIN - curvature) / CURVATURE_LIMIT,
                            (MSC_LIMIT - MSC_MARGIN - jnp.sum(curvature**2 * speed, axis=1) / jnp.sum(speed, axis=1))
                            / MSC_LIMIT,
                            (coil_distance - COIL_DISTANCE_LIMIT - DISTANCE_MARGIN) / COIL_DISTANCE_LIMIT,
                            (self_distance - SELF_CLEARANCE) / COIL_DISTANCE_LIMIT,
                            (jnp.min(speed, axis=1) - 1e-4) / LENGTH_LIMIT))


def chunked_jacfwd(fun, chunk=32):
    """Forward-mode Jacobian of ``fun`` with ``chunk`` input directions at a time: memory of the point rows' distance
    kernels grows with the directions pushed at once (35 GiB for all of a QI design)."""
    import jax
    import jax.numpy as jnp

    def jac(x):
        n = x.size
        basis = jnp.eye(-(-n // chunk) * chunk, n, dtype=x.dtype).reshape(-1, chunk, n)
        columns = jax.lax.map(jax.vmap(lambda v: jax.jvp(fun, (x,), (v,))[1]), basis)
        return columns.reshape(-1, columns.shape[-1])[:n].T
    return jac


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    import jax
    import jax.numpy as jnp

    points = resampled(coils, DISTANCE_POINTS).gamma
    targets = surface.gamma.reshape(-1,3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p-targets)**2,axis=1))+1e-30),
                               points.reshape(-1,3)))


def continuous_surface_distance(points, rbc, zbs, nfp, *, grid=(48, 48), iterations=10):
    """Coil-to-plasma clearance between the continuous coil and the continuous surface (E3), and the runner-up gap.

    ``points`` (curves, samples, 3) are coil samples, uniform in the parameter of each closed curve; the curve is
    their periodic (trigonometric) interpolant, exact for a Fourier curve sampled above its Nyquist rate.
    ``rbc``/``zbs`` ``(2 ntor + 1, mpol)`` are the boundary, ``R + i Z = sum (rbc cos + i zbs sin)(m theta - n nfp
    phi)``.  A grid search over the whole torus finds each sample's nearest surface point; from the seven best
    samples, Newton on ``(s, theta, phi)`` (coil parameter and surface angles) refines the closest pair of the
    continuous curve and surface, and the smallest of the seven is the clearance.  The pair is held fixed for
    differentiation (the envelope theorem).  The value is a minimum of local minima, continuous in the coils and the
    boundary; it has a kink, but no jump, where the closest pair switches (the parabola through three samples used
    before jumped by about 1e-5 when the closest sample changed).  Returns ``(clearance, gap)``: ``gap`` is how much
    farther the nearest sample more than two samples away (on any coil) is, so a small ``gap`` flags a closest-pair
    switch nearby.
    """
    import jax
    import jax.numpy as jnp

    ntor = (rbc.shape[0] - 1) // 2
    m = jnp.arange(rbc.shape[1])
    n = (jnp.arange(rbc.shape[0]) - ntor) * nfp

    def surface(theta, phi):
        angle = m[None, :] * theta - n[:, None] * phi
        R = jnp.sum(rbc * jnp.cos(angle))
        Z = jnp.sum(zbs * jnp.sin(angle))
        return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z])

    th = jnp.linspace(0, 2 * jnp.pi, grid[0], endpoint=False)
    ph = jnp.linspace(0, 2 * jnp.pi, grid[1] * nfp, endpoint=False)
    TH, PH = jnp.meshgrid(th, ph, indexing="ij")
    S = jax.vmap(jax.vmap(surface))(TH, PH).reshape(-1, 3)
    pts = jnp.asarray(points)
    ncurve, nsample = pts.shape[0], pts.shape[1]
    flat = pts.reshape(-1, 3)
    d2 = jax.lax.map(lambda q: jnp.min(jnp.sum((q - S) ** 2, axis=1)), flat)
    nearest = jax.lax.map(lambda q: jnp.argmin(jnp.sum((q - S) ** 2, axis=1)), jax.lax.stop_gradient(flat))
    order = jnp.argsort(jax.lax.stop_gradient(d2))[:7]  # the coarse search's best candidates, refined below
    coef = jnp.fft.fft(pts, axis=1) / nsample  # (curves, wavenumbers, 3): the trigonometric interpolant
    wave = jnp.fft.fftfreq(nsample, 1.0 / nsample)
    if nsample % 2 == 0:  # the Nyquist term split evenly between +-N/2 keeps the interpolant real and smooth
        coef = coef.at[:, nsample // 2].multiply(0.5)
        coef = jnp.concatenate([coef, coef[:, nsample // 2:nsample // 2 + 1]], axis=1)
        wave = jnp.concatenate([wave, -wave[nsample // 2:nsample // 2 + 1]])

    def curve(c, u, coef=coef):
        """Point of curve ``c`` at parameter ``u`` (sample units: u = i is sample i)."""
        return jnp.real(jnp.sum(coef[c] * jnp.exp(2j * jnp.pi * wave * u / nsample)[:, None], axis=0))

    def pair(j):
        """Distance between coil ``c`` and the surface, the closest pair refined by Newton from sample ``j`` and
        its nearest grid point (held fixed for AD)."""
        c = j // nsample
        z0 = jnp.array([(j % nsample).astype(float), TH.ravel()[nearest[j]], PH.ravel()[nearest[j]]])
        fixed = jax.lax.stop_gradient(coef)
        half = lambda z: 0.5 * jnp.sum((curve(c, z[0], fixed) - surface(z[1], z[2])) ** 2)  # noqa: E731
        z = jax.lax.stop_gradient(z0)
        for _ in range(iterations):
            # Newton on |C - S|^2 / 2 (Gauss-Newton fails when the distance is comparable to a radius of curvature);
            # a gradient step where the Hessian is not positive definite
            g, H = jax.grad(half)(z), jax.hessian(half)(z)
            newton = -jnp.linalg.solve(H + 1e-14 * jnp.eye(3), g)
            ok = (jnp.linalg.eigvalsh(H)[0] > 0) & jnp.all(jnp.isfinite(newton))
            z = z + jnp.where(ok, newton, -g / jnp.maximum(jnp.linalg.norm(H), 1e-300))
        z = jax.lax.stop_gradient(z)
        return jnp.sqrt(jnp.sum((curve(c, z[0]) - surface(z[1], z[2])) ** 2) + 1e-30)

    cand = jnp.stack([pair(j) for j in order])
    best = jnp.argmin(jax.lax.stop_gradient(cand))
    clearance = cand[best]
    k = order[best]
    c, i = k // nsample, k % nsample
    sample = jnp.arange(ncurve * nsample)
    near = (sample // nsample == c) & (jnp.abs(((sample % nsample) - i + nsample // 2) % nsample - nsample // 2) <= 2)
    second = jnp.sqrt(jnp.min(jnp.where(near, jnp.inf, jax.lax.stop_gradient(d2))))
    return clearance, second - jax.lax.stop_gradient(clearance)


def coil_constraint(coils_from_x):
    """Pure-coil rows require no equilibrium solves or adjoint right-hand sides."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import NonlinearConstraint
    rows = coil_point_inequalities if COIL_POINT_ROWS else coil_inequalities
    fun = jax.jit(lambda x: rows(coils_from_x(x)))
    jac = jax.jit(chunked_jacfwd(fun) if COIL_POINT_ROWS else jax.jacrev(fun))  # thousands of point rows: forward
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))


# ---- the optimization ------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--current-iota-weight", type=float, help="w of the loss term w/2 iota_I^2 "
                        "(default: CURRENT_IOTA_WEIGHT, 1e-2 for the -jbs cases)")
    parser.add_argument("--vacuum-iota-floor", type=float, help="floor on the boundary's vacuum |iota| "
                        "(default: VACUUM_IOTA_FLOOR, 0.23 for qa4-beta)")
    parser.add_argument("--iota-ceiling", type=float, help="ceiling on max |iota| (default: IOTA_CEILING)")
    parser.add_argument("--check-jacobian", type=int, default=0, metavar="N",
                        help="compare the model's implicit boundary Jacobian with central differences of tight "
                        "rows along N random directions and the weakest one (jacobian_check.json), at the seed "
                        "boundary or --jacobian-at, without solving the free boundary; then stop")
    parser.add_argument("--jacobian-following", action="store_true",
                        help="--check-jacobian along boundary-only directions with the bootstrap current re-solved to "
                        "conditional stationarity at every point (the response the variable projection follows), "
                        "against the reduced model (I - P_c) J_b d and against the fixed-current difference")
    parser.add_argument("--jacobian-dump", action="store_true",
                        help="--check-jacobian also saves every direction's J d and differences (jacobian_check.npz)")
    parser.add_argument("--jacobian-at", type=Path, metavar="RECORD",
                        help="a certificate record (.npz, fbrecord) whose model coordinates --check-jacobian uses")
    parser.add_argument("--seed-start", nargs="+", metavar="KIND",
                        help="fit the seed's boundary from another start (start-dependence qualification): "
                        "'random EPS SEED' or 'scale EPS'; the deck stays the reference")
    parser.add_argument("--polish-rounds", type=int, help="certificate polish rounds per trial (default: the model's, "
                        "freeboundary_vc.POLISH_ROUNDS); a finite-beta seed can need about 100")
    parser.add_argument("--polish-ftol", type=float, help="the polish target (default freeboundary_vc.POLISH_FTOL); "
                        "derivative qualification needs about 1e-5")
    parser.add_argument("--clearance", choices=("continuous", "points"), default="continuous",
                        help="coil-plasma clearance: the continuous closest point (default; smooth where the closest "
                        "pair is unique) or the minimum over sampled coil and surface points (non-smooth at every closest-pair switch)")
    parser.add_argument("--newton-polish", type=float, default=1e-8, metavar="STEP_TOL",
                        help="refine every certified fit by Newton steps on its cost with the exact Hessian until the "
                        "scaled step is at most STEP_TOL (or the cost's floor), so the fit's stationary point and its "
                        "exact design gradient are determined (ThreeTermFreeBoundaryProblem._newton_refine)")
    parser.add_argument("--check-exact-products", action="store_true",
                        help="compare the second-order adjoint products H v, F_d^T v (ThreeTermFreeBoundaryProblem."
                        "exact_products) with Richardson difference products (step --difference-h) along the weakest, "
                        "strongest and a random direction, with timings (exact_products_check.json), then stop")
    parser.add_argument("--exact-products-timing", type=int, nargs="*", default=[], metavar="M",
                        help="with --check-exact-products: also time compiled calls of M random columns")
    parser.add_argument("--newton-track", action="store_true",
                        help="record the exact gradient at every Newton iterate")
    parser.add_argument("--newton-max-steps", type=int, default=60, help="Newton polish steps at most")
    parser.add_argument("--newton-quasi", action=argparse.BooleanOptionalAction, default=True,
                        help="Newton polish: the first step with the exact Hessian, then damped BFGS updates from the "
                        "exact gradient (one tight solve per step), refreshed exactly when stalled or every 10 steps; "
                        "the record's design gradient still uses the exact Hessian (--no-newton-quasi: exact every step)")
    parser.add_argument("--design-trust", type=float, metavar="R",
                        help="reject (before solving) any trial design farther than R (Euclidean, design parameters) "
                        "from the accepted one, so the optimizer's line search shortens the step (default: no limit)")
    parser.add_argument("--newton-response", type=float, nargs=2, metavar=("VTOL", "GTOL"),
                        help="the Krylov Newton polish also converges once the values change by at most VTOL on two "
                        "iterates and two exact design gradients agree to GTOL (relative)")
    parser.add_argument("--trial-pin", type=float, metavar="W",
                        help="two-stage trials: a fit with the boundary's flat (relabelling) directions pinned at the "
                        "accepted point's (pin curvature W times the Hessian's largest eigenvalue), then the exact "
                        "Newton polish of the unpinned cost, which alone defines the trial (default: polish from the "
                        "accepted boundary)")
    parser.add_argument("--newton-warm", action="store_true",
                        help="trials start Newton from the accepted point's converged boundary with its kept Hessian "
                        "(no fit; falls back to the fit when Newton does not converge)")
    parser.add_argument("--check-rows-c2", action="store_true",
                        help="at the accepted state, the second derivative of |rows|^2/2 (interface and bootstrap rows "
                        "separately) along a random state direction: autodiff vs central differences of its exact "
                        "gradient (rows_c2_check.json), then stop")
    parser.add_argument("--check-soft-modes", type=int, default=0, metavar="K",
                        help="at the accepted point, the exact Hessian's K softest directions split into normal (shape) "
                        "and tangential (relabelling) surface displacement (soft_modes.json), then stop")
    parser.add_argument("--check-horizontal-gradient", type=int, nargs="+", metavar="K",
                        help="at the accepted point, the exact design gradient with the exact Hessian's K softest "
                        "directions projected out of mu = H^-1 Q_b^T, for each K (the first is the reference; "
                        "horizontal_gradient.json), then stop")
    parser.add_argument("--check-labelling-sensitivity", type=int, default=0, metavar="K",
                        help="at the accepted point, move the boundary along the exact Hessian's K softest directions "
                        "(scaled steps 0.03..3) and record the cost and every value (labelling_sensitivity.json), then stop")
    parser.add_argument("--check-cost-gradient", action="store_true",
                        help="at the accepted boundary, central differences of the fit's cost along the descent and a "
                        "random direction (steps 1e-2..1e-8, scaled) against g.d (cost_gradient_check.json), then stop")
    parser.add_argument("--check-g-noise", action="store_true",
                        help="the fit's stationarity g at the accepted boundary from three equilibrium starts (hot, "
                        "cold, the record's own state), and the scaled Newton step their differences imply "
                        "(g_noise_check.json), then stop")
    parser.add_argument("--check-stationarity-gradient", action="store_true",
                        help="every row's exact derivative of the converged stationarity at the seed, with the "
                        "Gauss-Newton one (ThreeTermFreeBoundaryProblem.stationarity_design_gradient; "
                        "stationarity_gradient_check.json, with the reverse-pass vs linearization consistency check), "
                        "then stop")
    parser.add_argument("--difference-h", type=float, default=1e-3,
                        help="central-difference step (scaled coordinates) of --check-exact-products' reference")
    parser.add_argument("--check-gradient-cold", action="store_true",
                        help="--check-gradient trials start from the seed's boundary, not the Gauss-Newton prediction "
                        "(a predicted start that already certifies makes the difference reproduce the formula)")
    parser.add_argument("--check-gradient-steps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4], metavar="H",
                        help="central-difference steps of --check-gradient")
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
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="save the accepted point to OUTPUT/checkpoint.npz every N accepted steps (0: never)")
    parser.add_argument("--resume", type=Path, metavar="CHECKPOINT",
                        help="continue from a checkpoint.npz instead of solving the seed design: the problem must be "
                        "built the same way (e.g. --seed-run with the checkpoint's run), or this stops; the "
                        "accepted equilibrium is restored and certified; SLSQP continues with its Hessian when "
                        "slsqp_state.npz beside the checkpoint is of that point, else it starts afresh from it")
    parser.add_argument("--ns", type=int, help="radial resolution (default: RESOLUTION)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"), help="default: RESOLUTION")
    parser.add_argument("--boundary-modes", type=int, metavar="K",
                        help="boundary coefficients up to m, |n| <= K (default: the interior's --modes); an interior "
                        "above the boundary's resolution keeps its truncation out of the optimized boundary")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bootstrap", action="store_true")
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
    parser.add_argument("--gauge", choices=("condensation", "equal_arc"), default=None,
                        help="hold the boundary's poloidal angle exactly by VMEC's spectral condensation (a gauge, not "
                        "penalty rows: every Newton iterate is projected onto it); use with --label-weight 0")
    parser.add_argument("--labelling", choices=("start", "equal_arc"), default="start",
                        help="the boundary's poloidal labelling rows: anchored to the start boundary (start) or "
                        "equal arc length of the current surface (equal_arc); see ThreeTermFreeBoundaryModel")
    parser.add_argument("--m1-complement", choices=("solved", "frozen"), default="solved",
                        help="the interior's m = 1 complement: solved with the evolved dofs (unique root, default) "
                        "or VMEC2000's released (frozen) complement")
    parser.add_argument("--label-weight", type=float,
                        help="weight of the labelling rows (default: the model's, 1e-2; 0 drops them)")
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

    def coils_from_u(u):
        return coils.with_dofs(jnp.concatenate((x0 + COIL_STEP * u, coils.dofs_currents)))

    def normal_field_rms(new):
        return weighted_rms(interface.weights, total_normal_field(interface, coil_field(new)))

    def objective(u):
        new = coils_from_u(u)
        rows = jnp.concatenate([coil_inequalities(new), jnp.atleast_1d(
            (surface_distance(new, surface) - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN)
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


def check_jacobian(inp, options, chart, directions, out, record=None, steps=(1e-2, 1e-3, 1e-4), dump=False,
                   following=False):
    """The model's implicit boundary Jacobian against central differences of tight rows (WP5/C5).

    At the seed design's field and plasma parameters and the seed boundary (or ``record``'s model coordinates),
    without solving the free boundary.  Directions: ``directions`` random unit vectors in the scaled model
    coordinates, then the weakest right singular vector of the column-scaled Jacobian.  For each and each step:
    ``|D - J d| / |J d|`` over all rows and by row block, with ``D`` the central difference.  Each perturbed
    interior root is solved tight (certified).  Written to ``jacobian_check.json``."""
    import jax.numpy as jnp

    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel

    model_options = {k: v for k, v in options.items()
                     if k not in ("boundary_ftol", "boundary_max_nfev", "boundary_max_residual", "boundary_xtol",
                                  "checkpoint", "seed_start", "semantic_identity")}
    model = ThreeTermFreeBoundaryModel(inp, **model_options)
    x_design = jnp.asarray(chart.x0)
    params, field = chart.plasma_params_at(model.params0, x_design), chart(x_design)
    b = np.asarray(model.x0, dtype=float)
    if record is not None:
        b = np.asarray(np.load(record, allow_pickle=False)["x"], dtype=float)
    rows, aux = model.evaluate(b, params, field, tight=True)
    J = np.asarray(model.linearize(b, aux, field)[0])[:, :b.size]
    scale = np.asarray(model.x_scale, dtype=float)
    if following:
        return _check_jacobian_following(model, params, field, b, rows, aux, J, scale, directions, out, steps, record)
    Js = J * scale
    _, sv, vt = np.linalg.svd(Js / np.linalg.norm(Js, axis=0), full_matrices=False)
    rng = np.random.default_rng(0)
    dirs = [rng.standard_normal(b.size) for _ in range(directions)]
    dirs.append(vt[-1] / np.linalg.norm(Js, axis=0))
    records, arrays = [], {}
    for k, d in enumerate(dirs):
        d = d / np.linalg.norm(d) * scale
        Jd = J @ d
        arrays[f"Jd_{k}"] = Jd
        for h in steps:
            pair = [model.evaluate(b + sign * h * d, params, field, tight=True) for sign in (1, -1)]
            if any(p is None for p in pair):
                records.append(dict(direction=k, h=h, failed=True))
                continue
            diff = (np.asarray(pair[0][0]) - np.asarray(pair[1][0])) / (2 * h)
            arrays[f"diff_{k}_{h:.0e}"] = diff
            row = dict(direction=k, kind="weakest" if k == directions else "random", h=h,
                       relative=float(np.linalg.norm(diff - Jd) / np.linalg.norm(Jd)))
            for block, b0, b1 in model.row_blocks:
                nrm = float(np.linalg.norm(Jd[b0:b1]))
                if nrm > 0:
                    row[block] = float(np.linalg.norm(diff[b0:b1] - Jd[b0:b1]) / nrm)
            records.append(row)
            print(f"[jacobian check] direction {k} ({row['kind']}) h={h:.0e}: relative {row['relative']:.2e}",
                  flush=True)
    meta = dict(resolution=model.resolution(), singular_values=[float(sv[0]), float(sv[-1])],
                record=None if record is None else str(record), options=model.options)
    (out / "jacobian_check.json").write_text(json.dumps(dict(meta=meta, checks=records), indent=1, default=str) + "\n")
    if dump:
        blocks = {name: np.array([b0, b1]) for name, b0, b1 in model.row_blocks}
        np.savez(out / "jacobian_check.npz", rows=np.asarray(rows), **arrays, **{f"block_{k}": v for k, v in blocks.items()})


def _check_jacobian_following(model, params, field, b, rows0, aux0, J, scale, directions, out, steps, record):
    """--jacobian-following: boundary-only directions, the current re-solved to conditional stationarity."""
    nb = int(model.n_boundary)
    cur = np.arange(nb, b.size)
    J_b, J_c = J[:, :nb], J[:, nb:]
    Q, _ = np.linalg.qr(J_c)
    reduced = J_b - Q @ (Q.T @ J_b)  # Kaufman: the response with the current following, to first order

    def follow(bb):
        """``bb`` with its current coordinates at conditional stationarity (Gauss-Newton, fresh columns)."""
        got = model.evaluate(bb, params, field, tight=True)
        for _ in range(20):
            if got is None:
                return None, None
            r, aux = got
            Jc, aux = model.current_columns(bb, aux, field)
            Qc, Rc = np.linalg.qr(np.asarray(Jc))
            g = Qc.T @ np.asarray(r)
            if g @ g <= 1e-14 * (np.asarray(r) @ np.asarray(r)):
                return bb, np.asarray(r)
            bb = bb.copy()
            bb[cur] -= np.linalg.solve(Rc, g)
            got = model.evaluate(bb, params, field, tight=True)
        return bb, None

    b0, r0 = follow(b)
    rs = reduced * scale[:nb]
    _, sv, vt = np.linalg.svd(rs / np.linalg.norm(rs, axis=0), full_matrices=False)
    rng = np.random.default_rng(0)
    dirs = [rng.standard_normal(nb) for _ in range(directions)] + [vt[-1] / np.linalg.norm(rs, axis=0)]
    records = []
    for k, d in enumerate(dirs):
        d = d / np.linalg.norm(d) * scale[:nb]
        full = np.zeros(b.size)
        full[:nb] = d
        Rd, Jd = reduced @ d, J_b @ d
        for h in steps:
            plus, minus = follow(b0 + h * full), follow(b0 - h * full)
            fixed = [model.evaluate(b0 + s * h * full, params, field, tight=True) for s in (1, -1)]
            row = dict(direction=k, kind="weakest" if k == directions else "random", h=h)
            if plus[1] is not None and minus[1] is not None:
                diff = (plus[1] - minus[1]) / (2 * h)
                row["following"] = float(np.linalg.norm(diff - Rd) / np.linalg.norm(Rd))
                for block, b0_, b1_ in model.row_blocks:
                    nrm = float(np.linalg.norm(Rd[b0_:b1_]))
                    if nrm > 0:
                        row["following_" + block] = float(np.linalg.norm(diff[b0_:b1_] - Rd[b0_:b1_]) / nrm)
            if all(f is not None for f in fixed):
                diff = (np.asarray(fixed[0][0]) - np.asarray(fixed[1][0])) / (2 * h)
                row["fixed_current"] = float(np.linalg.norm(diff - Jd) / np.linalg.norm(Jd))
            records.append(row)
            print(f"[jacobian following] direction {k} ({row['kind']}) h={h:.0e}: following "
                  f"{row.get('following', float('nan')):.2e}, fixed current {row.get('fixed_current', float('nan')):.2e}",
                  flush=True)
    meta = dict(resolution=model.resolution(), record=None if record is None else str(record), options=model.options,
                base_closure_converged=r0 is not None)
    (out / "jacobian_following.json").write_text(json.dumps(dict(meta=meta, checks=records), indent=1, default=str)
                                                 + "\n")


def check_gradient(problem, directions, out, steps=(1e-2, 1e-3, 1e-4), cold=False):
    """Implicit design gradients of the loss and every quantity at the seed against central differences of
    re-solved trials, along ``directions`` random unit directions in the optimizer's coordinates, at each step in
    ``steps``; the relative difference ``|implicit - difference| / |difference|`` is written to
    ``gradient_check.json`` (the loss on its own, the quantities as one vector).

    By default a trial starts from the accepted point's first-order (Gauss-Newton) boundary prediction; a trial whose
    prediction already passes the certificate is accepted without a step, and its difference then reproduces the
    Gauss-Newton formula by construction (not an independent check). Every trial's fit evaluations and accepted steps
    are recorded (``trials``) so that can be seen. ``cold`` starts every trial from the seed's own boundary instead,
    so each fit has to move by the whole response: the difference is then the solver's actual response."""
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
            trials = []
            for s in (1, -1):
                key = np.asarray(problem._x(x + s * h * d), dtype=float)
                if cold:  # from the seed's boundary, without the Gauss-Newton prediction or its Jacobian
                    problem._solve(key, starts=[np.asarray(problem._anchor["b"])])
                sol = problem._cache.get(key.tobytes()) if hasattr(problem, "_cache") else None
                trials.append(None if sol is None else dict(nfev=sol.get("nfev"), accepted=sol.get("accepted"),
                                                           reason=sol.get("reason")))
            values = [np.r_[problem.fun(x + s * h * d), problem.constraint_values(x + s * h * d)] for s in (1, -1)]
            for s, t in zip((1, -1), trials):
                if t is None:
                    key = np.asarray(problem._x(x + s * h * d), dtype=float).tobytes()
                    sol = problem._cache.get(key) if hasattr(problem, "_cache") else None
                    trials[(1 - s) // 2] = None if sol is None else dict(
                        nfev=sol.get("nfev"), accepted=sol.get("accepted"), reason=sol.get("reason"))
            fd = (values[0] - values[1]) / (2 * h)
            loss = abs(implicit[0] - fd[0]) / max(abs(fd[0]), 1e-300)
            rows = np.linalg.norm(implicit[1:] - fd[1:]) / max(np.linalg.norm(fd[1:]), 1e-300)
            records.append(dict(direction=k, h=h, loss_implicit=float(implicit[0]), loss_difference=float(fd[0]),
                                loss_relative=float(loss), rows_relative=float(rows),
                                rows_implicit=implicit[1:].tolist(), rows_difference=fd[1:].tolist(),
                                cold=bool(cold), trials=trials))
            moved = [t["accepted"] for t in trials if t is not None]
            print(f"[gradient check] direction {k} h={h:.0e}: loss {implicit[0]:.6e} vs {fd[0]:.6e} "
                  f"(relative {loss:.1e}), quantities relative {rows:.1e}"
                  f"{' (cold)' if cold else ''} trial steps {moved}", flush=True)
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
    for name in ("seed_run", "restart", "coils", "resume"):
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
    args = parse_args(argv)
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
    mpol, ntor, ns = RESOLUTION
    inp, redl = seed_input(), None
    if args.max_iterations:  # before the seed is built: its beta ramp and current blend solve with the same cap
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    if args.restart is not None:
        if args.bootstrap:  # kinetic profiles exist only for bootstrap runs
            pin_profiles(args.restart)
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.seed_run is not None:
        if args.bootstrap:  # kinetic profiles exist only for bootstrap runs
            pin_profiles(args.seed_run)
        inp = replace(vj.VmecInput.from_file(str(args.seed_run / "input.run")), lfreeb=False)
        if (int(inp.mpol), int(inp.ntor)) != tuple(RESOLUTION[:2]):  # --modes applies to a seed run too
            inp = inp.change_resolution(mpol=RESOLUTION[0], ntor=RESOLUTION[1], ntheta=GRID[0], nzeta=GRID[1])
        if int(np.atleast_1d(inp.ns_array)[-1]) != int(RESOLUTION[2]):
            inp = replace(inp, ns_array=np.array([int(RESOLUTION[2])]),
                          ftol_array=np.atleast_1d(inp.ftol_array)[-1:], niter_array=np.atleast_1d(inp.niter_array)[-1:])
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
    if args.bootstrap:
        save_profiles(redl_profiles(inp)[0], out / "profiles.json")

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
    chart = CoilChart(coils0, current_dofs=current_dofs, scales=scales,
                      phiedge=float(inp.phiedge) if FREE_PHIEDGE else None,
                      plasma_current=current, plasma_current_spline=True)
    qs = target_residual()
    mirror = (opt.mirror_ratio,) if MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if IOTA_CEILING else ()
    vacuum = vacuum_iota(inp)
    vacuum_floor = ((lambda state, runtime: jnp.min(vacuum(state, runtime))),) if VACUUM_IOTA_FLOOR else ()

    def boundary_arrays(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)  # private: no public traced LCFS of a state
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return rbc, zbs

    def boundary_surface(state, runtime):
        rbc, zbs = boundary_arrays(state, runtime)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])

    ext_rows, ext_terms = load_extension(args.extension)

    def extension_cost(state, runtime):
        return sum((0.5 * w * jnp.sum(jnp.ravel(f(state, runtime)) ** 2) for f, w in ext_terms.values()), 0.0)

    def loss(state, runtime, x):
        residuals = qs.residuals_state(state, runtime)
        current = 0.5 * CURRENT_IOTA_WEIGHT * current_iota(state, runtime) ** 2 if CURRENT_IOTA_WEIGHT else 0.0
        return 0.5 * jnp.vdot(residuals, residuals) + current + extension_cost(state, runtime)

    def clearance(state, runtime, x):
        if args.clearance == "points":
            return surface_distance(chart.coils_from_x(x), boundary_surface(state, runtime))
        points = resampled(chart.coils_from_x(x), DISTANCE_POINTS).gamma
        return continuous_surface_distance(points, *boundary_arrays(state, runtime), int(inp.nfp))[0]

    def aspect(state, runtime, x):
        return opt.aspect_ratio(state, runtime)

    from vmex.core import freeboundary_vc as fvc

    if args.polish_rounds is not None:
        fvc.POLISH_ROUNDS = int(args.polish_rounds)
    if args.polish_ftol is not None:
        fvc.POLISH_FTOL = float(args.polish_ftol)
    tt_options = dict(nphi=args.vc_grid, ntheta=args.vc_grid, chunk=args.chunk,
                      quadrature=args.quadrature or (4 * int(inp.nfp) * args.vc_grid, 2 * args.vc_grid),
                      trial_ftol=args.trial_forward_ftol, bootstrap=profiles,
                      bootstrap_helicity=REDL_HELICITY, boundary_ftol=args.trial_ftol,
                      boundary_max_residual=args.max_boundary_residual, labelling=args.labelling, gauge=args.gauge,
                      max_mode=args.boundary_modes,
                      checkpoint=args.resume, m1_complement=args.m1_complement,
                      seed_start=None if args.seed_start is None else (args.seed_start[0], *map(float, args.seed_start[1:])),
                      newton_polish=None if args.newton_polish is None else dict(
                          step_tol=args.newton_polish, track=args.newton_track, verbose=args.newton_track,
                          warm=args.newton_warm, quasi=args.newton_quasi, max_steps=args.newton_max_steps,
                          pin=args.trial_pin, response=args.newton_response),
                      exact_gradient=args.newton_polish is not None,
                      **({} if args.label_weight is None else {"label_weight": args.label_weight}))
    if args.check_jacobian:
        check_jacobian(inp, tt_options, chart, args.check_jacobian, out, args.jacobian_at, dump=args.jacobian_dump,
                       following=args.jacobian_following)
        return
    problem = opt.FreeBoundaryProblem.from_loss(
        inp, loss, chart.x0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_params_at,
        scales=chart.scales, names=chart.dof_names,
        quantities=(min_abs_iota, opt.major_radius, opt.axis_field_strength, *mirror, *ceiling, *vacuum_floor,
                    *([bootstrap_mismatch(inp, redl, "gpu")] if redl is not None else []),
                    *([min_d_merc] if MERCIER_FLOOR is not None else []), *([current_iota] if CURRENT_IOTA_WEIGHT else []),
                    *(f for f, *_ in ext_rows.values())),
        parameter_quantities=(clearance, aspect), boundary_condition="three_term",
        three_term_options=tt_options)
    problem.design_trust = args.design_trust
    model = problem.model
    print(f"[model] {model.n_boundary} boundary and {model.x0.size - model.n_boundary} bootstrap-current coordinates, "
          f"{chart.size} design coordinates; built in {time.monotonic() - started:.0f} s", flush=True)
    if args.resume and args.newton_polish is not None:  # the restored record is exactly the saved one: refine it
        t_refine = time.monotonic()
        change = problem.refine_accepted()
        if change != change:  # NaN: not converged, the restored record kept
            print(f"[newton polish] restored accepted point kept as saved: {problem.stats.get('refine_unconverged')}",
                  flush=True)
        nw = problem._record(problem.accepted.parameters).get("newton", {})
        print(f"[newton polish] restored accepted point refined: values change {change:.2e}; "
              f"{len(nw.get('steps', []))} steps, converged {nw.get('converged')}, {nw.get('evaluations')} "
              f"evaluations, Hessian {nw.get('seconds_hessian', 0):.0f} s, {time.monotonic() - t_refine:.0f} s; "
              "scaled steps " + " ".join(f"{st['step']:.1e}" for st in nw.get("steps", []))
              + "; costs " + " ".join(f"{st.get('cost', np.nan):.6e}" for st in nw.get("steps", [])), flush=True)
        steps = nw.get("steps", [])
        if steps and "gradients" in steps[-1]:
            last = np.array(steps[-1]["gradients"])
            for k, st in enumerate(steps):
                G = np.array(st["gradients"])
                drift = np.linalg.norm(G - last, axis=1) / np.maximum(np.linalg.norm(last, axis=1), 1e-300)
                print(f"[newton polish] iterate {k}: step {st['step']:.1e}, cost {st['cost']:.9e}, exact gradient vs "
                      f"last iterate max {drift.max():.1e} (rows " + " ".join(f"{e:.0e}" for e in drift) + ")",
                      flush=True)

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
        *([(VACUUM_IOTA_FLOOR + IOTA_MARGIN, np.inf, VACUUM_IOTA_FLOOR)] if vacuum_floor else []),
        *[(-np.inf, REDL_TOLERANCE, REDL_TOLERANCE)] * nredl,
        *([(MERCIER_FLOOR + MERCIER_MARGIN, np.inf, 1.0)] if MERCIER_FLOOR is not None else []),
        *([(-np.inf, np.inf, 1.0)] if CURRENT_IOTA_WEIGHT else []),  # logged; the loss carries it
        *extension_bounds,
        (COIL_SURFACE_DISTANCE_LIMIT + DISTANCE_MARGIN, np.inf, COIL_SURFACE_DISTANCE_LIMIT),
        (aspect_lower, aspect_upper, 0.5 * (aspect_upper - aspect_lower)))
    coil_rows = coil_constraint(chart.coils_from_x)
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
            vac = np.asarray(vacuum(state_x, runtime_x))
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
                   **({"redl_mismatch": float(h[3 + len(mirror) + len(ceiling) + len(vacuum_floor)])} if nredl else {}),
                   **({"redl_max_relative": model.bootstrap_residual(state_x, params_x)} if fold else {}),
                   coil_surface_distance_m=float(h[-2]), aspect=float(h[-1]),
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   phiedge_factor=float(chart.phiedge_at(jnp.asarray(x)) / chart.phiedge) if FREE_PHIEDGE else 1.0,
                   bn=res.normal, pressure_balance=res.pressure, sheet_current=res.sheet_current,
                   label_speed_ratio=label_speed_ratio(params_x), boundary_jacobian_cond=boundary_jacobian_condition(),
                   step_seconds=now - last["time"], elapsed_seconds=now - started, **problem.stats,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        phases = {k: problem.stats.get(f"seconds_{k}", 0.0) for k in ("fit", "certify", "gradient")}
        row.update({f"step_seconds_{k}": v - last.get(k, 0.0) for k, v in phases.items()})
        last.update(time=now, **phases)
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
              + f"{row['step_seconds']:.1f}s (fit {row['step_seconds_fit']:.1f}s, "
              f"certify {row['step_seconds_certify']:.1f}s, gradient {row['step_seconds_gradient']:.1f}s; "
              + (f"exact gradient residual {row['exact_gradient_true_residual']:.1e}; "
                 if "exact_gradient_true_residual" in row else "")
              + (f"warm Newton trials {row.get('warm_newton_trials', 0)} (unconverged "
                 f"{row.get('warm_newton_unconverged', 0)}, failed {row.get('warm_newton_failed', 0)}), "
                 if "warm_newton_trials" in row or "warm_newton_failed" in row else "")
              + f"trials {row['trials']}, boundary evaluations {row['boundary_evaluations']}, "
              f"peak {row['peak_gpu_gib']:.1f} GiB)", flush=True)
        if row.get("last_rejection") and row["last_rejection"] != last.get("rejection"):
            print(f"[trial rejected] {row['last_rejection']}", flush=True)
            last["rejection"] = row["last_rejection"]
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")
        if args.checkpoint_every and row["step"] % args.checkpoint_every == 0:
            problem.save_checkpoint(out / "checkpoint.npz")
            # SLSQP's whole state at this accepted point, saved atomically beside the checkpoint: --resume continues
            # with its Hessian. Without a snapshot of this point (the start, or a call restarted after a rejected
            # trial before its first step) an older state would be of another point: removed
            snapshot = slsqp_latest.get("state")
            if snapshot is not None and np.array_equal(snapshot["accepted"], x):
                opt.save_slsqp_state(out / "slsqp_state.tmp.npz", snapshot)
                os.replace(out / "slsqp_state.tmp.npz", out / "slsqp_state.npz")
            elif (out / "slsqp_state.npz").exists():
                os.remove(out / "slsqp_state.npz")

    if args.checkpoint_every and (args.check_gradient or args.check_stationarity_gradient or args.check_exact_products
                                  or args.check_cost_gradient or args.check_labelling_sensitivity
                                  or args.check_soft_modes
                                  or args.check_g_noise or args.check_rows_c2):
        problem.save_checkpoint(out / "checkpoint.npz")  # the certified seed, reusable after a check (--resume)
    if args.check_rows_c2:  # are the rows twice differentiable? autodiff second derivative vs differences of the gradient
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        model = problem.model
        cfg = model.cfg
        frozen = jax.lax.stop_gradient(sol["state"])
        P = im._dof_projector(cfg, sol["mask"])
        edge = im._edge_mask(cfg)
        params_b, field_s = sol["params"], problem.field(jnp.asarray(x_seed))
        z0 = P(frozen)
        parts = dict(interface=lambda st, rt, p: model.rows_at(st, rt, field_s))
        if model.bootstrap is not None:
            parts["bootstrap"] = lambda st, rt, p: model.bootstrap.rows(st, rt, p)
        rng = np.random.default_rng(0)
        leaves, tree = jax.tree.flatten(z0)
        direction = jax.tree.unflatten(tree, [jnp.asarray(rng.standard_normal(np.shape(a))) for a in leaves])
        direction = P(direction)
        norm = float(np.sqrt(sum(float(jnp.vdot(a, a)) for a in jax.tree.leaves(direction))))
        direction = jax.tree.map(lambda a: a / norm, direction)
        record = {}
        for name, rows_of in parts.items():
            def f(z, rows_of=rows_of):
                st = im._assemble(z, im.runtime_from_params(params_b, cfg), frozen, P, edge)
                r = rows_of(st, im.runtime_from_params(params_b, cfg), params_b)
                return 0.5 * jnp.vdot(r, r)
            grad = jax.jit(jax.grad(f))
            exact = jax.jvp(grad, (z0,), (direction,))[1]
            flat = lambda t: np.concatenate([np.ravel(np.asarray(a)) for a in jax.tree.leaves(t)])  # noqa: E731
            e = flat(exact)
            rows = []
            for h in (1e-3, 1e-4, 1e-5, 1e-6):
                plus = grad(jax.tree.map(lambda a, d: a + h * d, z0, direction))
                minus = grad(jax.tree.map(lambda a, d: a - h * d, z0, direction))
                fd = (flat(plus) - flat(minus)) / (2 * h)
                rows.append(dict(h=h, relative=float(np.linalg.norm(fd - e) / max(np.linalg.norm(e), 1e-300))))
            record[name] = rows
            print(f"[rows C2] {name}: |d2 f v| {np.linalg.norm(e):.3e}; exact vs differences of the exact gradient: "
                  + " ".join(f"h={r['h']:.0e} {r['relative']:.1e}" for r in rows), flush=True)
        (out / "rows_c2_check.json").write_text(json.dumps(record, indent=1) + "\n")
        return
    if args.check_soft_modes:  # are the cost's soft directions relabellings (tangential) or shape (normal) modes?
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        scale = np.asarray(problem.model.x_scale, dtype=float)
        b0 = np.asarray(sol["x"], dtype=float)
        nb = problem.model.n_boundary
        H, _ = problem.dense_hessian(x_seed, sol)
        lam, V = np.linalg.eigh(0.5 * (H + H.T) * np.outer(scale, scale))
        N, a = map(np.asarray, problem.model._normal_operator(jnp.asarray(b0)))
        dR, dZ = (np.asarray(t).reshape(-1, b0.size) for t in jax.jacfwd(problem.model.surface)(jnp.asarray(b0)))
        rows = []
        for k in range(args.check_soft_modes):
            v = V[:, k] * scale
            normal = float(np.linalg.norm(N @ v[:N.shape[1]]))
            total = float(np.sqrt(np.linalg.norm(dR[:, :nb] @ v[:nb]) ** 2 + np.linalg.norm(dZ[:, :nb] @ v[:nb]) ** 2))
            rows.append(dict(eigenvalue=float(lam[k]), normal_fraction=normal / max(total, 1e-300)))
        frac = np.array([r["normal_fraction"] for r in rows])
        counts = {f"<{t:g}": int(np.sum(lam[:len(rows)] < t * lam[-1])) for t in (1e-8, 1e-7, 1e-6, 1e-5)}
        print(f"[soft modes] largest eigenvalue {lam[-1]:.3e}; counts below fraction of it {counts}", flush=True)
        for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)):
            sel = (frac >= lo) & (frac < hi)
            print(f"[soft modes] normal fraction {lo:.2f}-{hi:.2f}: {int(sel.sum())} of the {len(rows)} softest "
                  f"(eigenvalues {lam[:len(rows)][sel].min() if sel.any() else 0:.1e}.."
                  f"{lam[:len(rows)][sel].max() if sel.any() else 0:.1e})", flush=True)
        (out / "soft_modes.json").write_text(json.dumps(dict(largest=float(lam[-1]), modes=rows), indent=1) + "\n")
        return
    if args.check_horizontal_gradient:  # the exact gradient with the K softest (labelling) directions projected out
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        scale = np.asarray(problem.model.x_scale, dtype=float)
        xj = jnp.asarray(x_seed)
        b0 = np.asarray(sol["x"], dtype=float)
        params_bar = problem._pullback(sol["state"], sol["mask"], sol["params"], xj)
        G_b, G_x = map(np.asarray, problem._coordinates(sol["params"], jnp.asarray(b0), xj, params_bar))
        direct = np.asarray(problem._direct(sol["state"], sol["params"], xj))
        H, _ = problem.dense_hessian(x_seed, sol)
        lam, V = np.linalg.eigh(0.5 * (H + H.T) * np.outer(scale, scale))
        Qs = G_b.T * scale[:, None]
        record = dict(eigenvalues=lam[:220].tolist(), runs={})
        full = None
        for K in args.check_horizontal_gradient:
            Vk, lk = V[:, K:], lam[K:]
            mu = scale[:, None] * (Vk @ ((Vk.T @ Qs) / lk[:, None]))
            _, Fd_mu, _ = problem.exact_products(x_seed, sol, mu, check=False)
            grads = direct + G_x - np.asarray(Fd_mu).T
            if full is None:
                full = grads
            rel = (np.linalg.norm(grads - full, axis=1) / np.maximum(np.linalg.norm(full, axis=1), 1e-300)).tolist()
            record["runs"][K] = dict(gradients=grads.tolist(), relative_to_first=rel)
            print(f"[horizontal gradient] K={K} (eigenvalues from {lam[K]:.2e}): per-row change vs K="
                  f"{args.check_horizontal_gradient[0]}: " + " ".join(f"{v:.1e}" for v in rel), flush=True)
        (out / "horizontal_gradient.json").write_text(json.dumps(record) + "\n")
        return
    if args.check_labelling_sensitivity:  # how much do the values move along the cost's flat directions?
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        scale = np.asarray(problem.model.x_scale, dtype=float)
        xj = jnp.asarray(x_seed)
        params_s, field_s = problem._params_at(xj), problem.field(xj)
        b0 = np.asarray(sol["x"], dtype=float)
        H, _ = problem.dense_hessian(x_seed, sol)
        lam, V = np.linalg.eigh(0.5 * (H + H.T) * np.outer(scale, scale))
        got0 = problem.model.evaluate(b0, params_s, field_s, tight=True)
        cost0 = 0.5 * float(got0[0] @ got0[0])
        values0 = np.asarray(problem._rows(got0[1][0], got0[1][2], xj), dtype=float)
        record = dict(cost=cost0, values=values0.tolist(), eigenvalues=lam[:12].tolist(), largest=float(lam[-1]),
                      directions=[])
        for k in range(args.check_labelling_sensitivity):
            rows = []
            for delta in (0.03, 0.1, 0.3, 1.0, 3.0):
                got = problem.model.evaluate(b0 + delta * V[:, k] * scale, params_s, field_s, tight=True)
                if got is None:
                    rows.append(dict(delta=delta, failed=True))
                    continue
                cost = 0.5 * float(got[0] @ got[0])
                values = np.asarray(problem._rows(got[1][0], got[1][2], xj), dtype=float)
                rel = np.abs(values - values0) / np.maximum(np.abs(values0), 1e-300)
                rows.append(dict(delta=delta, cost_change=(cost - cost0) / cost0, value_change=rel.tolist()))
                print(f"[labelling sensitivity] direction {k} (eigenvalue {lam[k]:.2e}) delta {delta:g}: cost "
                      f"{(cost - cost0) / cost0:+.2e}, objective {rel[0]:.2e}, values max {rel.max():.2e} "
                      f"(row {int(rel.argmax())})", flush=True)
            record["directions"].append(dict(eigenvalue=float(lam[k]), steps=rows))
            (out / "labelling_sensitivity.json").write_text(json.dumps(record, indent=1) + "\n")
        return
    if args.check_cost_gradient:  # is g = d(cost)/db consistent with the cost the Newton polish measures?
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        scale = np.asarray(problem.model.x_scale, dtype=float)
        xj = jnp.asarray(x_seed)
        params_s, field_s = problem._params_at(xj), problem.field(xj)
        b0 = np.asarray(sol["x"], dtype=float)
        g0, _, r0 = problem._cost_gradients(b0, params_s, field_s, x_seed)
        cost0 = 0.5 * float(r0 @ r0)
        rng = np.random.default_rng(1)
        gs = g0 * scale
        directions = dict(descent=-gs / np.linalg.norm(gs), random=rng.standard_normal(scale.size))
        directions["random"] /= np.linalg.norm(directions["random"])
        record = dict(cost=cost0, g_norm_scaled=float(np.linalg.norm(gs)), directions={})
        for name, d in directions.items():
            rows = []
            for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                cp = problem.model.evaluate(b0 + h * d * scale, params_s, field_s, tight=True)
                cm = problem.model.evaluate(b0 - h * d * scale, params_s, field_s, tight=True)
                if cp is None or cm is None:
                    rows.append(dict(h=h, failed=True))
                    continue
                fp, fm = 0.5 * float(cp[0] @ cp[0]), 0.5 * float(cm[0] @ cm[0])
                fd = (fp - fm) / (2 * h)
                rows.append(dict(h=h, fd=fd, predicted=float(gs @ d), plus=fp - cost0, minus=fm - cost0,
                                 relative_error=float(abs(fd - gs @ d) / max(abs(gs @ d), 1e-300))))
                print(f"[cost gradient] {name} h={h:.0e}: FD {fd:.6e}, g.d {float(gs @ d):.6e}, rel err "
                      f"{rows[-1]['relative_error']:.1e}, f+ - f0 {fp - cost0:.3e}, f- - f0 {fm - cost0:.3e}",
                      flush=True)
            record["directions"][name] = rows
        (out / "cost_gradient_check.json").write_text(json.dumps(record, indent=1) + "\n")
        return
    if args.check_g_noise:  # the fit's stationarity g from two equilibrium starts at one boundary, and its Newton effect
        import scipy.linalg
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        nb, scale = problem.model.x0.size, np.asarray(problem.model.x_scale, dtype=float)
        xj = jnp.asarray(x_seed)
        params_s, field_s = problem._params_at(xj), problem.field(xj)
        b0 = np.asarray(sol["x"], dtype=float)
        model = problem.model
        outs = {}
        for name, seed in (("hot", None), ("cold", model.seed[0]), ("record", sol["state"])):
            for memo in (im._LAST_SOLVE, im._LAST_REFINED):  # a fresh solve from this start, not the memo
                memo.pop(model.cfg, None)
            got = model.evaluate(b0, params_s, field_s, seed=seed, tight=True)
            rows, aux = got
            state, mask, params_b, _ = aux
            r = jnp.asarray(rows)
            params_bar = problem._cost_pullback(state, mask, params_b, r, xj)
            b_bar, _ = problem._coordinates(params_b, jnp.asarray(b0), xj, params_bar)
            outs[name] = (np.asarray(rows), np.asarray(b_bar)[0])
        H, asym = problem.dense_hessian(x_seed, sol)
        record = dict(asymmetry=asym, pairs={})
        for a, b in (("hot", "cold"), ("hot", "record"), ("cold", "record")):
            (ra, ga), (rb, gb) = outs[a], outs[b]
            dg = ga - gb
            step = scipy.linalg.solve(H, dg, assume_a="sym") / scale
            rec = dict(rows=float(np.linalg.norm(ra - rb) / np.linalg.norm(rb)),
                       g=float(np.linalg.norm(dg) / max(np.linalg.norm(gb), 1e-300)),
                       newton_step_from_noise=float(np.linalg.norm(step)),
                       newton_step_from_g=float(np.linalg.norm(scipy.linalg.solve(H, gb, assume_a="sym") / scale)))
            record["pairs"][f"{a}-{b}"] = rec
            print(f"[g noise] {a} vs {b}: rows {rec['rows']:.1e}, g {rec['g']:.1e}; scaled Newton step of the "
                  f"difference {rec['newton_step_from_noise']:.1e} (of g itself {rec['newton_step_from_g']:.1e})",
                  flush=True)
        (out / "g_noise_check.json").write_text(json.dumps(record, indent=1) + "\n")
        return
    if args.check_exact_products:  # second-order adjoint products vs Richardson difference products
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        nb, scale = problem.model.x0.size, np.asarray(problem.model.x_scale, dtype=float)
        xj = jnp.asarray(x_seed)
        params_s, field_s = problem._params_at(xj), problem.field(xj)
        b0 = np.asarray(sol["x"], dtype=float)
        _, _, Vt = np.linalg.svd(np.asarray(sol["J"])[:, :nb] * scale, full_matrices=False)
        rng = np.random.default_rng(0)
        names = ("weakest", "strongest", "random")
        V = np.stack([Vt[-1] * scale, Vt[0] * scale, rng.standard_normal(nb) * scale], axis=1)
        t0 = time.monotonic()
        problem.exact_products(x_seed, sol, V)  # compile and run
        t1 = time.monotonic()
        HV, FdV, consistency = problem.exact_products(x_seed, sol, V)
        t2 = time.monotonic()
        print(f"[exact products] first-order consistency: g {consistency['g']:.2e}, c {consistency['c']:.2e}; "
              f"compile+3 columns {t1 - t0:.0f} s, 3 columns (compiled, incl. the consistency evaluation) "
              f"{t2 - t1:.1f} s", flush=True)
        record = dict(consistency=consistency, seconds_first=t1 - t0, seconds_three=t2 - t1, h=args.difference_h,
                      directions={})
        for j, name in enumerate(names):
            v = V[:, j]
            def central(step):
                eps = step / float(np.linalg.norm(v / scale))
                gp, cp, _ = problem._cost_gradients(b0 + eps * v, params_s, field_s, x_seed)
                gm, cm, _ = problem._cost_gradients(b0 - eps * v, params_s, field_s, x_seed)
                return (gp - gm) / (2 * eps), (cp - cm) / (2 * eps)
            t3 = time.monotonic()
            (h1, f1), (h2, f2) = central(args.difference_h), central(args.difference_h / 2)
            t4 = time.monotonic()
            Hr, Fr = (4 * h2 - h1) / 3, (4 * f2 - f1) / 3
            rec = dict(H=float(np.linalg.norm(HV[:, j] - Hr) / np.linalg.norm(Hr)),
                       F=float(np.linalg.norm(FdV[:, j] - Fr) / max(np.linalg.norm(Fr), 1e-300)),
                       H_plain=float(np.linalg.norm(HV[:, j] - h1) / np.linalg.norm(h1)),
                       seconds_richardson=t4 - t3)
            record["directions"][name] = rec
            print(f"[exact products] {name}: exact vs Richardson H {rec['H']:.2e}, F_d^T {rec['F']:.2e} "
                  f"(vs plain h: H {rec['H_plain']:.2e}); Richardson pair {t4 - t3:.1f} s", flush=True)
        record["timing"] = {}
        for m in args.exact_products_timing:  # the fixed per-call cost and the per-column cost
            W = rng.standard_normal((nb, m)) * scale[:, None]
            problem.exact_products(x_seed, sol, W)  # compile for m columns
            t5 = time.monotonic()
            problem.exact_products(x_seed, sol, W)
            record["timing"][m] = time.monotonic() - t5
            print(f"[exact products] {m} columns: {record['timing'][m]:.1f} s (compiled)", flush=True)
        (out / "exact_products_check.json").write_text(json.dumps(record, indent=1) + "\n")
        return
    if args.check_stationarity_gradient:  # every row's exact derivative of the converged stationarity
        x_seed = problem.accepted.parameters
        sol = problem._record(x_seed)
        nb = problem.model.x0.size
        xj = jnp.asarray(x_seed)
        params_s, field_s = problem._params_at(xj), problem.field(xj)
        g0, c0, r0 = problem._cost_gradients(sol["x"], params_s, field_s, x_seed)  # the reverse-pass products
        J0 = np.asarray(sol["J"])
        g_lin = J0[:, :nb].T @ np.asarray(sol["rows"])  # the same from the stored linearization (independent)
        consistency = dict(stationarity=float(np.linalg.norm(g0 - g_lin) / max(np.linalg.norm(g_lin), 1e-300)),
                           rows=float(np.linalg.norm(r0 - np.asarray(sol["rows"])) / np.linalg.norm(sol["rows"])))
        print(f"[stationarity gradient] reverse-pass g vs linearization J^T r: {consistency['stationarity']:.2e}, "
              f"rows {consistency['rows']:.2e}", flush=True)
        t0 = time.monotonic()
        got = problem.stationarity_design_gradient(x_seed)
        record = dict(consistency=consistency, true_residual=got["true_residual"], asymmetry=got.get("asymmetry"),
                      hessian_condition=got.get("condition"), seconds_hessian=got.get("seconds_hessian"),
                      gradients=np.asarray(got["gradients"]).tolist(), gauss_newton=np.asarray(got["gauss_newton"]).tolist(),
                      rows=got["rows"], seconds=time.monotonic() - t0, certificate=sol.get("certificate"),
                      newton=sol.get("newton"))
        if sol.get("newton"):
            nw = sol["newton"]
            print(f"[newton polish] {len(nw['steps'])} steps, converged {nw['converged']}, {nw['evaluations']} "
                  f"evaluations (Hessian {nw.get('seconds_hessian', 0):.0f} s), {nw['seconds']:.0f} s; scaled steps "
                  + " ".join(f"{st['step']:.1e}" for st in nw["steps"]), flush=True)
        (out / "stationarity_gradient_check.json").write_text(json.dumps(record, default=float) + "\n")
        print(f"[stationarity gradient] {len(got['rows'])} rows, true residual max {max(got['true_residual']):.1e}, "
              f"Hessian condition {got['condition']:.1e}, {record['seconds']:.0f} s", flush=True)
        return
    if args.check_gradient:
        check_gradient(problem, args.check_gradient, out, steps=tuple(args.check_gradient_steps),
                       cold=args.check_gradient_cold)
        return
    slsqp_latest = {}  # the latest SLSQP snapshot (opt.minimize's state_callback), saved with each checkpoint
    log_step()
    # --resume: SLSQP continues with the saved state of the restored point (slsqp_state.npz beside the checkpoint),
    # i.e. with its quasi-Newton Hessian instead of the identity; a finite-beta step is 45-60 min, so a 2 h segment
    # is 1-3 steps, too few to rebuild it. The state is taken only when it is of the restored accepted point exactly
    # (bitwise: the design vector the checkpoint restored). --newton-polish's refine_accepted() above Newton-refines
    # that point's equilibrium but keeps its design vector, so the state is still of this point: SciPy evaluates the
    # objective, constraints and gradients at its start itself, and the resumed method takes those refined values
    # there as if its last evaluation had returned them (its BFGS pair is then the saved step and the refined
    # gradient difference). Anything else (another point or problem size, a missing or unreadable file): an
    # identity Hessian, logged.
    resume, resume_reason = None, "no --resume"
    if args.resume is not None:
        state_file = Path(args.resume).parent / "slsqp_state.npz"
        if not state_file.exists():
            resume_reason = f"no {state_file}"
        else:
            try:
                resume = opt.load_slsqp_state(state_file)
            except Exception as error:  # noqa: BLE001 -- an unreadable state is not worth stopping the run
                resume_reason = f"{state_file} unreadable: {error}"
            else:
                if not np.array_equal(resume["accepted"], problem.accepted.parameters):
                    resume_reason = (f"{state_file} is of another point (largest difference "
                                     f"{np.max(np.abs(resume['accepted'] - problem.accepted.parameters)):.1e})"
                                     if resume["accepted"].shape == problem.accepted.parameters.shape
                                     else f"{state_file} is of another problem size")
                    resume = None
    if resume is not None:
        print(f"[slsqp] continuing with the saved state of {state_file} (SLSQP iteration "
              f"{int(resume['state']['iter'])}, accepted step {problem.accepted_step})", flush=True)
    else:
        print(f"[slsqp] identity Hessian ({resume_reason})", flush=True)
    while True:  # a rejected trial ends an SLSQP call; restart it from the accepted point (an identity Hessian)
        before = problem.accepted_step
        options = dict(maxiter=args.steps - before, ftol=OPTIMIZER_FTOL,
                       state_callback=lambda state: slsqp_latest.update(state=state))
        resumed = resume is not None
        try:
            result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                                  callback=lambda x: log_step(),
                                  options=dict(options, **({"resume": resume} if resumed else {})))
        except ValueError as error:
            if not resumed or "the SLSQP state is of another" not in str(error):
                raise
            print(f"[slsqp] identity Hessian (the saved state is refused: {error})", flush=True)
            resume = None
            continue
        resume = None
        if resumed and result.stop_reason == "equilibrium_trial_rejected" and problem.accepted_step == before:
            # the saved state's next trial was rejected: as an uninterrupted run would, restart from the identity
            print("[slsqp] identity Hessian (the resumed state's trial was rejected)", flush=True)
            continue
        if (result.stop_reason != "equilibrium_trial_rejected" or problem.accepted_step == before
                or problem.accepted_step >= args.steps):
            break
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
