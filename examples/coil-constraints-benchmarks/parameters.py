"""Cases of the coil-constraint benchmarks, selected by ``COIL_CASE`` (default ``ellipse5``).

The module-level values are the ``ellipse5`` case; the blocks at the end override
them per case. Every case has R0 = ``RADIUS_TARGET`` = 1 m and B0 = 1 T: the
coil currents are scaled once so their linked mu0 I / 2 pi is B0 R0 (the edge
R B_phi) and then held fixed, while the free arm varies PHIEDGE, so the plasma
size stays free and the field strength (and beta) holds. MSC is the mean
squared curvature, in 1/m^2.

=============== ================================ ======== ======== ============= ======================
case            configuration and seed           --beta   model    iota          coils
=============== ================================ ======== ======== ============= ======================
ellipse5        QA, nfp 2, aspect 4.9-5.1, from  volume   Redl     >= 0.41       3 x order 16: 5 m,
                ``input.rotating_ellipse``                                       5 /m, 5 /m^2, 0.15 m
                                                                                 apart, 0.20 m clear
ellipse5-beta7  as ellipse5                      on axis  Redl     >= 0.16       as ellipse5
qa3             QA, nfp 3, aspect 5.9-6.1        volume   Redl     >= 0.41       3 x order 8: 3.5 m,
qh              QH (1, -1), nfp 4, 5.9-6.1       volume   Redl     >= 1.1        8 /m, 10 /m^2, 0.08 m
qi              QI, nfp 4, 7.9-8.1, mirror 0.21  volume   Redl     >= 0.51       apart, 0.15 m clear
qa4-beta        QA, nfp 2, aspect 3.5-4.5        volume   Redl     >= 0.27       4 x order 12, limits
qi6-beta        QI, nfp 4, 5.9-6.1, mirror 0.21  volume   DKX      0.86 - 0.98   from the plasma size
=============== ================================ ======== ======== ============= ======================

"--beta" is how the scripts read ``--beta`` (``BETA_DEFINITION``: <beta>, or
WOUT ``betaxis``); "model" the bootstrap current of ``--bootstrap``
(``BOOTSTRAP_MODEL``; DKX needs the optional ``dkx`` package). The iota floors
and ceiling keep the profile off the low-order rationals where a vacuum field
breaks into islands the nested-surface equilibrium cannot see: QH between
iota = 1 and 8/7, QI above 1/2, qi6-beta in the Stellaris band below the 4/4
islands. ``qa3``, ``qh``, ``qi``, ``qa4-beta`` and ``qi6-beta`` seed from a
rotating ellipse (``SEED`` = (nfp, aspect, b / a_eff)) and use their own
stage-two coils, ``coils.<case>.json`` (``fit_coils.py``). A ``-tok`` suffix
(``qa4-beta-tok``, ``qi6-beta-tok``) seeds the same case from a circular
tokamak with a 0.05 m helical ripple instead. For ``qa4-beta*`` and
``qi6-beta*`` the coil length, curvature and MSC limits are
``COIL_LIMIT_FACTORS`` times the circumference 2 pi (a + d), the curvature
1 / (a + d) and its square of a circle d = ``COIL_SURFACE_DISTANCE_LIMIT``
outside the widest allowed plasma, a = R0 / aspect_min.
"""
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = os.environ.get("COIL_CASE", "ellipse5")
RESOLUTION = (8, 8, 51)            # MPOL, NTOR, NS of the optimization solves
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
INPUT_FILE = HERE / "input.rotating_ellipse"
COILS_FILE = HERE / "coils.initial.json"
BETA_DEFINITION = "volume"         # --beta is <beta>; "axis": WOUT betaxis
B0 = 1.0                           # T: coil currents give R B_phi = B0 R0; the seed PHIEDGE matches it
FREE_PHIEDGE = True                # free arm: PHIEDGE is a design variable (False: fixed PHIEDGE and currents)

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and CURTOR
# are design variables, and the mismatch sum_j R_j^2 against the BOOTSTRAP_MODEL current (Redl
# or DKX, in Redl's normalized form) is held under REDL_TOLERANCE. The seed's Picard loop is Redl's.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
PICARD_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAX = 10, 1e-3, 1.0
BOOTSTRAP_BETA_STEP = 0.01        # a larger --beta is ramped in with its bootstrap current, in steps of at most this
BOOTSTRAP_BETA_START = None       # set: the ramp first doubles beta from this, so a seed with little vacuum
                                  # iota stays under its eps iota^2 limit while the bootstrap current grows
OHMIC_CURRENT = None              # set (A): the seed ramp's prescribed current, blended into the bootstrap one
CURRENT_KNOTS, CURRENT_STEP = 8, 0.05   # step relative to the largest knot value and to |CURTOR|
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None minimizes the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # examples/optimization/QI_optimization.py
MIRROR_LIMIT, MIRROR_MARGIN = None, 0.001  # upper limit on the edge mirror ratio (Bmax - Bmin) / (Bmax + Bmin)
BOOTSTRAP_MODEL = "redl"          # "dkx": the DKX kinetic <j.B> replaces Redl in the self-consistency row
DKX_SURFACES, DKX_COLLISION_OPERATOR = (0.25, 0.5, 0.75), 0  # 0: momentum-conserving Fokker-Planck

# Physical targets. The free-boundary arm imposes them as hard inequalities.
IOTA_FLOOR, IOTA_MARGIN = 0.41, 0.0005
IOTA_CEILING = None                # upper limit on max |iota|
IOTA_AXIS = False                  # True: floor and ceiling also bound VMEC's extrapolated axis and edge iota (iotaf)
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001

# Coils and their hard engineering limits.
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 16, 256
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
    COILS_FILE = HERE / f"coils.{CASE}.json"
    N_COILS, COIL_ORDER = 3, 8
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
elif CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta"):
    # Finite-beta, self-consistent bootstrap cases (Redl for the QA, DKX for the QI), 4 order-12 coils per half period.
    COILS_FILE = HERE / f"coils.{CASE}.json"
    # (1.8, 2.5, 1.2): mid-range of Wechsung et al. (2022), Jorge et al. (2023) and Wiedman et al. (2024)
    N_COILS, COIL_ORDER, COIL_LIMIT_FACTORS = 4, 12, (1.8, 2.5, 1.2)
    COIL_FIT_MAXITER = 1000  # at 200 the qa4-beta refit left B.n/|B| ~3e-3 and the first free solve could fail
    PICARD_ITERATIONS, PICARD_RELAX, BOOTSTRAP_BETA_STEP = 30, 0.5, 0.005
    # The half-mesh minimum at s = 0.01 left the axis iota 1-2.5% under the floor; R0 is held to 1 mm.
    IOTA_AXIS, RADIUS_TOLERANCE, RADIUS_MARGIN = True, 1e-3, 1e-4
    if CASE.startswith("qa4-beta"):
        SEED, IOTA_FLOOR, ASPECT_RANGE = (2, 4.0, 0.5), 0.27, (3.5, 4.5)  # min |iota| sits on axis, near its vacuum value
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.10, 0.20
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
    OHMIC_CURRENT = 1.0e5 if CASE.startswith("qa4-beta") else 6.0e4
if COIL_LIMIT_FACTORS is not None:
    _radius = RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT
    LENGTH_LIMIT = COIL_LIMIT_FACTORS[0] * 2 * math.pi * _radius
    CURVATURE_LIMIT, MSC_LIMIT = COIL_LIMIT_FACTORS[1] / _radius, COIL_LIMIT_FACTORS[2] / _radius**2
