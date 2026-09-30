"""Case shared by the free- and fixed-boundary coil-constraint benchmarks.

Default: vacuum QA from a rotating ellipse, 1 m major radius, three independent
order-16 coils. MSC is mean SQUARED curvature, in inverse square metres.

``COIL_CASE=qa3``, ``qh`` and ``qi`` start from rotating ellipses at R = 1 m
and B0 ~ 1 T: nfp 3 at aspect 6 for QA, nfp 4 at aspect 6 for QH, helicity
(1, -1), and nfp 4 at aspect 8 for the constructed QI residual with a
mirror-ratio limit. Their iota floors keep the profile off the low-order
rationals a vacuum field breaks into islands at: QH above iota = 1 (its seed,
b = 0.9 a_eff, starts at 1.016), QI above 1/2. Each has its own stage-two
coils, three order-8 coils per half period.

B0 = 1 T in both cases: the coil currents are scaled once so their linked
mu0 I / 2 pi is B0 R0 (the edge R B_phi) and then held fixed, and the free arm
varies PHIEDGE, so the plasma size stays free while the field strength and,
at finite beta, beta hold.

``COIL_CASE=ellipse5-beta7`` keeps the ellipse with an iota floor of 0.16 and ``--beta`` on axis.
``COIL_CASE=qa6`` selects the aspect-6 case: the Landreman & Paul (2021) QA
boundary at R = 1 m, B0 = 1 T set through PHIEDGE (R B_phi = B0 R0 outside the
plasma), ``--beta`` read as on-axis beta (WOUT ``betaxis``), and the coil limits
of Jorge et al. (2023, section 4.2) with three order-6 coils, except a 6.5 m
per-coil length for this nfp-2 device.
``COIL_CASE=qa4-beta`` and ``qi6-beta`` are the finite-beta, self-consistent bootstrap
cases: a compact nfp-2 QA (aspect 3.5-4.5, Redl) and an nfp-4 QI at aspect 6, whose
bootstrap current comes from the DKX drift-kinetic solver (Redl assumes quasisymmetry).
Their coil length limit follows the plasma: LENGTH_FACTOR times the circumference
2 pi (a + d) of a circle d = COIL_SURFACE_DISTANCE_LIMIT outside the widest allowed
plasma, a = R / aspect_min.
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
NITER, DELT = None, None            # override the deck's iteration budget / time step when set
FREE_PHIEDGE = True                # free arm: PHIEDGE is a design variable (False: fixed PHIEDGE and currents)

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent Redl bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and
# CURTOR are design variables, and the Redl mismatch sum_j R_j^2 is held under REDL_TOLERANCE.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = tuple(0.1 + 0.8 * i / 7 for i in range(8))
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
PICARD_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAX = 10, 1e-3, 1.0
BOOTSTRAP_BETA_STEP = 0.01        # a larger --beta is ramped in with its bootstrap current, in steps of at most this
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
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001

# Coils and their hard engineering limits.
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 16, 256
COIL_STEP = 0.05                   # coordinate scale of the coil Fourier modes
LENGTH_LIMIT = 5.0                 # m, each independent coil
LENGTH_FACTOR = None               # set: LENGTH_LIMIT = LENGTH_FACTOR 2 pi (R / aspect_min + COIL_SURFACE_DISTANCE_LIMIT)
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, to the current plasma boundary
# Interior margins of the sampled constraints; endpoint checks use the limits.
CURVATURE_MARGIN, MSC_MARGIN, LENGTH_MARGIN, DISTANCE_MARGIN = 0.10, 0.02, 1e-5, 0.001

if CASE == "qa6":
    INPUT_FILE = HERE.parents[1] / "examples/data/input.LandremanPaul2021_QA_lowres"
    BETA_DEFINITION = "axis"
    NITER, DELT = 30000, 0.7       # one NS stage at FTOL 1e-15 instead of the deck's multigrid ladder
    IOTA_FLOOR = 0.42
    ASPECT_RANGE = (5.9, 6.1)
    COIL_ORDER = 6
    LENGTH_LIMIT = 6.5              # m: nfp 2, 3 coils/half-period (Jorge 2023: 5.5 m at nfp 3, 2 coils)
    COIL_DISTANCE_LIMIT = 0.10
    COIL_SURFACE_DISTANCE_LIMIT = 0.15
elif CASE == "ellipse5-beta7":
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
elif CASE in ("qa4-beta", "qi6-beta"):
    # Finite-beta, self-consistent bootstrap cases (Redl for the QA, DKX for the QI), 4 order-12 coils per half period.
    COILS_FILE = HERE / f"coils.{CASE}.json"
    N_COILS, COIL_ORDER, LENGTH_FACTOR = 4, 12, 1.8
    PICARD_ITERATIONS, PICARD_RELAX, BOOTSTRAP_BETA_STEP = 30, 0.5, 0.005
    if CASE == "qa4-beta":
        SEED, IOTA_FLOOR, ASPECT_RANGE = (2, 4.0, 0.5), 0.42, (3.5, 4.5)
        CURVATURE_LIMIT, MSC_LIMIT = 5.0, 5.0               # Wechsung et al. (2022), QUASR
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.10, 0.20
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 6.0, 0.5), None, "QI", (5.9, 6.1), 0.21
        IOTA_FLOOR, BOOTSTRAP_MODEL = 0.51, "dkx"
        CURVATURE_LIMIT, MSC_LIMIT = 12.0, 20.0             # between the nfp-4 QH designs and Stellaris
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
elif CASE != "ellipse5":
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
if LENGTH_FACTOR is not None:
    LENGTH_LIMIT = LENGTH_FACTOR * 2 * math.pi * (RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT)
