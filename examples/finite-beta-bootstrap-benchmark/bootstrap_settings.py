"""Matched finite-beta/Redl case; SI density and eV temperatures.

Pressure and PHIEDGE are frozen after preparing a common reference. These
profiles and beta are not optimization variables or optimization constraints.
"""

from main_reference import P

IOTA_FLOOR = 0.28  # finite-beta target; vacuum parameters remain unchanged
BETA_AXIS_INITIAL = 0.03
BETA_CALIBRATION_RTOL = 1e-3
DENSITY_AXIS_M3 = 1e20
TEMPERATURE_INITIAL_EV = 20.0  # starting guess; calibrated once, before either arm
ZEFF = 1.0
DENSITY_SHAPE = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
TEMPERATURE_SHAPE = (1.0, -1.0)
TI_OVER_TE = 1.0
HELICITY_N = 0
CURRENT_DEGREE = 12
CURRENT_SCALE_A = 1e5
REDL_SURFACES = tuple(0.02 + 0.96 * i / 48 for i in range(49))
REDL_N_LAMBDA = 64
REDL_CLOSURE_RTOL = 1e-8
REDL_INTERIOR_MISMATCH_LIMIT = 0.02
ROOT_ATOL = 1e-12
NEWTON_STEPS = 12
NEWTON_EXTRA_STEPS = 4
TRIAL_DENSE_REBUILDS = 1  # per speculative segment; strict derivatives remain unrestricted
NEWTON_STAGNATION_STEPS = 2
LINEAR_REFRESH_HORIZON = 12
LINEAR_REUSE_BUDGET_FRACTION = 0.2
CONTINUATION_INITIAL = 1.0  # identical trial recovery policy for fixed and free
CONTINUATION_MIN_STEP = 0.25
CONTINUATION_MAX_ATTEMPTS = 6
LINEAR_RTOL = 1e-10
LINEAR_GATE = 1e-9
LINEAR_RESTART = 80
LINEAR_CYCLES = 4
DENSE_BATCH_SIZE = 32
SAVED_LINEARIZATION = True
SAVED_KRYLOV_ACTIONS = True
NEWTON_RTOL = 1e-6
NEWTON_GATE = 1e-5
DEVICE_SOLVES = True
DENSE_ASSEMBLY = "device"
ADJOINT_RHS_BATCH_SIZE = 2
MAX_ROOT_DOFS = 20000
MAX_TRIALS = 500
COIL_FIT_STEPS = 200
MAX_BOUNDARY_MODE = P.RESOLUTION[0]
BOUNDARY_STEP_M = 0.01
BOUNDARY_SPECTRAL_ALPHA = 1.2
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
NORMAL_FIELD_WEIGHT = 1e3
NORMAL_FIELD_EXCESS_WEIGHT = 2e5
NORMAL_FIELD_MARGIN = 0.008
NORMAL_FIELD_LIMIT = 0.01
# Fixed-only interface objective. The pressure residual is
# (B_out^2 - B_in^2 - 2*mu0*p_edge) / <B_in^2 + 2*mu0*p_edge>_area.
# These are starting weights and endpoint limits, not a guarantee that the
# optimizer reaches them. Existing normal-only run objectives are different.
BOUNDARY_OBJECTIVE_VERSION = "normal-plus-pressure-v1"
PRESSURE_BALANCE_WEIGHT = 1e3
PRESSURE_BALANCE_EXCESS_WEIGHT = 2e5
PRESSURE_BALANCE_MARGIN = 0.008
PRESSURE_BALANCE_LIMIT = 0.01
INTERFACE_GRID = (37, 32)
VERIFY_INTERFACE_GRID = (73, 64)
VC_DIGITS = 4
CALIBRATION_STEPS = 16
SEED_FORCE_TOLERANCE = 1e-11  # initial Picard seed only; final coupled certificate stays 1e-15
SEED_MAX_ITERATIONS = 12000
PICARD_STEPS = 30
PICARD_RTOL = 1e-7
