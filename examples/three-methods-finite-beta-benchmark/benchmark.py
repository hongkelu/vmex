"""Historical scalar-loss regression constants; not production configuration."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'benchmark.json').read_text())
TARGET_BETA = CONFIG['beta_target']
BETA_RELATIVE_TOLERANCE = CONFIG['beta_endpoint_relative_tolerance']
MAX_MODE = CONFIG['numerics']['max_boundary_mode']
NPHI, NTHETA = (CONFIG['numerics'][k] for k in ('virtual_casing_nphi', 'virtual_casing_ntheta'))
VC_DIGITS = CONFIG['numerics']['virtual_casing_digits']
PARAMETER_STEP, COIL_STEP, ESS_ALPHA = (CONFIG['numerics'][k] for k in ('boundary_scale', 'coil_scale', 'ess_alpha'))
ASPECT_TARGET, ASPECT_WEIGHT = (CONFIG['plasma'][k] for k in ('aspect_target', 'aspect_weight'))
IOTA_FLOOR, IOTA_MARGIN, IOTA_WEIGHT = (CONFIG['plasma'][k] for k in ('minimum_abs_iota', 'iota_margin', 'iota_penalty_weight'))
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = (CONFIG['plasma'][k] for k in ('major_radius_target_m', 'major_radius_tolerance_m', 'major_radius_margin_m'))
LENGTH_TARGET, LENGTH_WEIGHT = (CONFIG['coil_penalties'][k] for k in ('length_target_m', 'length_weight'))
CURVATURE_LIMIT, CURVATURE_OBJECTIVE_LIMIT, CURVATURE_WEIGHT = (CONFIG['coil_penalties'][k] for k in ('curvature_limit_per_m', 'curvature_hinge_per_m', 'curvature_weight'))
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = (CONFIG['coil_penalties'][k] for k in ('coil_coil_min_m', 'coil_coil_weight'))
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = (CONFIG['coil_penalties'][k] for k in ('coil_plasma_min_m', 'coil_plasma_weight'))
NORMAL_FIELD_WEIGHT, NORMAL_FIELD_LIMIT, NORMAL_FIELD_OBJECTIVE_LIMIT, NORMAL_FIELD_LIMIT_WEIGHT = (CONFIG['coil_penalties'][k] for k in ('normal_field_weight', 'normal_field_max_limit', 'normal_field_smooth_max_hinge', 'normal_field_smooth_max_weight'))
STATE_FIELDS = ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')


def physical_values(state, runtime):
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core.statephysics import major_radius
    return jnp.stack((opt.volume_average_beta(state,runtime),
                      opt.min_abs_iota(state,runtime),major_radius(state,runtime)))


def inequalities(values):
    import jax.numpy as jnp
    _, iota, radius = values
    width = RADIUS_TOLERANCE-RADIUS_MARGIN
    return jnp.stack(((iota-IOTA_FLOOR-IOTA_MARGIN)/IOTA_FLOOR,
        (radius-RADIUS_TARGET+width)/RADIUS_TOLERANCE,
        (RADIUS_TARGET+width-radius)/RADIUS_TOLERANCE))
