"""Shared physical constraints for the two scalar comparison arms.

SLSQP enforces these as inequalities, independently of the objective weights.
Small interior margins protect the requested limits on the NS=101 check.
"""
IOTA_FLOOR = 0.19
RADIUS_TARGET = 1.0  # physical major radius, metres; not RBC(0,0)
RADIUS_TOLERANCE = 0.01
IOTA_MARGIN = 0.0005
RADIUS_MARGIN = 0.001
FORCE_TOLERANCE = 1e-11  # production force tolerance; derivative tests run separately


def physical_values(state, runtime):
    import jax.numpy as jnp
    from vmex import optimize as opt
    return jnp.stack((opt.min_abs_iota(state, runtime), opt.major_radius(state, runtime)))


def inequalities(values):
    """Dimensionless favourable-positive rows: iota, radius lower, radius upper."""
    import jax.numpy as jnp
    iota, radius = values
    width = RADIUS_TOLERANCE - RADIUS_MARGIN
    return jnp.stack(((iota-IOTA_FLOOR-IOTA_MARGIN)/IOTA_FLOOR,
                      (radius-RADIUS_TARGET+width)/RADIUS_TOLERANCE,
                      (RADIUS_TARGET+width-radius)/RADIUS_TOLERANCE))


def diagnostics(state, runtime):
    import numpy as np
    iota, radius = map(float, physical_values(state, runtime))
    return dict(major_radius_m=radius, radius_error_m=radius-RADIUS_TARGET,
        iota_constraint_slack=iota-IOTA_FLOOR,
        radius_constraint_slack_m=RADIUS_TOLERANCE-abs(radius-RADIUS_TARGET),
        constraints_feasible=int(iota >= IOTA_FLOOR and abs(radius-RADIUS_TARGET) <= RADIUS_TOLERANCE),
        optimizer_constraints_feasible=int(np.min(inequalities((iota,radius))) >= -1e-8))
