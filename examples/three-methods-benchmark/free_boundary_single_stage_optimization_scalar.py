#!/usr/bin/env python
"""Match the fixed scalar run with coils determining the plasma boundary.

Edit Settings below, then run this file. Currents and PHIEDGE remain fixed.
All objective targets are soft penalties. Unlike the fixed arm, R00 can move.
The numerical driver owns certified solves, the adjoint, BFGS and output files.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    aspect_target: float = 5.0
    aspect_weight: float = 1.0
    iota_floor: float = 0.19
    iota_weight: float = 10.0
    normal_field_weight: float = 1000.0
    normal_field_limit: float = 0.01
    normal_field_objective_limit: float = 0.008
    normal_field_limit_weight: float = 200000.0
    length_target: float = 5.0
    length_weight: float = 1.0
    curvature_limit: float = 7.0
    curvature_objective_limit: float = 6.9
    curvature_weight: float = 10.0
    coil_distance_limit: float = 0.15
    coil_distance_weight: float = 1000.0
    coil_surface_distance_limit: float = 0.2
    coil_surface_distance_weight: float = 1000.0
    method: str = 'BFGS'
    maxiter: int = 100
    max_trials: int = 300
    device: str = 'cpu'
    parameter_bound: float = 5.0
    coil_step: float = 0.05
    nphi: int = 37
    ntheta: int = 32
    n_coils: int = 3
    coil_order: int = 5
    n_segments: int = 64
    adjoint_batch_size: int = 32
    matrixfree_rhs_batch_size: int = 3  # independent adjoint rows; dense default is unchanged
    root_tolerance: float = 2e-06
    equilibrium_ftol: float = 1e-11  # user-selected optimization force/edge tolerance
    gradient_check_ftol: float = 1e-20  # independent reference solves only


settings = Settings()


def plasma_costs(state, runtime):
    """QA, aspect error and the minimum-absolute-iota floor."""
    import jax.numpy as jnp
    from vmex import optimize as opt
    import numpy as np
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    rows = qs.residuals_state(state, runtime)
    return jnp.stack([
        0.5 * jnp.vdot(rows, rows),
        0.5 * settings.aspect_weight * (opt.aspect_ratio(state, runtime)-settings.aspect_target)**2,
        0.5 * settings.iota_weight * jnp.maximum(settings.iota_floor-opt.min_abs_iota(state, runtime), 0)**2,
    ])


def normalized_normal_field(coils, surf):
    import jax
    import jax.numpy as jnp
    from essos.fields import BiotSavart
    magnetic_field = jax.vmap(BiotSavart(coils).B)(surf.gamma.reshape(-1, 3)).reshape(surf.gamma.shape)
    return jnp.sum(magnetic_field * surf.unitnormal, axis=2) / jnp.linalg.norm(magnetic_field, axis=2)

def coil_costs(coils, surf):
    """Six weighted coil penalties, evaluated on the solved free boundary."""
    import jax
    import jax.numpy as jnp
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    normal_field = normalized_normal_field(coils, surf)
    area_weights = surf.area_element / jnp.sum(surf.area_element)
    normal_rows = (jnp.sqrt(area_weights) * normal_field).ravel()
    smooth_maximum = jax.scipy.special.logsumexp(2000.0 * jnp.sqrt(normal_field**2 + 1.0e-12)) / 2000.0
    return jnp.stack([
        0.5 * settings.normal_field_weight * jnp.vdot(normal_rows, normal_rows),
        0.5 * settings.normal_field_limit_weight * jnp.maximum(smooth_maximum - settings.normal_field_objective_limit, 0.0)**2,
        0.5 * settings.length_weight * jnp.sum((coils.length[:settings.n_coils] - settings.length_target)**2),
        0.5 * settings.curvature_weight * jnp.sum(jnp.maximum(coils.curvature[:settings.n_coils] - settings.curvature_objective_limit, 0)**2),
        0.5 * settings.coil_distance_weight * loss_coil_separation(coils, settings.coil_distance_limit, block_size=32),
        0.5 * settings.coil_surface_distance_weight * loss_coil_surface_distance(coils, surf, settings.coil_surface_distance_limit, block_size=32)])


if __name__ == "__main__":
    from _free_boundary_scalar import run
    raise SystemExit(run(Path(__file__).resolve(), settings, plasma_costs, coil_costs))
