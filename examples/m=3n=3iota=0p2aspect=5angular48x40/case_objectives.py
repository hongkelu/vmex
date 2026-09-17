"""Original objective definitions retained for checkpoint and legacy callers."""
QA_SURFACES = (0.25, 0.5, 0.75, 1.0)
PHYSICAL_TARGETS = (0.2, 5.0, -0.17506474574437714)

def physical_rows(state, runtime):
    """The three physical quantities constrained by this example."""
    import jax.numpy as jnp
    from vmex.core import statephysics

    return jnp.array(
        [
            statephysics.mean_iota(state, runtime),
            statephysics.aspect_ratio(state, runtime),
            statephysics.on_axis_magnetic_field(state, runtime),
        ]
    )

def resolve_targets(initial_physical):
    """Use the stated targets; resume keeps the original checkpoint targets."""
    import numpy as np

    values = np.asarray(initial_physical, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or values[2] == 0:
        raise ValueError("finite, nonzero initial signed B0 required")
    return np.array(PHYSICAL_TARGETS)

def make_rows(runtime, targets, loss_scale):
    """Define the objective residual and normalized constraint errors.

    The optimizer minimizes F = 0.5 * rows[0]**2. The other three rows are
    constraints. loss_scale is fixed at initialization and retained on resume.
    """
    import jax
    import jax.numpy as jnp
    from case import CONSTRAINT_SCALES
    from vmex.core.optimize import QuasisymmetryRatioResidual

    qs = QuasisymmetryRatioResidual(QA_SURFACES, helicity_m=1, helicity_n=0)

    def rows(state):
        qa = jnp.linalg.norm(qs.residuals_state(state, runtime)) / loss_scale
        constraints = (physical_rows(state, runtime) - targets) / CONSTRAINT_SCALES
        return jnp.r_[qa, constraints]

    return jax.jit(rows)
