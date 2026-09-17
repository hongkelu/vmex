"""Independent QA steps, tolerance-based restoration, and measured acceptance."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Policy:
    constraint_relative_tolerance: float = .01
    maximum_coil_step_m: float = .001
    maximum_current_fraction_step: float = .01
    max_trials: int = 6
    backtrack_factor: float = .5
    armijo_fraction: float = 1e-4
    minimum_qa_decrease: float = 1e-10
    violation_reduction_fraction: float = 1e-4
    projected_gradient_atol: float = 1e-6
    projected_gradient_rtol: float = 1e-4
    restoration_fraction_interior: float = .001
    restoration_fraction_boundary: float = .02
    restoration_boundary_start: float = .8
    restoration_budget_fraction: float = .2
    restoration_descent_fraction: float = .5

    def __post_init__(self):
        for name in ('constraint_relative_tolerance', 'maximum_coil_step_m',
                     'maximum_current_fraction_step', 'minimum_qa_decrease',
                     'projected_gradient_atol', 'projected_gradient_rtol'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('invalid positive policy value: ' + name)
        if not 1 <= self.max_trials <= 8 or int(self.max_trials) != self.max_trials:
            raise ValueError('max_trials must be an integer in 1..8')
        for name in ('backtrack_factor', 'armijo_fraction', 'violation_reduction_fraction'):
            if not 0 < getattr(self, name) < 1:
                raise ValueError('invalid fraction: ' + name)
        for name in ('restoration_fraction_interior', 'restoration_fraction_boundary',
                     'restoration_boundary_start', 'restoration_budget_fraction',
                     'restoration_descent_fraction'):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError('invalid restoring fraction: ' + name)
        if self.restoration_fraction_interior > self.restoration_fraction_boundary:
            raise ValueError('boundary restoring weight must not be smaller')


def displacement(delta):
    delta = np.asarray(delta)
    if delta.shape != (111,) or not np.all(np.isfinite(delta)):
        raise ValueError('expected finite 111-vector')
    t = np.arange(75)/75
    basis = np.ones((75, 9))
    for k in range(1, 5):
        basis[:, 2*k-1] = np.sin(2*np.pi*k*t)
        basis[:, 2*k] = np.cos(2*np.pi*k*t)
    return np.einsum('cdk,sk->csd', delta[3:].reshape(4, 3, 9), basis)


def motion_bounds(delta):
    """Conservative displacement bound for every coil angle, plus current cap.

    Each Fourier harmonic is bounded by the spectral norm of its sine/cosine
    coefficient matrix; the triangle inequality bounds their sum. Stellarator
    symmetry copies are isometries, so the same bound covers all physical coils.
    Current coordinates are fractions of each coil's original nominal current.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (111,) or not np.all(np.isfinite(delta)):
        raise ValueError('expected finite 111-vector')
    coefficients = delta[3:].reshape(4, 3, 9)
    bound = np.linalg.norm(coefficients[:, :, 0], axis=1)
    for k in range(1, 5):
        bound += np.linalg.svd(coefficients[:, :, 2*k-1:2*k+1], compute_uv=False)[:, 0]
    return float(np.max(bound)), float(np.max(np.abs(delta[:3])))


def limit_direction(delta, policy, *, expand):
    motion, current = motion_bounds(delta)
    ratios = [policy.maximum_coil_step_m/motion if motion else np.inf,
              policy.maximum_current_fraction_step/current if current else np.inf]
    factor = min(ratios)
    if not expand:
        factor = min(1., factor)
    if not np.isfinite(factor):
        return np.zeros(111)
    return np.asarray(delta)*factor


def constraint_state(values, targets, constraint_scales, policy):
    values, targets, scales = map(np.asarray, (values, targets, constraint_scales))
    if (values.shape != (4,) or targets.shape != (3,) or scales.shape != (3,)
            or not all(np.all(np.isfinite(x)) for x in (values, targets, scales))
            or np.any(scales <= 0) or np.any(targets == 0)):
        raise ValueError('invalid physical constraints')
    tolerances = policy.constraint_relative_tolerance*np.abs(targets)
    errors = values[1:]*scales
    ratios = np.abs(errors)/tolerances
    return dict(feasible=bool(np.all(ratios <= 1.)), errors=errors.tolist(),
                tolerances=tolerances.tolist(), ratios=ratios.tolist(),
                violation=float(max(0., np.max(ratios)-1.)))


@dataclass(frozen=True)
class Direction:
    delta: np.ndarray
    mode: str
    projected_gradient_norm: float
    objective_directional_derivative: float
    constraint_info: dict
    restoring_weight: float = 0.
    combined_cap_factor: float = 1.
    predicted_constraint_change: tuple = ()


def proposal(values, jacobian, scales, targets, constraint_scales, policy):
    from single_stage_problem import proposal as implementation
    return implementation(values, jacobian, scales, targets, constraint_scales, policy)


def converged(direction, initial_projected_gradient_norm, policy):
    threshold = max(policy.projected_gradient_atol,
                    policy.projected_gradient_rtol*initial_projected_gradient_norm)
    return bool(direction.constraint_info['feasible'] and
                direction.projected_gradient_norm <= threshold), threshold


def acceptance(before, after, direction, alpha, targets, constraint_scales, policy):
    from single_stage_problem import acceptance as implementation
    return implementation(before, after, direction, alpha, targets, constraint_scales, policy)


# One exception identity, also when older callers load this module by file path.
from single_stage_support import TrialRejected as TrialRejected


@dataclass(frozen=True)
class SearchResult:
    candidate: object
    values: np.ndarray | None
    delta: np.ndarray | None
    trials: tuple


def backtrack(before, direction, targets, constraint_scales, policy, evaluate, record):
    from single_stage_problem import backtrack as implementation
    return implementation(before, direction, targets, constraint_scales, policy, evaluate, record)
