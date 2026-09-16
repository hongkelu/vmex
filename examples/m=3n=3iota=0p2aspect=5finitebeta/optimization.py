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


def proposal(values, jacobian, scales, targets, constraint_scales, policy):
    """QA tangent in the feasible band; pure equality restoration outside it.

    QA sizing depends on geometry/current caps, never the equality-error norm.
    Projection uses the three equality Jacobian rows in scaled coordinates.
    Thus convergence means equality-tangent stationarity, not a full inequality
    KKT certificate for the tolerance band.
    """
    values, scales, jacobian = map(np.asarray, (values, scales, jacobian))
    info = constraint_state(values, targets, constraint_scales, policy)
    if (scales.shape != (111,) or np.any(scales <= 0)
            or not np.all(np.isfinite(scales)) or jacobian.shape != (4, 111)
            or not np.all(np.isfinite(jacobian))):
        raise ValueError('invalid scaled Jacobian')
    jac = jacobian*scales[None, :]
    u, singular, vt = np.linalg.svd(jac[1:], full_matrices=False)
    if singular[-1] <= 1e-12*max(1., singular[0]):
        raise ValueError('equality Jacobian is rank deficient')
    gradient = float(values[0])*jac[0]
    projected = gradient-vt.T@(vt@gradient)
    norm = float(np.linalg.norm(projected))
    if info['feasible']:
        direction = -projected*scales
        mode = 'qa'
    else:
        direction = -(vt.T@((u.T@values[1:])/singular))*scales
        mode = 'restoration'
    delta = limit_direction(direction, policy, expand=(mode == 'qa'))
    derivative = float((float(values[0])*jacobian[0])@delta)
    return Direction(delta, mode, norm, derivative, info)


def converged(direction, initial_projected_gradient_norm, policy):
    threshold = max(policy.projected_gradient_atol,
                    policy.projected_gradient_rtol*initial_projected_gradient_norm)
    return bool(direction.constraint_info['feasible'] and
                direction.projected_gradient_norm <= threshold), threshold


def acceptance(before, after, direction, alpha, targets, constraint_scales, policy):
    old = direction.constraint_info
    try:
        new = constraint_state(after, targets, constraint_scales, policy)
    except ValueError:
        return False, dict(reason='nonfinite_candidate_metrics')
    qa_before, qa_after = .5*float(before[0])**2, .5*float(after[0])**2
    if not np.isfinite(qa_after):
        return False, dict(reason='nonfinite_candidate_objective')
    if old['feasible']:
        required = max(policy.minimum_qa_decrease,
                       policy.armijo_fraction*alpha*max(0., -direction.objective_directional_derivative))
        passed = new['feasible'] and qa_before-qa_after >= required
        reason = 'qa_decrease_and_feasible' if passed else ('constraint_violation' if not new['feasible'] else 'insufficient_qa_decrease')
    else:
        required = policy.violation_reduction_fraction*alpha*old['violation']
        passed = new['feasible'] or (old['violation']-new['violation'] >= max(1e-12, required))
        reason = 'violation_reduced' if passed else 'insufficient_violation_reduction'
    return bool(passed), dict(reason=reason, before=old, after=new,
                             qa_before=qa_before, qa_after=qa_after,
                             required_decrease=required)


class TrialRejected(Exception):
    """Expected numerical rejection of a candidate; permits bounded backtracking."""


@dataclass(frozen=True)
class SearchResult:
    candidate: object
    values: np.ndarray | None
    delta: np.ndarray | None
    trials: tuple


def backtrack(before, direction, targets, constraint_scales, policy, evaluate, record):
    """Evaluate each reduced proposal from the same accepted base state.

    evaluate(delta, trial_index) must return (candidate, actual_rows) without
    mutating the base state. Only a returned accepted candidate may be promoted.
    """
    trials = []
    for index in range(1, policy.max_trials+1):
        alpha = policy.backtrack_factor**(index-1)
        delta = alpha*direction.delta
        motion, current = motion_bounds(delta)
        if motion > policy.maximum_coil_step_m*(1+1e-12) or current > policy.maximum_current_fraction_step*(1+1e-12):
            raise ValueError('proposal exceeds the geometry/current budget')
        entry = dict(trial=index, alpha=alpha, maximum_coil_bound_m=motion,
                     maximum_current_fraction=current)
        if not np.any(delta):
            entry.update(accepted=False, reason='zero_proposal')
            trials.append(entry); record(entry)
            break
        try:
            candidate, values = evaluate(delta, index)
            passed, details = acceptance(before, values, direction, alpha,
                                         targets, constraint_scales, policy)
            entry.update(accepted=passed, **details)
        except TrialRejected as exc:
            candidate = values = None
            entry.update(accepted=False, reason='numerical_rejection', error=str(exc))
        trials.append(entry); record(entry)
        if entry['accepted']:
            return SearchResult(candidate, np.asarray(values), delta, tuple(trials))
    return SearchResult(None, None, None, tuple(trials))
