"""One scalar optimizer and acceptance policy for both finite-beta arms.

The scalar objective and physical constraint rows retain total coupled Redl
derivatives. VMEX FunctionProblem owns acceptance. Both arms use the same
SLSQP invocation, constraint construction, row scaling, and optimizer options.
"""

from types import SimpleNamespace
import numpy as np
from scipy.optimize import NonlinearConstraint
from vmex import optimize as opt
from vmex.core.errors import TrialRejected
from linear_root import RootFailure
import bootstrap_settings as S


def build_stage(campaign):
    linear = getattr(getattr(campaign, "model", None), "linear", None)
    if hasattr(linear, "initialize_accepted"):
        linear.initialize_accepted(campaign.y, campaign.x, remaining=campaign.args.accepted_steps)

    def checked(function, x):
        try:
            return function(x)
        except RootFailure as error:
            campaign.event(event="optimizer_trial_rejected", error=str(error))
            raise TrialRejected(str(error)) from error

    def evaluate(x):
        return checked(campaign.evaluate, x)

    def jacobian(x):
        return checked(campaign.jacobian, x)

    problem = opt.FunctionProblem.from_functions(
        campaign.x.copy(),
        fun=lambda x: float(evaluate(x)["rows"][0]),
        grad=lambda x: jacobian(x)[0],
        residual=lambda x: np.asarray(evaluate(x)["rows"][1:]),
        residual_jac=lambda x: np.asarray(jacobian(x)[1:]),
        scales=np.ones_like(campaign.x),
    ).with_acceptance(campaign.accept)
    width = S.P.RADIUS_TOLERANCE - S.P.RADIUS_MARGIN
    plasma_constraint = problem.nonlinear_constraint(
        [
            S.IOTA_FLOOR + S.P.IOTA_MARGIN,
            S.P.RADIUS_TARGET - width,
            S.P.COIL_SURFACE_DISTANCE_LIMIT + S.P.DISTANCE_MARGIN,
        ],
        [np.inf, S.P.RADIUS_TARGET + width, np.inf],
        scales=[S.IOTA_FLOOR, S.P.RADIUS_TOLERANCE, S.P.COIL_SURFACE_DISTANCE_LIMIT],
    )
    coil_constraint = NonlinearConstraint(
        lambda x: np.asarray(campaign.coil_rows(x)),
        0,
        np.inf,
        jac=lambda x: np.asarray(campaign.coil_jac(x)),
    )
    return SimpleNamespace(
        campaign=campaign,
        problem=problem,
        joint_problem=problem,
        x0=campaign.x.copy(),
        constraints=[plasma_constraint, coil_constraint],
        coil_constraint=coil_constraint,
        # CaseRun.accept writes exactly one checkpoint after certified acceptance.
        record_step=lambda *_: None,
        options=dict(maxiter=campaign.args.accepted_steps, ftol=S.P.OPTIMIZER_FTOL),
        initial_steps=campaign.steps,
    )


def run_optimizer(stage, args, *, arm):
    if arm not in ("fixed", "free"):
        raise ValueError(f"unknown finite-beta arm: {arm}")
    return opt.minimize(
        stage.problem,
        method="SLSQP",
        constraints=stage.constraints,
        callback=stage.record_step,
        options=stage.options,
    )


def report_result(stage, result):
    campaign = stage.campaign
    _, iota, radius, clearance = campaign.evaluate(campaign.x)["rows"]
    width = S.P.RADIUS_TOLERANCE - S.P.RADIUS_MARGIN
    slack = np.array(
        [
            (iota - S.IOTA_FLOOR - S.P.IOTA_MARGIN) / S.IOTA_FLOOR,
            (radius - S.P.RADIUS_TARGET + width) / S.P.RADIUS_TOLERANCE,
            (S.P.RADIUS_TARGET + width - radius) / S.P.RADIUS_TOLERANCE,
            (clearance - S.P.COIL_SURFACE_DISTANCE_LIMIT - S.P.DISTANCE_MARGIN) / S.P.COIL_SURFACE_DISTANCE_LIMIT,
        ]
    )
    return dict(
        optimizer_success=bool(result.success),
        message=str(result.message),
        stop_reason=getattr(result, "stop_reason", None),
        iterations=int(result.nit),
        accepted_steps=campaign.steps,
        new_accepted_steps=campaign.steps - stage.initial_steps,
        trial_evaluations=campaign.trials,
        terminal_point_was_accepted=bool(np.array_equal(result.x, campaign.x)),
        derivative_qualified=False,
        profile_adjustments_during_optimization=0,
        minimum_optimizer_slack=float(min(np.min(slack), np.min(campaign.coil_rows(campaign.x)))),
    )


def optimize(campaign):
    """Compatibility entry for preparation/qualification and existing callers."""
    stage = build_stage(campaign)
    return report_result(stage, run_optimizer(stage, campaign.args, arm=campaign.arm))
