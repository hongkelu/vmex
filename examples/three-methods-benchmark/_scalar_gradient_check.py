"""Centered differences that retain partial evidence and report failed solves."""

import numpy as np


def centered_checks(evaluate, center, direction, analytic, constraint_analytic, *, steps=(0.003, 0.001), save):
    """Evaluate independent endpoint records; never difference missing results.

    ``evaluate(parameters)`` returns value, constraints and solve diagnostics.
    The caller supplies an accepted-root solve without its tangent predictor.
    This helper does not set solve tolerances or relax derivative acceptance.
    """
    checks, endpoints = [], []
    for h in steps:
        pair = []
        for sign in (1, -1):
            try:
                endpoint = evaluate(center + sign * h * direction)
                values = [endpoint["value"], *endpoint.get("constraints", [])]
                if not np.all(np.isfinite(values)):
                    raise ValueError("nonfinite finite-difference endpoint")
            except Exception as error:
                failure = dict(h=h, sign=sign, type=type(error).__name__, message=str(error))
                save(dict(passed=False, checks=checks, endpoints=endpoints, failure=failure))
                raise RuntimeError(f"finite-difference solve failed at h={h}, sign={sign:+d}: {error}") from error
            pair.append(endpoint)
            endpoints.append(dict(h=h, sign=sign, **endpoint))
        plus, minus = pair
        fd = (plus["value"] - minus["value"]) / (2 * h)
        check = dict(
            h=h,
            analytic=float(analytic),
            finite_difference=float(fd),
            relative_error=float(abs(fd - analytic) / max(abs(analytic), abs(fd), 1e-8)),
        )
        if constraint_analytic is not None:
            fd = (np.asarray(plus["constraints"]) - np.asarray(minus["constraints"])) / (2 * h)
            errors = np.abs(fd - constraint_analytic) / np.maximum(
                np.maximum(np.abs(fd), np.abs(constraint_analytic)), 1e-8
            )
            check.update(
                constraint_analytic=constraint_analytic.tolist(),
                constraint_fd=fd.tolist(),
                constraint_relative_errors=errors.tolist(),
            )
        checks.append(check)
    return checks, endpoints
