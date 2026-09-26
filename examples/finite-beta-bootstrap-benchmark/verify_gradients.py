#!/usr/bin/env python
"""Separate independent endpoint checks; never imported by ordinary production."""

import argparse
import numpy as np
from case import arguments, run, write


def verify(campaign):
    import jax.numpy as jnp

    x = campaign.x.copy()
    y = campaign.y.copy()
    def checked_rows(root, point):
        rows = campaign.rows(root, point)
        if campaign.arm == "fixed":
            rows = jnp.r_[rows, campaign.interface_rows(root, point)]
        return rows

    names = ["objective", "min_abs_iota", "major_radius", "clearance"]
    if campaign.arm == "fixed":
        names += ["normal_field_loss", "pressure_balance_loss"]
    _, jac = campaign.model.linear.derivative(checked_rows, y, x)
    rng = np.random.default_rng(0)
    checks = []
    directions = [rng.normal(size=x.size)]
    if campaign.model.nb:
        boundary = np.zeros_like(x)
        boundary[: campaign.model.nb] = rng.normal(size=campaign.model.nb)
        directions.append(boundary)
        coils = np.zeros_like(x)
        coils[campaign.model.nb :] = rng.normal(size=x.size - campaign.model.nb)
        directions.append(coils)
    for direction in directions:
        direction /= np.linalg.norm(direction)
        exact = jac @ direction
        for h in (1e-4, 5e-5):
            endpoints = []
            for sign in (-1, 1):
                point = x + sign * h * direction
                # Each endpoint starts independently from the same accepted root.
                root = campaign.model.linear.solve(y.copy(), point)
                campaign.model.certify(root, point)
                endpoints.append(np.asarray(checked_rows(root, point)))
            fd = (endpoints[1] - endpoints[0]) / (2 * h)
            error = np.abs(fd - exact) / np.maximum(np.maximum(np.abs(fd), np.abs(exact)), 1e-8)
            checks.append(
                dict(
                    h=h,
                    finite_difference=fd.tolist(),
                    adjoint=exact.tolist(),
                    relative_error=error.tolist(),
                    passed=bool(np.all(np.abs(fd - exact) <= 1e-7 + 1e-3 * np.maximum(np.abs(fd), np.abs(exact)))),
                )
            )
    report = dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        row_names=names,
        optimizer_steps=0,
        contract=__import__("case").contract(),
    )
    write(campaign.out / "gradient_verification.json", report)
    if not report["passed"]:
        raise RuntimeError("independent coupled-gradient check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("fixed", "free"), required=True)
    chosen, remaining = parser.parse_known_args()
    args = arguments(chosen.arm, remaining)
    args.verify_gradients = True
    raise SystemExit(run(chosen.arm, args))
