#!/usr/bin/env python
"""Fresh constrained coil geometry for an authenticated finite-beta reference.

The plasma equilibrium/profiles and coil currents are held fixed. Start with
native ESSOS circles, fit the virtual-casing normal field with the same hard
engineering inequalities, and export only the best sampled-feasible candidate.
This is stage two preparation; it does not run single-stage optimization.
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np


def normal_on_fixed_interface(coils, vc):
    """Total normal field for a fixed plasma target; currents stay in the graph."""
    import jax
    import jax.numpy as jnp
    from essos.fields import BiotSavart

    field = BiotSavart(coils)
    external = jax.vmap(field.B)(jnp.moveaxis(vc.gamma, 0, -1).reshape(-1, 3))
    total = vc.B_plasma + jnp.moveaxis(external.reshape((*vc.gamma.shape[1:], 3)), -1, 0)
    return jnp.sum(total * vc.normal, axis=0) / jnp.linalg.norm(total, axis=0)


def spectral_scales(mode, count, step):
    """Metre-valued mode scaling, not a geometry penalty or a coil constraint."""
    return np.tile([step] + [step / k**1.2 for k in range(1, mode + 1) for _ in range(2)], 3 * count)


def choose_feasible(records):
    candidates = [
        r
        for r in records
        if np.isfinite(r["objective"]) and np.isfinite(r["minimum_scaled_slack"]) and r["minimum_scaled_slack"] >= -1e-8
    ]
    if not candidates:
        raise RuntimeError("no sampled-feasible coil candidate")
    return min(candidates, key=lambda r: r["objective"])


def feasible_segment(anchor, proposal, inequalities):
    """Recover a checked feasible coil on a segment; never waive a constraint."""
    anchor, proposal = np.asarray(anchor), np.asarray(proposal)

    def valid(x):
        rows = np.asarray(inequalities(x))
        return bool(np.all(np.isfinite(rows)) and np.min(rows) >= -1e-8)

    if not valid(anchor):
        raise RuntimeError("feasibility recovery needs a valid anchor")
    if valid(proposal):
        return proposal.copy()
    low, high = 0.0, 1.0
    for _ in range(18):
        mid = (low + high) / 2
        if valid(anchor + mid * (proposal - anchor)):
            low = mid
        else:
            high = mid
    return anchor + low * (proposal - anchor)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--allow-prepared-code-update", action="store_true")
    a = p.parse_args(argv)
    if a.iterations < 1:
        p.error("iterations must be positive")
    return a


def run(args):
    from case import setup, load_prepared, contract, write, sha, normal_cost

    out = setup(args)
    started = time.monotonic()
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.virtual_casing import plan_vc_precision
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize, NonlinearConstraint
    from main_reference import coil_modules
    from physics import EquilibriumCurrentRoot
    import bootstrap_settings as S

    def event(**data):
        row = dict(elapsed_seconds=time.monotonic() - started, **data)
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        inp, old, seed, profiles = load_prepared(args.prepared, allow_code_update=args.allow_prepared_code_update)
        limits, _ = coil_modules()
        model = EquilibriumCurrentRoot(inp, old, seed, profiles, "fixed", event=event)
        y = model.linear.solve(model.initial_y, model.x0)
        certificate = model.certify(y, model.x0)
        state, rt, _, _ = model.objects(jnp.asarray(y), jnp.asarray(model.x0))
        inp = model.input_at(y, model.x0)
        # The target is fixed during a coil-only fit. Compute virtual casing
        # once, without freezing it during later single-stage optimization.
        data = vj.surface_field_data_from_state(
            inp, state, runtime=rt, nphi=S.INTERFACE_GRID[0], ntheta=S.INTERFACE_GRID[1]
        )
        vc = vj.PlasmaVacuumInterface.from_surface_data(
            data, precision=plan_vc_precision(data, digits=S.VC_DIGITS), digits=S.VC_DIGITS
        )
        rbc, zbs, _, _ = opt.boundary_from_state(state, rt)
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=limits.SURFACE_GRID[0], ntheta=limits.SURFACE_GRID[1]
        )

        def normal(coils):
            return normal_on_fixed_interface(coils, vc)

        def inequalities(coils):
            return jnp.r_[
                limits.coil_inequalities(coils),
                (limits.surface_distance(coils, surface) - S.P.COIL_SURFACE_DISTANCE_LIMIT - S.P.DISTANCE_MARGIN)
                / S.P.COIL_SURFACE_DISTANCE_LIMIT,
            ]

        stats = jax.jit(
            lambda c: jnp.array(
                [
                    normal_cost(normal(c), vc.weights),
                    jnp.sqrt(jnp.sum(vc.weights * normal(c) ** 2)),
                    jnp.max(jnp.abs(normal(c))),
                    jnp.min(inequalities(c)),
                ]
            )
        )

        def metrics(c):
            v = np.asarray(stats(c))
            if not np.all(np.isfinite(v)):
                raise RuntimeError("nonfinite coil metrics")
            return dict(
                zip(["objective", "normal_field_rms", "normal_field_max", "minimum_scaled_slack"], map(float, v))
            )

        old_metrics = metrics(old)
        write(out / "previous_coils.json", old_metrics)
        candidates = []
        for radius in (0.5, 0.6, 0.65):
            curves = CreateEquallySpacedCurves(
                S.P.N_COILS,
                S.P.COIL_ORDER,
                S.P.RADIUS_TARGET,
                radius,
                n_segments=S.P.N_SEGMENTS,
                nfp=inp.nfp,
                stellsym=True,
            )
            c = Coils(curves, old.dofs_currents_raw)
            path = out / f"circles-{radius:.2f}.json"
            c.to_json(str(path))
            row = dict(radius_m=radius, path=path.name, **metrics(c))
            candidates.append(row)
            event(event="circle_candidate", **row)
        chosen = choose_feasible(candidates)
        fitted = Coils.from_json(str(out / chosen["path"]))
        fitted.to_json(str(out / "coils.initial.json"))
        best_metrics = metrics(fitted)
        stage_reports = []
        for mode in (S.P.COIL_ORDER,):
            chart = opt.CoilParameters.from_coils(
                fitted, current_dofs=(), max_coil_mode=mode, scales=spectral_scales(mode, S.P.N_COILS, S.P.COIL_STEP)
            )
            scales = jnp.asarray(chart.scales)

            def coils_at(u):
                return chart.coils_from_x(u * scales)

            value = jax.jit(jax.value_and_grad(lambda u: normal_cost(normal(coils_at(u)), vc.weights)))
            cons = jax.jit(lambda u: inequalities(coils_at(u)))
            cons_jac = jax.jit(jax.jacrev(cons))
            best = [dict(x=np.zeros(chart.size), **best_metrics)]
            iteration = [0]

            def consider(u):
                cost = float(value(jnp.asarray(u))[0])
                slack = float(jnp.min(cons(jnp.asarray(u))))
                row = dict(objective=cost, minimum_scaled_slack=slack)
                candidate = np.asarray(u)
                if np.isfinite(cost) and cost < best[0]["objective"] and slack < -1e-8:
                    candidate = feasible_segment(best[0]["x"], candidate, cons)
                    cost = float(value(jnp.asarray(candidate))[0])
                    slack = float(jnp.min(cons(jnp.asarray(candidate))))
                    row.update(recovered_objective=cost, recovered_minimum_scaled_slack=slack)
                if np.isfinite(cost) and np.isfinite(slack) and slack >= -1e-8 and cost < best[0]["objective"]:
                    best[0] = dict(x=np.array(candidate, copy=True), objective=cost, minimum_scaled_slack=slack)
                    coils_at(jnp.asarray(candidate)).to_json(str(out / f"coils.best-mode-{mode:02d}.json"))
                return row

            def callback(u):
                iteration[0] += 1
                event(event="coil_fit_step", mode=mode, iteration=iteration[0], **consider(u))

            def objective(u):
                f, g = value(jnp.asarray(u))
                return float(f), np.asarray(g)

            result = minimize(
                objective,
                np.zeros(chart.size),
                jac=True,
                method="SLSQP",
                constraints=[
                    NonlinearConstraint(lambda u: np.asarray(cons(u)), 0, np.inf, jac=lambda u: np.asarray(cons_jac(u)))
                ],
                callback=callback,
                options=dict(maxiter=args.iterations, ftol=S.P.OPTIMIZER_FTOL),
            )
            consider(result.x)
            fitted = coils_at(jnp.asarray(best[0]["x"]))
            best_metrics = metrics(fitted)
            fitted.to_json(str(out / f"coils.stage-{mode:02d}.json"))
            stage_reports.append(
                dict(
                    mode=mode,
                    success=bool(result.success),
                    message=str(result.message),
                    iterations=int(result.nit),
                    **best_metrics,
                )
            )
            event(event="coil_fit_stage_done", **stage_reports[-1])
            write(out / "fit_stages.json", stage_reports)
        fitted.to_json(str(out / "coils.json"))
        np.testing.assert_array_equal(np.asarray(fitted.dofs_currents_raw), np.asarray(old.dofs_currents_raw))
        fine_surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=limits.VERIFY_SURFACE_GRID[0], ntheta=limits.VERIFY_SURFACE_GRID[1]
        )
        geometry = limits.verify(fitted, fine_surface, path=out / "coil_verification.json")
        from case import normal_values, interface

        bn, weights = normal_values(
            inp, state, rt, fitted, interface(inp, state, rt, S.VERIFY_INTERFACE_GRID), S.VERIFY_INTERFACE_GRID
        )
        fine_normal = dict(
            normal_field_rms=float(jnp.sqrt(jnp.sum(weights * bn**2))), normal_field_max=float(jnp.max(jnp.abs(bn)))
        )
        ready = bool(
            geometry["all_reported_checks_pass"]
            and fine_normal["normal_field_max"] <= S.NORMAL_FIELD_LIMIT
            and best_metrics["minimum_scaled_slack"] >= -1e-8
        )
        report = dict(
            previous=old_metrics,
            new=best_metrics,
            refined=fine_normal,
            geometry_checks_pass=geometry["all_reported_checks_pass"],
            ready_for_single_stage=ready,
            coil_currents_unchanged=True,
            stages=stage_reports,
            initial_radius_m=chosen["radius_m"],
        )
        write(out / "coil_fit.json", report)
        inp.to_indata(out / "input.fixed")
        np.savez_compressed(out / "state.npz", **{n: np.asarray(getattr(state, n)) for n in state.__dataclass_fields__})
        profile = json.loads((args.prepared / "profiles.json").read_text())
        write(out / "profiles.json", profile)
        write(
            out / "lineage.json",
            dict(
                parent=str(args.prepared.resolve()),
                parent_manifest_sha256=sha(args.prepared / "prepared.json"),
                fresh_circle_geometry=True,
                physics_recalibrated=False,
                coil_currents_unchanged=True,
                accepted_reference_recertified=True,
            ),
        )
        if ready:
            write(
                out / "prepared.json",
                dict(
                    schema="finite-beta-redl-prepared/v1",
                    contract=contract(),
                    certificate=certificate,
                    artifacts={n: sha(out / n) for n in ("input.fixed", "state.npz", "profiles.json", "coils.json")},
                    independent_derivative_qualified=False,
                    initial_coils_feasible=True,
                ),
            )
        event(event="coil_fit_finished", **report)
        return 0 if ready else 2
    except Exception as error:
        write(out / "failure.json", dict(error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
