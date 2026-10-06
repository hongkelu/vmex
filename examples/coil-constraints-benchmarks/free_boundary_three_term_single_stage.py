#!/usr/bin/env python
"""Free-boundary single-stage coil optimization on sheet-current-free equilibria.

The free arm of ``free_boundary_single_stage_optimization.py`` with its
equilibrium replaced: every trial is the free boundary that satisfies all three
plasma-vacuum interface conditions (B.n = 0, pressure balance and no sheet
current, ``vmex.core.freeboundary_vc.ThreeTermFreeBoundaryModel``), not a VMEC + NESTOR
solve, which balances only |B| and leaves the tangential field jump free.  The
design variables (coil shapes, PHIEDGE and, with ``--bootstrap``, the current
spline values and CURTOR), the loss, the constraint rows and their bounds are
that script's.

Each trial starts from the latest solved boundary, after a first-order prediction
of the boundary change, and takes trust-region Gauss-Newton steps with its Jacobian;
each SLSQP gradient linearizes the solved boundary once: the state tangents of
the boundary and plasma coordinates (one block factorization) give the
interface Jacobian and the objective and constraint rows' derivatives, and by
the implicit function theorem of the boundary least squares a row ``h`` has the
design gradient

    dh/dx = dh/dx|_boundary - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

with ``r`` the interface rows, ``J_b`` their boundary Jacobian and ``dr/dx``
through the plasma parameters (state tangents) and the coil field (one reverse
pass at the fixed state).  Nothing recompiles when the coils or plasma
parameters change.

    COIL_CASE=qa4-beta python free_boundary_three_term_single_stage.py --beta 0.025 --bootstrap --output runs/three-term-qa4
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402
from _common import (bootstrap_input, bootstrap_mismatch, finite_beta_input, max_abs_iota,  # noqa: E402
                     min_abs_iota, redl_profiles, resize_coils, restart_input, scale_coil_currents, seed_input,
                     target_residual)
from free_boundary_single_stage_optimization import PHIEDGE_STEP, OPTIMIZER_FTOL, fit_coils_to_plasma  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--coils", type=Path, default=P.COILS_FILE)
    parser.add_argument("--no-coil-fit", action="store_true", help="use --coils as given, without the seed refit")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--ns", type=int, help="radial resolution (default: P.RESOLUTION)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"), help="default: P.RESOLUTION")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--restart", type=Path, help="continue a run from its input.run, coils.json and wout.nc")
    parser.add_argument("--seed-run", type=Path, help="start from another run's input.run and coils.initial.json "
                        "(its calibrated deck and fitted coils), skipping the bootstrap ramp and the coil fit")
    parser.add_argument("--vc-grid", type=int, default=48, help="virtual-casing grid per field period")
    parser.add_argument("--chunk", type=int, default=8, help="Jacobian columns per batch (memory)")
    parser.add_argument("--quadrature", type=int, nargs=2, metavar=("NT", "NP"),
                        help="virtual-casing singular quadrature (default 4 nfp grid x grid: 384 x 48 for nfp 2, the "
                        "4-digit plan of the qa4-beta seed, within 1%% of 4x finer on its optimized boundary)")
    parser.add_argument("--max-iterations", type=int, help="VMEC iteration cap of every solve (default: the deck's)")
    parser.add_argument("--trial-ftol", type=float, default=1e-3,
                        help="relative cost change at which a trial's boundary steps stop")
    parser.add_argument("--trial-forward-ftol", type=float, default=1e-11,
                        help="force residual of the trials' equilibria (each accepted point is re-solved to the "
                        "deck's tolerance before its gradient)")
    parser.add_argument("--trial-verbose", action="store_true", help="print every boundary step of every trial")
    parser.add_argument("--profile-rows", action="store_true",
                        help="time the rows, their gradient and the linearization at the seed (with and without the "
                        "bootstrap row), report each compiled program's memory, then stop")
    parser.add_argument("--check-gradient", type=int, default=0, metavar="N",
                        help="before optimizing, compare the gradients with central differences along N random "
                        "directions, then stop")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    P.RESOLUTION = (*(args.modes or P.RESOLUTION[:2]), args.ns or P.RESOLUTION[2])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import scipy.optimize
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im
    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel
    import _coil_constraints as coil_limits

    started = time.monotonic()
    mpol, ntor, ns = P.RESOLUTION
    inp, redl = seed_input(), None
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.seed_run is not None:
        inp = replace(vj.VmecInput.from_file(str(args.seed_run / "input.run")), lfreeb=False)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
        args.coils = args.seed_run / "coils.initial.json"
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, "gpu")
    else:
        inp, fixed = finite_beta_input(inp, args.beta, "gpu")
    if args.max_iterations:
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    inp.to_indata(out / "input.run")

    coils = resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS)
    if args.restart is None and args.seed_run is None:
        rbtor = abs(float(fixed.wout.rbtor)) if args.beta > 0 else P.B0 * float(inp.rbc[inp.ntor, 0])
        coils = scale_coil_currents(coils, rbtor)
        if args.beta > 0 and not args.no_coil_fit:
            coils = fit_coils_to_plasma(coils, fixed.wout, inp)
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    current = np.r_[np.asarray(inp.ac_aux_f)[: P.CURRENT_KNOTS - 1], inp.curtor] if args.bootstrap else None
    scales = np.r_[[PHIEDGE_STEP] * P.FREE_PHIEDGE, [P.CURRENT_STEP] * (0 if current is None else current.size),
                   P.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=(), scales=scales,
                                          phiedge=float(inp.phiedge) if P.FREE_PHIEDGE else None,
                                          plasma_current=current, plasma_current_spline=True)
    nplasma = chart.nphiedge + chart.nplasma
    qs = target_residual()

    quadrature = args.quadrature or (4 * int(inp.nfp) * args.vc_grid, args.vc_grid)  # quad_nt: a multiple of nfp grid
    model = ThreeTermFreeBoundaryModel(inp, nphi=args.vc_grid, ntheta=args.vc_grid, trial_ftol=args.trial_forward_ftol,
                               chunk=args.chunk, quadrature=quadrature)
    print(f"[model] {model.x0.size} boundary coordinates, {nplasma} plasma coordinates, "
          f"{chart.size - nplasma} coil coordinates; built in {time.monotonic() - started:.0f} s", flush=True)

    # ---- rows of the run: loss residual, then the constraint quantities in the free arm's order --------------------
    mirror = (opt.mirror_ratio,) if P.MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if P.IOTA_CEILING else ()
    plasma_rows = (min_abs_iota, opt.major_radius, *mirror, *ceiling,
                   *([bootstrap_mismatch(inp, redl, "gpu")] if redl is not None else []))

    def boundary_surface(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=coil_limits.SURFACE_GRID[0],
                                              ntheta=coil_limits.SURFACE_GRID[1])

    def quantities(state, params, x):
        """``[QA residuals..., plasma rows..., clearance, aspect]`` at a fixed-boundary state."""
        runtime = im.runtime_from_params(params, model.cfg)
        clearance = coil_limits.surface_distance(chart.coils_from_x(x), boundary_surface(state, runtime))
        return jnp.concatenate([qs.residuals_state(state, runtime),
                                jnp.stack([f(state, runtime) for f in plasma_rows]
                                          + [clearance, opt.aspect_ratio(state, runtime)])])

    nq = int(np.asarray(qs.residuals_state(model.seed[0], im.runtime_from_params(model.params0, model.cfg))).size)
    nrows = len(plasma_rows) + 2
    quantities_jit = jax.jit(quantities)
    # The loss and constraint rows are few scalars of expensive functions (QI residual, DKX): their gradients
    # by reverse mode through one shared block factorization, then onto the boundary and plasma coordinates.
    quantity_pullback = jax.jit(lambda state, mask, params, cots, x: model.pullback(
        quantities, state, mask, params, cots, x))

    @jax.jit
    def to_coordinates(params, x_b, x, params_bar):
        _, boundary_vjp = jax.vjp(lambda z: model.with_boundary(params, z), x_b)
        _, plasma_vjp = jax.vjp(lambda z: chart.plasma_params_at(params, z), x)
        return jax.vmap(lambda bar: jnp.concatenate([boundary_vjp(bar)[0], plasma_vjp(bar)[0][:nplasma]]))(
            params_bar)
    coil_gradient = jax.jit(lambda state, params, x, cot: jax.vjp(lambda xx: quantities(state, params, xx), x)[1](cot)[0])
    rows_coil_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(lambda xx: model.rows(state, params, chart(xx)), x)[1](
        lam)[0])
    rows_coil_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(lambda xx: model.rows(state, params, chart(xx)), (x,),
                                                                  (dx,))[1])

    def plasma_params(x):
        return chart.plasma_params_at(model.params0, jnp.asarray(x))

    plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
        lambda xx: chart.plasma_params_at(params, xx), (x,), (e,))[1])(jnp.eye(chart.size)[:nplasma]))

    # ---- the free-boundary solve of a design point, cached ----------------------------------------------------------
    cache = {}
    anchor = {}  # the latest linearized solution: boundary, Jacobian (boundary and plasma columns), design point
    counters = dict(trials=0, failed=0, lm_evaluations=0, linearizations=0, solve_seconds=0.0, linearize_seconds=0.0)

    def solve(x):
        key = np.asarray(x, dtype=float).tobytes()
        if key in cache:
            return cache[key]
        t = time.monotonic()
        counters["trials"] += 1
        x = np.asarray(x, dtype=float)
        params = plasma_params(x)
        field = chart(jnp.asarray(x))
        x_b, J = None, None
        if anchor:
            # First-order boundary prediction from the anchor's linearization.
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[:nplasma] + np.asarray(rows_coil_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            x_b = anchor["x_b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0]
            J = anchor["J_b"]
        result = None
        # From the predicted boundary, then (if its equilibrium fails) from the anchor's; a trial far above the
        # last floor (a large coil change) goes on with fresh Jacobians.
        for start in ([x_b, anchor["x_b"]] if anchor else [None]):
            try:
                result = model.solve_boundary(params, field, x0=start, jacobian=J, ftol=args.trial_ftol, max_nfev=40,
                                              target_cost=4 * anchor["cost"] if anchor else None,
                                              verbose=int(args.trial_verbose))
                break
            except (vj.VmecError, RuntimeError) as error:  # an uncertified equilibrium ends this SLSQP call
                print(f"[trial] failed: {str(error)[:300]}", flush=True)
        counters["solve_seconds"] += time.monotonic() - t
        if result is None:
            counters["failed"] += 1
            cache[key] = None
            return None
        counters["lm_evaluations"] += result["nfev"]
        state, mask, params_x, _ = result["aux"]
        values = np.asarray(quantities_jit(state, params_x, jnp.asarray(x)))
        cache[key] = dict(result, values=values, state=state, mask=mask, params=params_x)
        return cache[key]

    def linearize(x):
        """Design gradients of the loss and every row at ``x``'s solution (implicit function theorem)."""
        sol = solve(x)
        if "gradients" in sol:
            return sol["gradients"]
        t = time.monotonic()
        counters["linearizations"] += 1
        state, params_x = sol["state"], sol["params"]
        extra = plasma_directions(params_x, jnp.asarray(x))
        field = chart(jnp.asarray(x))
        nb = model.x0.size

        def tight_linearization():
            """Linearize at the solution, re-solved to the deck's tolerance; its values come from that state."""
            J, dz, params_batch, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra)
            state, mask, params_x, _ = aux
            sol.update(aux=aux, state=state, mask=mask, params=params_x,
                       rows=np.asarray(model._rows(state, params_x, field)),
                       values=np.asarray(quantities_jit(state, params_x, jnp.asarray(x))))
            return J, dz, params_batch

        J, dz, params_batch = tight_linearization()
        # Trials stop early on a Jacobian from an earlier point; polish this one onto the floor with its own
        # Jacobian, and linearize again where the polish went (if it gained), so the gradient belongs to it.
        cost = 0.5 * sol["rows"] @ sol["rows"]
        polished = model.solve_boundary(plasma_params(x), field, x0=sol["x"], jacobian=J[:, :nb], ftol=1e-4,
                                        max_nfev=20)
        counters["lm_evaluations"] += polished["nfev"]
        if 0.5 * polished["rows"] @ polished["rows"] < 0.9 * cost:
            sol.update(x=polished["x"], aux=polished["aux"])
            J, dz, params_batch = tight_linearization()
            counters["linearizations"] += 1
        state, params_x = sol["state"], sol["params"]
        J_b, J_p = J[:, :nb], J[:, nb:]
        values = sol["values"]
        q = values[:nq]
        # loss 0.5 |q|^2, then each row: (1 + nrows) x (nb + np) fixed-boundary derivatives
        cotangents = jnp.asarray(np.vstack([np.r_[q, np.zeros(nrows)], np.c_[np.zeros((nrows, nq)), np.eye(nrows)]]))
        params_bar = quantity_pullback(state, sol["mask"], params_x, cotangents, jnp.asarray(x))
        G = np.asarray(to_coordinates(params_x, jnp.asarray(sol["x"]), jnp.asarray(x), params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G[:, :nb].T)  # residual-space multipliers, one column per row
        gradients = np.zeros((1 + nrows, chart.size))
        for i in range(1 + nrows):
            cot_i = jnp.asarray(np.r_[q if i == 0 else np.zeros(nq), np.eye(nrows)[i - 1] if i else np.zeros(nrows)])
            direct = np.asarray(coil_gradient(state, params_x, jnp.asarray(x), cot_i))
            through_rows = np.asarray(rows_coil_vjp(state, params_x, jnp.asarray(x), jnp.asarray(lam[:, i])))
            gradients[i] = direct - through_rows
            gradients[i, :nplasma] += G[i, nb:] - lam[:, i] @ J_p
        anchor.update(x=np.asarray(x, dtype=float).copy(), x_b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                      params=params_x, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        counters["linearize_seconds"] += time.monotonic() - t
        return gradients

    # ---- SLSQP problem ------------------------------------------------------------------------------------------------
    aspect_lower, aspect_upper = P.ASPECT_RANGE
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN
    nredl = int(redl is not None)
    bounds = [(P.IOTA_FLOOR + P.IOTA_MARGIN, np.inf, P.IOTA_FLOOR),
              (P.RADIUS_TARGET - width, P.RADIUS_TARGET + width, P.RADIUS_TOLERANCE),
              *([(-np.inf, P.MIRROR_LIMIT - P.MIRROR_MARGIN, P.MIRROR_LIMIT)] if mirror else []),
              *([(-np.inf, P.IOTA_CEILING - P.IOTA_MARGIN, P.IOTA_FLOOR)] if ceiling else []),
              *[(-np.inf, P.REDL_TOLERANCE, P.REDL_TOLERANCE)] * nredl,
              (P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN, np.inf, P.COIL_SURFACE_DISTANCE_LIMIT),
              (aspect_lower, aspect_upper, 0.5 * (aspect_upper - aspect_lower))]
    lower, upper, row_scale = map(np.asarray, zip(*bounds))
    has_lower, has_upper = np.isfinite(lower), np.isfinite(upper)
    u_scale = chart.scales  # optimizer coordinates u = x / scales

    def failed():
        raise _TrialRejected

    def fun(u):
        sol = solve(u * u_scale)
        if sol is None:
            failed()
        q = sol["values"][:nq]
        return 0.5 * float(q @ q)

    def jac(u):
        if solve(u * u_scale) is None:
            failed()
        return linearize(u * u_scale)[0] * u_scale

    def ineq(u):
        sol = solve(u * u_scale)
        if sol is None:
            failed()
        h = sol["values"][nq:]
        return np.r_[((h - lower) / row_scale)[has_lower], ((upper - h) / row_scale)[has_upper]]

    def ineq_jac(u):
        if solve(u * u_scale) is None:
            failed()
        g = linearize(u * u_scale)[1:] * u_scale
        return np.vstack([(g / row_scale[:, None])[has_lower], (-g / row_scale[:, None])[has_upper]])

    coil_rows = coil_limits.constraint(chart.coils_from_x)
    constraints = [dict(type="ineq", fun=ineq, jac=ineq_jac),
                   dict(type="ineq", fun=lambda u: coil_rows.fun(u * u_scale),
                        jac=lambda u: coil_rows.jac(u * u_scale) * u_scale)]

    # ---- logging -----------------------------------------------------------------------------------------------------
    state = dict(step=0, time=time.monotonic(), x=chart.x0.copy())
    solver_seen = {}

    def save(tag, x):
        sol = solve(x)
        chart.coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        from vmex.core.optimize import solve_equilibrium
        deck = im.input_with_params(model.fixed, sol["params"])
        vj.write_wout(str(out / f"wout{tag}.nc"), solve_equilibrium(deck, initial_state=sol["state"]).wout)

    def log_step(x):
        sol = solve(x)
        if sol is None:
            raise RuntimeError("the accepted point has no certified free-boundary solution")
        h = sol["values"][nq:]
        q = sol["values"][:nq]
        res = model.boundary_residual(sol["state"], sol["params"], chart(jnp.asarray(x)))
        now = time.monotonic()
        # forward solves of the trials and of the tight re-solves: count, VMEC iterations and seconds by part
        solver = {}
        for tag, cfg in (("trial", model.cfg_trial), ("tight", model.cfg)):
            stats = dict(im._SOLVE_STATS.get(cfg, {}))
            last = solver_seen.get(tag, {})
            solver[tag] = {k: round(v - last.get(k, 0), 2) for k, v in stats.items()
                           if isinstance(v, (int, float)) and v - last.get(k, 0)}
            solver_seen[tag] = stats
            if model.cfg_trial is model.cfg:
                break
        row = dict(step=state["step"], qa=float(q @ q), min_abs_iota=float(h[0]), major_radius_m=float(h[1]),
                   **({"redl_mismatch": float(h[2 + len(mirror) + len(ceiling)])} if nredl else {}),
                   coil_surface_distance_m=float(h[-2]), aspect=float(h[-1]),
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   phiedge_factor=float(chart.phiedge_at(jnp.asarray(x)) / chart.phiedge) if P.FREE_PHIEDGE else 1.0,
                   bn=res.normal, pressure_balance=res.pressure, sheet_current=res.sheet_current,
                   step_seconds=now - state["time"], elapsed_seconds=now - started, **counters, solver=solver,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        for k in ("trials", "failed", "lm_evaluations", "linearizations"):
            counters[k] = 0
        counters["solve_seconds"] = counters["linearize_seconds"] = 0.0
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"R={row['major_radius_m']:.5f} aspect={row['aspect']:.4f} clearance={row['coil_surface_distance_m']:.4f} "
              f"coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              + (f"redl={row['redl_mismatch']:.2e} " if nredl else "")
              + f"B.n={res.normal:.1e} K={res.sheet_current:.1e} {row['step_seconds']:.1f}s", flush=True)
        print(f"[timing] trials {row['trials']} ({row['solve_seconds']:.0f} s, {row['lm_evaluations']} boundary "
              f"evaluations), linearizations {row['linearizations']} ({row['linearize_seconds']:.0f} s), "
              f"peak {row['peak_gpu_gib']:.1f} GiB; forward solves {json.dumps(solver)}", flush=True)
        if state["step"] % args.save_every == 0:
            save(f".step{state['step']}", x)
        state.update(step=state["step"] + 1, time=now, x=np.asarray(x).copy())
        if len(cache) > 64:  # keep the accepted point only
            keep = np.asarray(x, dtype=float).tobytes()
            for k in [k for k in cache if k != keep]:
                del cache[k]

    log_step(chart.x0)
    u = chart.x0 / u_scale
    if args.profile_rows:
        sol = solve(chart.x0)
        state, mask, params_x, x = sol["state"], sol["mask"], sol["params"], jnp.asarray(chart.x0)
        no_bootstrap = [f for f in plasma_rows if f is not plasma_rows[-1]] if redl is not None else list(plasma_rows)

        def without(state, params, x):
            runtime = im.runtime_from_params(params, model.cfg)
            return jnp.concatenate([qs.residuals_state(state, runtime),
                                    jnp.stack([f(state, runtime) for f in no_bootstrap])])

        def timed(name, fn, repeat=3):
            jax.block_until_ready(fn())
            t = time.perf_counter()
            for _ in range(repeat):
                jax.block_until_ready(fn())
            print(f"[profile] {name}: {(time.perf_counter() - t) / repeat:.2f} s", flush=True)

        n_all, n_without = nq + nrows, nq + len(no_bootstrap)
        cot_all = jnp.asarray(np.eye(n_all)[[0] + list(range(nq, n_all))])
        cot_without = jnp.asarray(np.eye(n_without)[[0] + list(range(nq, n_without))])
        pull_without = jax.jit(lambda s_, m_, p_, c_: model.pullback(lambda a, b: without(a, b, x), s_, m_, p_, c_))
        timed("rows (all)", lambda: quantities_jit(state, params_x, x))
        timed("rows (no bootstrap row)", lambda: jax.jit(without)(state, params_x, x))
        timed(f"reverse gradient of {cot_all.shape[0]} rows (all)",
              lambda: quantity_pullback(state, mask, params_x, cot_all, x), repeat=1)
        timed(f"reverse gradient of {cot_without.shape[0]} rows (no bootstrap)",
              lambda: pull_without(state, mask, params_x, cot_without), repeat=1)
        extra = plasma_directions(params_x, x)
        timed("interface linearization (boundary + plasma columns)",
              lambda: model.linearize(sol["x"], sol["aux"], chart(x), extra=extra)[0], repeat=1)

        def memory(name, fn, *a):
            m = fn.lower(*a).compile().memory_analysis()
            print(f"[memory] {name}: temp {m.temp_size_in_bytes / 2**30:.2f} GiB, arguments "
                  f"{m.argument_size_in_bytes / 2**30:.2f} GiB, output {m.output_size_in_bytes / 2**30:.2f} GiB",
                  flush=True)

        field = chart(x)
        batch = model.boundary_directions(params_x, sol["x"])
        dz = model._tangents(params_x, state, mask, batch)
        memory("interface rows", model._rows, state, params_x, field)
        memory("quantities (QI, plasma rows, clearance, aspect)", quantities_jit, state, params_x, x)
        memory(f"state tangents ({model.x0.size} boundary directions)", model._tangents, params_x, state, mask, batch)
        memory("rows along the tangents", model._push_rows, state, mask, params_x, dz, batch, field)
        memory(f"reverse gradient of {cot_all.shape[0]} rows", quantity_pullback, state, mask, params_x, cot_all, x)
        memory("coil gradient of a row", coil_gradient, state, params_x, x, cot_all[0])
        memory("interface rows' coil pullback", rows_coil_vjp, state, params_x, x,
               jnp.zeros(np.asarray(sol["rows"]).size))
        print(f"[memory] peak so far {(jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use', 0) / 2**30:.2f} "
              "GiB", flush=True)
        return
    if args.check_gradient:
        # Directional derivatives of the loss and rows: gradient . d against central differences of
        # re-solved boundaries (each trial tight, from the seed solution's linearization).
        args.trial_ftol = 1e-8
        g0 = linearize(chart.x0) * u_scale  # loss and rows, per optimizer coordinate
        rng = np.random.default_rng(0)
        for k in range(args.check_gradient):
            d = rng.normal(size=u.size)
            d /= np.linalg.norm(d)
            for h in (1e-2, 3e-3):
                plus, minus = (solve((u + s * h * d) * u_scale) for s in (1, -1))
                values = [np.r_[0.5 * sol["values"][:nq] @ sol["values"][:nq], sol["values"][nq:]]
                          for sol in (plus, minus)]
                fd = (values[0] - values[1]) / (2 * h)
                exact = g0 @ d
                print(f"[gradient check] direction {k} h {h:g}: loss {exact[0]:+.6e} vs {fd[0]:+.6e}; rows rel. err "
                      + " ".join(f"{abs(e - f) / max(abs(f), 1e-14):.1e}" for e, f in zip(exact[1:], fd[1:])),
                      flush=True)
        return
    stop = dict(message="", success=False)
    while state["step"] <= args.steps:
        before = state["step"]
        try:
            result = scipy.optimize.minimize(fun, u, jac=jac, method="SLSQP", constraints=constraints,
                                             callback=lambda uk: log_step(uk * u_scale),
                                             options=dict(maxiter=args.steps + 1 - before, ftol=OPTIMIZER_FTOL))
            stop = dict(message=str(result.message), success=bool(result.success))
            break
        except _TrialRejected:
            print("[slsqp] trial rejected: restarting from the accepted point", flush=True)
            u = state["x"] / u_scale
            if state["step"] == before:
                stop = dict(message="trial rejected before any progress", success=False)
                break
    save("", state["x"])
    summary = dict(accepted_steps=state["step"] - 1, **stop, elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


class _TrialRejected(Exception):
    """A trial point without a certified free-boundary solution."""


if __name__ == "__main__":
    main()
