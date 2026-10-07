"""The fixed-boundary single stage with the bootstrap current solved in every equilibrium.

``single_stage_optimization.py --bootstrap`` with ``P.BOOTSTRAP_IN_SOLVE`` (Redl) runs this instead of
its ``VmecProblem`` path.  The design variables are the boundary modes up to ``MAX_MODE`` (RBC(0,0)
fixed), PHIEDGE and the coil shapes, as there; the current is no design variable and no constraint
row: every trial solves the fixed-boundary equilibrium together with a current that is Redl's on every
half-grid surface (``ThreeTermFreeBoundaryModel(fixed_boundary=True, bootstrap=)``, ``I'(0) = 0``).
The objective and the other rows are that script's.  A row ``h`` then has the design gradient

    dh/dp = dh/dp|_current - lambda^T dR/dp,   lambda = J_c (J_c^T J_c)^-1 dh/dc,

with ``R`` the current's self-consistency rows, ``J_c`` their Jacobian in the current values and
``dR/dp`` along the boundary and PHIEDGE (state tangents); coil-only rows are differentiated directly.
"""

import json
import time
from dataclasses import replace

import parameters as P
from _common import (FIELD_STRENGTH_TOLERANCE, NORMAL_FIELD_CONSTRAINT, boundary_diagnostics, coil_field,
                     max_abs_iota, min_abs_iota, redl_profiles, target_residual, weighted_rms)


def run(args, inp, coils0, out, *, max_mode, ess_alpha, boundary_step, coil_step, normal_field_weight,
        optimizer_ftol, vc_digits, nphi, ntheta, started):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im
    from vmex.core import virtual_casing as vc
    from vmex.core.fields import surface_currents
    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel
    from vmex.core.optimize import _ess_scale, boundary_arrays_from_x, pack_boundary, solve_equilibrium
    from vmex.core.statephysics import _field_chain
    import _coil_constraints as coil_limits

    checking = bool(getattr(args, "check_gradient", 0))  # central differences need every trial solved tight
    model = ThreeTermFreeBoundaryModel(inp, bootstrap=redl_profiles(inp)[0], bootstrap_helicity=P.REDL_HELICITY,
                                       fixed_boundary=True,
                                       chunk=args.chunk, trial_ftol=None if checking else 1e-11)
    current_ftol = 1e-14 if checking else 1e-8
    fixed, nc = model.fixed, model.x0.size
    vary_phiedge = P.FREE_PHIEDGE
    phiedge0 = float(fixed.phiedge)
    xb0 = pack_boundary(fixed, max_mode)
    nb = xb0.size
    npl = nb + int(vary_phiedge)                    # plasma design coordinates: boundary, then PHIEDGE
    x_coils0 = np.asarray(coils0.curves.dofs).ravel()
    x0 = np.concatenate([xb0, np.zeros(int(vary_phiedge)), x_coils0])
    scales = np.concatenate([boundary_step * _ess_scale(fixed, max_mode, ess_alpha), [0.05] * int(vary_phiedge),
                             np.full(x_coils0.size, coil_step)])
    print(f"[model] {nc} bootstrap-current coordinates, {npl} plasma and {x_coils0.size} coil design coordinates",
          flush=True)

    def plasma_params(params, x):
        rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], max_mode)
        params = replace(params, rbc=rbc, zbs=zbs)
        return replace(params, phiedge=phiedge0 * (1.0 + x[nb])) if vary_phiedge else params

    def coils_from_x(x):
        return coils0.with_dofs(jnp.concatenate((x[npl:], coils0.dofs_currents)))

    def surface_from_x(x, n_phi=nphi, n_theta=ntheta):
        rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], max_mode)
        return surfacerzfourier_from_boundary(rbc, zbs, fixed.nfp, nphi=n_phi, ntheta=n_theta)

    state0, _ = model.seed
    runtime0 = im.runtime_from_params(model.params0, model.cfg)
    # A fixed singular quadrature (4 nfp nphi x 2 ntheta, the three-term arm's default): the planner's error estimate
    # alone can exhaust a GPU on an optimized boundary.
    precision = vc.plan_vc_precision(vc.surface_field_data_from_state(fixed, state0, runtime=runtime0, nphi=nphi,
                                                                      ntheta=ntheta), digits=vc_digits,
                                     quad_nt=4 * int(fixed.nfp) * nphi, quad_np=2 * ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    loop = float(fixed.rbc[fixed.ntor, 0]) * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    b_phi = np.sum(np.asarray(coil_field(coils0)(jnp.asarray(loop))) * np.stack(
        [-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1), axis=-1)
    linked_rbtor = abs(float(fixed.rbc[fixed.ntor, 0]) * float(np.mean(b_phi)))

    qs = target_residual()
    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    def total_normal_field_rms(coils, state, runtime):
        data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=vc_digits, precision=precision)
        normal = interface.bnormal_residual(coil_field(coils)) / jnp.linalg.norm(data.B_total, axis=0)
        return weighted_rms(interface.weights, normal)

    def rbtor_ratio(state, runtime):
        fields = _field_chain(state, runtime)[3]
        currents = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=runtime.trig,
                                    s=jnp.asarray(runtime.setup.s_full), signgs=runtime.setup.signgs)
        return jnp.abs(currents.rbtor) / linked_rbtor

    def quantities(state, params, x):
        """``[QA residuals..., B.n rms, plasma rows...]`` (rows scaled so c >= 0 is feasible)."""
        runtime = im.runtime_from_params(params, model.cfg)
        coils = coils_from_x(x)
        iota, aspect, radius = (min_abs_iota(state, runtime), opt.aspect_ratio(state, runtime),
                                opt.major_radius(state, runtime))
        nf = total_normal_field_rms(coils, state, runtime)
        strength = rbtor_ratio(state, runtime) - 1.0
        rows = [(iota - P.IOTA_FLOOR - P.IOTA_MARGIN) / P.IOTA_FLOOR,
                (aspect - aspect_lower) / aspect_scale, (aspect_upper - aspect) / aspect_scale,
                (radius - P.RADIUS_TARGET + width) / P.RADIUS_TOLERANCE,
                (P.RADIUS_TARGET + width - radius) / P.RADIUS_TOLERANCE]
        if P.MIRROR_LIMIT:
            rows.append((P.MIRROR_LIMIT - P.MIRROR_MARGIN - opt.mirror_ratio(state, runtime)) / P.MIRROR_LIMIT)
        if P.IOTA_CEILING:
            rows.append((P.IOTA_CEILING - P.IOTA_MARGIN - max_abs_iota(state, runtime)) / P.IOTA_FLOOR)
        rows += [1.0 - nf / NORMAL_FIELD_CONSTRAINT, (FIELD_STRENGTH_TOLERANCE - strength) / FIELD_STRENGTH_TOLERANCE,
                 (FIELD_STRENGTH_TOLERANCE + strength) / FIELD_STRENGTH_TOLERANCE]
        return jnp.concatenate([qs.residuals_state(state, runtime), jnp.stack([nf] + rows)])

    nq = int(np.asarray(qs.residuals_state(state0, runtime0)).size)
    nrows = 5 + bool(P.MIRROR_LIMIT) + bool(P.IOTA_CEILING) + 3
    quantities_jit = jax.jit(quantities)
    quantity_pullback = jax.jit(lambda state, mask, params, cots, x: model.pullback(
        quantities, state, mask, params, cots, x))
    direct_gradient = jax.jit(lambda state, params, x, cot: jax.vjp(lambda xx: quantities(state, params, xx), x)[1](
        cot)[0])

    @jax.jit
    def to_coordinates(params, v, x, params_bar):
        _, current_vjp = jax.vjp(lambda z: model.with_boundary(params, z), v)
        _, plasma_vjp = jax.vjp(lambda z: plasma_params(params, z), x)
        return jax.vmap(lambda bar: jnp.concatenate([current_vjp(bar)[0], plasma_vjp(bar)[0][:npl]]))(params_bar)

    plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
        lambda xx: plasma_params(params, xx), (x,), (e,))[1])(jnp.eye(x.size)[:npl]))

    def coil_rows(x):
        coils, surface = coils_from_x(x), surface_from_x(x)
        clearance = coil_limits.surface_distance(coils, surface)
        return jnp.concatenate([coil_limits.coil_inequalities(coils), jnp.atleast_1d(
            (clearance - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN) / P.COIL_SURFACE_DISTANCE_LIMIT)])

    coil_rows_jit, coil_rows_jac = jax.jit(coil_rows), jax.jit(jax.jacrev(coil_rows))

    cache, anchor = {}, {}
    counters = dict(trials=0, failed=0, evaluations=0, linearizations=0)

    def solve(x):
        key = np.asarray(x, dtype=float).tobytes()
        if key in cache:
            return cache[key]
        counters["trials"] += 1
        params = plasma_params(model.params0, jnp.asarray(x))
        try:
            result = model.solve_boundary(params, None, x0=anchor.get("v"), jacobian=anchor.get("J_c"),
                                          ftol=current_ftol, max_nfev=30)
        except (vj.VmecError, RuntimeError) as error:
            print(f"[trial] failed: {str(error)[:300]}", flush=True)
            counters["failed"] += 1
            cache[key] = None
            return None
        counters["evaluations"] += result["nfev"]
        state, mask, params_x, _ = result["aux"]
        values = np.asarray(quantities_jit(state, params_x, jnp.asarray(x)))
        cache[key] = dict(result, values=values, state=state, mask=mask, params=params_x)
        return cache[key]

    def linearize(x):
        """Design gradients of the objective and every plasma row at ``x``'s solution."""
        sol = solve(x)
        if "gradients" in sol:
            return sol["gradients"]
        counters["linearizations"] += 1
        xj = jnp.asarray(x)
        extra = plasma_directions(sol["params"], xj)
        J, dz, batch, aux = model.linearize(sol["x"], sol["aux"], None, extra=extra)
        # polish the current with this Jacobian, and linearize again where it went if it gained
        polished = model.solve_boundary(plasma_params(model.params0, xj), None, x0=sol["x"], jacobian=J[:, :nc],
                                        ftol=1e-10, max_nfev=10)
        if 0.5 * polished["rows"] @ polished["rows"] < 0.9 * 0.5 * sol["rows"] @ sol["rows"]:
            sol.update(x=polished["x"], aux=polished["aux"])
            J, dz, batch, aux = model.linearize(sol["x"], sol["aux"], None, extra=extra)
        state, mask, params_x, _ = aux
        sol.update(aux=aux, state=state, mask=mask, params=params_x,
                   values=np.asarray(quantities_jit(state, params_x, xj)))
        J_c, J_p = J[:, :nc], J[:, nc:]
        values = sol["values"]
        q, nf = values[:nq], values[nq]
        # objective 0.5 |q|^2 + 0.5 w nf^2, then each row
        cots = np.zeros((1 + nrows, values.size))
        cots[0, :nq], cots[0, nq] = q, normal_field_weight * nf
        cots[1:, nq + 1:] = np.eye(nrows)
        params_bar = quantity_pullback(state, mask, params_x, jnp.asarray(cots), xj)
        G = np.asarray(to_coordinates(params_x, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_c)
        lam = Q @ np.linalg.solve(R.T, G[:, :nc].T)
        gradients = np.zeros((1 + nrows, x.size))
        for i in range(1 + nrows):
            gradients[i] = np.asarray(direct_gradient(state, params_x, xj, jnp.asarray(cots[i])))
            gradients[i, :npl] += G[i, nc:] - lam[:, i] @ J_p
        anchor.update(v=np.asarray(sol["x"]).copy(), J_c=J_c)
        sol["gradients"] = gradients
        return gradients

    class Rejected(Exception):
        pass

    def fun(u):
        sol = solve(x0 + scales * u)
        if sol is None:
            raise Rejected
        q, nf = sol["values"][:nq], sol["values"][nq]
        return 0.5 * float(q @ q) + 0.5 * normal_field_weight * float(nf) ** 2

    def jac(u):
        return linearize(x0 + scales * u)[0] * scales

    def plasma_rows(u):
        sol = solve(x0 + scales * u)
        return np.full(nrows, -1.0) if sol is None else sol["values"][nq + 1:]

    def plasma_jac(u):
        if solve(x0 + scales * u) is None:
            return np.zeros((nrows, u.size))
        return linearize(x0 + scales * u)[1:] * scales

    constraints = [dict(type="ineq", fun=plasma_rows, jac=plasma_jac),
                   dict(type="ineq", fun=lambda u: np.asarray(coil_rows_jit(jnp.asarray(x0 + scales * u))),
                        jac=lambda u: np.asarray(coil_rows_jac(jnp.asarray(x0 + scales * u))) * scales)]
    last = dict(time=time.monotonic(), step=0, u=np.zeros_like(x0))

    def save(tag, x):
        sol = solve(x)
        coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        deck = im.input_with_params(fixed, sol["params"])
        vj.write_wout(str(out / f"wout{tag}.nc"), solve_equilibrium(deck, initial_state=sol["state"]).wout)

    def log_step(u):
        x = x0 + scales * np.asarray(u, dtype=float)
        sol = solve(x)
        if sol is None:
            raise RuntimeError("the accepted point has no certified equilibrium")
        state, params = sol["state"], sol["params"]
        runtime = im.runtime_from_params(params, model.cfg)
        values = sol["values"]
        coils = coils_from_x(jnp.asarray(x))
        now = time.monotonic()
        row = dict(step=last["step"], qa=float(values[:nq] @ values[:nq]),
                   objective=fun(np.asarray(u, dtype=float)), min_abs_iota=float(min_abs_iota(state, runtime)),
                   aspect=float(opt.aspect_ratio(state, runtime)), major_radius_m=float(opt.major_radius(state, runtime)),
                   total_normal_field_rms=float(values[nq]), rbtor_ratio=float(rbtor_ratio(state, runtime)),
                   coil_surface_distance_m=float(coil_limits.surface_distance(coils, surface_from_x(jnp.asarray(x)))),
                   coil_minimum_scaled_slack=float(np.min(np.asarray(coil_rows_jit(jnp.asarray(x))))),
                   plasma_minimum_scaled_slack=float(np.min(values[nq + 1:])),
                   redl_max_relative=model.bootstrap_residual(state, params), curtor=float(params.curtor),
                   beta=float(opt.volume_average_beta(state, runtime)), **counters,
                   step_seconds=now - last["time"], elapsed_seconds=now - started,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        for k in counters:
            counters[k] = 0
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"aspect={row['aspect']:.4f} R={row['major_radius_m']:.5f} B.n={row['total_normal_field_rms']:.2e} "
              f"RBphi={row['rbtor_ratio']:.5f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"plasma_slack={row['plasma_minimum_scaled_slack']:.4f} redl(max rel)={row['redl_max_relative']:.1e} "
              f"CURTOR={row['curtor']:.0f}A {row['step_seconds']:.1f}s", flush=True)
        if args.save_every and row["step"] % args.save_every == 0:
            save(f".step{row['step']}", x)
        last.update(time=now, step=last["step"] + 1, u=np.asarray(u, dtype=float).copy())
        if len(cache) > 64:
            keep = np.asarray(x, dtype=float).tobytes()
            for k in [k for k in cache if k != keep]:
                del cache[k]

    log_step(last["u"])
    if getattr(args, "check_gradient", 0):
        g0 = linearize(x0) * scales
        rng = np.random.default_rng(0)
        for k in range(args.check_gradient):
            d = rng.normal(size=x0.size)
            if k == 0:
                d[npl:] = 0.0  # first along the plasma coordinates only
            d /= np.linalg.norm(d)
            for h in (1e-2, 3e-3):
                vals = [np.r_[fun(s * h * d), plasma_rows(s * h * d)] for s in (1, -1)]
                fd = (vals[0] - vals[1]) / (2 * h)
                exact = g0 @ d
                print(f"[gradient check] direction {k} h {h:g}: objective {exact[0]:+.6e} vs {fd[0]:+.6e}; rows rel. "
                      "err " + " ".join(f"{abs(e - f) / max(abs(f), 1e-14):.1e}" for e, f in zip(exact[1:], fd[1:])),
                      flush=True)
        return
    u, stop = last["u"], dict(message="", success=False)
    while last["step"] <= args.steps:
        before = last["step"]
        try:
            result = minimize(fun, u, jac=jac, method="SLSQP", constraints=constraints, callback=log_step,
                              options=dict(maxiter=args.steps + 1 - before, ftol=optimizer_ftol))
            stop = dict(message=str(result.message), success=bool(result.success))
            break
        except Rejected:
            print("[slsqp] trial rejected: restarting from the accepted point", flush=True)
            u = last["u"]
            if last["step"] == before:
                stop = dict(message="trial rejected before any progress", success=False)
                break
    x = x0 + scales * last["u"]
    save("", x)
    coils = coils_from_x(jnp.asarray(x))
    coils.to_json(str(out / "coils.json"))
    summary = dict(accepted_steps=last["step"] - 1, **stop, elapsed_seconds=time.monotonic() - started,
                   endpoint=boundary_diagnostics(vj.read_wout(str(out / "wout.nc")), coils))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
