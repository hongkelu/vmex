"""The production driver with the three-term labelling weight overridden (start-dependence decision tests).

    python label_probe.py W run   OUT <driver arguments>   # a normal optimization run at label_weight W
    python label_probe.py W cold  OUT <driver arguments>   # build the problem (solve the seed design) and stop
    python label_probe.py W noise OUT <driver arguments>   # LM noise / finite-difference test at the restart design
    python label_probe.py W curvature OUT <driver arguments> # the least-squares curvature the gradient leaves out
    python label_probe.py W exactgrad OUT <driver arguments> # the loss gradient with that curvature, against FD
    python label_probe.py W twostart OUT <driver arguments> # the restart design's boundary fit from two starts
                                  # (TWOSTART_STARTS=restart: that start only; CONVERGED_MAX_NFEV caps the full fit)

(cwd: a vmex tree with the driver.)  W = "default" keeps the code's weight.  ``cold`` reports the seed design's
loss, constraint values and interface residuals; ``noise`` is finish_noise.py's LM half.
"""
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

from vmex import optimize as opt
from vmex.core import freeboundary_vc as fv

W, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
del sys.argv[1:4]
out.mkdir(parents=True, exist_ok=True)
import os  # noqa: E402

if os.environ.get("BOUNDARY_FTOL") or os.environ.get("BOUNDARY_MAX_NFEV"):  # the problem's trial fits, tightened
    from_loss = fv.ThreeTermFreeBoundaryProblem.from_loss.__func__

    def tightened(cls, *args, **kwargs):
        if os.environ.get("BOUNDARY_FTOL"):
            kwargs["boundary_ftol"] = float(os.environ["BOUNDARY_FTOL"])
        if os.environ.get("BOUNDARY_MAX_NFEV"):
            kwargs["boundary_max_nfev"] = int(os.environ["BOUNDARY_MAX_NFEV"])
        return from_loss(cls, *args, **kwargs)

    fv.ThreeTermFreeBoundaryProblem.from_loss = classmethod(tightened)
if os.environ.get("SB_JACOBIAN_FTOL") or os.environ.get("SB_REFRESH"):  # the boundary fit itself, tightened
    solve_boundary = fv.ThreeTermFreeBoundaryModel.solve_boundary

    def tightened_fit(self, *args, **kwargs):
        if os.environ.get("SB_JACOBIAN_FTOL"):
            kwargs["jacobian_ftol"] = float(os.environ["SB_JACOBIAN_FTOL"])
        if os.environ.get("SB_REFRESH"):
            kwargs["refresh_on_stall"] = True
        return solve_boundary(self, *args, **kwargs)

    fv.ThreeTermFreeBoundaryModel.solve_boundary = tightened_fit
if W != "default" or os.environ.get("LABELLING"):
    init = fv.ThreeTermFreeBoundaryModel.__init__

    def __init__(self, *args, **kwargs):
        if W != "default":
            kwargs["label_weight"] = float(W)
        if os.environ.get("LABELLING"):
            kwargs["labelling"] = os.environ["LABELLING"]
        init(self, *args, **kwargs)

    fv.ThreeTermFreeBoundaryModel.__init__ = __init__


def cold(problem, **_):
    x0 = np.asarray(problem.accepted.parameters, dtype=float)
    res = problem.boundary_residual(x0)
    row = dict(W=W, loss=float(problem.fun(x0)), constraints=np.asarray(problem.constraint_values(x0)).tolist(),
               bn=res.normal, pressure=res.pressure, sheet=res.sheet_current, stats=problem.stats)
    print(json.dumps(row), flush=True)
    (out / "cold.json").write_text(json.dumps(row, indent=1))
    sys.exit(0)


T = np.r_[-np.logspace(-2, -6, 5), 0.0, np.logspace(-6, -2, 5)]
H = [float(h) for h in os.environ.get("NOISE_H", "1e-2,1e-3,1e-4,1e-5,1e-6").split(",")]


def noise(problem, **_):
    x0 = np.asarray(problem.accepted.parameters, dtype=float)
    d = np.asarray(problem.scales) * np.random.default_rng(0).standard_normal(x0.size)
    family = os.environ.get("NOISE_FAMILY")  # phiedge | current (= profile + curtor) | profile | curtor | coils
    if family:
        plasma = np.asarray(problem._plasma)
        keep = dict(phiedge=plasma[:1], current=plasma[1:], curtor=plasma[-1:], profile=plasma[1:-1],
                    coils=np.setdiff1d(np.arange(x0.size), plasma))[family]
        mask = np.zeros(x0.size, dtype=bool)
        mask[keep] = True
        d = np.where(mask, d, 0.0)
    d /= np.linalg.norm(d / np.asarray(problem.scales))
    t0 = time.perf_counter()
    f0, g = problem.value_and_grad(x0)
    gd = float(np.dot(g, d))
    line, bounds = [], []
    for t in T:
        line.append(problem.fun(x0 + t * d))
        bounds.append(np.asarray(problem._record(x0 + t * d)["x"]))
        print(f"t={t:+.0e} loss={line[-1]:.12e}", flush=True)
    bounds = np.asarray(bounds)
    lin = np.array([np.polyfit(T, bounds[:, k], 1) for k in range(bounds.shape[1])])
    b_fit = T[:, None] * lin[:, 0][None] + lin[:, 1][None]
    fd = [(problem.fun(x0 + h * d) - problem.fun(x0 - h * d)) / (2 * h) for h in H]
    row = dict(W=W, loss0=float(f0), g_dot_d=gd, line=line, fd=fd, fd_rel_err=[abs(v - gd) / abs(gd) for v in fd],
               boundary_scatter=float(np.sqrt(np.mean(np.sum((bounds - b_fit) ** 2, axis=1)))),
               seconds=time.perf_counter() - t0, stats=problem.stats)
    print(json.dumps(row), flush=True)
    (out / "noise.json").write_text(json.dumps(row, indent=1))
    sys.exit(0)


def twostart(problem, **_):
    """One design (the restart's coils, PHIEDGE and current), its boundary fitted from the restart boundary and from
    the run's seed boundary (input.run), as a trial fit (the problem's settings) and fully converged."""
    from dataclasses import replace as _replace

    import jax.numpy as jnp

    from vmex.core.input import VmecInput
    from vmex.core.optimize import pack_boundary

    model, x0 = problem.model, np.asarray(problem.accepted.parameters, dtype=float)
    xj = jnp.asarray(x0)
    params, field = problem._params_at(xj), problem.field(xj)
    seed = VmecInput.from_file(Path(sys.argv[sys.argv.index("--restart") + 1]) / "input.run")
    b_seed = pack_boundary(_replace(model.fixed, rbc=seed.rbc, zbs=seed.zbs), model.max_mode, vary_major_radius=True)
    starts = dict(restart=np.asarray(model.x0), seed=np.asarray(b_seed))
    starts = {k: starts[k] for k in os.environ.get("TWOSTART_STARTS", "restart,seed").split(",")}
    settings = dict(trial=dict(ftol=problem.boundary_ftol, max_nfev=problem.boundary_max_nfev),
                    converged=dict(ftol=1e-10, jacobian_ftol=1e-10, refresh_on_stall=True,
                                   max_nfev=int(os.environ.get("CONVERGED_MAX_NFEV", 400))))
    rows = []
    for how, kw in settings.items():
        for name, b in starts.items():
            t0 = time.perf_counter()
            fit = model.solve_boundary(params, field, x0=b, **kw)
            state, _, params_b, _ = fit["aux"]
            values = np.asarray(problem._rows(state, params_b, xj))
            res = model.boundary_residual(state, params_b, field)
            row = dict(W=W, labelling=os.environ.get("LABELLING", "start"), fit=how, start=name,
                       loss=float(values[0]), constraints=values[1:].tolist(), bn=res.normal, pressure=res.pressure,
                       sheet=res.sheet_current, nfev=int(fit["nfev"]), njev=int(fit["njev"]),
                       converged=bool(fit["converged"]), seconds=round(time.perf_counter() - t0),
                       start_distance=float(np.linalg.norm((b - starts["restart"]) / model.x_scale)))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / "twostart.json").write_text(json.dumps(rows, indent=1))
    sys.exit(0)


def curvature(problem, **_):
    """Size of the least-squares curvature term the implicit gradient leaves out, at the restart design: for the
    loss row's multiplier mu = (J^T J)^-1 dh/db (and a random direction), |S mu| / |J^T J mu| with
    S = sum_i r_i grad^2 r_i from central differences of the boundary Jacobian."""
    import jax.numpy as jnp

    model, x0 = problem.model, np.asarray(problem.accepted.parameters, dtype=float)
    xj = jnp.asarray(x0)
    problem.value_and_grad(x0)
    sol, anchor = problem._solve(x0), problem._anchor
    state, mask, params = sol["state"], sol["mask"], sol["params"]
    params_bar = problem._pullback(state, mask, params, xj)
    G_b = np.asarray(problem._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar)[0])
    J, b, nb = anchor["J_b"], np.asarray(anchor["b"]), model.x0.size
    P, field = problem._params_at(xj), problem.field(xj)
    r = np.asarray(model.evaluate(b, P, field, tight=True)[0])
    Q, R = np.linalg.qr(J)
    mu = np.linalg.solve(R, np.linalg.solve(R.T, G_b[0]))
    rows, scale = [], np.asarray(model.x_scale)
    from vmex.core.optimize import _dof_modes
    modes = _dof_modes(model.fixed, model.max_mode)
    names = [f"rbc{m},{n}" for m, n in modes] + [f"zbs{m},{n}" for m, n in modes] + ["rbc0,0"]
    weight = (mu / scale) ** 2 / np.sum((mu / scale) ** 2)
    top = np.argsort(weight)[::-1][:12]
    composition = dict(top=[(names[i], round(float(weight[i]), 4)) for i in top],
                       m0=float(sum(w for w, k in zip(weight, names) if k.startswith(("rbc0,", "zbs0,")))),
                       rbc00=float(weight[-1]), singular_values=np.linalg.svd(J * scale[None],
                                                                              compute_uv=False)[-5:].tolist())
    print(json.dumps(dict(loss_multiplier_composition=composition)), flush=True)
    for name, v in (("loss multiplier", mu), ("random", scale * np.random.default_rng(0).standard_normal(nb))):
        for step in (1e-3, 1e-4):
            t = step / np.linalg.norm(v / scale)
            Js = []
            for sign in (1, -1):
                got = model.evaluate(b + sign * t * v, P, field)
                Js.append(model.linearize(b + sign * t * v, got[1], field)[0][:, :nb])
            Sv = (Js[0] - Js[1]).T @ r / (2 * t)
            JJv = J.T @ (J @ v)
            row = dict(direction=name, step=step, S_v=float(np.linalg.norm(Sv)), JtJ_v=float(np.linalg.norm(JJv)),
                       ratio=float(np.linalg.norm(Sv) / np.linalg.norm(JJv)), residual=float(np.linalg.norm(r)),
                       loss_gradient_change=float(abs(mu @ Sv) / abs(mu @ JJv)) if name == "loss multiplier" else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / "curvature.json").write_text(json.dumps(rows, indent=1))
    sys.exit(0)


def exactgrad(problem, **_):
    """The loss gradient with the least-squares curvature in the k weakest boundary directions (EXACT_K, a list),
    against the Gauss-Newton gradient and, along the noise mode's seeded family directions, against the finite
    differences in FD_DIR/<family>/noise.json.  S v and the mixed term are central differences (EXACT_EPS, scaled
    coordinates) of exact adjoint gradients J(b')^T r and d/dc <r, r(b', c)>, so they include the second-order
    response through the interior equilibrium."""
    import jax.numpy as jnp

    ks = [int(k) for k in os.environ.get("EXACT_K", "8,24,48").split(",")]
    eps = float(os.environ.get("EXACT_EPS", "1e-4"))
    model, x0 = problem.model, np.asarray(problem.accepted.parameters, dtype=float)
    xj = jnp.asarray(x0)
    problem.value_and_grad(x0)
    sol, anchor = problem._solve(x0), problem._anchor
    G_gn = np.asarray(sol["gradients"])
    state, mask, params = sol["state"], sol["mask"], sol["params"]
    P, field = problem._params_at(xj), problem.field(xj)
    b, J, J_p = np.asarray(anchor["b"]), np.asarray(anchor["J_b"]), np.asarray(anchor["J_p"])
    scale = np.asarray(model.x_scale)
    r = np.asarray(model._rows(state, params, field))
    params_bar = problem._pullback(state, mask, params, xj)
    G_b, G_x = map(np.asarray, problem._coordinates(params, jnp.asarray(b), xj, params_bar))
    direct = np.asarray(problem._direct(state, params, xj))
    plasma = np.asarray(problem._plasma)

    def first_order(lam_res):
        out = direct[0] + G_x[0] - np.asarray(problem._field_vjp(state, params, xj, jnp.asarray(lam_res)))
        out[plasma] -= lam_res @ J_p
        return out

    nb = model.x0.size

    def grads_at(bp):  # d/db' and d/dc of <r, r(b', c)> at the boundary bp (r held fixed), from linearize's columns
        got = model.evaluate(bp, P, field, tight=True)
        extra = problem._plasma_directions(got[1][2], xj) if plasma.size else None
        Jf, _, _, aux = model.linearize(bp, got[1], field, extra=extra)
        model._last_linearize = None
        st, _, pr, _ = aux
        gx = np.array(problem._field_vjp(st, pr, xj, jnp.asarray(r)))
        gx[plasma] += np.asarray(Jf)[:, nb:].T @ r
        return np.asarray(Jf)[:, :nb].T @ r, gx

    Js = J * scale[None]
    Gs = scale * G_b[0]
    Vt = np.linalg.svd(Js, full_matrices=False)[2]
    kmax = max(ks)
    V = Vt[-kmax:][::-1].T  # weakest first
    Z = np.zeros_like(V)
    for j in range(kmax):
        step = eps * scale * V[:, j]
        Z[:, j] = scale * (grads_at(b + step)[0] - grads_at(b - step)[0]) / (2 * eps)
        print(json.dumps(dict(hvp=j, ratio=float(np.linalg.norm(Z[:, j]) / np.linalg.norm(Js.T @ (Js @ V[:, j]))))), flush=True)
    for k in ks:  # is the fit a strict minimum in the weak subspace? eigenvalues of V^T (J^T J + S) V
        Vk, Zk = V[:, :k], Z[:, :k]
        H = Vk.T @ (Js.T @ (Js @ Vk)) + 0.5 * (Vk.T @ Zk + Zk.T @ Vk)
        print(json.dumps(dict(k=k, weak_hessian_eigenvalues=np.linalg.eigvalsh(H).tolist(),
                              weak_JtJ_eigenvalues=np.linalg.eigvalsh(Vk.T @ (Js.T @ (Js @ Vk))).tolist())), flush=True)
    lam_gn = np.linalg.solve(Js.T @ Js, Gs)
    rows = [dict(k=0, check_gn=float(np.max(np.abs(first_order(Js @ lam_gn) - G_gn[0])) / np.max(np.abs(G_gn[0]))))]
    gradients = {0: first_order(Js @ lam_gn)}
    for k in ks:
        Vk, Zk = V[:, :k], Z[:, :k]
        S = Zk @ Vk.T + Vk @ Zk.T - Vk @ (Vk.T @ Zk) @ Vk.T
        lam = np.linalg.solve(Js.T @ Js + 0.5 * (S + S.T), Gs)
        step = eps * scale * lam / np.linalg.norm(lam)
        mixed = (grads_at(b + step)[1] - grads_at(b - step)[1]) / (2 * eps) * np.linalg.norm(lam)
        gradients[k] = first_order(Js @ lam) - mixed
        rows.append(dict(k=k, multiplier_change=float(np.linalg.norm(lam - lam_gn) / np.linalg.norm(lam_gn))))
    fd_dir = os.environ.get("FD_DIR")
    for family in ("coils", "phiedge", "current"):
        d = np.asarray(problem.scales) * np.random.default_rng(0).standard_normal(x0.size)
        keep = dict(phiedge=plasma[:1], current=plasma[1:], coils=np.setdiff1d(np.arange(x0.size), plasma))[family]
        d = np.where(np.isin(np.arange(x0.size), keep), d, 0.0)
        d /= np.linalg.norm(d / np.asarray(problem.scales))
        fd = None
        if fd_dir and Path(fd_dir, f"grad-{family}", "noise.json").exists():
            fd = json.loads(Path(fd_dir, f"grad-{family}", "noise.json").read_text())["fd"][2]  # h = 1e-4
        for k, g in gradients.items():
            rows.append(dict(family=family, k=k, g_dot_d=float(g @ d), fd=fd,
                             rel_err=None if fd is None else float(abs(g @ d - fd) / abs(fd))))
    for row in rows:
        print(json.dumps(row), flush=True)
    (out / "exactgrad.json").write_text(json.dumps(rows, indent=1))
    sys.exit(0)


def gnstep(problem, **_):
    """How far the optimizer's trial fits are from their stationary point: at the restart design and nearby trial
    designs, the Gauss-Newton step |J^+ r| (scaled boundary coordinates) and the loss at the trial fit, and the same
    after a tightly converged fit (ftol 1e-10) of the same design, with that fit's evaluation count."""
    import jax.numpy as jnp

    model, x0 = problem.model, np.asarray(problem.accepted.parameters, dtype=float)
    scale = np.asarray(model.x_scale)
    nb = model.x0.size
    problem.value_and_grad(x0)

    def step(x_b, aux, field):
        J = np.asarray(model.linearize(x_b, aux, field)[0])[:, :nb] * scale[None]
        model._last_linearize = None
        r = np.asarray(model._rows(aux[0], aux[2], field))
        return float(np.linalg.norm(np.linalg.lstsq(J, r, rcond=None)[0])), r

    rng = np.random.default_rng(1)
    rows = []
    for t in (0.0, 1e-3, 1e-2, 3e-2):
        d = np.asarray(problem.scales) * rng.standard_normal(x0.size)
        x = x0 + t * d / np.linalg.norm(d / np.asarray(problem.scales))
        sol = problem._solve(x)
        if sol is None:
            rows.append(dict(t=t, trial="rejected"))
            continue
        xj = jnp.asarray(x)
        field, P = problem.field(xj), problem._params_at(xj)
        s_trial, r_trial = step(sol["x"], sol["aux"], field)
        tight = model.solve_boundary(P, field, x0=sol["x"], ftol=1e-10, jacobian_ftol=1e-10, max_nfev=400,
                                     refresh_on_stall=True)
        s_tight, r_tight = step(tight["x"], tight["aux"], field)
        values = np.asarray(problem._rows(tight["aux"][0], tight["aux"][2], xj))
        rows.append(dict(t=t, step_trial=s_trial, step_tight=s_tight, tight_nfev=int(tight["nfev"]),
                         tight_reason=tight.get("reason"), boundary_move=float(np.linalg.norm((tight["x"] - sol["x"]) / scale)),
                         loss_trial=float(sol["values"][0]), loss_tight=float(values[0]),
                         cost_trial=float(r_trial @ r_trial), cost_tight=float(r_tight @ r_tight)))
        print(json.dumps(rows[-1]), flush=True)
    (out / "gnstep.json").write_text(json.dumps(rows, indent=1))
    sys.exit(0)


if mode == "twostart":
    opt.minimize = twostart
elif mode == "cold":
    opt.minimize = cold
elif mode == "noise":
    opt.minimize = noise
elif mode == "curvature":
    opt.minimize = curvature
elif mode == "exactgrad":
    opt.minimize = exactgrad
elif mode == "gnstep":
    opt.minimize = gnstep
sys.argv[0] = "examples/optimization/single_stage_free_boundary_optimization_three_term.py"
runpy.run_path(sys.argv[0], run_name="__main__")
