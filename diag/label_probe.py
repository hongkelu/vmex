"""The production driver with the three-term labelling weight overridden (start-dependence decision tests).

    python label_probe.py W run   OUT <driver arguments>   # a normal optimization run at label_weight W
    python label_probe.py W cold  OUT <driver arguments>   # build the problem (solve the seed design) and stop
    python label_probe.py W noise OUT <driver arguments>   # LM noise / finite-difference test at the restart design
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
H = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def noise(problem, **_):
    x0 = np.asarray(problem.accepted.parameters, dtype=float)
    d = np.asarray(problem.scales) * np.random.default_rng(0).standard_normal(x0.size)
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


if mode == "twostart":
    opt.minimize = twostart
elif mode == "cold":
    opt.minimize = cold
elif mode == "noise":
    opt.minimize = noise
sys.argv[0] = "examples/optimization/single_stage_free_boundary_optimization_three_term.py"
runpy.run_path(sys.argv[0], run_name="__main__")
