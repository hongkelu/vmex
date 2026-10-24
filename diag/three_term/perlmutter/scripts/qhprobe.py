"""Run the campaign driver with the certification traced: every LM finish verbose (with its projection misses), and
every certify() round's state dumped (rows, Jacobian, x, row blocks) to OUT/cert_<n>.npz before it decides.
usage: python qhprobe.py OUT <driver args...>   (cwd: a run tree with the driver)"""
import runpy
import sys
from pathlib import Path

import numpy as np

from vmex.core import freeboundary_vc as fv

out = Path(sys.argv[1]); del sys.argv[1]
out.mkdir(parents=True, exist_ok=True)
_lm = fv._levenberg_marquardt


def traced_lm(*args, **kwargs):
    kwargs["verbose"] = 1
    res = _lm(*args, **kwargs)
    print(f"[lm] reason {res['reason']} nfev {res['nfev']} accepted {res['accepted']} cost "
          f"{0.5 * res['rows'] @ res['rows']:.6e} project {kwargs.get('project') is not None} "
          f"misses {res.get('projections', res.get('projection_misses'))}", flush=True)
    return res


fv._levenberg_marquardt = traced_lm
import os
for _name in ("POLISH_MAX_NFEV", "STALL_STEPS"):
    if os.environ.get(_name):
        setattr(fv, _name, int(os.environ[_name]))
        print(_name, getattr(fv, _name), flush=True)
if os.environ.get("POLISH_ROUNDS"):
    fv.POLISH_ROUNDS = int(os.environ["POLISH_ROUNDS"])
    print("POLISH_ROUNDS", fv.POLISH_ROUNDS, flush=True)
_cert = fv._fit_certificate
count = [0]


def traced_cert(J, r, normal=None):
    c = _cert(J, r, normal)
    count[0] += 1
    np.savez(out / f"cert_{count[0]:03d}.npz", J=J, r=r)
    print(f"[cert {count[0]}] cost {0.5 * r @ r:.6e} stationarity {c['stationarity']:.3e} step {c['step']:.3e} "
          f"condition {c['condition']:.3e} normal_step {c.get('normal_step')}", flush=True)
    return c


fv._fit_certificate = traced_cert
_lin = fv.ThreeTermFreeBoundaryModel.linearize
nlin = [0]


def saving_linearize(self, x, aux, field, extra=None):
    nlin[0] += 1
    np.save(out / "last_x.npy", np.asarray(x))  # the model coordinates of the latest linearization
    return _lin(self, x, aux, field, extra)


fv.ThreeTermFreeBoundaryModel.linearize = saving_linearize
if os.environ.get("START_X"):  # skip the cold fit: start the first boundary solve at a saved point
    _sb = fv.ThreeTermFreeBoundaryModel.solve_boundary
    used = [False]

    def resume(self, params, field, *, x0=None, jacobian=None, **kwargs):
        if x0 is None and jacobian is None and not used[0]:
            used[0] = True
            x = np.load(os.environ["START_X"])
            got = self.evaluate(x, params, field, tight=True)
            print(f"[resume] cost {0.5 * got[0] @ got[0]:.6e} at START_X", flush=True)
            return dict(x=x, rows=got[0], aux=got[1], nfev=1, njev=0, accepted=0, converged=False, radius=None,
                        reason="stalled", jacobian=None)
        return _sb(self, params, field, x0=x0, jacobian=jacobian, **kwargs)

    fv.ThreeTermFreeBoundaryModel.solve_boundary = resume
if os.environ.get("NORMAL_TOL"):  # the problem's trial certificates accept a remaining normal step below it
    _fl = fv.ThreeTermFreeBoundaryProblem.from_loss.__func__

    def with_normal_tol(cls, *args, **kwargs):
        kwargs["boundary_normal_tol"] = float(os.environ["NORMAL_TOL"])
        return _fl(cls, *args, **kwargs)

    fv.ThreeTermFreeBoundaryProblem.from_loss = classmethod(with_normal_tol)
if os.environ.get("CALIBRATE"):  # at the certified seed: how much does the remaining GN step change the outputs?
    import jax.numpy as jnp

    _init = fv.ThreeTermFreeBoundaryProblem.__init__

    def calibrated_init(self, model, loss, x0, **kwargs):
        _init(self, model, loss, x0, **kwargs)
        sol = self._cache[np.asarray(x0, dtype=float).tobytes()]
        n = model.x0.size
        J, r = sol["J"][:, :n], sol["rows"]
        step = -np.linalg.lstsq(J, r, rcond=None)[0]
        params, field, xj = self._params_at(jnp.asarray(x0)), self.field(jnp.asarray(x0)), jnp.asarray(x0)
        base = model.boundary_residual(sol["state"], sol["params"], field)
        print(f"[calibrate] certificate {sol['certificate']}", flush=True)
        print(f"[calibrate] values {np.array2string(sol['values'], precision=6)}; B.n {base.normal:.4e} "
              f"K {base.sheet_current:.4e}", flush=True)
        for alpha in (0.5, 1.0):
            got = model.evaluate(sol["x"] + alpha * step, params, field, tight=True)
            if got is None:
                print(f"[calibrate] alpha {alpha}: no equilibrium", flush=True)
                continue
            state, _, params_b, _ = got[1]
            values = np.asarray(self._rows(state, params_b, xj))
            res = model.boundary_residual(state, params_b, field)
            rel = np.abs(values - sol["values"]) / np.maximum(np.abs(sol["values"]), 1e-12)
            print(f"[calibrate] alpha {alpha}: cost {0.5 * got[0] @ got[0]:.6e} (was {0.5 * r @ r:.6e}); "
                  f"B.n {res.normal:.4e} K {res.sheet_current:.4e}; values max rel change {rel.max():.3e} "
                  f"(loss {rel[0]:.3e}); per value {np.array2string(rel, precision=2)}", flush=True)
        if os.environ.get("CALIBRATE") == "stop":
            raise SystemExit(0)

    fv.ThreeTermFreeBoundaryProblem.__init__ = calibrated_init
if os.environ.get("TRAJECTORY"):  # the outputs at every certified polish round (the observed convergence)
    import jax.numpy as jnp

    PROBLEM = {}
    _lin2 = fv.ThreeTermFreeBoundaryModel.linearize

    def recording_linearize(self, x, aux, field, extra=None):
        result = _lin2(self, x, aux, field, extra)
        problem = PROBLEM.get("p")
        if problem is not None and hasattr(problem, "_rows") and extra is not None:
            state, _, params_b, _ = result[3]
            values = np.asarray(problem._rows(state, params_b, jnp.asarray(PROBLEM["x0"])))
            res = self.boundary_residual(state, params_b, field)
            rows = np.asarray(self._rows(state, params_b, field))
            prev = PROBLEM.get("values")
            change = "" if prev is None else (
                f"; values rel change {np.max(np.abs(values - prev) / np.maximum(np.abs(prev), 1e-12)):.3e} "
                f"(loss {abs(values[0] - prev[0]) / abs(prev[0]):.3e})")
            PROBLEM["values"] = values
            print(f"[trajectory] cost {0.5 * rows @ rows:.6e} B.n {res.normal:.5e} K {res.sheet_current:.5e} "
                  f"loss {values[0]:.8e}{change}", flush=True)
        return result

    fv.ThreeTermFreeBoundaryModel.linearize = recording_linearize
    _init2 = fv.ThreeTermFreeBoundaryProblem.__init__

    def trajectory_init(self, model, loss, x0, **kwargs):
        PROBLEM.update(p=self, x0=np.asarray(x0, dtype=float))
        try:
            _init2(self, model, loss, x0, **kwargs)
        except Exception as error:  # the plateau is refused: the trajectory is the result
            print(f"[trajectory] ended: {error}", flush=True)
            raise SystemExit(0)

    fv.ThreeTermFreeBoundaryProblem.__init__ = trajectory_init
if os.environ.get("MAX_MODE"):  # fit the boundary up to this mode only
    _minit = fv.ThreeTermFreeBoundaryModel.__init__

    def max_mode_init(self, *args, **kwargs):
        kwargs["max_mode"] = int(os.environ["MAX_MODE"])
        _minit(self, *args, **kwargs)

    fv.ThreeTermFreeBoundaryModel.__init__ = max_mode_init
    print("MAX_MODE", os.environ["MAX_MODE"], flush=True)
sys.argv[0] = "examples/optimization/single_stage_free_boundary_optimization_three_term.py"
runpy.run_path(sys.argv[0], run_name="__main__")
