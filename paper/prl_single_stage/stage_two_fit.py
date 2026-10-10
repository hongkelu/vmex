"""Stage-two baseline: coils fitted to a finished fixed-boundary (stage-one) finite-beta equilibrium.

    COIL_CASE=qa4-beta python stage_two_fit.py --wout W --input INPUT --coils START --out DIR [--weights 1e3,1e5,1e7]
        [--maxiter 3000] [--normal-scale 1e-3] [--order 12]

The objective is the driver's own finite-beta coil refit (fit_coils_to_plasma): the RMS of (B_coils + B_plasma).n/|B|
on the stage-one LCFS, B_plasma from virtual casing, plus a quadratic penalty on the SAME coil-engineering rows the
single-stage runs satisfy as hard constraints (length, curvature, MSC, coil-coil, coil-surface clearance, margins
included).  The penalty weight is continued upward so the result ends inside the feasible set; the final rows are
reported.  Coil currents stay at the value fixed by the equilibrium's RBTOR, as in the single-stage runs.
Writes DIR/coils.json (ESSOS), DIR/stage_two.json (fit report) and a pseudo-run (DIR/input.run, DIR/wout.step0.nc,
DIR/coils.step0.json) for  eval_steps_freeboundary.py DIR DIR/eval --steps 0  = the equilibrium the coils make.
GPU job: run inside an allocation.
"""
import argparse
import importlib.util
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

T = Path("/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization/single_stage_free_boundary_optimization_three_term.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", required=True, help="stage-one equilibrium (fixed boundary, finite beta)")
    ap.add_argument("--input", required=True, help="VMEC input deck of that equilibrium (resolution, profiles)")
    ap.add_argument("--coils", required=True, help="starting ESSOS coils")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="1e3,1e5,1e7", help="penalty-weight continuation")
    ap.add_argument("--maxiter", type=int, default=3000, help="L-BFGS-B iterations per weight")
    ap.add_argument("--normal-scale", type=float, default=1e-3, help="B.n/|B| unit of the objective")
    ap.add_argument("--order", type=int, default=None, help="coil Fourier order (default: the case's)")
    ap.add_argument("--slsqp", type=int, default=0, metavar="MAXITER",
                    help="after the penalty continuation, polish with SLSQP: minimise the squared normal field subject to "
                         "the coil rows and the clearance as HARD constraints (the single-stage formulation)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import jax.numpy as jnp
    import vmex as vj
    from essos.coils import Coils

    spec = importlib.util.spec_from_file_location("tdriver", T)
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    if args.order:
        td.COIL_ORDER = args.order
    td.COIL_FIT_MAXITER, td.COIL_FIT_NORMAL_SCALE = args.maxiter, args.normal_scale

    w = vj.read_wout(args.wout)
    inp = vj.VmecInput.from_file(args.input)
    inp = td.current_from_wout(replace(td.boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)
    coils = td.resize_coils(Coils.from_json(args.coils), td.COIL_ORDER, td.N_SEGMENTS)
    coils = td.scale_coil_currents(coils, abs(float(w.rbtor)))
    print(f"case {os.environ.get('COIL_CASE')}: limits length {td.LENGTH_LIMIT:.3f} m/coil, curvature {td.CURVATURE_LIMIT:.3f}, "
          f"MSC {td.MSC_LIMIT:.3f}, coil-coil {td.COIL_DISTANCE_LIMIT}, coil-surface {td.COIL_SURFACE_DISTANCE_LIMIT}; "
          f"order {td.COIL_ORDER}; stage-one QS seen below", flush=True)

    from essos.surfaces import surfacerzfourier_from_boundary
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=td.SURFACE_GRID[0], ntheta=td.SURFACE_GRID[1])

    def report(c):
        m = {k: np.asarray(v).tolist() for k, v in td.coil_metrics(c).items()}
        rows = np.asarray(td.coil_inequalities(c))
        cs = float(td.surface_distance(c, surface))
        return dict(metrics=m, min_row=float(rows.min()), coil_surface_distance=cs,
                    feasible_with_margins=bool(rows.min() >= 0 and cs >= td.COIL_SURFACE_DISTANCE_LIMIT + td.DISTANCE_MARGIN),
                    within_limits=bool(max(m["length"]) <= td.LENGTH_LIMIT and max(m["peak"]) <= td.CURVATURE_LIMIT
                                       and max(m["msc"]) <= td.MSC_LIMIT and m["coil_distance"] >= td.COIL_DISTANCE_LIMIT
                                       and cs >= td.COIL_SURFACE_DISTANCE_LIMIT))

    history = [dict(stage="start", **report(coils))]
    print("start:", json.dumps(history[-1]), flush=True)
    t0 = time.time()
    for wgt in [float(x) for x in args.weights.split(",")]:
        td.COIL_FIT_WEIGHT = wgt
        coils = td.fit_coils_to_plasma(coils, w, inp)
        history.append(dict(stage=f"weight {wgt:g}", seconds=time.time() - t0, **report(coils)))
        print(json.dumps(history[-1]), flush=True)
        coils.to_json(str(out / "coils.json"))
    if args.slsqp:
        import jax
        import vmex as vj
        from scipy.optimize import NonlinearConstraint, minimize

        interface = vj.PlasmaVacuumInterface.from_wout(w, nphi=37, ntheta=32)
        x0 = jnp.asarray(coils.curves.dofs).ravel()

        def coils_from_u(u):
            return coils.with_dofs(jnp.concatenate((x0 + td.COIL_STEP * u, coils.dofs_currents)))

        def objective(u):
            new = coils_from_u(u)
            return 0.5 * (td.weighted_rms(interface.weights, td.total_normal_field(interface, td.coil_field(new))) / args.normal_scale)**2

        def rows(u):
            new = coils_from_u(u)
            return jnp.concatenate([td.coil_inequalities(new), jnp.atleast_1d(
                (td.surface_distance(new, surface) - td.COIL_SURFACE_DISTANCE_LIMIT - td.DISTANCE_MARGIN) / td.COIL_SURFACE_DISTANCE_LIMIT)])

        vg, rf, rj = jax.jit(jax.value_and_grad(objective)), jax.jit(rows), jax.jit(jax.jacrev(rows))
        con = NonlinearConstraint(lambda u: np.asarray(rf(jnp.asarray(u))), 0, np.inf, jac=lambda u: np.asarray(rj(jnp.asarray(u))))
        t1 = time.time()
        res = minimize(lambda u: tuple(map(np.asarray, vg(jnp.asarray(u)))), np.zeros(x0.size), jac=True, method="SLSQP",
                       constraints=[con], options=dict(maxiter=args.slsqp, ftol=1e-14))
        coils = coils_from_u(jnp.asarray(res.x))
        nf = float(np.sqrt(2 * res.fun) * args.normal_scale)
        print(f"[slsqp] {res.nit} iterations ({res.message}): normal field RMS {nf:.3e}, {time.time() - t1:.0f} s", flush=True)
        history.append(dict(stage=f"slsqp {res.nit}", seconds=time.time() - t0, normal_field_rms=nf, **report(coils)))
        print(json.dumps(history[-1]), flush=True)
        coils.to_json(str(out / "coils.json"))
    # pseudo-run for the free-boundary scorer
    inp.to_indata(out / "input.run")
    shutil.copy(args.wout, out / "wout.step0.nc")
    shutil.copy(out / "coils.json", out / "coils.step0.json")
    json.dump(dict(case=os.environ.get("COIL_CASE"), overrides=os.environ.get("COIL_CASE_OVERRIDES"), wout=args.wout,
                   input=args.input, start_coils=args.coils, order=td.COIL_ORDER, weights=args.weights,
                   normal_scale=args.normal_scale, maxiter=args.maxiter, history=history,
                   limits=dict(length=td.LENGTH_LIMIT, curvature=td.CURVATURE_LIMIT, msc=td.MSC_LIMIT,
                               coil_distance=td.COIL_DISTANCE_LIMIT, coil_surface=td.COIL_SURFACE_DISTANCE_LIMIT)),
              open(out / "stage_two.json", "w"), indent=1)
    print("wrote", out / "stage_two.json", flush=True)


if __name__ == "__main__":
    main()
