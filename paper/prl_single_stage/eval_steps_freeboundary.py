"""What the optimizer saw vs what the coils make: free-boundary re-solve of a run's saved steps.

    python eval_steps_freeboundary.py <run_dir> <outdir> [--every 10] [--steps 0,5,10] [--last]

For each saved step N of a fixed-arm (or free-arm) run: the deck is input.run with wout.stepN's boundary,
PHIEDGE and current profile (the driver's restart deck), the coils are coils.stepN.json, and the three-term
free-boundary equilibrium of those coils is solved from that boundary.  Rows (outdir/steps.jsonl):
  qa_seen    two-term QS of wout.stepN (what the optimizer scored; for the fixed arm its fixed-boundary target)
  qa_actual  two-term QS of the free-boundary equilibrium the coils make
  bn / pressure_balance / sheet_current  residuals of the coils on the step's boundary (initial) and after the solve
  lcfs_mm    max distance between the target boundary and the actual LCFS (mm at R0 = 1 m), on phi = 0 and half period
GPU job: run inside an allocation.  Re-uses the driver's deck helpers.
"""
import argparse
import importlib.util
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

T = Path("/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization/single_stage_free_boundary_optimization_three_term.py")
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
VC_GRID = 48


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r, phi, z):
        import jax.numpy as jnp

        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement) * inv_radius3[..., None]
        current_shape = (1,) * (xyz.ndim - 1) + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


def lcfs_mm(w_a, w_b, nfp):
    """Max over two cross-sections of the min distance from boundary a to boundary b, in mm."""
    theta = np.linspace(0, 2 * np.pi, 181)
    worst = 0.0
    for phi in (0.0, np.pi / nfp):
        curves = []
        for w in (w_a, w_b):
            ang = np.outer(theta, np.asarray(w.xm)) - np.asarray(w.xn)[None] * phi
            curves.append((np.cos(ang) @ np.asarray(w.rmnc)[-1], np.sin(ang) @ np.asarray(w.zmns)[-1]))
        (Ra, Za), (Rb, Zb) = curves
        worst = max(worst, float(np.max(np.min(np.hypot(Ra[:, None] - Rb[None], Za[:, None] - Zb[None]), axis=1))))
    return 1e3 * worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("outdir")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--last", action="store_true")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import vmex as vj
    from essos.coils import Coils
    from vmex import optimize as opt
    from vmex.core.freeboundary_vc import solve_free_boundary_three_term

    jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])
    spec = importlib.util.spec_from_file_location("tdriver", T)
    tdriver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdriver)

    saved = sorted(int(m.group(1)) for p in run.glob("wout.step*.nc") if (m := re.search(r"step(\d+)\.nc$", p.name))
                   and (run / f"coils.step{m.group(1)}.json").exists())
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    elif args.last:
        steps = saved[-1:]
    else:
        steps = [s for s in saved if s % args.every == 0] + ([saved[-1]] if saved and saved[-1] % args.every else [])
    done = set()
    log = out / "steps.jsonl"
    if log.exists():
        done = {json.loads(l)["step"] for l in log.read_text().splitlines() if l.strip()}
    qs = opt.QuasisymmetryRatioResidual(QA_SURFACES, 1, 0)
    inp0 = vj.VmecInput.from_file(run / "input.run")
    print(f"{run}: steps {steps} (saved {len(saved)}, done {sorted(done)})", flush=True)
    for n in steps:
        if n in done:
            continue
        w = vj.read_wout(str(run / f"wout.step{n}.nc"))
        inp = tdriver.current_from_wout(replace(tdriver.boundary_from_wout(inp0, w), phiedge=float(w.phi[-1]),
                                                lfreeb=True, mgrid_file="field(direct)"), w)
        coils = Coils.from_json(str(run / f"coils.step{n}.json"))
        field = DirectCoilField(np.asarray(coils.gamma), np.asarray(coils.gamma_dash), np.asarray(coils.currents))
        nfp = int(inp.nfp)
        t0 = time.time()
        row = dict(step=n, qa_seen=float(qs.total(w)), beta_seen=float(w.betatotal), iota_edge_seen=float(np.asarray(w.iotaf)[-1]))
        try:
            fit = solve_free_boundary_three_term(inp, external_field=field, nphi=VC_GRID, ntheta=VC_GRID, max_nfev=60,
                                                 trial_ftol=1e-11, chunk=8, quadrature=(4 * nfp * VC_GRID, 2 * VC_GRID),
                                                 verbose=0)
            wa = fit.equilibrium.wout
            res, res0 = fit.boundary_residual, fit.initial_boundary_residual
            row.update(qa_actual=float(qs.total(wa)), beta_actual=float(wa.betatotal),
                       iota_edge_actual=float(np.asarray(wa.iotaf)[-1]), aspect_actual=float(wa.aspect),
                       bn0=float(res0.normal), pb0=float(res0.pressure), K0=float(res0.sheet_current),
                       bn=float(res.normal), pb=float(res.pressure), K=float(res.sheet_current),
                       lcfs_mm=lcfs_mm(w, wa, nfp), nfev=int(fit.nfev), seconds=time.time() - t0)
            vj.write_wout(str(out / f"wout_actual.step{n}.nc"), wa)
        except Exception as e:  # a step whose coils support no equilibrium is a result too
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}", seconds=time.time() - t0)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
