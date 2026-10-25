"""Is a run's 8 x 8 objective right?  Cold NL free-boundary solve of its saved coils, and resolution checks.

usage: COIL_CASE=qa4-beta-merc python nl_coils_eval.py RUN_DIR OUT_DIR --steps 10 20 ... [--helicity 1 0]

For each saved step N (``wout.stepN.nc`` + ``coils.stepN.json``), one JSON line in OUT_DIR/nl_coils.jsonl:
- ``seen``: the run's own 8 x 8 objective (QS of wout.stepN);
- ``seen_m12``: the run's boundary re-solved fixed-boundary with the interior at M = 12;
- ``nl``: the NL (no labelling rows) three-term free-boundary equilibrium of the step's coils, solved cold from the
  step's boundary at the run's resolution, and its objective; ``nl_m12``: that boundary at M = 12;
- the NL solve's interface residuals and the max LCFS distance between the run's and NL's boundary (mm at R0 = 1).
The deck is the driver's restart deck (input.run with the step's boundary, PHIEDGE and current profile).
"""
import argparse
import importlib.util
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

DRIVER = Path(__file__).resolve().parents[4] / "examples/optimization/single_stage_free_boundary_optimization_three_term.py"
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
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--helicity", type=int, nargs=2, default=(1, 0))
    ap.add_argument("--resolution", type=int, default=12, help="interior M of the resolution checks")
    ap.add_argument("--ftol", type=float, default=1e-7, help="the NL fit's relative cost-change tolerance")
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--interior", type=int, help="interior mpol = ntor of the NL solve (default: the run's)")
    ap.add_argument("--boundary-modes", type=int, help="boundary modes of the NL solve (default: all of the interior's)")
    ap.add_argument("--m1-complement", default="solved", choices=("solved", "frozen", "gauged"))
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import vmex as vj
    from essos.coils import Coils
    from vmex import optimize as opt
    from vmex.core.freeboundary_vc import solve_free_boundary_three_term

    jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])
    spec = importlib.util.spec_from_file_location("tdriver", DRIVER)
    tdriver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdriver)
    qs = opt.QuasisymmetryRatioResidual(np.array([i / 10 for i in range(1, 11)]), *args.helicity)
    inp0 = vj.VmecInput.from_file(run / "input.run")

    def at_resolution(inp, w, M):  # the wout's boundary, fixed-boundary, interior at M
        fixed = replace(tdriver.boundary_from_wout(inp, w), lfreeb=False).change_resolution(mpol=M, ntor=M, ntheta=GRID, nzeta=GRID)
        eq = opt.solve_equilibrium(fixed, raise_on_max_iterations=True)
        return float(qs.total(eq.wout))

    for n in args.steps:
        w = vj.read_wout(str(run / f"wout.step{n}.nc"))
        inp = tdriver.current_from_wout(replace(tdriver.boundary_from_wout(inp0, w), phiedge=float(w.phi[-1]),
                                                lfreeb=True, mgrid_file="field(direct)"), w)
        if args.interior:
            inp = inp.change_resolution(mpol=args.interior, ntor=args.interior, ntheta=GRID, nzeta=GRID)
        coils = Coils.from_json(str(run / f"coils.step{n}.json"))
        field = DirectCoilField(np.asarray(coils.gamma), np.asarray(coils.gamma_dash), np.asarray(coils.currents))
        nfp = int(inp.nfp)
        row = dict(step=n, seen=float(qs.total(w)), interior=args.interior, boundary_modes=args.boundary_modes,
                   m1=args.m1_complement)
        t0 = time.time()
        try:
            row["seen_m12"] = at_resolution(inp, w, args.resolution)
        except Exception as e:
            row["seen_m12_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            fit = solve_free_boundary_three_term(inp, external_field=field, nphi=VC_GRID, ntheta=VC_GRID,
                                                 label_weight=0.0, ftol=args.ftol, max_nfev=args.max_nfev,
                                                 max_mode=args.boundary_modes, m1_complement=args.m1_complement,
                                                 trial_ftol=1e-11, chunk=8, quadrature=(4 * nfp * VC_GRID, 2 * VC_GRID))
            wa = fit.equilibrium.wout
            res, res0 = fit.boundary_residual, fit.initial_boundary_residual
            row.update(nl=float(qs.total(wa)), bn0=float(res0.normal), bn=float(res.normal), pb=float(res.pressure),
                       K=float(res.sheet_current), lcfs_mm=lcfs_mm(w, wa, nfp), nfev=int(fit.nfev))
            vj.write_wout(str(out / f"wout_nl{args.tag}.step{n}.nc"), wa)
            try:
                row["nl_m12"] = at_resolution(inp, wa, args.resolution)
            except Exception as e:
                row["nl_m12_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        except Exception as e:  # coils that support no equilibrium are a result too
            row["nl_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        row["seconds"] = time.time() - t0
        with open(out / f"nl_coils{args.tag}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    main()
