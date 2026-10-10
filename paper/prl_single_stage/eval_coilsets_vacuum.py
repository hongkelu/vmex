"""Vacuum free-boundary evaluation of fixed coil sets on the LP QA deck.

    python eval_coilsets_vacuum.py <coils.json> <outdir> [--check check.npz]

Solves the three-term free boundary (B.n = 0, pressure balance, K = 0) of the
coils' field with the Landreman-Paul QA lowres deck (nfp 2, A 6, vacuum, 8x8
modes, NS ladder 16/31/50), starting from the deck boundary, then scores the
equilibrium the coils actually make: the three interface residuals, the
two-term quasisymmetry residual on s = 0.1..1.0 (profile and total), iota,
aspect ratio, volume.  The same scores of the deck's fixed-boundary target are
written once to <outdir>/../target.json.  GPU job: run inside an allocation.
"""
import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path("/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data")
DECK = DATA / "input.LandremanPaul2021_QA_lowres"
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
VC_GRID = 48


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Filament Biot-Savart field as a pytree (copy of the three-term driver's)."""

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


MODES, NS_FINAL = (8, 8), None


def deck():
    import vmex as vj

    inp = vj.VmecInput.from_file(DECK)
    mpol, ntor = MODES
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=max(64, 4 * mpol + 8), nzeta=max(64, 4 * ntor + 8))
    if NS_FINAL and NS_FINAL > int(np.max(inp.ns_array)):  # one more multigrid rung at the requested NS
        inp = replace(inp, ns_array=np.r_[np.asarray(inp.ns_array), NS_FINAL],
                      ftol_array=np.r_[np.asarray(inp.ftol_array), np.asarray(inp.ftol_array)[-1]],
                      niter_array=np.r_[np.asarray(inp.niter_array), np.asarray(inp.niter_array)[-1]])
    return inp


def scores(eq, qs):
    w = eq.wout
    q = np.asarray(qs.residuals_state(eq.solution, eq.solver_context))
    prof = np.asarray(qs.profile_state(eq.solution, eq.solver_context))
    return dict(qa_total=float(q @ q), qa_profile=prof.tolist(), qa_surfaces=list(QA_SURFACES),
                iota_axis=float(np.asarray(w.iotaf)[0]), iota_edge=float(np.asarray(w.iotaf)[-1]),
                iota_min=float(np.min(np.abs(np.asarray(w.iotaf)[1:]))), aspect=float(w.aspect),
                volume_m3=float(w.volume_p), rmajor=float(w.Rmajor_p), b0=float(w.b0), volavgB=float(w.volavgB))


def check_coils(coils, path):
    """The ESSOS coil set must equal the numpy reconstruction (same points, same currents)."""
    ref = np.load(path)
    g, I = np.asarray(coils.gamma), np.asarray(coils.currents)
    assert g.shape == ref["gamma"].shape, (g.shape, ref["gamma"].shape)
    worst = 0.0
    for gi, Ii in zip(g, I):
        d = np.max(np.linalg.norm(ref["gamma"] - gi[None], axis=-1), axis=1)   # same parametrisation
        k = int(np.argmin(d))
        worst = max(worst, float(d[k]), abs(float(ref["currents"][k]) - float(Ii)) / abs(float(Ii)))
    print(f"coil load check: worst mismatch {worst:.2e}", flush=True)
    assert worst < 1e-8, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coils")
    ap.add_argument("outdir")
    ap.add_argument("--check", default=None)
    ap.add_argument("--modes", type=int, nargs=2, default=(8, 8), metavar=("MPOL", "NTOR"))
    ap.add_argument("--ns", type=int, default=None, help="extra final NS rung above the deck's 50")
    ap.add_argument("--vc-grid", type=int, default=48)
    args = ap.parse_args()
    global MODES, NS_FINAL, VC_GRID
    MODES, NS_FINAL, VC_GRID = tuple(args.modes), args.ns, args.vc_grid
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import vmex as vj
    from essos.coils import Coils
    from vmex import optimize as opt
    from vmex.core.freeboundary_vc import solve_free_boundary_three_term

    print("devices:", jax.devices(), flush=True)
    jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])
    coils = Coils.from_json(args.coils)
    if args.check:
        check_coils(coils, args.check)
    field = DirectCoilField(np.asarray(coils.gamma), np.asarray(coils.gamma_dash), np.asarray(coils.currents))
    qs = opt.QuasisymmetryRatioResidual(QA_SURFACES, 1, 0)

    target_json = out.parent / "target.json"
    if not target_json.exists():
        t0 = time.time()
        eq_t = opt.solve_equilibrium(deck())
        row = dict(name="target_LP_QA_lowres", seconds=time.time() - t0, **scores(eq_t, qs))
        vj.write_wout(str(out.parent / "wout_target.nc"), eq_t.wout)
        target_json.write_text(json.dumps(row, indent=1) + "\n")
        print("TARGET " + json.dumps({k: v for k, v in row.items() if k != "qa_profile"}), flush=True)

    inp = replace(deck(), lfreeb=True, mgrid_file="field(direct)")
    nfp = int(inp.nfp)
    t0 = time.time()
    fit = solve_free_boundary_three_term(inp, external_field=field, nphi=VC_GRID, ntheta=VC_GRID, max_nfev=60,
                                         trial_ftol=1e-11, chunk=8, quadrature=(4 * nfp * VC_GRID, 2 * VC_GRID),
                                         verbose=1)
    seconds = time.time() - t0
    res, res0 = fit.boundary_residual, fit.initial_boundary_residual
    row = dict(name=out.name, coils=str(args.coils), seconds=seconds, nfev=int(fit.nfev), njev=int(fit.njev),
               cost=float(fit.cost),
               bn=float(res.normal), pressure_balance=float(res.pressure), sheet_current=float(res.sheet_current),
               bn_initial=float(res0.normal), pressure_balance_initial=float(res0.pressure),
               sheet_current_initial=float(res0.sheet_current),
               **scores(fit.equilibrium, qs))
    vj.write_wout(str(out / "wout.nc"), fit.equilibrium.wout)
    (out / "row.json").write_text(json.dumps(row, indent=1) + "\n")
    print("ROW " + json.dumps({k: v for k, v in row.items() if k != "qa_profile"}), flush=True)


if __name__ == "__main__":
    main()
