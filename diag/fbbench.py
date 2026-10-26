"""Free-boundary solve accuracy and start independence across configurations (measurement only).

    python fbbench.py OUT CONFIG [CONFIG ...]

Methods: NESTOR + VMEX (from the deck), and the three-term solve (the equal-arc "C" labelling was removed with
labelling="equal_arc") with no labelling rows ("NL": start labelling at weight 0), and with VMEC's condensation rows ("D" at 1e-3, "DautoR"
at the automatic weight of ratio R), each from several starting boundaries; FBB_METHODS=C,NL,... picks a subset.  Every result is
scored the same way: a fixed-boundary equilibrium on its boundary; the three interface conditions on an N x N grid
with one fixed singular quadrature per configuration (planned on the deck's equilibrium); field lines in vacuum;
axis/edge iota, R00, and the LCFS distance to the NESTOR result with a dense reference curve; per method, the
largest LCFS distance between its starts ("spread", independent of the labelling).  FBB_FIXED_MODEL=1 builds each
method's problem once (from the deck) and varies only the initial boundary: otherwise every start rebuilds the model
from itself (A's labelling reference, the quadrature plan and the axis then change with the start).

CONFIG: cth_vac, cth_beta, lpqa_vac, lpqa_b05, lpqa_b25, ncsx (repository data), or prod:RUN:STEP for a production
run's design (RUN a directory with input.run, coils.stepSTEP.json and wout.stepSTEP.nc); append @M for the
resolution MPOL = NTOR = M (boundary truncated or zero-padded).  Six starts at M = 6 and 8, four otherwise.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_three_term as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.multigrid import _nestor_boundary  # noqa: E402

sys.path.insert(0, "tests")
from test_freeboundary_vc import _vacuum_flux_surface  # noqa: E402

DATA = Path("examples/data")
N = 64
METHODS = {"NL": dict(labelling="start", label_weight=0.0)}
# The fits' tolerances (solve_free_boundary_three_term's defaults; FBB_FTOL=1e-10 FBB_MAX_NFEV=300 for tight fits)
FIT = dict(ftol=float(os.environ.get("FBB_FTOL", "1e-4")), jacobian_ftol=float(os.environ.get("FBB_JACOBIAN_FTOL", "1e-2")),
           max_nfev=int(os.environ.get("FBB_MAX_NFEV", "60")))
LABELLINGS = {"A": "start"}
if os.environ.get("FBB_METHODS"):  # a subset, e.g. FBB_METHODS=NL,A:1e-2 (label:weight for A)
    METHODS = {k: METHODS[k] if k in METHODS else dict(labelling=LABELLINGS[k.split(":")[0]],
                                                       label_weight=float(k.split(":")[1]))
               for k in os.environ["FBB_METHODS"].split(",")}


@dataclass(frozen=True)
class Filament:
    gamma: object
    gamma_dash: object
    currents: object

    def b_cyl(self, r, phi, z):
        xyz = jnp.stack(jnp.broadcast_arrays(r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
        d = xyz[..., None, None, :] - self.gamma
        dB = jnp.cross(self.gamma_dash, d) * (jnp.sum(d * d, axis=-1) ** -1.5)[..., None]
        B = 1e-7 * jnp.mean(jnp.sum(dB * self.currents[:, None, None], axis=-3), axis=-2)
        c, s = jnp.cos(phi), jnp.sin(phi)
        return c * B[..., 0] + s * B[..., 1], -s * B[..., 0] + c * B[..., 1], B[..., 2]


jax.tree_util.register_dataclass(Filament, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])


def coil_field(path):
    from essos.coils import Coils

    c = Coils.from_json(str(path))
    return Filament(jnp.asarray(c.gamma), jnp.asarray(c.gamma_dash), jnp.asarray(c.currents))


def ladder(inp, mpol, ns):
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=max(16, 2 * mpol + 4))
    return replace(inp, ns_array=np.array([ns]), niter_array=np.array([20000]), ftol_array=np.array([1e-11]))


def config(name):
    """(deck, field, vacuum?, number of starts, three-term options), at the resolution of an ``@M`` suffix."""
    base, _, res = name.partition("@")
    inp, field, vacuum, k, options = _config(base)
    if res:
        M = int(res)
        ns = int(np.asarray(inp.ns_array)[-1])
        # an mgrid deck keeps its NZETA: it must divide the table's toroidal planes (VMEC2000's pairing rule)
        nzeta = int(inp.nzeta) if isinstance(field, vj.MgridField) else max(16, 2 * M + 4)
        inp = replace(inp.change_resolution(mpol=M, ntor=M, ntheta=2 * M + 6, nzeta=nzeta),
                      ns_array=np.array([ns]), niter_array=np.array([20000]),
                      ftol_array=np.array([float(np.asarray(inp.ftol_array)[-1])]))
        k = min(k, 6 if M in (6, 8) else 4)
    return inp, field, vacuum, k, options


def _config(name):
    if name.startswith("cth"):
        inp = replace(vj.VmecInput.from_file(DATA / "input.cth_like_free_bdy"), mgrid_file=str(DATA / "mgrid_cth_like.nc"))
        if name == "cth_vac":
            inp = replace(inp, pres_scale=0.0, curtor=0.0)
        return inp, vj.MgridField.from_input(inp, inp.mgrid_file, order=3), name == "cth_vac", 6, {}
    if name == "ncsx":
        inp = replace(vj.VmecInput.from_file(DATA / "input.ncsx_c09r00_free_lowres"),
                      mgrid_file=str(DATA / "mgrid_ncsx_c09r00_small.nc"))
        return inp, vj.MgridField.from_input(inp, inp.mgrid_file, order=3), False, 6, {}
    if name.startswith("lpqa"):
        deck, coils = {"lpqa_vac": ("input.LandremanPaul2021_QA_lowres", "ESSOS_biot_savart_LandremanPaulQA.json"),
                       "lpqa_b05": ("input.LandremanPaul2021_QA_beta0p5_bootstrap",
                                    "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"),
                       "lpqa_b25": ("input.LandremanPaul2021_QA_beta2p5_bootstrap",
                                    "ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json")}[name]
        inp = replace(ladder(vj.VmecInput.from_file(DATA / deck), 6, 51), lfreeb=True, mgrid_file="essos_coils(direct)")
        if name == "lpqa_vac":
            inp = replace(inp, phiedge=-0.025, curtor=0.0, am=np.zeros(21))
        return inp, coil_field(DATA / coils), name == "lpqa_vac", 6, {}
    _, run, step = name.split(":")
    run = Path(run)
    inp = vj.VmecInput.from_file(run / "input.run")
    w = vj.read_wout(run / f"wout.step{step}.nc")
    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, n, r, z in zip(np.asarray(w.xm, int), np.asarray(w.xn, int) // int(w.nfp), np.asarray(w.rmnc)[-1],
                          np.asarray(w.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=float(w.phi[-1]), lfreeb=True, mgrid_file="essos_coils(direct)",
                  raxis_c=np.asarray(w.raxis_cc)[:inp.ntor + 1], zaxis_s=np.asarray(w.zaxis_cs)[:inp.ntor + 1])
    if int(inp.ncurr) == 1:
        kind = str(w.pcurr_type).strip()
        if "spline" in kind or "line_segment" in kind:
            inp = replace(inp, pcurr_type=kind, ac_aux_s=np.asarray(w.ac_aux_s, float), ac_aux_f=np.asarray(w.ac_aux_f, float),
                          curtor=float(w.ctor))
        else:
            inp = replace(inp, pcurr_type=kind, ac=np.asarray(w.ac, float)[:np.size(inp.ac)], curtor=float(w.ctor))
    vacuum = float(np.max(np.abs(np.asarray(getattr(w, "presf", getattr(w, "pres", [0.0])))))) == 0.0
    n = 48
    return inp, coil_field(run / f"coils.step{step}.json"), vacuum, 4, dict(nphi=n, ntheta=n,
                                                                             quadrature=(4 * int(inp.nfp) * n, 2 * n))


def starts(inp, k):
    def scaled(f):
        r, z = np.array(inp.rbc), np.array(inp.zbs)
        r[:, 1:] *= f
        z[:, 1:] *= f
        return replace(inp, rbc=r, zbs=z)

    r = np.array(inp.rbc)
    r[inp.ntor, 0] += 0.01
    rng = np.random.default_rng(451)
    rp, zp = np.array(inp.rbc), np.array(inp.zbs)
    rp[:, 1:] *= 1 + 0.02 * rng.standard_normal(rp[:, 1:].shape)
    zp[:, 1:] *= 1 + 0.02 * rng.standard_normal(zp[:, 1:].shape)
    six = {"deck": inp, "nestor": None, "x0.97": scaled(0.97), "random2%": replace(inp, rbc=rp, zbs=zp),
           "x1.03": scaled(1.03), "shift+1cm": replace(inp, rbc=r)}
    return dict(list(six.items())[:k])


def lcfs_mm(a, b, nphi=8, ntheta=361):
    """Largest distance (mm) from a's LCFS to b's, against a dense (2881-point) curve of b on each plane."""
    worst = 0.0
    th, dense = np.linspace(0, 2 * np.pi, ntheta), np.linspace(0, 2 * np.pi, 2881)
    for phi in np.linspace(0, np.pi / int(a.nfp), nphi):
        def rz(w, t):
            ang = np.outer(t, np.asarray(w.xm)) - np.asarray(w.xn) * phi
            return np.cos(ang) @ np.asarray(w.rmnc)[-1], np.sin(ang) @ np.asarray(w.zmns)[-1]
        (R, Z), (R0, Z0) = rz(a, th), rz(b, dense)
        worst = max(worst, float(np.max(np.min(np.hypot(R[:, None] - R0[None], Z[:, None] - Z0[None]), axis=1))))
    return 1e3 * worst


def from_start(model, field, start):
    """The three-term fit of ``model`` (one problem definition) from the initial boundary of ``start``, as
    :func:`~vmex.core.freeboundary_three_term.solve_free_boundary_three_term` runs it."""
    from types import SimpleNamespace

    from vmex.core.optimize import pack_boundary

    nb = model.n_boundary
    x0 = np.r_[pack_boundary(start, model.max_mode, vary_major_radius=True), np.asarray(model.x0)[nb:]]
    out = model.solve_boundary(model.params0, field, x0=x0, **FIT, refresh_on_stall=True)
    params = out["aux"][2]
    return SimpleNamespace(input=model._im.input_with_params(model.fixed, params), jac=out["jacobian"],
                           nfev=out["nfev"], njev=out["njev"], converged=out["converged"], success=out["converged"],
                           model=model)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for cname in sys.argv[2:]:
        tag = cname.replace("/", "_").replace(":", "_")
        log = open(out / f"{tag}.jsonl", "a")

        def emit(row):
            row = dict(config=cname, **row)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()

        inp, field, vacuum, k, options = config(cname)
        fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        plan = {}

        def score(deck, label, ref=None, **extra):
            eq = opt.solve_equilibrium(replace(deck, lfreeb=False, mgrid_file="NONE"))
            data = vc.surface_field_data_from_state(fixed, eq.solution, runtime=eq.solver_context, nphi=N, ntheta=N)
            if "p" not in plan:
                plan["p"] = vc.plan_vc_precision(data, digits=4, quad_nt=4 * int(inp.nfp) * N, quad_np=2 * N)
            rows, weights = fvc.boundary_residual(fixed, eq.solution, field, runtime=eq.solver_context, nphi=N, ntheta=N,
                                                  precision=plan["p"])
            res = fvc.summarize_boundary_residual(rows, weights)
            row = dict(label=label, bn=res.normal, pressure=res.pressure, sheet=res.sheet_current,
                       iota_axis=float(np.asarray(eq.wout.iotaf)[0]), iota_edge=float(np.asarray(eq.wout.iotaf)[-1]),
                       r00=float(np.asarray(eq.wout.rmnc)[-1][0]), aspect=float(eq.wout.aspect), **extra)
            if vacuum:
                row["coil_bn"], drift = _vacuum_flux_surface(eq.wout, field, lines=32, transits=30, steps=128)
                row["fieldline_drift_mm"] = 1e3 * drift
            if ref is not None:
                row["lcfs_mm_to_nestor"] = lcfs_mm(eq.wout, ref)
            emit(row)
            return eq.wout

        score(fixed, "deck (fixed boundary)")  # plans the quadrature on the deck
        t0 = time.perf_counter()
        try:
            nestor_deck = _nestor_boundary(inp, external_field=field)
            seconds = round(time.perf_counter() - t0)
            ref = score(nestor_deck, "NESTOR + VMEX", seconds=seconds)
        except Exception as error:  # noqa: BLE001
            emit(dict(label="NESTOR + VMEX", failed=f"{type(error).__name__}: {str(error)[:200]}"))
            nestor_deck, ref = None, None
        for method, kw in METHODS.items():
            wouts, model = {}, None
            for sname, start in starts(inp, k).items():
                if sname == "nestor":
                    if nestor_deck is None:
                        continue
                    start = nestor_deck
                t0 = time.perf_counter()
                try:
                    if os.environ.get("FBB_FIXED_MODEL") == "1" and model is not None:
                        fit = from_start(model, field, start)
                    else:
                        fit = fvc.solve_free_boundary_three_term(inp, external_field=field, initial_boundary=start,
                                                                 **kw, **options, **FIT)
                        model = fit.model
                except Exception as error:  # noqa: BLE001
                    emit(dict(label=f"{method} from {sname}", failed=f"{type(error).__name__}: {str(error)[:200]}"))
                    continue
                J = np.asarray(fit.jac)
                sv = np.linalg.svd(J / np.maximum(np.linalg.norm(J, axis=0), 1e-300), compute_uv=False)
                wouts[sname] = score(fit.input, f"{method} from {sname}", ref=ref,
                                     seconds=round(time.perf_counter() - t0), nfev=int(fit.nfev), njev=int(fit.njev),
                                     converged=bool(fit.converged), success=bool(fit.success), cond=float(sv[0] / sv[-1]),
                                     label_weight=float(getattr(fit.model, "label_weight", np.nan)),
                                     fixed_model=os.environ.get("FBB_FIXED_MODEL") == "1")
            names = list(wouts)
            pairs = {f"{a}|{b}": max(lcfs_mm(wouts[a], wouts[b]), lcfs_mm(wouts[b], wouts[a]))
                     for i, a in enumerate(names) for b in names[i + 1:]}
            if pairs:
                emit(dict(label=f"{method} spread", lcfs_spread_mm=max(pairs.values()), pairs_mm=pairs))
        log.close()


if __name__ == "__main__":
    main()
