#!/usr/bin/env python
"""Re-solve a run's final coils at a higher resolution with the three-term free boundary, and export them for DESC.

    COIL_CASE=qa4-beta python three_term_highres.py <run dir> <out dir> [--modes 12 12] [--ns 101]

``<run dir>`` holds input.run, coils.json and wout.nc (a finished run). The deck's profiles and the run's final
boundary (as the start) are re-solved at ``--modes``/``--ns`` with every interface condition (B.n, pressure balance,
no sheet current). Writes ``<out>/wout_three_term.nc``, ``<out>/report.json`` (interface residuals, QA, iota, aspect,
LCFS distance to the run's own boundary, time, memory) and ``<out>/coils.npz`` (filament points, tangents and
currents of every coil, with the field at check points: the exact field for DESC).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--modes", type=int, nargs=2, default=(12, 12))
p.add_argument("--ns", type=int, default=101)
p.add_argument("--chunk", type=int, default=4)
p.add_argument("--quadrature", type=int, nargs=2, help="default 4 nfp 48 x 96")
p.add_argument("--export-only", action="store_true", help="write coils.npz and stop")
args = p.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from essos.coils import Coils  # noqa: E402
import vmex as vj  # noqa: E402
import _coil_constraints as coil_limits  # noqa: E402
from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel  # noqa: E402
from vmex.core.optimize import solve_equilibrium, unpack_boundary  # noqa: E402
from _common import min_abs_iota, restart_input, target_residual  # noqa: E402

args.out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - t0:7.0f}s] {msg}", flush=True)


mpol, ntor = args.modes
inp = restart_input(args.run).change_resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 6)
inp = replace(inp, ns_array=np.array([args.ns]), ftol_array=np.array([1e-13]), niter_array=np.array([100000]))
chart = coil_limits.CoilChart(Coils.from_json(str(args.run / "coils.json")), current_dofs=())
field = chart(jnp.asarray(chart.x0))
rng = np.random.default_rng(0)
check = jnp.asarray(np.c_[1.0 + 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64)])
from vmex.core import virtual_casing as vc  # noqa: E402
B_check = np.asarray(vc.external_B_cartesian(field, check.T[:, :, None]))[:, :, 0].T
np.savez(args.out / "coils.npz", gamma=np.asarray(field.gamma), gamma_dash=np.asarray(field.gamma_dash),
         currents=np.asarray(field.currents), check_points=np.asarray(check), check_B=B_check)
log(f"deck {mpol}x{ntor} ns {args.ns}; {np.asarray(field.gamma).shape[0]} coils exported")
if args.export_only:
    sys.exit(0)

model = ThreeTermFreeBoundaryModel(inp, chunk=args.chunk, quadrature=args.quadrature or (4 * int(inp.nfp) * 48, 96),
                           trial_ftol=1e-11)
log(f"model: {model.x0.size} boundary coordinates")
out = model.solve_boundary(model.params0, field, verbose=2, max_nfev=80)
seconds = time.perf_counter() - t0
state, _, params, _ = out["aux"]
deck = unpack_boundary(model.fixed, out["x"], model.max_mode, vary_major_radius=True)
eq = solve_equilibrium(deck, initial_state=state)
w = eq.wout
vj.write_wout(str(args.out / "wout_three_term.nc"), w)
res = model.boundary_residual(eq.solution, model.with_boundary(model.params0, jnp.asarray(out["x"])), field)
q = np.asarray(target_residual().residuals_state(eq.solution, eq.solver_context))
th, planes = np.linspace(0, 2 * np.pi, 721), np.linspace(0, np.pi / int(inp.nfp), 4)


def lcfs_mm(wa, wb):
    d = 0.0
    for ph in planes:
        a, b = (np.outer(th, np.asarray(x.xm, float)) - np.asarray(x.xn, float) * ph for x in (wa, wb))
        Ra, Za = np.cos(a) @ np.asarray(wa.rmnc)[-1], np.sin(a) @ np.asarray(wa.zmns)[-1]
        Rb, Zb = np.cos(b) @ np.asarray(wb.rmnc)[-1], np.sin(b) @ np.asarray(wb.zmns)[-1]
        d = max(d, float(np.max(np.min(np.hypot(Ra[:, None] - Rb[None], Za[:, None] - Zb[None]), axis=1))))
    return round(1e3 * d, 3)


report = dict(run=str(args.run), modes=[mpol, ntor], ns=args.ns, boundary_coordinates=int(model.x0.size),
              seconds=round(seconds), nfev=out["nfev"], njev=out["njev"], fsq=float(w.fsqr + w.fsqz + w.fsql),
              normal=res.normal, pressure=res.pressure, sheet_current=res.sheet_current, qa=float(q @ q),
              min_abs_iota=float(min_abs_iota(eq.solution, eq.solver_context)),
              iota_axis=float(abs(np.asarray(w.iotaf)[0])), iota_edge=float(abs(np.asarray(w.iotaf)[-1])),
              aspect=float(w.aspect), lcfs_mm_vs_run=lcfs_mm(w, vj.read_wout(str(args.run / "wout.nc"))),
              peak_gpu_gb=round((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9, 2))
(args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
log("DONE " + json.dumps(report))
