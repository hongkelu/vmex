#!/usr/bin/env python
"""WOUT figures and alpha losses of a three-term single-stage run's final equilibrium.

    COIL_CASE=qa4-beta python three_term_postprocess.py <run dir> [--ns 201] [--bootstrap] [--trace-args "..."]

The run's final boundary is the sheet-current-free free boundary of its final coils; this re-solves it as a
fixed boundary at ``--ns`` (the campaign's dense radial resolution) with the run's PHIEDGE, pressure and current
-- not with VMEC + NESTOR, which would leave that branch. Writes into ``<run>/postprocess``: ``wout_dense.nc``,
``dense.json`` (QA, min/max |iota|, aspect, major radius, bootstrap mismatch), the ``vmex.plot_wout`` figures
and, unless ``--no-trace``, ``trace/`` (``vmex --trace`` alpha losses, as postprocess.py's ``--trace``).
"""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--ns", type=int, default=201)
p.add_argument("--bootstrap", action="store_true", help="report the Redl mismatch of the run's profiles")
p.add_argument("--no-trace", action="store_true")
p.add_argument("--trace-args", default="")
args = p.parse_args()
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np  # noqa: E402
import vmex  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from _common import max_abs_iota, min_abs_iota, redl_profiles, restart_input, target_residual  # noqa: E402
from postprocess import trace  # noqa: E402

out = args.run.resolve() / "postprocess"
out.mkdir(exist_ok=True)
inp = restart_input(args.run)
deck = replace(inp, ns_array=np.array([args.ns]), ftol_array=np.array([1e-13]), niter_array=np.array([60000]))
eq = opt.solve_equilibrium(deck, initial_state=vmex.state_from_wout(vmex.read_wout(args.run / "wout.nc"), inp=deck,
                                                                     ns=args.ns))
w = eq.wout
path = out / "wout_dense.nc"
vmex.write_wout(str(path), w)
q = np.asarray(target_residual().residuals_state(eq.solution, eq.solver_context))
report = dict(ns=args.ns, fsq=float(w.fsqr + w.fsqz + w.fsql), qa=float(q @ q),
              min_abs_iota=float(min_abs_iota(eq.solution, eq.solver_context)),
              max_abs_iota=float(max_abs_iota(eq.solution, eq.solver_context)), aspect=float(w.aspect),
              major_radius_m=float(w.Rmajor_p), betatotal=float(w.betatotal))
if args.bootstrap:
    report["redl_mismatch"] = float(redl_profiles(deck)[1].total_state(eq.solution, eq.solver_context))
report["figures"] = [str(f) for f in vmex.plot_wout(str(path), out, name="three_term").values()]
(out / "dense.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if not args.no_trace:
    trace(path, out, args.trace_args)
print(f"wrote {out}", flush=True)
