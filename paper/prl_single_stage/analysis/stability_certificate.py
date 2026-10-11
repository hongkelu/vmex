"""Ideal-MHD stability certificate of a run step: Mercier (DMerc) and infinite-n ideal ballooning (vmex COBRA analogue).

    COIL_CASE=qa4-beta JAX_PLATFORMS=cpu python analysis/stability_certificate.py <run dir> [--step N] [--ns 71] [--out file.json]

Re-solves the step's LCFS with its pressure and current profiles as a fixed-boundary equilibrium (the equilibrium
the coils make, on a finer radial grid), then evaluates DMerc on every surface and the ballooning eigenvalue lambda
(> 0 unstable, normalized squared growth rate (gamma a_N / v_A)^2) on surfaces s = 0.3..0.95, 4 field lines alpha in
[0, pi], zeta0 scanned over [-pi/2, pi/2] (5 values), 121 points x 3 turns.  No finite-n code exists in vmex.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
T = P.parent / "vmex/examples/optimization/single_stage_free_boundary_optimization_three_term.py"
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--step", type=int, default=None)
ap.add_argument("--ns", type=int, default=71)
ap.add_argument("--ftol", type=float, default=1e-13)
ap.add_argument("--out", default=None)
args = ap.parse_args()
d = Path(args.run)
step = args.step if args.step is not None else max(int(re.search(r"step(\d+)", q.name).group(1)) for q in d.glob("wout.step*.nc"))
wout = d / f"wout.step{step}.nc"
os.environ.setdefault("COIL_CASE", "qa4-beta")

import jax.numpy as jnp
import vmex as vj
from vmex import optimize as opt
from vmex.core.stability import ballooning_lambda

spec = importlib.util.spec_from_file_location("tdriver", T)
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)

w = vj.read_wout(wout)
inp = vj.VmecInput.from_file(d / "input.run")
inp = td.current_from_wout(replace(td.boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)
inp = replace(inp, ns_array=np.array([args.ns]), ftol_array=np.array([args.ftol]), niter_array=np.array([20000]))
print(f"{d.name} step {step}: beta {float(w.betatotal):.4f}, re-solving fixed boundary at NS {args.ns}", flush=True)
eq = opt.solve_equilibrium(inp, verbose=False)
state, rt = eq.solution, eq.solver_context
phiedge = float(inp.phiedge)
dmerc = np.asarray(opt.d_merc_state(state, rt))
s = np.linspace(0, 1, dmerc.shape[0])
win = (s >= 0.1) & (s < 1.0)
S_B = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
js = [min(max(int(round(x * (args.ns - 1))), 2), args.ns - 2) for x in S_B]
lam = np.asarray(ballooning_lambda(state, rt, s_indices=js, zeta0s=np.linspace(-np.pi / 2, np.pi / 2, 5), npoints=121, nturns=3.0))
lam_s = lam.reshape(len(js), -1).max(axis=1)
res = dict(run=d.name, step=step, ns=args.ns, beta=float(w.betatotal), s=s.tolist(), dmerc=dmerc.tolist(),
           phiedge2_dmerc_min=float((phiedge**2 * dmerc)[win].min()), dmerc_frac_stable=float(np.mean(dmerc[win] > 0)),
           wout_dmerc_frac_stable=float(np.mean(np.asarray(w.DMerc)[win] > 0)) if hasattr(w, "DMerc") else None,
           ballooning_s=S_B, ballooning_lambda_max=lam_s.tolist(), ballooning_lambda_global=float(lam.max()),
           ballooning_unstable_surfaces=[x for x, l in zip(S_B, lam_s) if l > 0])
print(f"DMerc (>0 stable): frac of surfaces stable on s in [0.1,1) = {res['dmerc_frac_stable']:.2f}; "
      f"min PHIEDGE^2 DMerc = {res['phiedge2_dmerc_min']:+.3e}")
print("DMerc at s=0.1..0.9:", np.array2string(np.interp(np.arange(1, 10) / 10, s, dmerc), precision=3))
print("ballooning max lambda per surface:", " ".join(f"s={x:.2f}:{l:+.2e}" for x, l in zip(S_B, lam_s)))
print(f"=> {'ballooning UNSTABLE on ' + str(res['ballooning_unstable_surfaces']) if lam.max() > 0 else 'ballooning stable on all sampled surfaces'}")
out = Path(args.out) if args.out else d / f"stability.step{step}.json"
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
