"""Which LCFS is right?  The three interface conditions of candidate boundaries under one common, accurate model.

usage: COIL_CASE=qa4-beta-merc python interface_benchmark.py RUN_DIR STEP OUT.jsonl WOUT [WOUT ...]
           [--interior 16] [--boundary-modes 12] [--vc-grid 48 64] [--m1-complement gauged]

Each candidate ``WOUT``'s boundary (a gauge or NL solution of the same coils at any resolution) is evaluated, not
solved: the evaluator's deck (the run's restart deck at ``STEP``, interior at ``--interior``) takes the candidate's
boundary, its interior equilibrium is solved at that resolution, and the RMS B.n, pressure-balance and sheet-current
residuals against the step's coils are reported (the solver's ``initial_boundary_residual``).  The candidate with
the smallest residuals is the most accurate free boundary of those coils, whatever produced it.
"""
import argparse
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from nl_coils_eval import DRIVER, DirectCoilField


def checkpoint_boundary(path, deck):
    """``rbc``, ``zbs`` of ``deck``'s shape holding a checkpoint's boundary (its coordinates on its own deck)."""
    import tempfile
    import vmex as vj
    from vmex.core.optimize import boundary_arrays_from_x

    data = np.load(path, allow_pickle=False)
    definition = json.loads(str(data["meta"]))["definition"]
    with tempfile.NamedTemporaryFile("w", suffix=".run", delete=False) as f:
        f.write(definition["deck"])
    own = vj.VmecInput.from_file(f.name)
    b = np.asarray(data["b"], float)

    def count(K):
        return len([1 for m in range(min(K, own.mpol - 1) + 1) for n in range(-min(K, own.ntor), min(K, own.ntor) + 1)
                    if not (m == 0 and n <= 0)])

    K = next(K for K in range(1, max(own.mpol, own.ntor) + 1) if 2 * count(K) + 1 == b.size)  # the boundary modes
    nm = count(K)
    rbc, zbs = map(np.asarray, boundary_arrays_from_x(own, b[:2 * nm + 1], K,
                                                     vary_major_radius=True))
    out = {k: np.zeros_like(np.asarray(getattr(deck, k), dtype=float)) for k in ("rbc", "zbs")}
    for k, a in (("rbc", rbc), ("zbs", zbs)):
        for i in range(a.shape[0]):
            n = i - int(own.ntor)
            if abs(n) <= int(deck.ntor):
                out[k][n + int(deck.ntor), :min(a.shape[1], out[k].shape[1])] = a[i, :out[k].shape[1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("step", type=int)
    ap.add_argument("out")
    ap.add_argument("wouts", nargs="+")
    ap.add_argument("--interior", type=int, default=16)
    ap.add_argument("--boundary-modes", type=int, default=12)
    ap.add_argument("--vc-grid", type=int, nargs="+", default=[48])
    ap.add_argument("--m1-complement", default="gauged", choices=("solved", "frozen", "gauged"))
    args = ap.parse_args()

    import jax
    import vmex as vj
    from essos.coils import Coils
    from vmex.core.freeboundary_vc import solve_free_boundary_three_term

    jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])
    spec = importlib.util.spec_from_file_location("tdriver", DRIVER)
    tdriver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdriver)
    run = Path(args.run)
    inp0 = vj.VmecInput.from_file(run / "input.run")
    w_step = vj.read_wout(str(run / f"wout.step{args.step}.nc"))
    coils = Coils.from_json(str(run / f"coils.step{args.step}.json"))
    field = DirectCoilField(np.asarray(coils.gamma), np.asarray(coils.gamma_dash), np.asarray(coils.currents))
    for path in args.wouts:
        # the step's deck (PHIEDGE, current), the candidate's boundary, the evaluator's interior
        deck = tdriver.current_from_wout(replace(tdriver.boundary_from_wout(inp0, w_step), phiedge=float(w_step.phi[-1]),
                                                 lfreeb=True, mgrid_file="field(direct)"), w_step)
        deck = deck.change_resolution(mpol=args.interior, ntor=args.interior)
        if path.endswith(".npz"):  # a checkpoint: its boundary coordinates on its own deck
            deck = replace(deck, **checkpoint_boundary(path, deck))
        else:
            deck = tdriver.boundary_from_wout(deck, vj.read_wout(path))
        nfp = int(deck.nfp)
        for grid in args.vc_grid:
            t0 = time.time()
            row = dict(candidate=path, interior=args.interior, boundary_modes=args.boundary_modes, vc_grid=grid,
                       m1=args.m1_complement)
            try:
                fit = solve_free_boundary_three_term(deck, external_field=field, nphi=grid, ntheta=grid,
                                                     label_weight=0.0, max_nfev=1, trial_ftol=1e-11, chunk=8,
                                                     quadrature=(4 * nfp * grid, 2 * grid), max_mode=args.boundary_modes,
                                                     m1_complement=args.m1_complement)
                r = fit.initial_boundary_residual
                row.update(bn=float(r.normal), pb=float(r.pressure), K=float(r.sheet_current))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            row["seconds"] = time.time() - t0
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
