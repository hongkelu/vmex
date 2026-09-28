#!/usr/bin/env python
"""Post-process a finished run of either coil-constraint benchmark.

    python postprocess.py runs/free                   # free-boundary arm
    python postprocess.py runs/fixed --match-flux     # fixed-boundary arm

Reads what both optimization scripts write: ``metrics.jsonl``, the initial
coils, the ``coils.stepN.json`` / ``wout.stepN.nc`` checkpoints and the final
``coils.json`` / ``wout.nc``. Writes into ``<run>/postprocess``:

- ``history.csv``, ``loss.png`` (QA and objective) and ``constraints.png``
  (every constrained quantity against its limits, from ``parameters.py``);
- ``evolution.gif``: coils and LCFS at every checkpoint, coloured by |B| of
  the coil field, and ``initial_final.png``;
- ``wout_dense.nc``, ``dense.json`` and the ``vmex.plot_wout`` figures: the
  final (or latest checkpoint) coils solved free-boundary at ``--ns`` (default NS201), the like-for-
  like QA of both arms. A fixed-arm run need not enclose PHIEDGE; with
  ``--match-flux`` its currents are rescaled by the logged coil-flux ratio,
  which in vacuum changes the field strength only.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory written by either benchmark")
    parser.add_argument("--ns", type=int, default=201, help="radial resolution of the dense free-boundary solve")
    parser.add_argument("--no-dense", action="store_true", help="skip the dense free-boundary solve")
    parser.add_argument("--match-flux", action="store_true", help="rescale fixed-arm currents to enclose PHIEDGE")
    parser.add_argument("--seed-wout", type=Path, help="initial state of the dense solve (default: the run's wout.nc)")
    parser.add_argument("--max-iterations", type=int, default=12000, help="iteration cap of the dense solve")
    parser.add_argument("--flux-tolerance", type=float, help="flux band the fixed-arm run used, for the plot")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args(argv)


def read_history(run):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    keys = [k for k in rows[0] if all(isinstance(r.get(k), (int, float)) for r in rows)]
    return rows, keys


def limits(flux_tolerance=None):
    """Constrained quantities and their bounds, as the two benchmarks impose them."""
    import single_stage_optimization as fixed
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN
    band = (None, None) if flux_tolerance is None else (1 - flux_tolerance, 1 + flux_tolerance)
    return {
        "min_abs_iota": ("min |iota|", P.IOTA_FLOOR, None),
        "aspect": ("aspect ratio", *P.ASPECT_RANGE),
        "major_radius_m": ("major radius [m]", P.RADIUS_TARGET - width, P.RADIUS_TARGET + width),
        "coil_surface_distance_m": ("coil-plasma distance [m]", P.COIL_SURFACE_DISTANCE_LIMIT, None),
        "coil_minimum_scaled_slack": ("min coil slack", 0.0, None),
        "normal_field_rms": ("rms B.n/|B|", None, fixed.NORMAL_FIELD_CONSTRAINT),
        "flux_ratio": ("coil flux / PHIEDGE", *band),
        "phiedge_factor": ("PHIEDGE / seed PHIEDGE", None, None),
    }


def plot_history(rows, keys, out, flux_tolerance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    step = [r["step"] for r in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("qa", "QA"), ("objective", "objective")):
        if key in keys:
            axis.semilogy(step, [max(r[key], 1e-16) for r in rows], label=label)
    axis.set_xlabel("accepted step"), axis.grid(True, alpha=0.3), axis.legend()
    figure.tight_layout(), figure.savefig(out / "loss.png", dpi=200), plt.close(figure)

    panels = [(k, *spec) for k, spec in limits(flux_tolerance).items() if k in keys]
    columns = 3
    figure, axes = plt.subplots((len(panels) + columns - 1) // columns, columns,
                                figsize=(4 * columns, 3 * ((len(panels) + columns - 1) // columns)), squeeze=False)
    for axis, (key, label, lower, upper) in zip(axes.flat, panels):
        axis.plot(step, [r[key] for r in rows], lw=1)
        for bound in (lower, upper):
            if bound is not None:
                axis.axhline(bound, color="r", ls="--", lw=0.8)
        axis.set_title(label, fontsize=9), axis.set_xlabel("accepted step", fontsize=8), axis.grid(True, alpha=0.3)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.tight_layout(), figure.savefig(out / "constraints.png", dpi=200), plt.close(figure)


def checkpoints(run):
    """(label, coils, wout) for every saved step, then the final state."""
    steps = sorted(int(m.group(1)) for p in run.glob("coils.step*.json")
                   if (m := re.fullmatch(r"coils\.step(\d+)\.json", p.name)) and (run / f"wout.step{m.group(1)}.nc").exists())
    frames = [(f"step {s}", run / f"coils.step{s}.json", run / f"wout.step{s}.nc") for s in steps]
    if (run / "coils.json").exists() and (run / "wout.nc").exists():
        frames.append(("final", run / "coils.json", run / "wout.nc"))
    return frames


def plot_evolution(frames, out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    objects = [(SurfaceRZFourier.from_wout_file(str(w), nphi=64, ntheta=32), Coils.from_json(str(c)))
               for _, c, w in frames]

    def modB(x, pair):
        surface, coils = pair
        points = jnp.asarray(surface.gamma).reshape(-1, 3)
        return jnp.linalg.norm(jax.vmap(BiotSavart(coils).B)(points), axis=-1).reshape(surface.gamma.shape[:-1])

    vj.plot_optimization_movie(out / "evolution.gif", [np.array([i]) for i in range(len(objects))],
                               lambda x: objects[int(x[0])], color_factory=modB, color_label="|B| [T]")
    vj.plot_optimization_objects(out / "initial_final.png", (frames[0][0], *objects[0]), (frames[-1][0], *objects[-1]))


def dense_solve(frame, args, rows, out):
    import numpy as np
    import vmex
    from vmex import optimize as opt
    from vmex.core.freeboundary import _solve_free_boundary_stage, free_boundary_resolution
    from vmex.core.solver import prepare_runtime
    from vmex.core.statephysics import major_radius
    from vmex.core.wout import wout_from_state
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from free_boundary_single_stage_optimization import resize_coils

    label, coil_path, wout_path = frame
    mpol, ntor, _ = P.RESOLUTION
    run = Path(coil_path).parent
    deck = run / "input.run" if (run / "input.run").exists() else P.INPUT_FILE  # older runs: the case deck
    inp = vmex.VmecInput.from_file(deck).change_resolution(mpol=mpol, ntor=ntor, ntheta=P.GRID[0], nzeta=P.GRID[1])
    phiedge = float(vmex.read_wout(wout_path).phi[-1])  # a free-PHIEDGE run ends at its own PHIEDGE
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array([args.ns]), phiedge=phiedge,
                  ftol_array=np.array([P.EQUILIBRIUM_FTOL]), niter_array=np.array([args.max_iterations]))
    coils = resize_coils(Coils.from_json(str(coil_path)), P.COIL_ORDER, P.N_SEGMENTS)
    row = rows[-1] if label == "final" else next(r for r in rows if f"step {r['step']}" == label)
    scale = 1.0 / row["flux_ratio"] if args.match_flux else 1.0
    if scale != 1.0:
        coils = Coils(coils.curves, coils.dofs_currents_raw * scale, currents_scale=coils.currents_scale)
    field = BiotSavart(coils)
    seed = args.seed_wout or wout_path
    state = vmex.state_from_wout(vmex.read_wout(seed), inp=inp, ns=args.ns)
    resolution = free_boundary_resolution(inp, field, ns=args.ns)
    started = time.monotonic()
    stage = _solve_free_boundary_stage(
        inp, external_field=field, resolution=resolution, initial_state=state, ftol=P.EQUILIBRIUM_FTOL,
        max_iterations=args.max_iterations, include_edge_in_convergence=True, edge_force_tolerance=P.EQUILIBRIUM_FTOL,
        use_fft=False, error_on_no_convergence=False, jacobian_retries=0, allow_initial_axis_reguess=False)
    result = stage.result
    forces = {k: float(getattr(result, k)) for k in ("fsqr", "fsqz", "fsql", "fedge")}
    converged = bool(result.converged and result.vacuum is not None
                     and all(np.isfinite(v) and 0 <= v <= P.EQUILIBRIUM_FTOL for v in forces.values()))
    report = dict(frame=label, coils=str(coil_path), ns=args.ns, seed=str(seed), current_scale=scale, currents_A=np.asarray(coils.currents).tolist(),
                  converged=converged, iterations=int(result.iterations), seconds=time.monotonic() - started, **forces)
    if converged:
        wout = wout_from_state(inp=inp, state=result.state, niter=int(result.iterations), converged=True,
                               vacuum_output=result.vacuum, **{k: forces[k] for k in ("fsqr", "fsqz", "fsql")})
        path = vmex.write_wout(str(out / "wout_dense.nc"), wout)
        rt = prepare_runtime(inp, resolution)
        qs = opt.QuasisymmetryRatioResidual(np.asarray(P.QA_SURFACES), 1, 0)
        report.update(qa=float(qs.total(wout)), min_abs_iota=float(opt.min_abs_iota(result.state, rt)),
                      aspect=float(opt.aspect_ratio(result.state, rt)), major_radius_m=float(major_radius(result.state, rt)))
        report["figures"] = [str(p) for p in vmex.plot_wout(path, out, name="dense").values()]
    else:
        print("dense solve did not converge; seed it closer with --seed-wout", flush=True)
    (out / "dense.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    args = parse_args(argv)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if args.device == "gpu" else "cpu")
    run = args.run.resolve()
    out = run / "postprocess"
    out.mkdir(exist_ok=True)
    rows, keys = read_history(run)
    if args.match_flux and "flux_ratio" not in keys:
        raise SystemExit("--match-flux needs a fixed-arm run that logs flux_ratio")
    plot_history(rows, keys, out, args.flux_tolerance)
    frames = checkpoints(run)
    if frames:
        plot_evolution(frames, out)
    if not args.no_dense and frames:  # the final state, or the latest checkpoint of a running job
        print(json.dumps(dense_solve(frames[-1], args, rows, out)), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
