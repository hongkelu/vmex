"""Authenticate a completed free-boundary segment before restoring its endpoint."""
import csv
import hashlib
import json

import numpy as np


def load_endpoint(run, manifest, repo, settings, args, script):
    """Keep the original coil coordinates; restart only SciPy's optimizer state."""
    run = run.resolve()
    hashes = json.loads(manifest.read_text())["files_sha256"]

    def checked(name):
        path = run / name
        key = str(path.relative_to(repo))
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(key):
            raise ValueError(f"unauthenticated resume file: {name}")
        return path

    previous = json.loads(checked("provenance.json").read_text())
    summary = json.loads(checked("free_boundary_single_stage_scalar_optimization_summary.json").read_text())
    if summary["status"] not in ("iteration_budget_reached", "converged"):
        raise ValueError("resume requires a normally completed optimization segment")
    if previous["example_settings"] != settings:
        raise ValueError("resume settings differ")
    for name in ("constrained", "method", "ftol", "verify_ns", "verify_ftol"):
        if previous["settings"][name] != getattr(args, name):
            raise ValueError(f"resume setting differs: {name}")
    current = {script: previous["script_sha256"],
               script.parent / "input.rotating_ellipse": previous["input_sha256"],
               args.coils.resolve(): previous["stage_two"]["source_sha256"],
               **{repo / name: digest for name, digest in previous["core_sha256"].items()}}
    for name in ("_scalar_constraints.py", "_scalar_diagnostics.py"):
        current[script.parent / name] = hashlib.sha256(checked(name).read_bytes()).hexdigest()
    for path, digest in current.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"resume physics or coordinates changed: {path.name}")
    with checked("free_boundary_scalar_steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    step = int(summary["accepted"])
    if [int(row["step"]) for row in rows] != list(range(step + 1)):
        raise ValueError("accepted history is not contiguous")
    for index in range(step + 1):
        checked(f"accepted_{index:04d}.npz")
    checkpoint = checked(f"accepted_{step:04d}.npz")
    with np.load(checkpoint, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("nonfinite checkpoint")
    return arrays, dict(source_run=str(run), checkpoint_sha256=hashes[str(checkpoint.relative_to(repo))],
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        step_offset=summary.get("absolute_step", step), optimizer_state="fresh SLSQP state; Hessian not saved",
        previous_final=summary["final"], previous_verified=summary.get("verified"),
        previous_unmet=summary.get("unmet"), unchanged_physics=True)
