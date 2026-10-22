"""Run the single-stage driver with a complete, reproducible diagnostic record (work package 0).

usage: python fbrecord.py OUT <driver arguments without --output>
    (cwd: a vmex tree with examples/optimization/single_stage_free_boundary_optimization_three_term.py)

OUT must not exist.  It receives:

- ``manifest.json`` (written first): run id, argv, git revision and dirty-diff hash, sha256 of this probe, of the
  driver and of the library modules, dependency versions, every experimental override, GPU identity, host, Slurm job;
- ``config.json`` (after the model is built): the effective resolution, labelling, bootstrap closure identity, row and
  column layout; the run stops as ``configuration_mismatch`` if the requested --modes, --ns or --vc-grid did not take
  effect;
- ``records/NNNN_<kind>.npz``: every certificate round and outcome of every trial (``ThreeTermFreeBoundaryModel.recorder``),
  written atomically: coordinates, interior state, rows, Jacobian, scales, normal operator, boundary arrays, layouts,
  resolution, tolerances, certificate, values (diagnostic) and status;
- ``status.json`` (last): ``certified``, ``rejected``, ``budget_exhausted``, ``configuration_mismatch`` or
  ``unexpected_error`` (exit code 1, with the traceback).  The driver's own output goes to ``OUT/run``.

Experimental overrides (all recorded): POLISH_ROUNDS, POLISH_MAX_NFEV, STALL_STEPS (module constants), MAX_MODE
(the model's boundary max_mode).
"""
import datetime
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
import traceback
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np

DRIVER = Path("examples/optimization/single_stage_free_boundary_optimization_three_term.py")
OVERRIDES = ("POLISH_ROUNDS", "POLISH_MAX_NFEV", "STALL_STEPS", "STALL_RULE", "MAX_MODE")
PACKAGES = ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda12-plugin", "numpy", "scipy", "solvax", "booz_xform_jax",
            "essos", "dkx", "simsopt", "netCDF4", "equinox", "lineax")


def sha256(path):
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    """Output of a git command, or None."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, data):
    """Write JSON atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def gpu_identity():
    """The visible GPUs (nvidia-smi) and CUDA_VISIBLE_DEVICES."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), devices=out.strip().splitlines())


def versions():
    """Installed versions of the relevant packages."""
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def requested(argv):
    """The resolution the driver was asked for (--modes M N, --ns NS, --vc-grid G)."""
    def value(flag, count=1):
        if flag not in argv:
            return None
        i = argv.index(flag)
        return [int(v) for v in argv[i + 1:i + 1 + count]]
    return dict(modes=value("--modes", 2), ns=value("--ns"), vc_grid=value("--vc-grid"))


def main():
    """Run the probe."""
    out = Path(sys.argv[1]).resolve()
    driver_args = sys.argv[2:]
    if "--output" in driver_args:
        raise SystemExit("fbrecord.py sets --output itself (OUT/run)")
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run directory
    (out / "records").mkdir()
    run_id = f"{datetime.datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    library = [Path("vmex/core/freeboundary_vc.py"), Path("vmex/core/bootstrap.py"), Path("vmex/core/implicit.py")]
    diff = git("diff", "HEAD")
    write_json(out / "manifest.json", dict(
        run_id=run_id, argv=sys.argv, cwd=os.getcwd(), started=datetime.datetime.now().isoformat(),
        git_revision=git("rev-parse", "HEAD"), git_dirty_diff_sha256=hashlib.sha256((diff or "").encode()).hexdigest(),
        git_status=git("status", "--porcelain"),
        sha256={str(p): sha256(p) for p in [Path(__file__), DRIVER, *library] if p.exists()},
        versions=versions(), python=sys.version, host=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"),
        gpu=gpu_identity(), overrides={k: os.environ[k] for k in OVERRIDES if k in os.environ},
        environment={k: v for k, v in os.environ.items() if k.startswith(("COIL_CASE", "XLA", "JAX", "CUDA"))}))

    from vmex.core import freeboundary_vc as fv

    for name in ("POLISH_ROUNDS", "POLISH_MAX_NFEV", "STALL_STEPS"):
        if name in os.environ:
            setattr(fv, name, int(os.environ[name]))
    if "STALL_RULE" in os.environ:
        fv.STALL_RULE = os.environ["STALL_RULE"]
    counter = [0]
    certified = []

    def recorder(entry):
        counter[0] += 1
        kind = entry["kind"]
        if kind == "certified":
            certified.append(counter[0])
        arrays = {k: v for k, v in entry.items() if isinstance(v, np.ndarray)}
        state = entry.get("state") or []
        arrays.update({f"state{i}": np.asarray(a) for i, a in enumerate(state)})
        normal = entry.get("normal")
        if normal is not None:
            arrays["normal"] = np.asarray(normal)
        if entry.get("values") is not None:
            arrays["values"] = np.asarray(entry["values"])
        meta = {k: v for k, v in entry.items() if k not in arrays and k not in ("state", "normal", "values")}
        meta.update(run_id=run_id, sequence=counter[0], state_leaves=len(state))
        path = out / "records" / f"{counter[0]:04d}_{kind}.npz"
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta, default=str), **arrays)
        os.replace(tmp, path)

    init = fv.ThreeTermFreeBoundaryModel.__init__
    want = requested(driver_args)

    def recording_init(self, *args, **kwargs):
        if "MAX_MODE" in os.environ:
            kwargs["max_mode"] = int(os.environ["MAX_MODE"])
        init(self, *args, **kwargs)
        self.recorder = recorder
        config = dict(resolution=self.resolution(), labelling=self.options["labelling"],
                      label_weight=self.options["label_weight"], bootstrap=self.options["bootstrap"],
                      weights=self.options["weights"], net_current_weight=self.options["net_current_weight"],
                      row_blocks=self.row_blocks, column_blocks=self.column_blocks, requested=want)
        mismatch = []
        res = self.resolution()
        if want["modes"] and [res["mpol"], res["ntor"]] != want["modes"]:
            mismatch.append(f"--modes {want['modes']} but the model has mpol {res['mpol']}, ntor {res['ntor']}")
        if want["ns"] and res["ns"][-1] != want["ns"][0]:
            mismatch.append(f"--ns {want['ns'][0]} but the model's ns is {res['ns'][-1]}")
        if want["vc_grid"] and (res["nphi"], res["ntheta"]) != (want["vc_grid"][0], want["vc_grid"][0]):
            mismatch.append(f"--vc-grid {want['vc_grid'][0]} but the model's grid is {res['nphi']} x {res['ntheta']}")
        config["mismatch"] = mismatch
        write_json(out / "config.json", config)
        if mismatch:
            raise ConfigurationMismatch("; ".join(mismatch))

    fv.ThreeTermFreeBoundaryModel.__init__ = recording_init
    status = dict(run_id=run_id)
    code = 0
    sys.argv = [str(DRIVER), *driver_args, "--output", str(out / "run")]
    try:
        runpy.run_path(str(DRIVER), run_name="__main__")
        status.update(status="certified" if certified else "completed_without_certificate")
    except ConfigurationMismatch as error:
        status.update(status="configuration_mismatch", reason=str(error))
        code = 1
    except SystemExit as error:
        if error.code not in (0, None):
            status.update(status="unexpected_error", reason=f"SystemExit({error.code})")
            code = 1
        else:
            status.update(status="certified" if certified else "completed_without_certificate")
    except Exception as error:  # classified below; anything unrecognized is an unexpected error
        text = str(error)
        if type(error).__name__ == "VmecError" and "has no three-term free boundary" in text:
            exhausted = all("ran out of evaluations" in part for part in text.split(": ", 1)[-1].split("; "))
            status.update(status="budget_exhausted" if exhausted else "rejected", reason=text)
        else:
            status.update(status="unexpected_error", reason=f"{type(error).__name__}: {text}",
                          traceback=traceback.format_exc())
            code = 1
    status.update(records=counter[0], certified_records=certified, finished=datetime.datetime.now().isoformat())
    write_json(out / "status.json", status)
    print(f"[fbrecord] {status['status']} ({counter[0]} records) -> {out}", flush=True)
    sys.exit(code)


class ConfigurationMismatch(RuntimeError):
    """A requested resolution did not take effect."""


if __name__ == "__main__":
    main()
