"""Make a frozen fixture self-contained: exact kinetic profiles, current representation, closure convention, hashes.

usage: COIL_CASE=<case> python fixture_manifest.py FIXTURE_DIR "<driver arguments>"
    (cwd: a vmex tree with the driver; JAX_PLATFORMS=cpu is enough: nothing is solved)

Writes FIXTURE_DIR/profiles.json (the coefficients the driver's ``--seed-run`` path uses: ``redl_profiles(input.run)``,
or the profiles.json of the run itself when present) and FIXTURE_DIR/fixture.json.  The frozen input.run and
coils.initial.json are not changed; SHA256SUMS is extended to the new files.
"""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

DRIVER = "examples/optimization/single_stage_free_boundary_optimization_three_term.py"


def sha256(path):
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Run the probe."""
    folder, driver_args = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ""
    import vmex as vj

    module = runpy.run_path(DRIVER, run_name="fixture_manifest")  # the case table, without running the driver
    inp = vj.VmecInput.from_file(str(folder / "input.run"))
    bootstrap = "--bootstrap" in driver_args.split()
    fixture = dict(coil_case=os.environ.get("COIL_CASE"), driver_args=driver_args, nfp=int(inp.nfp),
                   mpol=int(inp.mpol), ntor=int(inp.ntor), ns=[int(v) for v in np.atleast_1d(inp.ns_array)],
                   phiedge=float(inp.phiedge), pres_scale=float(inp.pres_scale), ncurr=int(inp.ncurr),
                   current=dict(pcurr_type=str(inp.pcurr_type), curtor=float(inp.curtor),
                                ac=np.asarray(inp.ac, float).tolist(),
                                ac_aux_s=np.asarray(inp.ac_aux_s, float).tolist(),
                                ac_aux_f=np.asarray(inp.ac_aux_f, float).tolist()))
    if bootstrap:
        profiles = module["redl_profiles"](inp)[0]
        model = str(module.get("BOOTSTRAP_MODEL", "redl"))
        fixture["closure"] = dict(
            model=model, in_solve=bool(module.get("BOOTSTRAP_IN_SOLVE", False)) and model == "redl",
            helicity_n=int(module["REDL_HELICITY"]), g_convention="simsopt",
            note=("QA/QH: Redl half-mesh closure solved with the boundary (HalfMeshCurrent); QI: DKX kinetic mismatch "
                  "as an outer constraint on the current coordinates"))
        coefficients = {k: np.atleast_1d(np.asarray(getattr(profiles, k), float)).tolist()
                        for k in ("ne_coeffs", "Te_coeffs", "Ti_coeffs", "Zeff_coeffs")}
        (folder / "profiles.json").write_text(json.dumps(coefficients, indent=1) + "\n")
        fixture["profiles"] = dict(file="profiles.json", n0=coefficients["ne_coeffs"][0],
                                   T0=coefficients["Te_coeffs"][0],
                                   source="redl_profiles(input.run) of the driver (the --seed-run path)")
    else:
        fixture["closure"] = dict(model="none (vacuum: prescribed zero current)")
    files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "fixture.json"))
    fixture["sha256"] = {name: sha256(folder / name) for name in files}
    fixture["driver_sha256"] = sha256(DRIVER)
    (folder / "fixture.json").write_text(json.dumps(fixture, indent=1) + "\n")
    names = files + ["fixture.json"]
    (folder / "SHA256SUMS").write_text("".join(f"{sha256(folder / n)}  {n}\n" for n in names))
    print(f"{folder}: {', '.join(names)}")


if __name__ == "__main__":
    main()
