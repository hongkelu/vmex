"""What the rho gauge of the m = 1 complement (complement_gauge.py) changes physically, against the cold state.

usage: python complement_physics.py OUT.json   (cwd: a vmex tree; a GPU; env MPOL, NS, FTOL as in rows_gauge.py)

The cold start refined as is and refined in the rho gauge: iota, B_min/B_max per surface and the largest surface
distance (interior_gauge.py's measures, on wouts re-solved from the refined states), and the rows by block (RMS of
the change relative to the block's RMS, and the largest change relative to the largest row).
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

import jax.numpy as jnp

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from interior_gauge import distance, physical  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import implicit as im  # noqa: E402


def main():
    """Rows from three starts, with and without the cold complement; write OUT.json."""
    inp, field = t._small_vacuum()
    if os.environ.get("MPOL") or os.environ.get("NS"):
        inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_lowres", int(os.environ.get("MPOL", 2)),
                                          int(os.environ.get("NS", 7))), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    if os.environ.get("FTOL"):
        inp = dataclasses.replace(inp, ftol_array=np.array([float(os.environ["FTOL"])]), niter_array=np.array([50000]))
    model = fvc.ThreeTermFreeBoundaryModel(dataclasses.replace(inp, lfreeb=False), **t.SMALL)
    x, params = np.asarray(model.x0), model.params0
    mask = im._fixed_boundary_dof_mask(model.cfg)
    pos, neg = im._m1_pair_columns(model.cfg)
    _, aux = model.evaluate(x, params, field, tight=True)
    cold, params_x = aux[0], aux[2]
    v = np.asarray(cold.Z_sin)
    anti = 0.5 * (v[:, pos] - v[:, neg])
    w = np.sqrt(np.linspace(0.0, 1.0, v.shape[0]))
    sym = 0.5 * (v[:, pos] + v[:, neg])
    g = v.copy()
    g[:, pos], g[:, neg] = sym + w[:, None] * anti[-1], sym - w[:, None] * anti[-1]
    states = dict(plain=im._refined_state(model.cfg, params_x, cold, mask),
                  rho=im._refined_state(model.cfg, params_x, dataclasses.replace(cold, Z_sin=jnp.asarray(g)), mask))
    rows = {k: np.asarray(model._rows(st, params_x, field)) for k, st in states.items()}
    phys = {k: physical(opt.solve_equilibrium(model.fixed, initial_state=st).wout) for k, st in states.items()}
    out = dict(resolution=model.resolution(),
               iota=float(np.max(np.abs(phys["rho"]["iota"] - phys["plain"]["iota"]))),
               iota_scale=float(np.max(np.abs(phys["plain"]["iota"]))),
               bmin=float(np.max(np.abs(phys["rho"]["bmin"] - phys["plain"]["bmin"]))),
               bmax=float(np.max(np.abs(phys["rho"]["bmax"] - phys["plain"]["bmax"]))),
               surface_by_js=[distance(a, b) for a, b in zip(phys["rho"]["surfaces"], phys["plain"]["surfaces"])],
               rows_max_rel=float(np.max(np.abs(rows["rho"] - rows["plain"])) / np.max(np.abs(rows["plain"]))))
    for block, b0, b1 in model.row_blocks:
        d = rows["rho"][b0:b1] - rows["plain"][b0:b1]
        out[block] = float(np.sqrt(np.mean(d**2)) / max(np.sqrt(np.mean(rows["plain"][b0:b1] ** 2)), 1e-300))
    print(json.dumps({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in out.items() if k != "resolution"}),
          flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
