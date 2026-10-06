#!/usr/bin/env python3
"""DESC arm of ``run_three_term_resolution.py``: the free boundary without a sheet current at L = 2M, M = N.

    python benchmarks/three_term_resolution_desc.py M OUT [--grid 32]
    python benchmarks/run_three_term_resolution.py score M OUT desc OUT/mM/desc/wout_desc.nc

DESC (https://github.com/PlasmaControl/DESC, not a VMEX dependency) starts from
``OUT/mM/wout_start.nc``, with that run's pressure, enclosed current and toroidal
flux.  It uses the same external field, ``OUT/field.npz``, as the exact filament
sum B = 1e-7 sum_c I_c gamma'_c x d / |d|^3 of VMEX's ``DirectCoilField``
(checked to 1e-10).  The boundary is solved with all three interface conditions
and the sheet current potential held at zero (``FixSheetCurrent``).  This
writes ``OUT/mM/desc/report.json``, holding the wall time and peak GPU memory of
the free-boundary solve and of the whole run, with ``eq.h5`` and
``wout_desc.nc``.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from scipy.constants import mu_0  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("M", type=int)
p.add_argument("out", type=Path)
p.add_argument("--grid", type=int, default=32)
p.add_argument("--maxiter", type=int, default=100)
args = p.parse_args()
wout = args.out / f"m{args.M}" / "wout_start.nc"
out = args.out / f"m{args.M}" / "desc"
out.mkdir(parents=True, exist_ok=True)

from desc import set_device  # noqa: E402

set_device("gpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import desc  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.magnetic_fields import FourierCurrentPotentialField, MagneticFieldFromUser  # noqa: E402
from desc.objectives import (  # noqa: E402
    BoundaryError, FixCurrent, FixPressure, FixPsi, FixSheetCurrent, ForceBalance, ObjectiveFunction)
from desc.profiles import PowerSeriesProfile  # noqa: E402
from desc.utils import rpz2xyz, xyz2rpz_vec  # noqa: E402
from desc.vmec import VMECIO  # noqa: E402

t0 = time.perf_counter()
report = dict(L=2 * args.M, M=args.M, N=args.M, grid=args.grid, desc_version=desc.__version__,
              device=str(jax.devices()[0]))


def log(msg):
    print(f"[{time.perf_counter() - t0:7.0f}s] {msg}", flush=True)


def peak_gib():
    return round((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30, 2)


def dump():
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")


# ---- the external field as filaments -------------------------------------------------------------------------------
data = np.load(args.out / "field.npz")
gamma = jnp.asarray(data["gamma"]).reshape(-1, 3)
npts = data["gamma"].shape[-2]
weight = jnp.asarray(np.repeat(data["currents"], npts) / npts * 1e-7)
dgamma = jnp.asarray(data["gamma_dash"]).reshape(-1, 3) * weight[:, None]


def b_xyz(x):
    def one(pt):
        d = pt[None, :] - gamma
        return jnp.sum(jnp.cross(dgamma, d) / (jnp.sum(d * d, -1) ** 1.5)[:, None], 0)

    return jax.lax.map(one, x, batch_size=256)


class CoilsField(MagneticFieldFromUser):
    """DESC's BoundaryError passes ``transforms``, which MagneticFieldFromUser does not take; it is unused."""

    def compute_magnetic_field(self, coords, params=None, basis="rpz", source_grid=None, transforms=None,
                               chunk_size=None):
        return super().compute_magnetic_field(coords, params, basis, source_grid, chunk_size)


field = CoilsField(lambda coords, params: xyz2rpz_vec(b_xyz(rpz2xyz(coords)), phi=coords[:, 1]))
check = np.asarray(b_xyz(jnp.asarray(data["check_points"])))
report["field_check_rel"] = float(np.max(np.abs(check - data["check_B"])) / np.max(np.abs(data["check_B"])))
assert report["field_check_rel"] < 1e-10, report["field_check_rel"]

# ---- start: the common initial boundary and the deck's profiles ----------------------------------------------------
eq = VMECIO.load(str(wout), L=2 * args.M, M=args.M, N=args.M, profile="current")
with netCDF4.Dataset(wout) as nc:
    buco = np.asarray(nc["buco"][:], float)
ns = buco.size
s = np.concatenate([[0.0], (np.arange(1, ns) - 0.5) / (ns - 1), [1.0]])
# VMECIO.load flips the signgs = -1 wout to DESC's orientation; the enclosed current flips with it. A polynomial in
# s keeps I ~ s at the axis, which sets iota there.
current = -2 * np.pi / mu_0 * np.concatenate([[0.0], buco[1:], [1.5 * buco[-1] - 0.5 * buco[-2]]])
coef = np.linalg.lstsq(np.stack([s**k for k in range(1, 9)], 1), current, rcond=None)[0]
eq.current = PowerSeriesProfile(params=np.r_[0.0, coef], modes=2 * np.arange(9), name="current")
report["current_fit_rel"] = float(np.max(np.abs(eq.current(np.sqrt(s)) - current)) / np.max(np.abs(current)))
eq, _ = eq.solve(objective="force", optimizer="lsq-exact", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1)
log("fixed-boundary start solved")

# ---- free boundary without a sheet current -------------------------------------------------------------------------
eq.surface = FourierCurrentPotentialField.from_surface(eq.surface, M_Phi=args.M, N_Phi=args.M)
grid = LinearGrid(rho=np.array([1.0]), M=args.grid, N=args.grid, NFP=eq.NFP, sym=False)
boundary = BoundaryError(eq, field, field_fixed=True, source_grid=grid, eval_grid=grid, bs_chunk_size=64,
                         B_plasma_chunk_size=64)
objective = ObjectiveFunction(boundary, deriv_mode="batched", jac_chunk_size=25)
constraints = [ForceBalance(eq), FixPressure(eq), FixCurrent(eq), FixPsi(eq), FixSheetCurrent(eq)]
t = time.perf_counter()
eq, free = eq.optimize(objective, constraints, optimizer="proximal-lsq-exact", ftol=1e-8, xtol=1e-10,
                       maxiter=args.maxiter, verbose=3, copy=True,
                       options={"solve_options": {"ftol": 1e-10, "xtol": 1e-10, "maxiter": 300}})
eq, _ = eq.solve(objective="force", optimizer="lsq-exact", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300,
                 verbose=1)
report.update(seconds=round(time.perf_counter() - t, 1), nfev=int(free.nfev), message=str(free.message),
              total_seconds=round(time.perf_counter() - t0, 1), peak_gpu_gib=peak_gib())
objective.build(verbose=0)
f = np.asarray(objective.compute_unscaled(objective.x(eq))).reshape(-1, grid.num_nodes)
report["block_rms"] = [float(np.sqrt(np.mean(b**2))) for b in f]
log("free boundary: " + json.dumps(report))
eq.save(out / "eq.h5")
VMECIO.save(eq, str(out / "wout_desc.nc"), surfs=51, verbose=0)
dump()
log("DONE")
