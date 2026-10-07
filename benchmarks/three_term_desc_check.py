"""DESC free boundary without a sheet current for a VMEX run's final coils (the same filaments), vs the VMEX three-term solve.

    python benchmarks/three_term_desc_check.py <wout> <coils.npz> <out dir> [--L 24 --M 12 --N 12 --grid 32] [--compare name=wout ...]

<wout>: the run's final VMEX equilibrium (start boundary, pressure, enclosed current). <coils.npz>: from
three_term_highres.py, the filament points, tangents and currents with check values, B = 1e-7 sum_c I_c mean_p
gamma'_cp x d / |d|^3 (VMEX's DirectCoilField), checked here to 1e-10. The free boundary holds all three interface
conditions with the sheet potential fixed at zero (DESC's FixSheetCurrent), as VMEX's three-term solve. Writes
report.json (iota, interface residual blocks, LCFS distances to each --compare wout), eq.h5 and wout_desc.nc.
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
p.add_argument("wout", type=Path)
p.add_argument("coils", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--L", type=int, default=24)
p.add_argument("--M", type=int, default=12)
p.add_argument("--N", type=int, default=12)
p.add_argument("--grid", type=int, default=32)
p.add_argument("--maxiter", type=int, default=100)
p.add_argument("--ftol", type=float, default=1e-8)
p.add_argument("--compare", nargs="*", default=[], help="name=path of VMEX wouts to measure the LCFS against")
args = p.parse_args()

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

args.out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
report = dict(L=args.L, M=args.M, N=args.N, grid=args.grid, wout=str(args.wout), desc_version=desc.__version__,
              device=str(jax.devices()[0]))


def log(msg):
    print(f"[{time.perf_counter() - t0:7.0f}s] {msg}", flush=True)


def dump():
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")


# ---- the coils' filament field -------------------------------------------------------------------------------------
data = np.load(args.coils)
gamma = jnp.asarray(data["gamma"]).reshape(-1, 3)                      # (coils * points, 3)
npts = data["gamma"].shape[-2]
weight = jnp.asarray(np.repeat(data["currents"], npts) / npts * 1e-7)  # I_c / points * mu0 / 4 pi
dgamma = jnp.asarray(data["gamma_dash"]).reshape(-1, 3) * weight[:, None]


def b_xyz(x):
    def one(pt):
        d = pt[None, :] - gamma
        return jnp.sum(jnp.cross(dgamma, d) / (jnp.sum(d * d, -1) ** 1.5)[:, None], 0)

    return jax.lax.map(one, x, batch_size=256)


class CoilsField(MagneticFieldFromUser):
    """DESC master's BoundaryError passes ``transforms``, which MagneticFieldFromUser does not take; it is unused."""

    def compute_magnetic_field(self, coords, params=None, basis="rpz", source_grid=None, transforms=None,
                               chunk_size=None):
        return super().compute_magnetic_field(coords, params, basis, source_grid, chunk_size)


field = CoilsField(lambda coords, params: xyz2rpz_vec(b_xyz(rpz2xyz(coords)), phi=coords[:, 1]))
check = np.asarray(b_xyz(jnp.asarray(data["check_points"])))
report["field_check_rel"] = float(np.max(np.abs(check - data["check_B"])) / np.max(np.abs(data["check_B"])))
log(f"coils: {data['gamma'].shape[0]} filaments x {npts} points; |B_desc - B_vmex|/|B| = {report['field_check_rel']:.2e}")
assert report["field_check_rel"] < 1e-10

# ---- equilibrium: the VMEX boundary and profiles --------------------------------------------------------------------
eq = VMECIO.load(str(args.wout), L=args.L, M=args.M, N=args.N, profile="current")
with netCDF4.Dataset(args.wout) as nc:
    buco = np.asarray(nc["buco"][:], float)
    iota_vmex = np.abs(np.asarray(nc["iotaf"][:], float))
ns = buco.size
s = np.concatenate([[0.0], (np.arange(1, ns) - 0.5) / (ns - 1), [1.0]])
# VMECIO.load flips the signgs = -1 wout to DESC's orientation; the enclosed current flips with it. A polynomial in
# s keeps I ~ s at the axis, which sets iota there.
current = -2 * np.pi / mu_0 * np.concatenate([[0.0], buco[1:], [1.5 * buco[-1] - 0.5 * buco[-2]]])
coef = np.linalg.lstsq(np.stack([s**k for k in range(1, 9)], 1), current, rcond=None)[0]
eq.current = PowerSeriesProfile(params=np.r_[0.0, coef], modes=2 * np.arange(9), name="current")
report["current_fit_rel"] = float(np.max(np.abs(eq.current(np.sqrt(s)) - current)) / np.max(np.abs(current)))
eq, fixed = eq.solve(objective="force", optimizer="lsq-exact", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300,
                     verbose=1)


def iota_ends():
    g = LinearGrid(rho=np.array([0.0, 0.5, 1.0]), M=0, N=0, NFP=eq.NFP)
    return [abs(float(v)) for v in np.asarray(eq.compute("iota", grid=g)["iota"])]


report["fixed"] = dict(iota=iota_ends(), iota_vmex=[float(iota_vmex[0]), float(np.interp(0.25, np.linspace(0, 1, iota_vmex.size), iota_vmex)), float(iota_vmex[-1])])
log("fixed-boundary DESC solve on the VMEX boundary: |iota| axis/mid/edge " + json.dumps(report["fixed"]))
dump()

# ---- free boundary, no sheet current --------------------------------------------------------------------------------
eq.surface = FourierCurrentPotentialField.from_surface(eq.surface, M_Phi=args.M, N_Phi=args.N)
grid = LinearGrid(rho=np.array([1.0]), M=args.grid, N=args.grid, NFP=eq.NFP, sym=False)
boundary = BoundaryError(eq, field, field_fixed=True, source_grid=grid, eval_grid=grid, bs_chunk_size=64,
                         B_plasma_chunk_size=64)
objective = ObjectiveFunction(boundary, deriv_mode="batched", jac_chunk_size=25)
constraints = [ForceBalance(eq), FixPressure(eq), FixCurrent(eq), FixPsi(eq), FixSheetCurrent(eq)]
t = time.perf_counter()
eq, free = eq.optimize(objective, constraints, optimizer="proximal-lsq-exact", ftol=args.ftol, xtol=1e-10,
                       maxiter=args.maxiter, verbose=3, copy=True,
                       options={"solve_options": {"ftol": 1e-10, "xtol": 1e-10, "maxiter": 300}})
eq, polish = eq.solve(objective="force", optimizer="lsq-exact", ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300,
                      verbose=1)
objective.build(verbose=0)
f = np.asarray(objective.compute_unscaled(objective.x(eq))).reshape(-1, grid.num_nodes)
report["free"] = dict(nfev=int(free.nfev), message=str(free.message), seconds=round(time.perf_counter() - t),
                      iota=iota_ends(), block_rms=[float(np.sqrt(np.mean(b**2))) for b in f])
log("free boundary: " + json.dumps(report["free"]))
eq.save(args.out / "eq.h5")
dump()

# ---- LCFS distances (max over 4 planes of a half period of the distance to the other curve) -----------------------
th = np.linspace(0, 2 * np.pi, 721)
planes = np.linspace(0, np.pi / eq.NFP, 4)


def rz_wout(path, ph):
    with netCDF4.Dataset(path) as nc:
        xm, xn = np.asarray(nc["xm"][:], float), np.asarray(nc["xn"][:], float)
        rmnc, zmns = np.asarray(nc["rmnc"][:])[-1], np.asarray(nc["zmns"][:])[-1]
    a = np.outer(th, xm) - xn * ph
    return np.cos(a) @ rmnc, np.sin(a) @ zmns


def rz_desc(ph):
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([1.0]), theta=th, zeta=np.array([ph]), NFP=eq.NFP))
    return np.asarray(d["R"]), np.asarray(d["Z"])


report["lcfs_mm"] = {}
for item in [f"start={args.wout}", *args.compare]:
    name, _, path = item.partition("=")
    if not Path(path).exists():
        continue
    dist = 0.0
    for ph in planes:
        (Ra, Za), (Rb, Zb) = rz_desc(ph), rz_wout(path, ph)
        dist = max(dist, float(np.max(np.min(np.hypot(Ra[:, None] - Rb[None], Za[:, None] - Zb[None]), axis=1))))
    report["lcfs_mm"][name] = round(1e3 * dist, 3)
log("LCFS distance [mm]: " + json.dumps(report["lcfs_mm"]))
try:
    VMECIO.save(eq, str(args.out / "wout_desc.nc"), surfs=51, verbose=0)
except Exception as exc:  # noqa: BLE001  (the wout only feeds cross-evaluations)
    report["wout_error"] = repr(exc)
dump()
log("DONE " + json.dumps(dict(lcfs_mm=report["lcfs_mm"], free=report["free"])))
