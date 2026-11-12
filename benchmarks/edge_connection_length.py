"""Connection length to a wall against the distance from the separatrix on the midplane.

usage: edge_connection_length.py --run RUN --xpoint R Z --wall D [--distances 0.001 0.002 ...] [--maps 2 3 4 5 6]

The separatrix is the X-line's two legs towards Z = 0 on the plane phi = 0
(vmex.core.periodic_fieldlines.manifold_crossing): the unstable one crosses at ``R_u``, the stable one at ``R_s``.
Field lines are started on Z = 0 at each of ``--distances`` [m] beyond both crossings, away from the plasma, and
followed both ways until they are ``--wall`` [m] off the plasma boundary, enter the plasma or have run
``--max-length`` [m]; the lengths along the torus are printed, with ``|B|`` on the X-line over ``|B|`` where the
line meets the wall (total flux expansion: above 1 the field falls towards the wall).  This is the tokamak's connection length against the
distance from the separatrix at the midplane, so two runs compare at the same distances.  The lines are traced in the
field sampled on ``--table`` (vmex.core.periodic_fieldlines.tabulated_field); the X-line is one Newton step of the
field itself away from the table's, and that step is printed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from matplotlib.path import Path as Polygon  # noqa: E402
from essos.coils import Coils  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402
from vmex.core import periodic_fieldlines as pf  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.coil_design import SummedField, coil_pairs_from_dict  # noqa: E402
from vmex.core.extender import VmecExtender  # noqa: E402
from vmex.core.wout import read_wout  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--xpoint", type=float, nargs=2, required=True, metavar=("R", "Z"))
parser.add_argument("--wall", type=float, required=True, metavar="D")
parser.add_argument("--distances", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
parser.add_argument("--maps", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="return maps tried for each leg")
parser.add_argument("--max-length", type=float, default=600.0)
parser.add_argument("--planes", type=int, default=8, help="planes per field period where a line is tested")
parser.add_argument("--steps", type=int, default=64, help="Runge-Kutta steps per field period")
parser.add_argument("--table", type=int, nargs=3, default=[192, 64, 256], metavar=("NR", "NPHI", "NZ"))
parser.add_argument("--margin", type=float, default=0.26, help="the table's box around the boundary [m]")
parser.add_argument("--vc-grid", type=int, default=64)
parser.add_argument("--near-surface", default="routed", choices=("direct", "routed"))
args = parser.parse_args()

wout = read_wout(str(args.run / "wout.nc"))
nfp = int(wout.nfp)
period = 2.0 * np.pi / nfp
coil_field = BiotSavart(Coils.from_json(str(args.run / "coils.json")))
if (args.run / "pf.json").exists():
    coil_field = SummedField(coil_field, BiotSavart(coil_pairs_from_dict(json.loads((args.run / "pf.json").read_text()))))
surface = vc.surface_field_data_from_wout(wout, nphi=args.vc_grid, ntheta=args.vc_grid)
total = VmecExtender.from_surface_data(surface, external_field=lambda xyz: jax.vmap(coil_field.B)(xyz), accuracy_check="off",
                                       **({"near_surface": "direct"} if args.near_surface == "direct" else {}))
field = lambda p, xyz: total.B(xyz)  # noqa: E731


def section(phi):  # the plasma boundary on the plane phi
    theta = np.linspace(0.0, 2.0 * np.pi, 361)[:, None]
    angle = np.asarray(wout.xm)[None] * theta - np.asarray(wout.xn)[None] * phi
    return np.stack((np.cos(angle) @ np.asarray(wout.rmnc)[-1], np.sin(angle) @ np.asarray(wout.zmns)[-1]), -1)


planes = np.arange(args.planes) * period / args.planes
edges = [section(phi) for phi in planes]
low = np.min([e.min(axis=0) for e in edges], axis=0) - args.margin
high = np.max([e.max(axis=0) for e in edges], axis=0) + args.margin
def in_plasma(xyz):  # the table's nodes inside the boundary: the virtual-casing field vanishes there
    xyz = np.asarray(xyz)
    radius, angle = np.hypot(xyz[:, 0], xyz[:, 1]), np.round(np.arctan2(xyz[:, 1], xyz[:, 0]), 12)
    inside = np.zeros(len(xyz), dtype=bool)
    for phi in np.unique(angle):
        pick = angle == phi
        inside[pick] = Polygon(section(phi)).contains_points(np.stack((radius[pick], xyz[pick, 2]), -1))
    return inside


fast = pf.tabulated_field(field, None, nfp=nfp, r_range=(low[0], high[0]), z_max=max(-low[1], high[1]),
                          shape=tuple(args.table), mask=in_plasma)

r, z = np.asarray([args.xpoint[0]]), np.asarray([args.xpoint[1]])
r, z, M, _ = pf.return_map_fixed_point(fast, None, r, z, nfp=nfp, steps=8 * args.steps, iterations=10)
_, _, _, step = pf.return_map_fixed_point(field, None, r, z, nfp=nfp, steps=8 * args.steps, iterations=1)
line = pf.PeriodicFieldline(r, z, jnp.zeros(1), jnp.zeros(()), jnp.zeros(()))
print(f"X-point ({float(r[0]):.5f}, {float(z[0]):.5f}) Tr M {float(jnp.trace(M[0])):.4f} in the table, "
      f"{float(step[0]):.1e} m from the field's", flush=True)

inside = Polygon(edges[0])
centre = edges[0][:, 0].mean()
seeds, names = [], []
for stable in (False, True):
    for maps in args.maps:
        R, sine, miss = (float(v) for v in pf.manifold_crossing(fast, None, line, M[0], nfp=nfp, iterations=maps,
                                                                stable=stable, steps=2 * args.steps))
        if miss < 1e-6:
            break
    else:
        raise SystemExit(f"no crossing of the {'stable' if stable else 'unstable'} leg in {args.maps} maps")
    # the plasma is on the side of the boundary's nearest point on Z = 0
    crossings = edges[0][np.abs(edges[0][:, 1]) < 0.01, 0]
    away = np.sign(R - crossings[np.argmin(np.abs(crossings - R))])
    gap = float(pf.surface_distance(jnp.asarray([[R, 0.0, 0.0]]), surface.gamma, nfp))
    name = "stable" if stable else "unstable"
    print(f"{name} leg: {maps} maps, crosses Z = 0 at R {R:.5f} (sine {sine:+.5f}), {100 * gap:.2f} cm off the plasma",
          flush=True)
    seeds += [(R + away * d, 0.0) for d in args.distances]
    names += [(name, d) for d in args.distances]
seeds = np.asarray(seeds)

chunk = 5
angles = jnp.arange(chunk * args.planes + 1) * period / args.planes
substeps = max(4, args.steps // args.planes)
length_of_period = period * float(np.asarray(wout.rmnc)[0, 0])
lengths, ends, ratios = {}, {}, {}


def strength(R, Z, phi):  # |B| in the table
    xyz = jnp.asarray(np.stack((R * np.cos(phi), R * np.sin(phi), Z), -1))
    return np.linalg.norm(np.asarray(fast(None, xyz)), axis=-1)


x_r, x_z = (np.asarray(v)[:, 0] for v in pf.trace_fieldline(fast, None, r, z, jnp.asarray(planes), substeps=4 * args.steps))
b_x = strength(x_r, x_z, planes)   # along the X-line, on the planes
print(f"|B| on the X-line: {b_x.min():.4f} to {b_x.max():.4f} T", flush=True)
for sign in (1.0, -1.0):
    advance = jax.jit(lambda a, b, sign=sign: pf.trace_fieldline(fast, None, a, b, sign * angles, substeps=substeps))
    a, b = seeds[:, 0].copy(), seeds[:, 1].copy()
    living, life, end = np.ones(len(seeds), dtype=bool), np.zeros(len(seeds)), np.full(len(seeds), "kept", dtype=object)
    ratio = np.full(len(seeds), np.nan)   # |B| on the X-line over |B| where the line meets the wall, same plane
    for start in range(0, int(np.ceil(args.max_length / length_of_period)), chunk):
        path = [np.asarray(v) for v in advance(jnp.asarray(a), jnp.asarray(b))]
        for k in range(1, chunk * args.planes + 1):
            plane = int(round(sign * k)) % args.planes
            pr, pz = path[0][k], path[1][k]
            ok = np.isfinite(pr) & np.isfinite(pz)
            pr, pz = np.where(ok, pr, 0.0), np.where(ok, pz, 0.0)
            xyz = np.stack((pr * np.cos(planes[plane]), pr * np.sin(planes[plane]), pz), -1)
            far = np.asarray(pf.surface_distances(jnp.asarray(xyz), surface.gamma, nfp)) >= args.wall
            plasma = Polygon(edges[plane]).contains_points(np.stack((pr, pz), -1))
            struck = living & far & ~plasma & ok
            if struck.any():
                ratio[struck] = b_x[plane] / strength(pr[struck], pz[struck], planes[plane])
            for mask, label in ((~ok, "lost"), (plasma, "plasma"), (far & ~plasma, "wall")):
                end[living & mask & (end == "kept")] = label
            living &= ok & ~far & ~plasma
            life[living] = start + k / args.planes
        a, b = np.where(living, path[0][-1], seeds[:, 0]), np.where(living, path[1][-1], seeds[:, 1])
        if not living.any():
            break
    lengths[sign], ends[sign], ratios[sign] = life * length_of_period, end, ratio

print("leg       distance [mm]   forward [m] (end)    backward [m] (end)   shorter [m]   B_X / B_wall (forward)")
for i, (name, d) in enumerate(names):
    print(f"{name:9s} {1e3 * d:8.1f}   {lengths[1.0][i]:10.1f} ({ends[1.0][i]:6s})   {lengths[-1.0][i]:10.1f} "
          f"({ends[-1.0][i]:6s})   {min(lengths[1.0][i], lengths[-1.0][i]):8.1f}   {ratios[1.0][i]:8.3f}", flush=True)
