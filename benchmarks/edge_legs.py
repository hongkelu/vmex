"""The legs of a run's X-line: where they cross Z = 0, at which angle, and the line's height spread.

usage: edge_legs.py --run RUN --xpoint R Z [--vc-grid 64 128] [--near-surface direct routed] [--maps 1 2 3 4]

For each field model (virtual-casing grid, near-surface rule) and number of return maps it prints, for the unstable
and the stable leg towards Z = 0 on the plane phi = 0, ``R`` of the crossing, the sine of the crossing angle and the
remaining ``|Z|`` (vmex.core.periodic_fieldlines.manifold_crossing).  The rows with a miss near zero are the leg; the
smallest such number of maps is the one for the driver's --edge-leg-maps.  The field is the coils' (coils.json and
the circular pairs of pf.json) plus the plasma's virtual-casing field of wout.nc.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
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
parser.add_argument("--vc-grid", type=int, nargs="+", default=[64])
parser.add_argument("--near-surface", nargs="+", default=["direct"])
parser.add_argument("--maps", type=int, nargs="+", default=[1, 2, 3, 4])
parser.add_argument("--steps", type=int, default=128, help="Runge-Kutta steps per field period")
args = parser.parse_args()

wout = read_wout(str(args.run / "wout.nc"))
nfp = int(wout.nfp)
coil_field = BiotSavart(Coils.from_json(str(args.run / "coils.json")))
if (args.run / "pf.json").exists():
    coil_field = SummedField(coil_field, BiotSavart(coil_pairs_from_dict(json.loads((args.run / "pf.json").read_text()))))

for grid in args.vc_grid:
    surface = vc.surface_field_data_from_wout(wout, nphi=grid, ntheta=grid)
    for rule in args.near_surface:
        total = VmecExtender.from_surface_data(surface, external_field=lambda xyz: jax.vmap(coil_field.B)(xyz),
                                               accuracy_check="off", near_surface=rule)
        field = lambda p, xyz: total.B(xyz)  # noqa: E731
        r, z, M, step = pf.return_map_fixed_point(field, None, np.asarray([args.xpoint[0]]), np.asarray([args.xpoint[1]]),
                                                  nfp=nfp, steps=4 * args.steps, iterations=10)
        M = M[0]
        line = pf.PeriodicFieldline(r, z, jnp.zeros(1), jnp.zeros(()), step[0])
        phi = np.linspace(0.0, 2.0 * np.pi / nfp, 65)
        height = np.asarray(pf.trace_fieldline(field, None, r, z, jnp.asarray(phi), substeps=16)[1])[:-1, 0]
        gap = float(pf.surface_distance(line.xyz, surface.gamma, nfp))
        print(f"[vc-grid {grid}, {rule}] X-point ({float(r[0]):.5f}, {float(z[0]):.5f}) Newton step {float(step[0]):.1e} "
              f"Tr M {float(jnp.trace(M)):.4f} gap at phi = 0 {gap:.4f} m, height: mean {height.mean():.4f} "
              f"spread {height.std():.4f} m", flush=True)
        for stable in (False, True):
            for maps in args.maps:
                R, sine, miss = (float(v) for v in pf.manifold_crossing(field, None, line, M, nfp=nfp, iterations=maps,
                                                                        stable=stable, steps=args.steps))
                reach = float(pf.surface_distance(jnp.asarray([[R, 0.0, 0.0]]), surface.gamma, nfp))
                print(f"    {'stable  ' if stable else 'unstable'} leg, {maps} maps: R {R:.5f} sine {sine:+.5f} "
                      f"miss {miss:.1e} m, {reach:.4f} m from the plasma", flush=True)
