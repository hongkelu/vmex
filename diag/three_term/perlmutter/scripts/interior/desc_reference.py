"""DESC fixed-boundary reference for wouts that share one boundary (WP1: which m = 1 treatment is closer).

usage: python desc_reference.py OUT.json WOUT_A.nc [WOUT_B.nc ...]   (env desc-latest; a GPU; env L, M, N)

The boundary, pressure and enclosed-current profile are taken from the first wout (DESC's VMECIO.load with the
current profile); DESC solves the fixed-boundary equilibrium at (L, M, N).  Reported per wout: the largest iota
difference on the wout's full mesh and the largest distance of its surfaces from DESC's at rho = sqrt(s), on 8 phi
cuts of 256 theta points.
"""
import json
import sys

import desc
import numpy as np
from netCDF4 import Dataset

desc.set_device("gpu")
from desc.grid import Grid  # noqa: E402
from desc.vmec import VMECIO  # noqa: E402


def wout_surfaces(path, rho, phis, th):
    """R, Z on full-mesh surfaces js (rho = sqrt(s_js)) at the given cuts, and iotaf."""
    with Dataset(path) as d:
        xm, xn = d["xm"][:], d["xn"][:]
        rmnc, zmns, iotaf = d["rmnc"][:], d["zmns"][:], d["iotaf"][:]
    out = []
    for js in range(rmnc.shape[0]):
        cuts = []
        for p in phis:
            a = xm[None, :] * th[:, None] - xn[None, :] * p
            cuts.append((np.cos(a) @ rmnc[js], np.sin(a) @ zmns[js]))
        out.append(cuts)
    return out, np.asarray(iotaf)


def distance(c1, c2):
    """Largest point-to-segment distance between two closed curves, over the cuts."""
    worst = 0.0
    for (R1, Z1), (R2, Z2) in zip(c1, c2):
        P, Q = np.stack([R1, Z1], 1)[::4], np.stack([R2, Z2], 1)
        seg = np.roll(Q, -1, 0) - Q
        t = np.clip(np.einsum("psk,sk->ps", P[:, None] - Q[None], seg) / np.sum(seg**2, 1)[None], 0, 1)
        d = np.min(np.linalg.norm(P[:, None] - (Q[None] + t[..., None] * seg[None]), axis=2), axis=1)
        worst = max(worst, float(d.max()))
    return worst


def main():
    """Solve DESC at (L, M, N) and compare every wout; write OUT.json."""
    import os
    L, M, N = (int(os.environ.get(k, v)) for k, v in (("L", 16), ("M", 8), ("N", 8)))
    eq = VMECIO.load(sys.argv[2], profile="current")
    eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
    eq.solve(ftol=1e-12, xtol=1e-14, gtol=1e-14, maxiter=500, verbose=1)
    nfp = int(eq.NFP)
    phis = np.linspace(0, 2 * np.pi / nfp, 8, endpoint=False)
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    with Dataset(sys.argv[2]) as d:
        ns = int(d["ns"][:])
    s = np.linspace(0, 1, ns)
    rho = np.sqrt(s[1:])
    ref = []
    for r in rho:
        cuts = []
        for p in phis:
            g = Grid(np.stack([np.full_like(th, r), th, np.full_like(th, p)], 1))
            c = eq.compute(["R", "Z"], grid=g)
            cuts.append((np.asarray(c["R"]), np.asarray(c["Z"])))
        ref.append(cuts)
    g = Grid(np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)], 1))
    iota_ref = np.asarray(eq.compute("iota", grid=g)["iota"])
    out = dict(desc=dict(L=L, M=M, N=N))
    for path in sys.argv[2:]:
        surf, iotaf = wout_surfaces(path, rho, phis, th)
        # VMEC's iota sign convention can differ from DESC's: compare magnitudes
        di = np.abs(np.abs(iotaf[1:]) - np.abs(iota_ref))
        dist = [distance(a, b) for a, b in zip(surf[1:], ref)]
        out[path] = dict(iota=float(di.max()), surface_max=float(max(dist)), surface_by_js=dist)
        print(path, out[path]["iota"], out[path]["surface_max"], flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
