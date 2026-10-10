"""Published LP QA coil sets -> ESSOS coil files at the vmex scale (R0 = 1 m).

Reads references/gil2026/coils_npz/<set>.npz (reactor scale, R0_target =
10.1266 m) and writes paper/coils/lpqa_<set>.json in ESSOS ``Coils.to_json``
layout, with the geometry divided by R0_target and the currents scaled so the
linked mu0 I / 2 pi equals the LP QA lowres deck's RBTOR, exactly as the
drivers' ``scale_coil_currents`` does for a fresh seed (sign included).  Also
writes lpqa_<set>_check.npz with the 16 scaled coils for a load check.
Pure numpy: safe on a login node.
"""
import json
from pathlib import Path

import numpy as np

MU0 = 4e-7 * np.pi
RBTOR = 1.0696471908533929   # wout_LandremanPaul2021_QA_lowres.nc: R B_phi at the edge
R0 = 1.0
NSEG = 256
SRC = Path("/pscratch/sd/h/hongkelu/freeboundary-single-stage/references/gil2026/coils_npz")
OUT = Path(__file__).resolve().parent / "coils"
OUT.mkdir(exist_ok=True)


def fourier_gamma(dofs, order, nq):
    """simsopt/ESSOS CurveXYZFourier: (3, 2o+1) -> (nq, 3) points and tangents."""
    t = np.arange(nq) / nq
    g = np.zeros((nq, 3)); dg = np.zeros((nq, 3))
    for i in range(3):
        g[:, i] = dofs[i, 0]
        for j in range(1, order + 1):
            w = 2 * np.pi * j
            g[:, i] += dofs[i, 2 * j - 1] * np.sin(w * t) + dofs[i, 2 * j] * np.cos(w * t)
            dg[:, i] += w * (dofs[i, 2 * j - 1] * np.cos(w * t) - dofs[i, 2 * j] * np.sin(w * t))
    return g, dg


def symmetric_copies(g, dg, I, nfp, stellsym):
    """All field-period and stellarator-symmetric copies (SIMSOPT convention)."""
    out = []
    for k in range(nfp):
        phi = 2 * np.pi * k / nfp
        rot = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]]).T
        out.append((g @ rot, dg @ rot, I))
        if stellsym:
            flip = rot @ np.diag([1.0, -1.0, -1.0])
            out.append((g @ flip, dg @ flip, -I))
    return out


def linked_rbphi(coils):
    """Mean of R B_phi on the loop R = R0, Z = 0 (signed)."""
    phi = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    pts = R0 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], -1)
    B = np.zeros_like(pts)
    for g, dg, I in coils:
        r = pts[:, None, :] - g[None]
        r3 = np.linalg.norm(r, axis=-1, keepdims=True) ** 3
        B += MU0 * I / (4 * np.pi) * np.sum(np.cross(dg[None], r) / r3, axis=1) / len(g)
    return R0 * float(np.mean(-np.sin(phi) * B[:, 0] + np.cos(phi) * B[:, 1]))


for npz in sorted(SRC.glob("*.npz")):
    d = np.load(npz)
    order, nfp, scale = int(d["order"]), int(d["nfp"]), float(d["R0_target"])
    base = d["base_dofs"].reshape(-1, 3, 2 * order + 1) / scale
    nbase = base.shape[0]
    currents = d["currents"][:nbase] / scale          # same B after shrinking the geometry
    coils = []
    for b, I in zip(base, currents):
        g, dg = fourier_gamma(b, order, NSEG)
        coils += symmetric_copies(g, dg, I, nfp, True)
    factor = RBTOR / linked_rbphi(coils)
    currents = currents * factor
    coils = [(g, dg, I * factor) for g, dg, I in coils]
    name = f"lpqa_{npz.stem}"
    data = dict(nfp=nfp, stellsym=True, order=order, n_segments=NSEG,
                dofs_curves_raw=base.tolist(), scaling_type=2, scaling_factor=0.0, scale_fixed=1.0,
                dofs_currents_raw=currents.tolist(), currents_scale=None)
    json.dump(data, open(OUT / f"{name}.json", "w"), indent=1)
    np.savez(OUT / f"{name}_check.npz", gamma=np.array([c[0] for c in coils]), currents=np.array([c[2] for c in coils]))
    print(f"{name:22s} {len(coils)} coils, order {order}, I_base = {np.round(currents, 1)} A, "
          f"linked R Bphi -> {linked_rbphi(coils):.4f} (target {RBTOR:.4f})")
