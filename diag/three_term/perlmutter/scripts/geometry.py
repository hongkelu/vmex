"""Normal vs tangential share of the weak directions of a dumped certificate (QH fixture geometry).

usage: python geometry.py <input.run> <cert.npz> [n_current]
For each singular direction of the scaled joint Jacobian (boundary part), the boundary displacement dX on a
64 x 64 grid over one field period: RMS of dX.n and of the tangential part, relative to the minor radius, and the
direction's share of the certificate's predicted Gauss-Newton gain.  Also the GN step's own boundary displacement.
"""
import sys
from dataclasses import replace

import numpy as np

import vmex as vj
from vmex.core.optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

inp = replace(vj.VmecInput.from_file(sys.argv[1]), lfreeb=False)
d = np.load(sys.argv[2])
J, r = d["J"], d["r"]
nc = int(sys.argv[3]) if len(sys.argv) > 3 else 53
nb = J.shape[1] - nc
max_mode = max(int(inp.mpol) - 1, int(inp.ntor))
x0 = np.asarray(pack_boundary(inp, max_mode, vary_major_radius=True))
scale = np.asarray(_ess_scale(inp, max_mode, 1.2, vary_major_radius=True))
assert x0.size == nb, (x0.size, nb)
nfp, ntor = int(inp.nfp), int(inp.ntor)
th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
ph = np.linspace(0, 2 * np.pi / nfp, 64, endpoint=False)
TH, PH = np.meshgrid(th, ph, indexing="ij")


def surface(x):
    rbc, zbs = map(np.asarray, boundary_arrays_from_x(inp, x, max_mode, vary_major_radius=True))
    R = np.zeros_like(TH); Z = np.zeros_like(TH)
    Rt = np.zeros_like(TH); Zt = np.zeros_like(TH); Rp = np.zeros_like(TH); Zp = np.zeros_like(TH)
    for i in range(rbc.shape[0]):
        n = i - ntor
        for m in range(rbc.shape[1]):
            a = m * TH - n * nfp * PH
            c, s = np.cos(a), np.sin(a)
            R += rbc[i, m] * c; Z += zbs[i, m] * s
            Rt += -m * rbc[i, m] * s; Zt += m * zbs[i, m] * c
            Rp += n * nfp * rbc[i, m] * s; Zp += -n * nfp * zbs[i, m] * c
    return R, Z, Rt, Zt, Rp, Zp


R, Z, Rt, Zt, Rp, Zp = surface(x0)
# Cartesian tangents: e_theta = (Rt cos, Rt sin, Zt), e_phi = (Rp cos - R sin, Rp sin + R cos, Zp)
cp, sp = np.cos(PH), np.sin(PH)
et = np.stack([Rt * cp, Rt * sp, Zt]); ep = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp])
nrm = np.cross(et, ep, axis=0); nrm /= np.linalg.norm(nrm, axis=0)
minor = np.sqrt(np.mean((R - R.mean()) ** 2 + Z**2)) * np.sqrt(2)


def displacement(dx):
    """Linear boundary displacement of the coordinate change dx (Cartesian, (3, 64, 64))."""
    eps = 1e-6 / max(np.max(np.abs(dx)), 1e-300)
    R1, Z1 = surface(x0 + eps * dx)[:2]
    dR, dZ = (R1 - R) / eps, (Z1 - Z) / eps
    return np.stack([dR * cp, dR * sp, dZ])


def shares(dx):
    dX = displacement(dx)
    n = np.sum(dX * nrm, axis=0)
    t = dX - n * nrm
    return np.sqrt(np.mean(n**2)) / minor, np.sqrt(np.mean(np.sum(t**2, axis=0))) / minor


U, s, Vt = np.linalg.svd(J, full_matrices=False)
g = U.T @ r
gain = g**2 / (r @ r)
print(f"minor radius {minor:.4f} m; certificate gain {gain.sum():.4f}; boundary {nb}, current {nc}")
print(" rank  s          gain     |dX.n|/a   |dX_t|/a   normal share   (unit scaled direction)")
for k in list(range(1, 11)) + [15, 20, 30, 50, 75, 100]:
    i = len(s) - k
    dn, dt = shares(Vt[i, :nb] * scale)
    print(f"{k:5d}  {s[i]:.3e}  {gain[i]:.2e}  {dn:.3e}  {dt:.3e}  {dn / np.hypot(dn, dt):.3f}")
step = Vt.T @ (-g / s)
dn, dt = shares(step[:nb] * scale)
print(f"GN step: |dX.n|/a {dn:.3e} (RMS), |dX_t|/a {dt:.3e}, normal share {dn / np.hypot(dn, dt):.3f}")
weak = len(s) - 100
step_w = Vt[weak:].T @ (-g[weak:] / s[weak:])
step_s = Vt[:weak].T @ (-g[:weak] / s[:weak])
for name, st in (("100 weakest", step_w), ("rest", step_s)):
    dn, dt = shares(st[:nb] * scale)
    print(f"GN step in the {name}: |dX.n|/a {dn:.3e}, |dX_t|/a {dt:.3e}; normal motion {dn * minor * 1e3:.3f} mm RMS")
