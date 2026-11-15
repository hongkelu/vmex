#!/usr/bin/env python
"""Free-boundary check of a planar TF + shaping (dipole) coil set against the plasma it was fitted to.

usage: free_boundary_planar_coils.py INPUT_FILE COILS_NPZ [OUTPUT_NAME]

``COILS_NPZ`` is what ESSOS's optimize_planar_coils_with_potential.py writes (encircling_* and shaping_* filaments).
Their Biot-Savart field is tabulated once into an in-memory mgrid and the fixed-boundary deck is re-solved with a free
boundary at the same pressure, current and toroidal flux. Printed: how far the last closed surface moved from the
fixed boundary, and aspect, beta, iota, mirror and Mercier of both.
Environment: FB_NS (51), FB_GRID (ir,jz,kp = 160,160,64), FB_NZETA (32), FB_FTOL (1e-11), FB_NITER (40000).
"""
import os
import sys
from dataclasses import replace

import jax
import numpy as np

import vmex as vj
from vmex import optimize as opt

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

input_file, coils_file = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "free_boundary_planar_coils"
NS = int(os.environ.get("FB_NS", 51))
IR, JZ, KP = (int(v) for v in os.environ.get("FB_GRID", "160,160,64").split(","))
NZETA = int(os.environ.get("FB_NZETA", 32))
FTOL, NITER = float(os.environ.get("FB_FTOL", 1e-11)), int(os.environ.get("FB_NITER", 40000))

### The coil field ############################################################

coils = np.load(coils_file)
groups = [tuple(jnp.asarray(coils[f"{k}_{q}"]) for q in ("gamma", "gamma_dash", "currents")) for k in ("encircling", "shaping")]


@jax.jit
def _field(points):
    def one(x):
        B = jnp.zeros(3)
        for gamma, gamma_dash, currents in groups:
            d = x - gamma
            B = B + 1e-7 * jnp.einsum("n,nsk->k", currents, jnp.cross(gamma_dash, d) / jnp.linalg.norm(d, axis=-1, keepdims=True) ** 3) / gamma.shape[1]
        return B
    return jax.lax.map(one, points, batch_size=4096)


def coil_field(points):
    return np.asarray(_field(jnp.asarray(points).reshape(-1, 3))).reshape(np.shape(points))


### Fixed-boundary reference and the free-boundary solve ######################

inp = vj.VmecInput.from_file(input_file)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([FTOL]), niter_array=np.array([NITER]), nzeta=NZETA)
fixed = opt.solve_equilibrium(inp, verbose=False).wout


def boundary_points(wout, ntheta=200, nphi=64):
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
                             np.linspace(0, 2 * np.pi / int(wout.nfp), nphi, endpoint=False), indexing="ij")
    angle = np.asarray(wout.xm)[:, None, None] * theta - np.asarray(wout.xn)[:, None, None] * phi
    R = np.einsum("m,mtp->tp", np.asarray(wout.rmnc)[-1], np.cos(angle))
    Z = np.einsum("m,mtp->tp", np.asarray(wout.zmns)[-1], np.sin(angle))
    return R, Z


R, Z = boundary_points(fixed)
minor = float(fixed.Aminor_p)
pad = 0.35 * minor                                                     # inside the shaping coils, which sit ~0.6 a out
grid = dict(rmin=float(R.min() - pad), rmax=float(R.max() + pad), zmin=float(Z.min() - pad), zmax=float(Z.max() + pad),
            ir=IR, jz=JZ, kp=KP, nfp=int(inp.nfp))
external = vj.MgridField.from_cartesian_field(coil_field, label="planar_coils", **grid)
print(f"coil field tabulated on R [{grid['rmin']:.3f}, {grid['rmax']:.3f}], Z [{grid['zmin']:.3f}, {grid['zmax']:.3f}], {IR} x {JZ} x {KP}", flush=True)

free_inp = replace(inp, lfreeb=True, mgrid_file="planar_coils(direct)")
result = vj.solve_free_boundary(free_inp, external_field=external, error_on_no_convergence=False)
free = vj.wout_from_state(inp=free_inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
                          niter=int(result.iterations), converged=bool(result.converged), vacuum_output=result.vacuum)
vj.write_wout(f"wout_{name}.nc", free)
vj.write_wout(f"wout_{name}_fixed.nc", fixed)

### Compare ###################################################################

Rf, Zf = boundary_points(free)
moved = np.sqrt((Rf[:, None, :] - R[None, :, :]) ** 2 + (Zf[:, None, :] - Z[None, :, :]) ** 2).min(axis=1)   # same toroidal plane


def describe(label, wout):
    s = np.linspace(0, 1, int(wout.ns))
    dmerc = np.asarray(wout.DMerc)
    keep = (s >= 0.2) & (s < 1)
    iota = np.asarray(wout.iotaf)
    bmnc = np.asarray(wout.bmnc)[-1]
    print(f"{label:6s}: aspect {float(wout.aspect):.3f}, R0 {float(wout.Rmajor_p):.4f}, a {float(wout.Aminor_p):.4f}, volume {float(wout.volume_p):.4f}, "
          f"beta {float(wout.betatotal):.3%}, iota axis {iota[0]:+.4f} edge {iota[-1]:+.4f}, axis R(0) {float(np.sum(np.asarray(wout.raxis_cc))):.4f}, "
          f"<B> {float(wout.volavgB):.4f}, min DMerc(s>=0.2) {dmerc[keep].min():+.3e}, edge |B| m=0 n=nfp over b00 {bmnc[(np.asarray(wout.xm_nyq) == 0) & (np.asarray(wout.xn_nyq) == int(wout.nfp))][0] / bmnc[0]:+.3f}",
          flush=True)


print(f"free-boundary solve: converged {bool(result.converged)}, {int(result.iterations)} iterations, "
      f"fsq {max(float(result.fsqr), float(result.fsqz), float(result.fsql)):.1e}")
describe("fixed", fixed)
describe("free", free)
print(f"last closed surface moved by mean {moved.mean() * 1e3:.2f} mm, max {moved.max() * 1e3:.2f} mm "
      f"({moved.mean() / minor:.2%} / {moved.max() / minor:.2%} of the minor radius {minor:.4f} m)")
print(f"Wrote wout_{name}.nc and wout_{name}_fixed.nc")
