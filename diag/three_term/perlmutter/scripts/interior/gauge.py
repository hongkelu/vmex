"""Decompose the difference between the cold and perturbed-start interior solutions on each surface:
dX = X_theta f(phi)  [m=0 relabelling]  +  other tangential  +  normal (physical)."""
import os
import sys
from dataclasses import replace

import numpy as np
import jax

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex import optimize as opt  # noqa: E402

if os.environ.get("KEEP_M1"):  # never release the constrained m=1 Z force (VMEC2000 releases it below fsqz 1e-6)
    from vmex.core import solver as _solver

    _solver.zero_m1_z_force = lambda force, enabled: force
    print("KEEP_M1: constrained m=1 Z force kept to convergence", flush=True)

m, ns = int(os.environ.get("MPOL", 4)), int(os.environ.get("NS", 15))
inp = replace(t._deck("input.LandremanPaul2021_QA_lowres", m, ns), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
              lfreeb=False, ftol_array=np.array([1e-13]), niter_array=np.array([50000]))
cold = opt.solve_equilibrium(inp)
seed = jax.tree.map(lambda a: a * (1 + 1e-3 * jax.numpy.cos(jax.numpy.arange(a.size).reshape(a.shape)))
                    if jax.numpy.issubdtype(a.dtype, jax.numpy.inexact) else a, cold.state)
pert = opt.solve_equilibrium(inp, initial_state=seed)
print("fsq cold", float(cold.wout.fsqr), float(cold.wout.fsqz), float(cold.wout.fsql), "pert", float(pert.wout.fsqr),
      float(pert.wout.fsqz), float(pert.wout.fsql), flush=True)
print("state max diff", max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in
                            zip(jax.tree.leaves(cold.state), jax.tree.leaves(pert.state))), flush=True)
w1, w2 = cold.wout, pert.wout
xm, xn, nfp = np.asarray(w1.xm), np.asarray(w1.xn), int(w1.nfp)
th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
ph = np.linspace(0, 2 * np.pi / nfp, 32, endpoint=False)
TH, PH = np.meshgrid(th, ph, indexing="ij")
A = xm[None, None, :] * TH[..., None] - xn[None, None, :] * PH[..., None]
C, S = np.cos(A), np.sin(A)


def surf(w, js):
    rc, zs = np.asarray(w.rmnc)[js], np.asarray(w.zmns)[js]
    return C @ rc, S @ zs, (-S * xm) @ rc, (C * xm) @ zs  # R, Z, R_theta, Z_theta


print(" js   |dX| rms     m=0 relabel   any relabel   normal rms     (fractions of |dX|^2)")
for js in range(2, int(w1.ns) - 1, 2):
    R1, Z1, Rt, Zt = surf(w1, js)
    R2, Z2 = surf(w2, js)[:2]
    dR, dZ = R2 - R1, Z2 - Z1
    tot = np.mean(dR**2 + dZ**2)
    g2 = Rt**2 + Zt**2
    f_phi = np.sum(dR * Rt + dZ * Zt, axis=0) / np.sum(g2, axis=0)  # best f(phi) per cut
    r0 = np.mean((dR - Rt * f_phi) ** 2 + (dZ - Zt * f_phi) ** 2)
    tang = (dR * Rt + dZ * Zt) / np.sqrt(g2)  # tangential (in-cut) component, any f(theta, phi)
    nrm = (dR * Zt - dZ * Rt) / np.sqrt(g2)  # in-cut normal component
    print(f"{js:3d}  {np.sqrt(tot):.2e}    {1 - r0 / tot:.3f}         {np.mean(tang**2) / tot:.3f}         "
          f"{np.sqrt(np.mean(nrm**2)):.2e} ({np.mean(nrm**2) / tot:.3f})", flush=True)
print("iota max diff", float(np.max(np.abs(np.asarray(w1.iotaf) - np.asarray(w2.iotaf)))))
