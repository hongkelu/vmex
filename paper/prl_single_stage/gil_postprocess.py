"""Gil et al. (2026) post-processing, reproduced from their Zenodo scripts/run_post_processing.py.

    python gil_postprocess.py <biot_savart.json> <outdir> [--no-vmec]

For the LP QA reactor-scale target (their input.LandremanPaul2021_QA_reactorScale_lowres):
  1. <|B.n|>, <|B.n|>/<|B|> on a 256x256 (phi, theta) grid of the target surface (their definition);
  2. a QFM surface of the coil field (their QFM_Generator: mpol = ntor = 12, L-BFGS, 800 iterations);
  3. fixed-boundary VMEC on that QFM surface;
  4. simsopt QuasisymmetryRatioResidual, helicity (1, 0), on the 51 surfaces np.arange(0, 1.01, 1.01/50):
     total and profile.
Writes <outdir>/gil_metrics.json.  Run with simsopt-env (MPICH_GPU_SUPPORT_ENABLED=0), CPU only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from simsopt import load
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux

TEST_DIR = Path("/pscratch/sd/h/hongkelu/finitebeta-single-stage/simsopt-editable/tests/test_files")
INPUT = TEST_DIR / "input.LandremanPaul2021_QA_reactorScale_lowres"
NPHI = NTHETA = 256
NS = 50


def qfm_generator(surface, coils, mpol=12, ntor=12, maxiter=800, tol=1e-15):
    """Verbatim logic of their QFM_Generator (stellsym=True)."""
    bs = BiotSavart(coils)
    sq = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=surface.nfp, stellsym=True,
                          quadpoints_phi=surface.quadpoints_phi, quadpoints_theta=surface.quadpoints_theta)
    for m in range(0, surface.mpol + 1):
        for n in range(-surface.ntor, surface.ntor + 1):
            sq.set_rc(m, n, surface.get_rc(m, n))
            sq.set_zs(m, n, surface.get_zs(m, n))
    tf = ToroidalFlux(sq, BiotSavart(coils))
    tf_target = tf.J()
    qfm = QfmSurface(bs, sq, tf, tf_target)
    qfm.minimize_qfm(tol=tol, maxiter=maxiter, method="LBFGS", constraint_weight=1)
    return sq, float(qfm.qfm_objective(x=sq.x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coils")
    ap.add_argument("outdir")
    ap.add_argument("--no-vmec", action="store_true")
    ap.add_argument("--surface-wout", default=None, help="seed the QFM search from this equilibrium's LCFS (scaled) "
                    "instead of the LP QA target; B.n stats are then reported on both surfaces")
    ap.add_argument("--surface-scale", type=float, default=10.1266)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    bfield = load(args.coils)
    coils = bfield.coils
    qp = np.linspace(0, 1, NPHI, endpoint=True)
    qt = np.linspace(0, 1, NTHETA, endpoint=True)
    s = SurfaceRZFourier.from_vmec_input(str(INPUT), quadpoints_phi=qp, quadpoints_theta=qt)
    bfield.set_points(s.gamma().reshape((-1, 3)))
    Bn = np.sum(bfield.B().reshape((NPHI, NTHETA, 3)) * s.unitnormal(), axis=2)
    modB = bfield.AbsB()
    rec = dict(coils=str(args.coils), ncoils=len(coils),
               BdotN=float(np.mean(np.abs(Bn))), BdotN_over_B=float(np.mean(np.abs(Bn)) / np.mean(modB)),
               max_BdotN_over_B=float(np.max(np.abs(Bn.reshape(-1)) / modB.reshape(-1))), mean_B=float(np.mean(modB)))
    print("B.n stats on the LP QA target:", {k: v for k, v in rec.items() if k != "coils"}, flush=True)
    if args.surface_wout:  # our free-boundary runs move the plasma: seed the QFM from the LCFS the coils made
        s = SurfaceRZFourier.from_wout(args.surface_wout, s=1, quadpoints_phi=qp, quadpoints_theta=qt)
        for m in range(0, s.mpol + 1):  # Surface.scale() is not a geometric scaling: scale the coefficients
            for n in range(-s.ntor, s.ntor + 1):
                if m == 0 and n < 0:
                    continue
                s.set_rc(m, n, s.get_rc(m, n) * args.surface_scale)
                if not (m == 0 and n == 0):
                    s.set_zs(m, n, s.get_zs(m, n) * args.surface_scale)
        bfield.set_points(s.gamma().reshape((-1, 3)))
        Bn = np.sum(bfield.B().reshape((NPHI, NTHETA, 3)) * s.unitnormal(), axis=2)
        modB = bfield.AbsB()
        rec.update(surface_wout=args.surface_wout, BdotN_own=float(np.mean(np.abs(Bn))),
                   BdotN_over_B_own=float(np.mean(np.abs(Bn)) / np.mean(modB)),
                   max_BdotN_over_B_own=float(np.max(np.abs(Bn.reshape(-1)) / modB.reshape(-1))))
        print("B.n stats on the run's own LCFS (v2):", {k: rec[k] for k in ("BdotN_own", "BdotN_over_B_own", "max_BdotN_over_B_own")}, flush=True)

    t0 = time.time()
    qfm_surf, qfm_obj = qfm_generator(s, coils)
    rec.update(qfm_objective=qfm_obj, qfm_seconds=time.time() - t0)
    print(f"QFM: objective {qfm_obj:.3e} in {rec['qfm_seconds']:.0f} s", flush=True)
    qfm_surf.to_vtk(str(out / "qfm_surface"))
    # the QFM boundary as plain Fourier coefficients (VMEC convention: cos(m theta - nfp n phi)), for the vmex path
    rc = [[int(m), int(n), float(qfm_surf.get_rc(m, n))] for m in range(qfm_surf.mpol + 1)
          for n in range(-qfm_surf.ntor, qfm_surf.ntor + 1) if not (m == 0 and n < 0)]
    zs = [[int(m), int(n), float(qfm_surf.get_zs(m, n))] for m in range(qfm_surf.mpol + 1)
          for n in range(-qfm_surf.ntor, qfm_surf.ntor + 1) if not (m == 0 and n <= 0)]
    (out / "qfm_surface.json").write_text(json.dumps(dict(nfp=int(qfm_surf.nfp), mpol=int(qfm_surf.mpol),
                                                          ntor=int(qfm_surf.ntor), rc=rc, zs=zs, input=str(INPUT))) + "\n")

    if not args.no_vmec:
        from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
        from simsopt.util.mpi import MpiPartition

        mpi = MpiPartition(ngroups=1)
        equil = Vmec(str(INPUT), mpi)
        equil.boundary = qfm_surf
        equil.run()
        surfaces = np.arange(0, 1.01, 1.01 / NS)
        qs = QuasisymmetryRatioResidual(equil, surfaces, helicity_m=1, helicity_n=0)
        rec.update(qs_total=float(qs.total()), qs_profile=[float(v) for v in qs.profile()], qs_surfaces=surfaces.tolist(),
                   iota_axis=float(equil.wout.iotaf[0]), iota_edge=float(equil.wout.iotaf[-1]),
                   aspect=float(equil.wout.aspect), volume=float(equil.wout.volume_p))
        print(f"QS total (51 surfaces) {rec['qs_total']:.3e}; iota {rec['iota_axis']:.4f} -> {rec['iota_edge']:.4f}; A {rec['aspect']:.3f}", flush=True)
    (out / "gil_metrics.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
