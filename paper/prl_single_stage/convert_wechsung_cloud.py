"""Wechsung et al. 2022 archive (all 8 random starts at 18/20/22/24 m, QA target) -> ESSOS coil files at R0 = 1 m.

    python convert_wechsung_cloud.py          (pure numpy; login-node safe)

Reads references/wechsung2022/.../archive/minimizers/<run>__curve_i.txt (SIMSOPT CurveXYZFourier full_x, order 16,
coils 0-3 = base coils) and __current_i.txt (relative currents), writes coils/wech_cloud_L<L>_ig<k>.json with the
currents scaled so the linked mu0 I / 2 pi equals the LP QA lowres deck's RBTOR (as convert_coilsets.py).
"""
import json
import re
from pathlib import Path

import numpy as np

from convert_coilsets import fourier_gamma, symmetric_copies, linked_rbphi, RBTOR, NSEG  # noqa: E402  (runs its conversion once)

SRC = Path("/pscratch/sd/h/hongkelu/freeboundary-single-stage/references/wechsung2022/florianwechsung-CoilsForPreciseQS-aea919f/archive/output")
OUT = Path(__file__).resolve().parent / "coils"
order, nfp, nbase = 16, 2, 4
runs = sorted(p.name for p in SRC.glob("well_False_*") if (p / "xmin.txt").exists())
rows = []
for run in runs:
    L = float(re.search(r"lengthbound_([\d.]+)", run).group(1)); ig = int(re.search(r"_ig_(\d+)_", run).group(1))
    # xmin layout (verified against the published minimizers): currents 1..3 (current 0 fixed at 1), then 4 x 99 curve dofs
    x = np.loadtxt(SRC / run / "xmin.txt")
    rel = np.r_[1.0, x[:3]]
    base = x[3:].reshape(nbase, 3, 2 * order + 1)
    coils = []
    for b, I in zip(base, rel):
        g, dg = fourier_gamma(b, order, NSEG)
        coils += symmetric_copies(g, dg, I, nfp, True)
    factor = RBTOR / linked_rbphi(coils)
    currents = rel * factor
    lengths = [float(np.mean(np.linalg.norm(fourier_gamma(b, order, 1024)[1], axis=1))) for b in base]
    name = f"wech_cloud_L{int(L)}_ig{ig}"
    json.dump(dict(nfp=nfp, stellsym=True, order=order, n_segments=NSEG, dofs_curves_raw=base.tolist(), scaling_type=2,
                   scaling_factor=0.0, scale_fixed=1.0, dofs_currents_raw=currents.tolist(), currents_scale=None),
              open(OUT / f"{name}.json", "w"), indent=1)
    rows.append(dict(name=name, L=L, ig=ig, total_length=float(sum(lengths)), currents=currents.tolist()))
    print(f"{name:22s} total length {sum(lengths):.3f} m (bound {L})  I = {np.round(currents, 0)}")
json.dump(rows, open(OUT / "wech_cloud_index.json", "w"), indent=1)
