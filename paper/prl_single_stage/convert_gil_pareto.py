"""Gil et al. 2026 Pareto-front coil sets (PRL Fig. 1: LP QA at R0 = 1 m, 5 coils/hfp, total length 25 / 30 m, force
thresholds 9-14 kN/m) -> ESSOS coil files coils/gilpar_L<L>_F<thr>.json + coils/gil_pareto_index.json.

    python convert_gil_pareto.py          (pure numpy; login-node safe)

Reads references/gil2026/auglag_extract/zenodo_repository/qa_pareto_{25,30}_comparison/*/biot_savart_*.json
(SIMSOPT graph, parsed without simsopt); currents are rescaled to the LP QA deck's RBTOR as in convert_coilsets.py.
The archive's own metrics (length per hfp, max curvature from the dofs) are stored for the record.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "references/gil2026"))
from dump_coils import Graph, coil_metrics  # noqa: E402
from convert_coilsets import fourier_gamma, symmetric_copies, linked_rbphi, RBTOR, NSEG  # noqa: E402

SRC = HERE.parent / "references/gil2026/auglag_extract/zenodo_repository"
OUT = HERE / "coils"
rows = []
for d in sorted(SRC.glob("qa_pareto_*_comparison/*/")):
    L = 30 if "_30_" in d.name else 25
    thr = re.search(r"threshold([0-9p.]+)kNm", d.name).group(1).replace("p", ".")
    fn = next((d / n for n in ("biot_savart_same_setup.json", "biot_savart_total_length.json") if (d / n).exists()), None)
    if fn is None:
        print("no R0 = 1 coil file in", d.name); continue
    G = Graph(fn)
    base, currents, metrics = [], [], []
    for cref in G.root["coils"]:
        coil = G.deref(cref)
        (g, dg, ddg), dofs, order = G.curve(coil["curve"])
        if dofs is not None:
            base.append(dofs.reshape(3, 2 * order + 1)); currents.append(G.current(coil["current"])); metrics.append(coil_metrics(dg, ddg))
    base, rel = np.array(base), np.array(currents)
    coils = []
    for b, I in zip(base, rel):
        gg, dgg = fourier_gamma(b, order, NSEG)
        coils += symmetric_copies(gg, dgg, I, 2, True)
    factor = RBTOR / linked_rbphi(coils)
    name = f"gilpar_L{L}_F{thr}"
    json.dump(dict(nfp=2, stellsym=True, order=order, n_segments=NSEG, dofs_curves_raw=base.tolist(), scaling_type=2,
                   scaling_factor=0.0, scale_fixed=1.0, dofs_currents_raw=(rel * factor).tolist(), currents_scale=None),
              open(OUT / f"{name}.json", "w"), indent=1)
    rows.append(dict(name=name, L_target=L, force_threshold_kNm=float(thr), source=str(fn.relative_to(SRC)), ncoils=len(base), order=order,
                     total_length=float(sum(m[0] for m in metrics)), max_kappa=float(max(m[1] for m in metrics)),
                     max_msc=float(max(m[2] for m in metrics)), currents=(rel * factor).tolist(), archive_currents=rel.tolist()))
    print(f"{name:18s} {fn.name:32s} L/hfp {rows[-1]['total_length']:.2f}  kappa_max {rows[-1]['max_kappa']:.1f}  MSC_max {rows[-1]['max_msc']:.2f}")
json.dump(rows, open(OUT / "gil_pareto_index.json", "w"), indent=1)
print(f"wrote {len(rows)} sets, index -> coils/gil_pareto_index.json")
