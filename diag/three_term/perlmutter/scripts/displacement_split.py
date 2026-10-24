"""How much of a boundary displacement is relabelling: the Newton polish's move from a seed to its refined point.

usage: python displacement_split.py SEED_CHECKPOINT REFINED_CHECKPOINT [MAX_MODE]   (one JSON line)

Both checkpoints hold the model coordinates ``b`` and the run's deck (``meta.definition.deck``). The displacement
``b_refined - b_seed`` is taken in the scaled coordinates (``_ess_scale``, as the fit and the certificate) and split
against the relabellings ``theta -> theta + f`` (``certificate_split.relabel_basis``: ``f = sin(m theta - n nfp phi)``,
``dR = -R_theta f``, ``dZ = -Z_theta f``, projected onto the boundary coordinates). Reported: the scaled length, the
share of it in the relabelling span, and the relabelling basis' projection errors (how well the coordinates hold a
relabelling at this truncation).
"""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from certificate_split import relabel_basis  # noqa: E402


def main():
    import vmex as vj
    from vmex.core.optimize import _ess_scale, boundary_arrays_from_x

    seed, refined = (np.load(p, allow_pickle=False) for p in sys.argv[1:3])
    max_mode = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    deck = json.loads(str(refined["meta"]))["definition"]["deck"]
    with tempfile.NamedTemporaryFile("w", suffix=".run", delete=False) as f:
        f.write(deck)
    inp = vj.VmecInput.from_file(f.name)
    b0, b1 = np.asarray(seed["b"], float), np.asarray(refined["b"], float)
    nb = b1.size
    scale = np.asarray(_ess_scale(inp, max_mode, 1.2, vary_major_radius=True), float)[:nb]
    rbc, zbs = map(np.asarray, boundary_arrays_from_x(inp, b1, max_mode, vary_major_radius=True))
    inp = replace(inp, rbc=rbc, zbs=zbs)

    def surface(C_, S_):
        def at(x):
            rb, zb = map(np.asarray, boundary_arrays_from_x(inp, x, max_mode, vary_major_radius=True))
            return np.einsum("tpnm,nm->tp", C_, rb).ravel(), np.einsum("tpnm,nm->tp", S_, zb).ravel()
        R0, Z0 = at(b1)
        cols = []
        for k in range(nb):
            e = np.zeros(nb)
            e[k] = 1.0
            R1, Z1 = at(b1 + e)
            cols.append(np.r_[R1 - R0, Z1 - Z0])
        return np.array(cols).T

    T_raw, errors = relabel_basis(rbc, zbs, int(inp.nfp), max_mode, surface)
    Qt, _ = np.linalg.qr(T_raw / scale[:, None])
    d = (b1 - b0) / scale
    share = float(np.linalg.norm(Qt.T @ d) / max(np.linalg.norm(d), 1e-300))
    print(json.dumps(dict(seed=sys.argv[1], refined=sys.argv[2], scaled_length=float(np.linalg.norm(d)),
                          relabel_share=share, n_relabel=int(Qt.shape[1]), n_boundary=nb,
                          relabel_projection_error=[float(np.median(errors)), float(max(errors))])), flush=True)


if __name__ == "__main__":
    main()
