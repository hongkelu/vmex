"""The exact condensation gauge on a saved boundary: does it pin the relabellings, and how far is the boundary from it?

usage: python gauge_check.py CHECKPOINT [MAX_MODE]   (one JSON line)

The gauge (``freeboundary_vc._gauge_rows``) is evaluated on the checkpoint's boundary (48 x 48 grid over one field
period, as the model's ``surface``); its Jacobian ``C`` is taken by autodiff in the scaled model coordinates.
Reported: the number of conditions, their size at this boundary, the singular values of ``C`` restricted to the
relabelling span (``certificate_split.relabel_basis``) and to its complement (the gauge should act on the first and
barely on the second), and the rank of ``C``.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from certificate_split import relabel_basis  # noqa: E402


def main():
    import jax
    import jax.numpy as jnp

    import vmex as vj
    from vmex.core.freeboundary_vc import _gauge_rows
    from vmex.core.optimize import _ess_scale, boundary_arrays_from_x

    data = np.load(sys.argv[1], allow_pickle=False)
    max_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    deck = json.loads(str(data["meta"]))["definition"]["deck"]
    with tempfile.NamedTemporaryFile("w", suffix=".run", delete=False) as f:
        f.write(deck)
    inp = vj.VmecInput.from_file(f.name)
    b = np.asarray(data["b"], float)
    nb = b.size
    scale = np.asarray(_ess_scale(inp, max_mode, 1.2, vary_major_radius=True), float)[:nb]
    nfp = int(inp.nfp)

    def surface(x):
        rbc, zbs = boundary_arrays_from_x(inp, x, max_mode, vary_major_radius=True)
        th = jnp.linspace(0.0, 2 * jnp.pi, 48, endpoint=False)
        ph = jnp.linspace(0.0, 2 * jnp.pi / nfp, 48, endpoint=False)
        n = (jnp.arange(rbc.shape[0]) - (rbc.shape[0] - 1) // 2) * nfp
        m = jnp.arange(rbc.shape[1])
        angle = m[None, None, None, :] * th[:, None, None, None] - n[None, None, :, None] * ph[None, :, None, None]
        return (jnp.einsum("tpnm,nm->tp", jnp.cos(angle), rbc), jnp.einsum("tpnm,nm->tp", jnp.sin(angle), zbs))

    def gauge(x):
        R, Z = surface(x)
        return _gauge_rows(R, Z, int(inp.mpol), int(inp.ntor))

    c = np.asarray(gauge(jnp.asarray(b)))
    C = np.asarray(jax.jacfwd(gauge)(jnp.asarray(b))) * scale  # scaled coordinates

    rbc, zbs = map(np.asarray, boundary_arrays_from_x(inp, b, max_mode, vary_major_radius=True))
    from dataclasses import replace
    inp_b = replace(inp, rbc=rbc, zbs=zbs)

    def surface_np(C_, S_):
        def at(x):
            rb, zb = map(np.asarray, boundary_arrays_from_x(inp_b, x, max_mode, vary_major_radius=True))
            return np.einsum("tpnm,nm->tp", C_, rb).ravel(), np.einsum("tpnm,nm->tp", S_, zb).ravel()
        R0, Z0 = at(b)
        cols = []
        for k in range(nb):
            e = np.zeros(nb)
            e[k] = 1.0
            R1, Z1 = at(b + e)
            cols.append(np.r_[R1 - R0, Z1 - Z0])
        return np.array(cols).T

    T_raw, _ = relabel_basis(rbc, zbs, int(inp.nfp), max_mode, surface_np)
    Qt, _ = np.linalg.qr(T_raw / scale[:, None])
    U, sv, _ = np.linalg.svd(np.eye(nb) - Qt @ Qt.T)
    B = U[:, sv > 0.5]
    s_T = np.linalg.svd(C @ Qt, compute_uv=False)
    s_B = np.linalg.svd(C @ B, compute_uv=False)
    s_C = np.linalg.svd(C, compute_uv=False)
    print(json.dumps(dict(checkpoint=sys.argv[1], conditions=int(c.size), gauge_max=float(np.abs(c).max()),
                          gauge_rms=float(np.sqrt(np.mean(c**2))), rank=int(np.sum(s_C > 1e-10 * s_C[0])),
                          C_sv_max_min=[float(s_C[0]), float(s_C[-1])],
                          relabel_sv_max_min=[float(s_T[0]), float(s_T[min(len(s_T), c.size) - 1])],
                          relabel_sv_tail=[float(v) for v in s_T[-12:]],
                          complement_sv_max=float(s_B[0]), n_relabel=int(Qt.shape[1]))), flush=True)


if __name__ == "__main__":
    main()
