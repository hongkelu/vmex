"""E1: native ESSOS BiotSavart against the driver's DirectCoilField at the same discretized coils.

usage: JAX_PLATFORMS=cpu python essos_parity.py COILS.json OUT.json   (cwd: a vmex tree)

Both fields are built from the same ESSOS Coils (the coil chart's map: Coils(Curves(dofs), currents)).  Compared
at random points near the coils' torus: B_cyl values; the JVP with respect to the curve dofs and the currents
(both through the same Coils construction); a central difference of the same; the number of compilations for
three different geometries (jit cache misses); and warm execution time.  Also with every current negated.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "examples/optimization")
from single_stage_free_boundary_optimization_three_term import DirectCoilField  # noqa: E402
from essos.coils import Coils, Curves  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402


def main():
    """Parity of values, derivatives, compilations and time; write OUT.json."""
    base = Coils.from_json(sys.argv[1])
    dofs0, cur0 = jnp.asarray(base.dofs_curves), jnp.asarray(base.dofs_currents_raw if hasattr(base, "dofs_currents_raw") else base.currents)
    nseg, nfp, ss = int(base.curves.n_segments), int(base.nfp), bool(base.stellsym)

    def coils(dofs, currents):
        return Coils(Curves(dofs, nseg, nfp, ss), currents)

    rng = np.random.default_rng(0)
    R0 = float(np.mean(np.hypot(np.asarray(base.gamma)[..., 0], np.asarray(base.gamma)[..., 1])))
    n = 64
    R = R0 * (1 + 0.15 * rng.uniform(-1, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    Z = 0.15 * R0 * rng.uniform(-1, 1, n)
    pts = tuple(map(jnp.asarray, (R, phi, Z)))

    def essos_b(dofs, currents):
        return jnp.stack(BiotSavart(coils(dofs, currents)).b_cyl(*pts))

    def direct_b(dofs, currents):
        c = coils(dofs, currents)
        return jnp.stack(DirectCoilField(jnp.asarray(c.gamma), jnp.asarray(c.gamma_dash), jnp.asarray(c.currents)).b_cyl(*pts))

    out = {}
    for sign in (1.0, -1.0):
        cur = sign * cur0
        be, bd = essos_b(dofs0, cur), direct_b(dofs0, cur)
        tdofs = jnp.asarray(rng.standard_normal(dofs0.shape)) * 1e-2 * jnp.max(jnp.abs(dofs0))
        tcur = jnp.asarray(rng.standard_normal(cur.shape)) * 1e-2 * jnp.max(jnp.abs(cur))
        je = jax.jvp(essos_b, (dofs0, cur), (tdofs, tcur))[1]
        jd = jax.jvp(direct_b, (dofs0, cur), (tdofs, tcur))[1]
        h = 1e-4
        fd = (direct_b(dofs0 + h * tdofs, cur + h * tcur) - direct_b(dofs0 - h * tdofs, cur - h * tcur)) / (2 * h)
        out[f"sign{int(sign)}"] = dict(
            value_rel=float(jnp.max(jnp.abs(be - bd)) / jnp.max(jnp.abs(bd))),
            jvp_rel=float(jnp.max(jnp.abs(je - jd)) / jnp.max(jnp.abs(jd))),
            jvp_vs_fd=float(jnp.max(jnp.abs(jd - fd)) / jnp.max(jnp.abs(jd))))
    # compilations and warm time over three geometries, each provider under one jit
    for name, f in (("essos", essos_b), ("direct", direct_b)):
        g = jax.jit(f)
        times = []
        try:
            for k in range(3):
                d = dofs0 * (1 + 1e-3 * k)
                t0 = time.perf_counter()
                jax.block_until_ready(g(d, cur0))
                times.append(time.perf_counter() - t0)
        except Exception as error:  # the provider cannot run under an outer jit
            out[name] = dict(jit_error=repr(error)[:300])
            continue
        out[name] = dict(cache_size=int(g._cache_size()) if hasattr(g, "_cache_size") else None,
                         first_s=times[0], warm_s=min(times[1:]))
    out["setup"] = dict(nfp=nfp, stellsym=ss, n_segments=nseg, n_points=n)
    print(json.dumps(out, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
