"""The rematerialized far-field sum equals the package's, in value and gradient."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("virtual_casing_jax")

import jax.numpy as jnp  # noqa: E402

from vmex.core.virtual_casing import _remat_fxd_u_eval  # noqa: E402


def test_remat_far_field_matches_package_value_and_gradient():
    from virtual_casing_jax import integrals

    rng = np.random.default_rng(0)
    X_src = jnp.asarray(rng.normal(size=(3, 1000)))
    X_trg = jnp.asarray(3.0 + rng.normal(size=(3, 70)))
    density = jnp.asarray(rng.normal(size=1000))
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=1000))
    kw = dict(chunk_size=256, target_chunk_size=32)  # both padded: 1000 = 3*256 + 232, 70 = 2*32 + 6

    ref = integrals.laplace_fxd_u_eval(X_src, X_trg, density, area, **kw)
    out = _remat_fxd_u_eval(X_src, X_trg, density, area, **kw)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

    def loss(f):
        return lambda xs, d: jnp.sum(jnp.sin(f(xs, X_trg, d, area, **kw)))

    g_ref = jax.grad(loss(integrals.laplace_fxd_u_eval), argnums=(0, 1))(X_src, density)
    g_out = jax.jit(jax.grad(loss(_remat_fxd_u_eval), argnums=(0, 1)))(X_src, density)
    for a, b in zip(g_out, g_ref):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)
