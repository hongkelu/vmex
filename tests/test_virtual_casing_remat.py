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


def test_scatter_free_singular_matches_package_value_and_tangent():
    """The singular evaluator without the polar scatter equals the package's: same value and JVP (a regrouped sum)."""
    from virtual_casing_jax import integrals

    nfp, nt, npol, trg_nt, trg_np = 2, 64, 32, 8, 16
    t = jnp.arange(nt) * 2 * jnp.pi / nt
    p = jnp.arange(npol) * 2 * jnp.pi / npol
    T, P = jnp.meshgrid(t, p, indexing="ij")

    def surface(a):
        R = 1.0 + a * jnp.cos(P) + 0.05 * jnp.cos(P - nfp * T)
        Z = a * jnp.sin(P) + 0.05 * jnp.sin(P - nfp * T)
        X = jnp.stack([R * jnp.cos(T), R * jnp.sin(T), Z])
        dX = jnp.concatenate([jnp.gradient(X, t, axis=1), jnp.gradient(X, p, axis=2)])
        return X, dX

    density = jnp.cos(P) + 0.3 * jnp.sin(2 * P - nfp * T)
    kw = dict(digits=4, patch_dim0=8, chunk_size=256, target_chunk_size=16)

    def run(f, a):
        X, dX = surface(a)
        return f(X, dX, density, trg_nt, trg_np, nfp, **kw)

    from vmex.core.virtual_casing import _fxd_u_eval_singular

    ref, dref = jax.jvp(lambda a: run(integrals.laplace_fxd_u_eval_singular, a), (0.3,), (1.0,))
    out, dout = jax.jvp(lambda a: run(_fxd_u_eval_singular, a), (0.3,), (1.0,))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-13 * float(jnp.abs(ref).max()))
    np.testing.assert_allclose(dout, dref, rtol=1e-10, atol=1e-13 * float(jnp.abs(dref).max()))


def test_scatter_free_singular_traces_under_jit():
    """The dense interpolation is built inside a jitted call (the patch tables can be tracers there)."""
    from vmex.core.virtual_casing import _fxd_u_eval_singular

    nfp, nt, npol = 2, 32, 16
    t = jnp.arange(nt) * 2 * jnp.pi / nt
    p = jnp.arange(npol) * 2 * jnp.pi / npol
    T, P = jnp.meshgrid(t, p, indexing="ij")

    @jax.jit
    def run(a):
        R = 1.0 + a * jnp.cos(P)
        X = jnp.stack([R * jnp.cos(T), R * jnp.sin(T), a * jnp.sin(P)])
        dX = jnp.concatenate([jnp.gradient(X, t, axis=1), jnp.gradient(X, p, axis=2)])
        return _fxd_u_eval_singular(X, dX, jnp.cos(P), 4, 8, nfp, digits=4, patch_dim0=8, chunk_size=128,
                                    target_chunk_size=8, orient=1.0)

    assert np.all(np.isfinite(np.asarray(run(0.3))))
