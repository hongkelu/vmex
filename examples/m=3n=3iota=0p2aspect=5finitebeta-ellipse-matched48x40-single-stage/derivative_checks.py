"""Bounded local derivative consistency checks; no equilibrium changes."""
import jax
import jax.numpy as jnp
import numpy as np


def contract_rows(rhs, tangent):
    return sum(np.asarray(jnp.sum(a * b, axis=tuple(range(1, a.ndim))))
               for a, b in zip(jax.tree.leaves(rhs), jax.tree.leaves(tangent)))


def check_state_rows(rows, state, mask, rhs, event):
    rng = np.random.default_rng(20260914)
    for direction in range(3):
        delta = jax.tree.map(
            lambda x, m: jnp.asarray(rng.normal(size=x.shape)) * m * 1e-5,
            state, mask)
        ad = contract_rows(rhs, delta)
        for h in (1., .5):
            plus = jax.tree.map(lambda x, d: x + h*d, state, delta)
            minus = jax.tree.map(lambda x, d: x - h*d, state, delta)
            fd = (np.asarray(rows(plus)) - np.asarray(rows(minus))) / (2*h)
            error = np.abs(ad-fd) / np.maximum(np.maximum(np.abs(ad), np.abs(fd)), 1e-6)
            passed = bool(np.all(np.isfinite(error)) and np.max(error) <= 1e-4)
            event('state_row_fd', direction=direction, h=h, ad=ad.tolist(),
                  fd=fd.tolist(), scaled_error=error.tolist(), passed=passed)
            if not passed:
                raise ValueError('host-eager scalar-row finite-difference check failed')


def check_duality(rhs, tangent, jacobian, delta, event):
    forward = contract_rows(rhs, tangent)
    reverse = np.asarray(jacobian) @ np.asarray(delta)
    error = np.abs(forward-reverse) / np.maximum(
        np.maximum(np.abs(forward), np.abs(reverse)), 1e-8)
    passed = bool(np.all(np.isfinite(error)) and np.max(error) <= 2e-4)
    event('tangent_adjoint_duality', forward=forward.tolist(), reverse=reverse.tolist(),
          scaled_error=error.tolist(), passed=passed)
    if not passed:
        raise ValueError('tangent/adjoint duality check failed; no trial solve')
