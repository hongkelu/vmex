"""Seed preconditioning must not reuse the seed operator or weaken acceptance."""

from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pytest

from vmex.core import freeboundary_continuation as fc, freeboundary_implicit as fbi
from vmex.core import _freeboundary_dense as dense, _freeboundary_matrixfree as mf
from tests.test_freeboundary_linearization import fixture

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_current_nonlinear_root_arbitrary_rows_and_seed_lifetime(monkeypatch):
    accepted, cfg, matrix, coupling = fixture(monkeypatch)
    residual = jax.jit(lambda z, p, f, *_: matrix @ z + 0.1 * z * z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_: residual)
    seed_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed_root.offload_factors()
    seed = seed_root.preconditioner()
    factors = seed._seed.factors[0].copy()
    seed_root.close()
    monkeypatch.setattr(dense, "_assemble_device", lambda *a, **k: pytest.fail("unexpected dense assembly"))
    state = jnp.array([0.2, -0.3])
    field = np.linalg.solve(coupling, -matrix @ state - 0.1 * state * state)
    other = NS(**(vars(accepted) | dict(state=state, parameters=field)))
    rhs = jnp.array([[1.0, 2.0], [3.0, 4.0], [-3.0, -4.0], [1.0, 2.0], [0.0, 0.0]])
    diagnostics = []
    root = fc.free_boundary_continuation_state_pullback(
        other, cfg, rhs, preconditioner=seed, return_linearization=True, diagnostics=diagnostics
    )
    current = np.asarray(matrix) + np.diag(0.2 * np.asarray(state))
    expected = -np.asarray(rhs) @ np.linalg.solve(current, coupling)
    np.testing.assert_allclose(root.field_jacobian, expected, rtol=1e-10, atol=1e-13)
    assert all(row["accepted"] for row in diagnostics)
    assert [row["iterations"] for row in diagnostics][2:] == [0, 0, 0]
    direction = jnp.array([0.3, -0.4])
    notes = []
    for alpha in (1.0, 0.5, 0.0):
        np.testing.assert_allclose(
            root.tangent(other, cfg, alpha * direction, diagnostics=notes),
            -np.linalg.solve(current, coupling @ (alpha * direction)),
            rtol=1e-10,
            atol=1e-13,
        )
    assert notes[1]["scaled_reuse"] and notes[1]["iterations"] == 0
    np.testing.assert_array_equal(seed._seed.factors[0], factors)
    seed.close()
    # A live current-root predictor owns its factor reference independently.
    assert np.all(np.isfinite(root.tangent(other, cfg, direction)))
    with pytest.raises(ValueError, match="closed"):
        fc.free_boundary_continuation_state_pullback(other, cfg, rhs, preconditioner=seed)
    root.close()
    with pytest.raises(ValueError, match="closed"):
        root.tangent(other, cfg, direction)


def test_preconditioner_rejects_foreign_config_mask_and_uncertified_root(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = root.preconditioner()
    with pytest.raises(ValueError, match="different continuation"):
        fc.free_boundary_continuation_state_pullback(accepted, NS(**vars(cfg)), jnp.eye(2), preconditioner=seed)
    other = NS(**(vars(accepted) | dict(dof_mask=jnp.array([1.0, 0.0]))))
    with pytest.raises(ValueError, match="active space"):
        fc.free_boundary_continuation_state_pullback(other, cfg, jnp.eye(2), preconditioner=seed)
    other = NS(**(vars(accepted) | dict(state=jnp.ones(2))))
    with pytest.raises(ValueError, match="root residual"):
        fc.free_boundary_continuation_state_pullback(other, cfg, jnp.eye(2), preconditioner=seed)
    for kw in (dict(rtol=0.0), dict(rtol=np.nan), dict(restart=0), dict(max_restarts=1.5)):
        with pytest.raises(ValueError):
            root.preconditioner(**kw)
    seed.close()
    root.close()


def test_true_residual_rejects_bad_krylov_answer_in_both_directions(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    dense_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = dense_root.preconditioner()
    root = fc.free_boundary_continuation_state_pullback(
        accepted, cfg, jnp.eye(2), preconditioner=seed, return_linearization=True
    )
    monkeypatch.setattr(mf, "solve", lambda action, z, space, lu, rhs, **kw: (jnp.zeros_like(rhs), 1))
    reports = []
    with pytest.raises(Exception, match="matrixfree_seed_lu"):
        fc.free_boundary_continuation_state_pullback(
            accepted, cfg, jnp.eye(2), preconditioner=seed, diagnostics=reports
        )
    assert reports and not reports[0]["accepted"]
    with pytest.raises(Exception, match="matrixfree_seed_lu"):
        root.tangent(accepted, cfg, jnp.ones(2))
    root.close()
    seed.close()
    dense_root.close()


def test_stable_tape_changes_numbers_without_new_executables():
    matrix = jnp.array([[3.0, 1.0], [-0.5, 2.0]])

    @jax.jit
    def residual(z, p, f, b, r, c):
        return matrix @ z + 0.01 * z * z + f

    space = dense._Space(jnp.arange(2), jnp.arange(2), jnp.zeros(2), jnp.ones(2))
    factors = jl.lu_factor(matrix)
    counts = []
    for i in range(3):
        z = jnp.array([0.2, 0.4]) + i * 0.1
        action = mf.prepare(z, None, jnp.zeros(2), z, None, None, residual=residual)
        current = np.asarray(matrix) + np.diag(0.02 * np.asarray(z))
        for transpose in (False, True):
            result, _ = mf.solve(
                action, z, space, factors, jnp.ones(2), transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
            )
            np.testing.assert_allclose(
                result, np.linalg.solve(current.T if transpose else current, np.ones(2)), rtol=1e-10
            )
        counts.append(mf.solve._cache_size())
    assert counts[0] == counts[1] == counts[2]


def test_redundant_paired_coordinates():
    space = dense._Space(jnp.array([0, 1]), jnp.array([0, 2]), jnp.array([0.0, -1.0]), jnp.array([1.0, 1 / np.sqrt(2)]))
    q = np.array([[1.0, 0.0], [0.0, 1 / np.sqrt(2)], [0.0, -1 / np.sqrt(2)]])
    active = np.array([[2.0, 0.5], [-0.8, 3.0]])
    action = jax.tree_util.Partial(jnp.matmul, jnp.asarray(q @ active @ q.T))
    factors = jl.lu_factor(jnp.asarray(active + np.eye(2) * 0.1))
    rhs = jnp.array([1.0, -2.0])
    for transpose in (False, True):
        solution, _ = mf.solve(
            action, jnp.zeros(3), space, factors, rhs, transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
        )
        np.testing.assert_allclose(solution, np.linalg.solve(active.T if transpose else active, rhs), rtol=1e-10)
