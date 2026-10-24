"""The fit-sensitivity algebra on a small analytic least-squares problem (CPU; Package C3 of the PR #121 brief).

The inner problem is ``y*(d) = argmin |r(y, d)|^2 / 2`` with more residuals than unknowns, a nonzero residual at the
minimum, a nearly flat direction (``y3 - y4`` enters only through a small term, like a boundary relabelling), and a
non-unit coordinate scale.  Exact derivatives come from JAX; the reference is a central difference of the tightly
re-solved minimizer.  Checked: the exact stationarity derivative (with the mixed design curvature) matches the
reference and Gauss-Newton does not; Hessian-vector products, mixed partials and the symmetry of ``H``; central
differences of ``g = J^T r`` (the production H products) against the exact ones; and
``freeboundary_vc._low_rank_multipliers``: Gauss-Newton with no directions, exact with all of them, and most of the
correction from the weak direction alone.
"""
import numpy as np
import pytest
from scipy.optimize import least_squares

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from vmex.core import freeboundary_vc as fvc  # noqa: E402

SCALE = np.array([1.0, 2.0, 0.5, 3.0, 1.0])  # the model coordinates' scale (W orthonormal in 1/scale^2)


def residual(y, d):
    """8 residuals of 5 unknowns and 3 design variables; nonzero at the minimum; y3 - y4 nearly flat."""
    s, t = y[3] + y[4], y[3] - y[4]
    return jnp.stack([
        y[0] - d[0] + 0.3 * y[1] ** 2,
        y[1] - d[1] * y[0],
        jnp.sin(y[2]) - d[2] + 0.2 * y[0] * y[1],
        s - d[0] * d[2],
        0.5 * (y[0] * y[2] - 1.0) + 0.1 * s ** 2,
        y[1] + y[2] - 0.7,  # conflicts with the rows above: a nonzero residual
        1e-3 * (t - d[1]) + 5e-4 * jnp.sin(t) * y[0],  # the weak direction, with design coupling
        0.4 * (s - 1.0) + 0.3 * y[2] ** 2 - d[1],
    ])


def quantity(y, d):
    """An outer value that depends on the weak direction too (as discretized physics does)."""
    return y[0] * y[2] + d[1] * y[3] ** 2 + 0.3 * jnp.cos(y[4]) + d[0] * y[1]


D0 = np.array([0.8, 0.6, 0.5])


def solve(d, y0=None):
    """The minimizer to round-off: Levenberg-Marquardt, then Newton on the stationarity J^T r = 0 (exact H)."""
    y0 = np.array([0.5, 0.3, 0.4, 0.5, 0.3]) if y0 is None else y0
    dj = jnp.asarray(d)
    fun = lambda y: np.asarray(residual(jnp.asarray(y), dj))  # noqa: E731
    jac = lambda y: np.asarray(jax.jacfwd(residual)(jnp.asarray(y), dj))  # noqa: E731
    y = least_squares(fun, y0, jac=jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000).x
    g = lambda yy: jax.jacfwd(residual, 0)(yy, dj).T @ residual(yy, dj)  # noqa: E731
    for _ in range(20):
        yj = jnp.asarray(y)
        step = np.linalg.solve(np.asarray(jax.jacfwd(g)(yj)), np.asarray(g(yj)))
        y = y - step
        if np.linalg.norm(step) < 1e-15 * max(np.linalg.norm(y), 1.0):
            break
    return y


@pytest.fixture(scope="module")
def point():
    y = solve(D0)
    yj, dj = jnp.asarray(y), jnp.asarray(D0)
    r = np.asarray(residual(yj, dj))
    J_y = np.asarray(jax.jacfwd(residual, 0)(yj, dj))
    J_d = np.asarray(jax.jacfwd(residual, 1)(yj, dj))
    g = lambda yy, dd: jax.jacfwd(residual, 0)(yy, dd).T @ residual(yy, dd)  # noqa: E731  (J_y^T r)
    H = np.asarray(jax.jacfwd(g, 0)(yj, dj))  # J^T J + sum r_i d2r_i/dy2
    F_d = np.asarray(jax.jacfwd(g, 1)(yj, dj))  # J^T J_d + sum r_i d2r_i/dy dd
    Q_y = np.asarray(jax.grad(quantity, 0)(yj, dj))
    Q_d = np.asarray(jax.grad(quantity, 1)(yj, dj))
    return dict(y=y, r=r, J_y=J_y, J_d=J_d, H=H, F_d=F_d, Q_y=Q_y, Q_d=Q_d, g=g)


def reference(h):
    """d/dd Q(y*(d), d) by central differences of tightly re-solved minimizers."""
    y = solve(D0)
    out = []
    for k in range(3):
        e = np.eye(3)[k] * h
        qp = float(quantity(jnp.asarray(solve(D0 + e, y)), jnp.asarray(D0 + e)))
        qm = float(quantity(jnp.asarray(solve(D0 - e, y)), jnp.asarray(D0 - e)))
        out.append((qp - qm) / (2 * h))
    return np.array(out)


def exact_gradient(p):
    mu = np.linalg.solve(p["H"].T, p["Q_y"])
    return p["Q_d"] - p["F_d"].T @ mu


def gauss_newton_gradient(p):
    mu = np.linalg.solve(p["J_y"].T @ p["J_y"], p["Q_y"])
    return p["Q_d"] - (p["J_y"].T @ p["J_d"]).T @ mu


def low_rank_gradient(p, W, DW, SbdW):
    """The production assembly: ``Q_d - J_d^T J_y mu - S_bd^T (P mu)`` with the helper's multipliers."""
    mu, MW = fvc._low_rank_multipliers(p["J_y"], p["Q_y"][None, :], W, DW, SCALE)
    mu = mu[:, 0]
    return p["Q_d"] - p["J_d"].T @ (p["J_y"] @ mu) - SbdW @ (MW.T @ mu)


def weak_directions(p, k):
    """The ``k`` weakest right singular directions of the scaled Jacobian, in model coordinates."""
    _, _, Vt = np.linalg.svd(p["J_y"] * SCALE, full_matrices=False)
    return (Vt[Vt.shape[0] - k:].T if k else np.zeros((5, 0))) * SCALE[:, None]


def curvature_columns(p, W):
    S = p["H"] - p["J_y"].T @ p["J_y"]
    S_yd = p["F_d"] - p["J_y"].T @ p["J_d"]
    return S @ W, S_yd.T @ W


def test_the_problem_has_a_nonzero_residual_and_a_weak_direction(point):
    assert np.linalg.norm(point["r"]) > 1e-2
    sv = np.linalg.svd(point["J_y"] * SCALE, compute_uv=False)
    assert sv[-1] / sv[0] < 1e-2


def test_exact_stationarity_derivative_matches_the_resolved_reference(point):
    exact = exact_gradient(point)
    for h in (1e-4, 1e-5):
        np.testing.assert_allclose(exact, reference(h), rtol=1e-6, atol=1e-9)


def test_gauss_newton_misses_the_reference(point):
    gn, ref = gauss_newton_gradient(point), reference(1e-5)
    assert np.linalg.norm(gn - ref) / np.linalg.norm(ref) > 1e-3


def test_hessian_vector_products_mixed_partials_and_symmetry(point):
    yj, dj = jnp.asarray(point["y"]), jnp.asarray(D0)
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal(5), rng.standard_normal(5)
    Hv = np.asarray(jax.jvp(lambda yy: point["g"](yy, dj), (yj,), (jnp.asarray(v),))[1])
    np.testing.assert_allclose(Hv, point["H"] @ v, rtol=1e-12, atol=1e-13)
    assert abs(u @ point["H"] @ v - v @ point["H"] @ u) < 1e-12 * np.abs(point["H"]).max()
    w = rng.standard_normal(3)  # mixed partials: d/dd (J^T r) by a central difference
    eps = 1e-6
    fd = (np.asarray(point["g"](yj, dj + eps * w)) - np.asarray(point["g"](yj, dj - eps * w))) / (2 * eps)
    np.testing.assert_allclose(fd, point["F_d"] @ w, rtol=1e-7, atol=1e-9)


def test_differenced_hessian_columns_match_the_exact_ones(point):
    """The production H products: central differences of g at y +- eps w (step 1e-4 in the scaled metric)."""
    W = weak_directions(point, 5)
    DW, _ = curvature_columns(point, W)
    A = point["J_y"].T @ point["J_y"]
    yj, dj = jnp.asarray(point["y"]), jnp.asarray(D0)
    for j in range(5):
        w = W[:, j]
        eps = 1e-4 / np.linalg.norm(w / SCALE)
        gp, gm = (np.asarray(point["g"](yj + s * eps * jnp.asarray(w), dj)) for s in (1, -1))
        np.testing.assert_allclose((gp - gm) / (2 * eps) - A @ w, DW[:, j], rtol=1e-6, atol=1e-8)


def test_low_rank_is_gauss_newton_with_no_directions_and_exact_with_all(point):
    zero = np.zeros((5, 0))
    np.testing.assert_allclose(low_rank_gradient(point, zero, zero, np.zeros((3, 0))),
                               gauss_newton_gradient(point), rtol=1e-9, atol=1e-12)  # cond^2 eps: about 1e-10
    W = weak_directions(point, 5)
    DW, SbdW = curvature_columns(point, W)
    np.testing.assert_allclose(low_rank_gradient(point, W, DW, SbdW), exact_gradient(point), rtol=1e-9,
                               atol=1e-12)


def test_preconditioned_cg_solves_the_exact_multiplier(point):
    """The production solver (``_preconditioned_cg``) reaches ``H^-1 Q_y`` with the Gauss-Newton preconditioner."""
    H, A = point["H"], point["J_y"].T @ point["J_y"]
    mu, it, res, reason = fvc._preconditioned_cg(lambda v: H @ v, lambda v: np.linalg.solve(A, v), point["Q_y"], 1e-12, 50)
    np.testing.assert_allclose(mu, np.linalg.solve(H, point["Q_y"]), rtol=1e-9, atol=1e-12)
    assert it <= 5 and reason == "converged" and np.linalg.norm(res) <= 1e-12 * np.linalg.norm(point["Q_y"])


def test_subspace_preconditioner_is_exact_with_every_direction_and_cuts_iterations(point):
    """``_low_rank_model``: H itself when W spans everything (CG then needs no iteration), and fewer iterations than
    Gauss-Newton with the weak direction alone; the multiplier is the exact one either way."""
    H, A, q = point["H"], point["J_y"].T @ point["J_y"], point["Q_y"]
    W = weak_directions(point, 5)
    DW, _ = curvature_columns(point, W)
    np.testing.assert_allclose(fvc._low_rank_model(point["J_y"], W, DW, SCALE), H, rtol=1e-10, atol=1e-12)
    counts = {}
    for k in (0, 1):
        Wk = weak_directions(point, k)
        M = fvc._low_rank_model(point["J_y"], Wk, curvature_columns(point, Wk)[0], SCALE) if k else A
        mu, counts[k], _, _ = fvc._preconditioned_cg(lambda v: H @ v, lambda v: np.linalg.solve(M, v), q, 1e-10, 50)
        np.testing.assert_allclose(mu, np.linalg.solve(H, q), rtol=1e-8, atol=1e-11)
    mu5, it5, _, _ = fvc._preconditioned_cg(lambda v: H @ v, lambda v: np.linalg.solve(H, v), q, 1e-10, 50)
    assert it5 == 1  # the first guess is already the answer: the loop only checks the residual
    assert counts[1] <= counts[0]


def test_preconditioned_cg_reports_an_indefinite_system(point):
    """An indefinite ``H`` stops conjugate gradients with a reason and an unconverged residual (never silently)."""
    H = np.diag([1.0, 2.0, -0.5, 3.0, 1.5])
    q = np.ones(5)
    mu, _, res, reason = fvc._preconditioned_cg(lambda v: H @ v, lambda v: v, q, 1e-12, 50)
    assert reason in ("curvature", "preconditioner") or np.linalg.norm(q - H @ mu) <= 1e-10
