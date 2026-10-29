"""The gauge Newton and the exact derivatives of the three-term free boundary.

The interface least squares of :mod:`vmex.core.freeboundary_three_terms` is
flat along the boundary's relabellings ``theta -> theta + f``, so a
Gauss-Newton fit converges only linearly there and its stationary point -- and
the derivative of that point with respect to the coils -- is not determined by
the fit.  With ``labelling="gauge"`` the angle is fixed by VMEC's spectral
condensation held exactly on the boundary (the conditions ``c(b) = 0`` of
:func:`~vmex.core.freeboundary_three_terms._gauge_rows`; Hirshman & Meier 1985),
and the fit's cost ``f(b) = |r(b)|^2 / 2`` is minimized on that manifold by a
trust-region Newton method (:func:`_newton_refine_gauged`): each step solves
``Z^T H_L Z dz = -Z^T g`` with ``Z`` the gauge's null space and ``H_L`` the
Lagrangian Hessian, every iterate is projected back onto the gauge, and the
reduced Hessian is definite at the minimum, so Newton converges quadratically.

The Hessian and the mixed design derivatives are exact
(:func:`exact_products`): with the equilibrium ``F(z, b, d) = 0`` (raw force,
state ``z``, boundary ``b``, design ``d``: plasma parameters and coil field),
its adjoint ``F_z^T mu = f_z`` and the Lagrangian ``L = f - mu^T F``, a column
``v`` of ``H`` and of ``F_d^T`` costs one state tangent, one forward-over-reverse
pass and one adjoint solve on a shared block factorization.  The design
gradient of a quantity ``Q`` at the converged point is then the implicit
function theorem on the gauge (:func:`_gauge_gradients`),

    dQ/dd = Q_d - F_d^T mu,   mu = S Z (Z^T S H_L S Z)^-1 Z^T S Q_b^T

(``S`` the coordinate scale).  The functions take a :class:`_Design` (the
model, the design's plasma-parameter and field maps and their jitted pieces);
a standalone solve is the design of no coordinates (:meth:`_Design.fixed`).
:func:`_accept` is the acceptance rule of a boundary fit and
:func:`_solve_point` the fit-then-accept loop over starts, shared by
:func:`~vmex.core.freeboundary_three_terms.solve_free_boundary_three_terms` and
:class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`.
With a bootstrap current the joint Hessian is round-off at the bootstrap
weight, and :func:`_newton_refine_projected` takes the Newton steps on the
boundary with the current eliminated (:func:`projected_hessian`).
"""

from __future__ import annotations

import time

import numpy as np

import jax
import jax.numpy as jnp

from .errors import TrialRejected, VmecError
from .freeboundary_three_terms import PROJECTION_CORRECTIONS

__all__ = ["dense_hessian", "exact_products", "least_squares_sensitivity", "projected_hessian"]

#: Columns per state-tangent solve of :func:`dense_hessian` and :func:`projected_hessian` (their ``chunk``):
#: each column in a batch holds a raw-force linearization, so a smaller batch needs less device memory.
EXACT_PRODUCTS_CHUNK = 16

#: The Newton polishes' floor: a predicted decrease at most this fraction of the cost is stationarity to the cost's
#: resolution (tight interior solves resolve it to ~1e-10).  Where the trust region has collapsed (steps rejected at every scale), ten times it:
#: a collapse whose predicted decrease is larger is a curved valley the model does not follow there, not a
#: stationary point.
NEWTON_FLOOR = 1e-9
#: The Newton polishes' initial trust radius (scaled coordinates).
NEWTON_RADIUS = 1e-1

#: The exact gauge: the largest |c| a projected boundary may keep, and the most Gauss-Newton steps of the projection.
GAUGE_TOL = 1e-12
GAUGE_PROJECT_STEPS = 20

#: The gauge Newton's SR1 update is skipped when ``|s^T (y - B s)| <= SR1_SKIP |s| |y - B s|`` (Nocedal & Wright 6.26).
SR1_SKIP = 1e-8
#: A quasi-Newton gauge iterate whose trust radius fell below this (or whose model is indefinite) refreshes the
#: Hessian (the Gauss-Newton ``J^T J`` at the iterate, then the exact dense one) instead of crawling.
QUASI_MIN_RADIUS = 1e-4
#: The final exact Newton step of the gauge polish (a quasi-Newton convergence): its conjugate-gradient tolerance
#: and iterations (each one exact-product call of one column).
FINAL_STEP_RTOL = 1e-6
FINAL_STEP_MAXITER = 30
#: Three accepted SR1 steps in a row, each longer than this fraction of the one before: linear convergence, so the
#: gauge Newton rebuilds its Gauss-Newton model (the converged point is the exact gradient's either way).
SR1_LINEAR_RATIO = 0.2
#: The relabelling onto the gauge (:func:`_gauge_relabel`): largest Levenberg-Marquardt step and largest relabelling
#: (rad) of the angle; a larger one is not the identity's continuous neighbour.
RELABEL_MAX_STEP = 0.5
RELABEL_MAX_SIZE = 1.5


class _Design:
    """A design ``x``: the maps of its plasma parameters and external field, and the jitted pieces of the fit's cost
    ``|r|^2 / 2`` that the gauge Newton and the exact derivatives share.

    ``params_at(x)`` is ``plasma_map(model.params0, x)`` (jitted) and ``field_at(x)`` the external field.
    :class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem` builds one from its
    ``plasma_from_parameters`` and ``field_from_parameters``; a standalone solve uses :meth:`fixed` (the deck's
    parameters and one field, ``x`` empty).  Nothing here recompiles when ``x`` changes.
    """

    def __init__(self, model, plasma_map, field_at):
        self.model, self.field_at = model, field_at
        self.params_at = jax.jit(lambda x: plasma_map(model.params0, x))
        #: ``ImplicitParams`` gradient of ``r . rows(state, params, field(x))`` through the equilibrium (``r``: the
        #: rows themselves, so this is the gradient of ``|r|^2 / 2`` at a fixed boundary).
        self.cost_pullback = jax.jit(lambda state, mask, params, r, x: model.pullback(
            model.rows, state, mask, params, r[None, :], field_at(x)))

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(lambda z: plasma_map(model.params0, z), x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self.coordinates = coordinates
        self.field_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(
            lambda z: model.rows(state, params, field_at(z)), x)[1](lam)[0])
        self.field_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(
            lambda z: model.rows(state, params, field_at(z)), (x,), (dx,))[1])
        self._exact_products_core = None

    @classmethod
    def fixed(cls, model, field):
        """The design of a standalone solve: the model's own plasma parameters and ``field``; ``x`` is empty."""
        return cls(model, lambda params, x: params, lambda x: field)


def _emit(opt):
    """The Newton polishes' progress lines go to ``opt["emit"]`` (the solver's ``emit``), else are printed."""
    return opt.get("emit") or (lambda message: print(message, flush=True))


def _trust_region_step(H, g, radius, lo, hi):
    """The step of ``min g.d + d.H.d / 2`` with ``|d| <= radius``: ``(H + lam I) d = -g`` with the shift ``lam``
    raised from ``hi`` until the step fits, then bisected on ``[lo, hi]`` (40 halvings) to the boundary."""
    import scipy.linalg

    eye = np.eye(g.size)
    while np.linalg.norm(scipy.linalg.solve(H + hi * eye, -g, assume_a="sym")) > radius:
        hi *= 10
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = scipy.linalg.solve(H + mid * eye, -g, assume_a="sym")
        lo, hi = (mid, hi) if np.linalg.norm(d) > radius or not np.all(np.isfinite(d)) else (lo, mid)
    return scipy.linalg.solve(H + hi * eye, -g, assume_a="sym")


def _cost_point(design, x, params, field, b):
    """``(cost, g, record)`` at boundary coordinates ``b``: the cost ``|r|^2 / 2``, its gradient ``g = J_b^T r`` (one
    tight interior solve and one reverse pass) and the record ``(state, mask, x, params)``; ``None`` if ``b`` has no
    certified interior root."""
    model = design.model
    got = model.evaluate(b, params, field, tight=True)
    if got is None:
        return None
    rows, aux = got
    state, mask, params_b, _ = aux
    r = jnp.asarray(rows)
    params_bar = design.cost_pullback(state, mask, params_b, r, jnp.asarray(x))
    b_bar, _ = design.coordinates(params_b, jnp.asarray(b), jnp.asarray(x), params_bar)
    return (0.5 * float(np.asarray(rows) @ np.asarray(rows)), np.asarray(b_bar)[0],
            dict(state=state, mask=mask, x=np.asarray(b, dtype=float), params=params_b))


def _gn_hessian(model, rec, field):
    """The Gauss-Newton Hessian ``J^T J`` of the fit's cost at the record ``rec`` (one linearization: the rows'
    Jacobian along the model coordinates), in the model coordinates."""
    J = np.asarray(model.linearize(np.asarray(rec["x"], dtype=float),
                                   (rec["state"], rec["mask"], rec["params"], True), field)[0])
    J = J[:, :model.x0.size]
    return J.T @ J


def _projected_gn_hessian(model, rec, field, bi, c):
    """:func:`projected_hessian`'s blocks from the Gauss-Newton ``J^T J`` at ``rec`` (one linearization): ``H_red``
    the boundary block with the current eliminated (``J_b`` projected off the current columns by a QR, so the
    weighted current block's round-off does not enter), ``H_cc`` and ``K = H_cc^-1 H_cb``.  A model for steps
    far from the minimum only: positive semidefinite, without the residual curvature."""
    import scipy.linalg

    J = np.asarray(model.linearize(np.asarray(rec["x"], dtype=float),
                                   (rec["state"], rec["mask"], rec["params"], True), field)[0])[:, :model.x0.size]
    Jb, Jc = J[:, bi], J[:, c]
    Q, _ = scipy.linalg.qr(Jc, mode="economic")
    Jp = Jb - Q @ (Q.T @ Jb)
    Hcc = Jc.T @ Jc
    return dict(Hred=0.5 * (Jp.T @ Jp + (Jp.T @ Jp).T), Hcc=0.5 * (Hcc + Hcc.T), K=np.linalg.solve(Hcc, Jc.T @ Jb),
                b=bi, c=c, leak=0.0)


def _gauge_relabel(model, b, stats):
    """``b`` relabelled onto the exact gauge: the same surface with the angle ``theta + f`` that satisfies the gauge
    (:attr:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.relabel`; Gauss-Newton on ``f``, geometry only), then held to GAUGE_TOL by
    :func:`_gauge_project`.  The shape changes only by the refit's truncation (a minimum-norm move in the
    coordinates instead changes the shape, and the projected point can have no equilibrium)."""
    b = np.asarray(b, dtype=float)
    fc = np.zeros(model.relabel_modes)
    # Levenberg-Marquardt from f = 0, every step lowering |c| and at most RELABEL_MAX_STEP: the small relabelling
    # continuously connected to the identity (undamped Gauss-Newton jumped to |f| ~ 200 rad from a rough start)
    c = np.asarray(model.relabel_gauge(jnp.asarray(b), jnp.asarray(fc)))
    mu = None
    for _ in range(4 * GAUGE_PROJECT_STEPS):
        if float(np.abs(c).max()) <= GAUGE_TOL:
            break
        Jf = np.asarray(model.relabel_gauge_jacobian(jnp.asarray(b), jnp.asarray(fc)))
        JTJ, JTc = Jf.T @ Jf, Jf.T @ c
        mu = 1e-8 * float(np.max(np.diag(JTJ))) if mu is None else mu
        for _ in range(30):
            step = -np.linalg.solve(JTJ + mu * np.eye(fc.size), JTc)
            if np.abs(step).max() > RELABEL_MAX_STEP:
                mu *= 10.0
                continue
            trial = np.asarray(model.relabel_gauge(jnp.asarray(b), jnp.asarray(fc + step)))
            if np.linalg.norm(trial) < np.linalg.norm(c):
                fc, c, mu = fc + step, trial, mu / 3.0
                break
            mu *= 10.0
        else:
            break
    stats["gauge_relabel_size"] = float(np.abs(fc).max())
    if float(np.abs(c).max()) > GAUGE_TOL or float(np.abs(fc).max()) > RELABEL_MAX_SIZE:
        raise TrialRejected(f"no small relabelling onto the gauge (|f| {np.abs(fc).max():.2e} rad, "
                            f"gauge residual {np.abs(c).max():.1e})")
    relabelled = np.asarray(model.relabel(jnp.asarray(b), jnp.asarray(fc)))
    return _gauge_project(model, relabelled)


def _gauge_project(model, b):
    """``b`` moved onto the exact gauge ``c(b) = 0`` by the smallest scaled move: Gauss-Newton steps on ``c``
    alone (minimum norm, geometry only: no equilibrium solve).  Returns the projected ``b``, or None if the gauge
    is not reached to GAUGE_TOL in GAUGE_PROJECT_STEPS."""
    scale = np.asarray(model.x_scale, dtype=float)
    b = np.asarray(b, dtype=float).copy()
    for _ in range(GAUGE_PROJECT_STEPS):
        c = np.asarray(model.gauge_rows(jnp.asarray(b)))
        if float(np.abs(c).max()) <= GAUGE_TOL:
            return b
        C = np.asarray(model.gauge_jacobian(jnp.asarray(b))) * scale
        b = b - np.linalg.lstsq(C, c, rcond=None)[0] * scale
    c = np.asarray(model.gauge_rows(jnp.asarray(b)))
    return b if float(np.abs(c).max()) <= GAUGE_TOL else None


def _gauge_reduced(model, b, g, H):
    """At a gauge-feasible ``b``: the null-space basis ``Z`` of the scaled gauge Jacobian, the multipliers ``lam``
    (``g = -C^T lam`` in the least-squares sense), the Lagrangian Hessian ``H_L = H + d2(lam . c)`` (model
    coordinates) and the reduced Hessian and gradient (scaled)."""
    import scipy.linalg

    scale = np.asarray(model.x_scale, dtype=float)
    C = np.asarray(model.gauge_jacobian(jnp.asarray(b)))
    lam = -np.linalg.lstsq(C.T, g, rcond=None)[0]
    H_L = np.asarray(H) + np.asarray(model.gauge_curvature(jnp.asarray(b), jnp.asarray(lam)))
    Z = scipy.linalg.null_space(C * scale)
    Hs = 0.5 * (H_L + H_L.T) * np.outer(scale, scale)
    return dict(Z=Z, lam=lam, H_L=H_L, Hr=Z.T @ Hs @ Z, gr=Z.T @ (g * scale))


def _newton_refine(design, x, params, field, extra, b, opt, *, stats, max_residual=None, values=None,
                   anchor_H=None, keep_hessian=None):
    """Newton steps on the fit's cost ``|r|^2 / 2`` from boundary coordinates ``b``, then its certificate.

    The Levenberg-Marquardt polish (Gauss-Newton) converges linearly along the weak (relabelling) directions, so a
    fit certified by its gain can still be far, in the scaled step, from its stationary point there; the response
    of the stationary point is then not determined by the fit.  :func:`_newton_refine_gauged` (or, with a
    bootstrap current, :func:`_newton_refine_projected`) converges it; the final point is certified by
    :meth:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.certify` (re-solved, linearized, no further polish), so the record, its
    Jacobian and its values come from the same code as every other trial.  ``sol["newton"]`` records the steps.
    ``opt``: ``step_tol``, ``max_steps``, ``quasi``, ``verbose``; ``anchor_H`` a previous point's cost Hessian (the
    quasi-Newton start); ``keep_hessian(record)`` receives the converged point's Hessian (the exact gradient's).
    """
    model = design.model
    options = dict(step_tol=1e-8, max_steps=10)
    options.update(opt or {})
    t0 = time.perf_counter()
    b = np.asarray(b, dtype=float).copy()
    common = dict(stats=stats, max_residual=max_residual, values=values, keep_hessian=keep_hessian)
    if model._projected is not None:
        return _newton_refine_projected(design, x, params, field, extra, b, options, t0, **common)
    if model.gauge is None:
        raise ValueError("Newton polish requires gauge='condensation'")
    return _newton_refine_gauged(design, x, params, field, extra, b, options, t0, anchor_H=anchor_H, **common)


def _newton_refine_gauged(design, x, params, field, extra, b, opt, t0, *, stats, max_residual=None, values=None,
                          anchor_H=None, keep_hessian=None):
    """Newton on the fit's cost on the exact gauge manifold ``c(b) = 0`` (:attr:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.gauge`).

    Every point it evaluates is feasible: ``b`` is projected onto the gauge first (:func:`_gauge_relabel`) and
    after every step.  Each step is the trust-region Newton step of the reduced problem: ``Z^T H_L Z dz = -Z^T g``
    (``Z`` the gauge's null space, scaled; ``H_L`` the dense exact Hessian plus the gauge's curvature times its
    multipliers), judged by the actual decrease of the cost at the projected point.  The labelling is fixed by
    the gauge, so the cost has no flat (relabelling) directions left and Newton converges quadratically.
    Converged when the reduced step is at most ``step_tol`` or its predicted decrease at most NEWTON_FLOOR of the
    cost (or ten times that with a collapsed radius); then certified as every other record, and the reduced
    Hessian is kept (``keep_hessian``) for the record's exact gradient."""
    import scipy.linalg

    model = design.model
    scale = np.asarray(model.x_scale, dtype=float)

    def point(bb):
        return _cost_point(design, x, params, field, bb)

    projected = _gauge_relabel(model, b, stats)
    if projected is None:
        raise TrialRejected("the boundary does not reach the gauge")
    if opt.get("verbose"):
        _emit(opt)(f"[newton gauge] relabelled onto the gauge: |f| max {stats['gauge_relabel_size']:.2e} rad, scaled "
              f"move {float(np.linalg.norm((projected - b) / scale)):.2e}")
    b = projected
    here = point(b)
    if here is None:
        raise TrialRejected("the gauge-projected boundary has no tight interior root")
    cost, g, rec = here
    radius, floor = NEWTON_RADIUS, NEWTON_FLOOR
    history, evaluations, converged, seconds_hessian, red = [], 1, False, 0.0, None
    # Quasi-Newton from the accepted point's exact Hessian (the gauge's minimum is not degenerate): SR1 on
    # the exact gradients' secants, a dense exact Hessian at the start of a run and when a step stalls or the trust
    # radius collapses (QUASI_MIN_RADIUS).
    H_prev = anchor_H
    quasi = bool(opt.get("quasi", True)) and isinstance(H_prev, np.ndarray) and H_prev.shape == (b.size, b.size)
    H_cost, exact, secant, refresh = (np.array(H_prev) if quasi else None), not quasi, None, False
    H_before, definite = None, False
    # model level: 0 quasi-Newton (SR1), 1 Gauss-Newton J^T J at the iterate (one linearization), 2 the exact
    # dense Hessian; a stale or indefinite model escalates one level (the converged point does not depend on it:
    # stationarity is the exact gradient's).  A cold start (no accepted Hessian) begins at the Gauss-Newton model
    # too: one dense exact Hessian is 255 exact products (~140-280 s at 8 x 8), and the escalation below reaches
    # it only where Gauss-Newton stalls.
    level = 0
    for _ in range(int(opt["max_steps"])):
        th = time.perf_counter()
        if H_cost is None or refresh:
            if level < 1:
                H_cost, level = _gn_hessian(model, rec, field), 1
            else:
                (H_cost, _), level = dense_hessian(design, x, rec), 2
            exact, secant, refresh = level == 2, None, False
        elif secant is not None:  # SR1 in the scaled coordinates (keeps indefinite curvature, unlike BFGS)
            s_, y_ = secant
            Hs_old = H_cost * np.outer(scale, scale)
            v = y_ - Hs_old @ s_
            denominator = float(s_ @ v)
            if abs(denominator) > SR1_SKIP * float(np.linalg.norm(s_) * np.linalg.norm(v)):
                H_before = H_cost if definite else None
                Hs_new = Hs_old + np.outer(v, v) / denominator
                H_cost = 0.5 * (Hs_new + Hs_new.T) / np.outer(scale, scale)
            exact, secant, level = False, None, 0
        red = _gauge_reduced(model, b, g, H_cost)
        Hr, gr = red["Hr"], red["gr"]
        eig = np.linalg.eigvalsh(Hr)
        if eig[0] <= 0 and not exact and H_before is not None:  # an update that broke a definite model near the
            # (non-degenerate) minimum is secant noise: the model before it
            H_cost, H_before = H_before, None
            red = _gauge_reduced(model, b, g, H_cost)
            Hr, gr = red["Hr"], red["gr"]
            eig = np.linalg.eigvalsh(Hr)
        H_before = None
        seconds_hessian += time.perf_counter() - th
        definite = bool(eig[0] > 0)
        newton = -scipy.linalg.solve(Hr, gr, assume_a="sym") if definite else None
        step = float(np.linalg.norm(newton)) if newton is not None else np.inf
        decrease = 0.5 * float(gr @ -newton) if newton is not None else np.inf
        history.append(dict(cost=cost, step=step, radius=radius, definite=definite, reduced_smallest=float(eig[0]),
                            reduced_condition=float(eig[-1] / eig[0]) if definite else float("inf"),
                            evaluations=evaluations))
        if opt.get("verbose"):
            _emit(opt)(f"[newton gauge] iterate {len(history) - 1}: cost {cost:.12e}, step {step:.2e}, radius "
                  f"{radius:.1e}, reduced smallest {eig[0]:.2e} condition {history[-1]['reduced_condition']:.1e}, "
                  f"{('SR1', 'Gauss-Newton', 'exact')[level]} Hessian, {seconds_hessian:.0f} s")
        if definite and (step <= opt["step_tol"] or decrease <= floor * max(cost, 1e-300)
                         or (exact and radius < 1e-6 and decrease <= 10 * floor * max(cost, 1e-300))):
            converged = True
            if opt["step_tol"] < step <= radius:
                # Below the floor the cost cannot judge the step, but the reduced gradient left here moves with the
                # design, so differences of the records miss the root's derivative (small vacuum problem: 1.8 % on
                # the loss, 10 % on mean iota, whose weak shape mode a quasi-Newton step leaves at 9e-8); the
                # final step is exact Newton (conjugate gradients on exact products, this model preconditioning),
                # which removes it quadratically.  Its Hessian is then the previous point's: only a
                # preconditioner for the record's exact gradient.
                if not exact:
                    y, count = _reduced_cg(design, x, rec, red, -gr[:, None], FINAL_STEP_RTOL, FINAL_STEP_MAXITER)
                    stats["final_step_cg"] = count
                    newton = newton if y is None else y[:, 0]
                target = _gauge_project(model, b + (red["Z"] @ newton) * scale)
                trial = point(target) if target is not None else None
                evaluations += 1
                if trial is not None and trial[0] <= cost + floor * max(cost, 1e-300):
                    b, (cost, g, rec) = target, trial
                    red, exact = _gauge_reduced(model, b, g, H_cost), False
                    history[-1]["final_step"] = step
            break
        if level < 2 and (radius < QUASI_MIN_RADIUS or not definite):  # stale, or far from the (non-degenerate)
            # minimum: a fresher model here (Gauss-Newton, then exact)
            refresh, radius = True, max(radius, 1e-3)
            history[-1]["refreshed"] = True
            continue
        accepted = False
        for _ in range(8):
            if definite and step <= radius:
                dz = newton
            else:
                dz = _trust_region_step(Hr, gr, radius, max(0.0, -float(eig[0])), max(1.0, float(np.abs(eig).max())))
            predicted = -(gr @ dz + 0.5 * dz @ Hr @ dz)
            target = _gauge_project(model, b + (red["Z"] @ dz) * scale)
            trial = point(target) if target is not None else None
            evaluations += 1
            rho = (cost - trial[0]) / predicted if trial is not None and predicted > 0 else -1.0
            if rho > 0.1:
                secant = ((target - b) / scale, (trial[1] - g) * scale)
                b, (cost, g, rec) = target, trial
                if rho > 0.75 and np.linalg.norm(dz) > 0.9 * radius:
                    radius *= 2
                accepted = True
                history[-1].update(taken=float(np.linalg.norm(dz)), rho=float(rho))
                taken = [h.get("taken") for h in history[-4:]]
                if (level == 0 and len(taken) == 4 and None not in taken
                        and all(t1 > SR1_LINEAR_RATIO * t0 for t0, t1 in zip(taken, taken[1:]))):
                    # three SR1 steps in a row shrank by less than 1/SR1_LINEAR_RATIO: the quasi-Newton
                    # model converges linearly; the next iterate rebuilds the Gauss-Newton model (one linearization)
                    refresh = True
                    history[-1]["refreshed"] = True
                break
            radius = 0.25 * min(radius, float(np.linalg.norm(dz)))
        if not accepted:
            history[-1]["stalled"] = True
            if level < 2:  # a stale model: a fresher one here (Gauss-Newton, then exact) before giving up
                refresh, radius = True, max(radius, 1e-3)
                continue
            break
    if not converged:
        raise TrialRejected(f"the gauge Newton polish did not converge ({len(history)} steps)")
    # The record's gradient reuses an exact reduced Hessian; a quasi-Newton one only preconditions its block CG.
    if keep_hessian is not None:
        keep_hessian(dict(b=np.asarray(rec["x"]).copy(), x=np.asarray(x, dtype=float).copy(), H=red["H_L"],
                          gauge=red, exact=exact, H_cost=H_cost))
    got = model.evaluate(b, params, field, tight=True)
    if got is None:
        raise TrialRejected("the gauge-polished boundary has no certified interior root")
    rows, aux = got
    fit = dict(x=b, aux=aux, rows=np.asarray(rows), nfev=evaluations, converged=True, reason="newton-gauge",
               accepted=len(history))
    out = model.certify(params, field, fit, gate=np.inf, xtol=None, polish=False, max_residual=max_residual,
                        extra=extra, values=values, stats=stats)
    out["newton"] = dict(steps=history, converged=converged, evaluations=evaluations, merit=cost,
                         subspace=int(red["Z"].shape[1]), products="exact", gauge=model.gauge,
                         seconds_hessian=seconds_hessian, seconds=time.perf_counter() - t0)
    stats["newton_evaluations"] = stats.get("newton_evaluations", 0) + evaluations
    stats["seconds_newton"] = stats.get("seconds_newton", 0.0) + out["newton"]["seconds"]
    return out


def _newton_refine_projected(design, x, params, field, extra, b, opt, t0, *, stats, max_residual=None, values=None,
                             keep_hessian=None):
    """:func:`_newton_refine` with a bootstrap current: Newton on the boundary with the current projected.

    The joint Hessian is round-off at the bootstrap weight (:func:`projected_hessian`), so each step is taken on
    the reduced cost ``min_c f(b, c)``: its gradient ``g_b - K^T g_c`` and Hessian ``H_red`` (exact at every
    iterate), in the same trust region on the cost. The current follows with ``-K d_b - H_cc^-1 g_c`` and is then
    re-solved at the trial boundary by Newton steps on its own block (``H_cc``, at most
    :data:`PROJECTION_CORRECTIONS`), so the trust region judges the reduced cost. Converged as the exact polish
    (Newton step or the cost's floor); the last projected Hessian is kept (``keep_hessian``) for the record's exact
    gradient.  On the exact gauge (``model.gauge``) the boundary steps lie in the gauge's null space ``Z_b`` (the
    gauge rows do not involve the current) with the gauge's curvature in the reduced Hessian, the start is
    relabelled onto the gauge and every trial boundary is projected back onto it before the current is re-solved."""
    import scipy.linalg

    model = design.model
    scale = np.asarray(model.x_scale, dtype=float)
    c = np.asarray(model._projected)
    bi = np.setdiff1d(np.arange(model.x0.size), c)
    sb = scale[bi]
    gauged = model.gauge is not None
    if gauged:
        b = _gauge_relabel(model, b, stats)

    def gauge_basis(bb, g_red):
        """``(Z_b, lam, curvature_bb)`` at a gauge-feasible ``bb`` for the reduced gradient ``g_red``; the
        identity without a gauge."""
        if not gauged:
            return np.eye(bi.size), None, 0.0
        C = np.asarray(model.gauge_jacobian(jnp.asarray(bb)))[:, bi]
        lam_g = -np.linalg.lstsq(C.T, g_red, rcond=None)[0]
        curv = np.asarray(model.gauge_curvature(jnp.asarray(bb), jnp.asarray(lam_g)))[np.ix_(bi, bi)]
        return scipy.linalg.null_space(C * sb), lam_g, curv

    def point(bb):  # (cost, g, sol-like record) at bb, or None
        return _cost_point(design, x, params, field, bb)

    def project(bb, here, Hcc, target=np.inf):  # the current re-solved at this boundary: Newton on its block
        # A correction is skipped when its predicted gain is below the cost's resolution (NEWTON_FLOOR, the floor
        # that also declares convergence, whose test includes the current's gain) or when even twice that gain
        # leaves the trial above ``target``, the cost its acceptance needs (each correction is a tight solve).
        used = 0
        for _ in range(PROJECTION_CORRECTIONS):
            dc = -np.linalg.solve(Hcc, here[1][c])
            gain = -0.5 * float(here[1][c] @ dc)
            if gain <= NEWTON_FLOOR * max(here[0], 1e-300) or here[0] - 2.0 * gain >= target:
                break
            trial = bb.copy()
            trial[c] += dc
            got = point(trial)
            used += 1
            if got is None or got[0] >= here[0]:
                break
            bb, here = trial, got
        return bb, here, used

    here = point(b)
    if here is None:
        raise TrialRejected("the certified fit has no tight interior root for its Newton polish")
    cost, g, rec = here
    radius = NEWTON_RADIUS
    history, evaluations, converged, seconds_hessian, P = [], 1, False, 0.0, None
    # Models, as in the gauge Newton without a current: level 0 SR1 on H_red from the exact reduced gradients'
    # secants (K and H_cc kept), 1 the Gauss-Newton blocks (:func:`_projected_gn_hessian`, one linearization
    # ~11 s), 2 the exact projected Hessian (~60-110 s at 8 x 8).  A start, an indefinite model or a stalled step
    # escalates one level ("model"); convergence is declared only on the exact Hessian ("exact" refresh first).
    # The cheaper models carry the steps far from the minimum, where an exact Hessian is refreshed only to be
    # replaced at the next indefinite or stalled model.
    exact, refresh, secant, asymmetry, level = False, "model", None, 0.0, 0
    for _ in range(int(opt["max_steps"])):
        th = time.perf_counter()
        if refresh:
            if refresh == "exact" or level >= 1:
                P, asymmetry = projected_hessian(design, x, rec)
                level = 2
            else:
                P, asymmetry, level = _projected_gn_hessian(model, rec, field, bi, c), 0.0, 1
            exact, refresh = level == 2, False
        elif secant is not None:
            s_, y_ = secant[0] / sb, secant[1] * sb
            Hsc = P["Hred"] * np.outer(sb, sb)
            v = y_ - Hsc @ s_
            denominator = float(s_ @ v)
            if abs(denominator) > SR1_SKIP * float(np.linalg.norm(s_) * np.linalg.norm(v)):
                Hsc = Hsc + np.outer(v, v) / denominator
                P = dict(P, Hred=0.5 * (Hsc + Hsc.T) / np.outer(sb, sb))
            exact, level = False, 0
        secant = None
        seconds_hessian += time.perf_counter() - th
        g_red = g[bi] - P["K"].T @ g[c]
        Zb, lam_g, curv = gauge_basis(b, g_red)
        Hs = Zb.T @ (0.5 * ((P["Hred"] + curv) + (P["Hred"] + curv).T) * np.outer(sb, sb)) @ Zb
        gs = Zb.T @ (g_red * sb)
        current_gain = 0.5 * float(g[c] @ np.linalg.solve(P["Hcc"], g[c]))
        try:
            newton = -scipy.linalg.solve(Hs, gs, assume_a="sym")
            eigenvalues = np.linalg.eigvalsh(Hs)
            definite = bool(np.all(eigenvalues > 0))
        except np.linalg.LinAlgError:
            newton, definite, eigenvalues = None, False, None
        step_newton = float(np.linalg.norm(newton)) if newton is not None else np.inf
        history.append(dict(cost=cost, step=step_newton, radius=radius, definite=definite,
                            stationarity=float(abs(gs @ newton)) if newton is not None else np.inf,
                            evaluations=evaluations, asymmetry=asymmetry, leak=P["leak"],
                            current_gain=current_gain))
        if eigenvalues is not None:
            history[-1].update(smallest_eigenvalues=eigenvalues[:5].tolist(),
                               largest_eigenvalue=float(eigenvalues[-1]))
        if opt.get("verbose"):
            eig = "" if eigenvalues is None else (" smallest eigenvalues " + " ".join(
                f"{e:.2e}" for e in eigenvalues[:5]) + f" largest {eigenvalues[-1]:.2e}")
            _emit(opt)(f"[newton projected] iterate {len(history) - 1}: cost {cost:.12e}, Newton step "
                  f"{step_newton:.2e}, definite {definite}, radius {radius:.1e}, current gain {current_gain:.1e}, "
                  f"leak {P['leak']:.1e}, {('SR1', 'Gauss-Newton', 'exact')[level]} model, Hessian {seconds_hessian:.0f} s "
                  f"total{eig}")
        history[-1].update(exact=exact, level=level)
        if not definite and not exact:
            refresh = "model"
            continue
        if definite and step_newton <= opt["step_tol"]:
            if not exact:
                refresh = "exact"
                continue
            converged = True
            break
        decrease = 0.5 * float(gs @ -newton) + current_gain
        floor = NEWTON_FLOOR * max(cost, 1e-300)
        if definite and (decrease <= floor or (radius < 1e-6 and decrease <= 10 * floor)):
            if not exact:
                refresh = "exact"
                continue
            converged = True
            history[-1]["floor"] = True
            break
        accepted, noise = False, 0.0
        for _ in range(8):
            if definite and step_newton <= radius:
                ds = newton
            else:
                ds = _trust_region_step(Hs, gs, radius, 0.0, max(1.0, float(np.abs(np.diag(Hs)).max())))
            predicted = -(gs @ ds + 0.5 * ds @ Hs @ ds) + current_gain
            d = np.zeros_like(b)
            d[bi] = (Zb @ ds) * sb
            d[c] = -P["K"] @ d[bi] - np.linalg.solve(P["Hcc"], g[c])
            b_try = b + d
            if gauged:  # back onto the gauge (a geometry-only move of the boundary; the current re-solves below)
                b_try = _gauge_project(model, b_try)
            trial = point(b_try) if b_try is not None else None
            evaluations += 1
            if b_try is None:
                b_try = b + d
            if trial is not None:
                b_try, trial, used = project(b_try, trial, P["Hcc"], target=cost - 0.1 * predicted)
                evaluations += used
            rho = (cost - trial[0]) / predicted if trial is not None and predicted > 0 else -1.0
            if trial is not None and np.isfinite(trial[0]):  # the trials' miss of their model: the cost's resolution
                noise = max(noise, abs((cost - trial[0]) - predicted))
            if rho > 0.1:
                g_new = trial[1]
                secant = ((b_try - b)[bi], (g_new[bi] - P["K"].T @ g_new[c]) - g_red)
                b, (cost, g, rec) = b_try, trial
                if rho > 0.75 and np.linalg.norm(ds) > 0.9 * radius:
                    radius *= 2
                accepted = True
                history[-1].update(taken=float(np.linalg.norm(ds)), rho=float(rho))
                break
            radius = 0.25 * min(radius, float(np.linalg.norm(ds)))
        if not accepted:
            if not exact:  # a model that stalls is refreshed one level up
                refresh = "model"
                continue
            if definite and decrease <= noise:  # the Newton gain is below what the cost resolves: stationary to the
                # evaluations' precision (rows known to ~1e-9 can resolve the cost only to ~1e-6 of itself, above
                # NEWTON_FLOOR, and every trial within it is then rejected)
                converged = True
                history[-1].update(floor="noise", noise=float(noise))
                break
            history[-1]["stalled"] = True
            break
    if keep_hessian is not None:
        keep_hessian(dict(b=np.asarray(rec["x"]).copy(), x=np.asarray(x, dtype=float).copy(), H=P, projected=True,
                          gauge_lam=lam_g, gauge_Z=Zb if gauged else None) if converged else None)
    if not converged:
        raise TrialRejected(f"the Newton polish did not converge ({len(history)} steps)")
    got = model.evaluate(b, params, field, tight=True)
    if got is None:
        raise TrialRejected("the Newton-refined boundary has no certified interior root")
    rows, aux = got
    fit = dict(x=b, aux=aux, rows=np.asarray(rows), nfev=evaluations, converged=True, reason="newton",
               accepted=len(history))
    # Newton's own convergence is the stationarity test (exact Hessian: a scaled step <= step_tol, or a predicted
    # decrease below the cost's floor). The certificate's Gauss-Newton gain divides by J^T J, ~1e9 weaker than
    # the Hessian along the flat directions, and turns a converged point's residual gradient into a false
    # "not stationary"; its other checks (re-solve, finiteness, rank, interface residuals) still apply.
    out = model.certify(params, field, fit, gate=np.inf, xtol=None, polish=False, max_residual=max_residual,
                        extra=extra, values=values, stats=stats)
    out["newton"] = dict(steps=history, converged=converged, evaluations=evaluations, merit=cost,
                         subspace=int(Zb.shape[1]) if P is not None else bi.size, products="exact",
                         seconds_hessian=seconds_hessian, seconds=time.perf_counter() - t0, warm=False, projected=True,
                         gauge=model.gauge)
    stats["newton_evaluations"] = stats.get("newton_evaluations", 0) + evaluations
    stats["seconds_newton"] = stats.get("seconds_newton", 0.0) + out["newton"]["seconds"]
    return out


def _cost_gradients(design, b, params, field, x):
    """At boundary coordinates ``b`` (fixed design ``x``): ``g = J_b^T r`` (the fit's stationarity), ``c = J_d^T r``
    (the design gradient of ``|r|^2 / 2`` at that fixed boundary: coil field and plasma parameters) and ``r``.

    One tight interior solve and one reverse pass through the equilibrium (:meth:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.pullback`
    with the rows as cotangent), instead of the Jacobian's every column.
    """
    got = design.model.evaluate(np.asarray(b, dtype=float), params, field, tight=True)
    if got is None:
        raise TrialRejected("a stationarity difference point has no certified interior root")
    rows, aux = got
    state, mask, params_b, _ = aux
    r = jnp.asarray(rows)
    params_bar = design.cost_pullback(state, mask, params_b, r, jnp.asarray(x))
    b_bar, x_bar = design.coordinates(params_b, jnp.asarray(b), jnp.asarray(x), params_bar)
    g = np.asarray(b_bar)[0]
    c = np.asarray(x_bar)[0] + np.asarray(design.field_vjp(state, params_b, jnp.asarray(x), r))
    return g, c, np.asarray(rows)


def dense_hessian(design, x, sol, chunk=None):
    """The fit's exact Hessian ``H = J_b^T J_b + sum_i r_i d2r_i/db2`` at the record ``sol``: every unit column (in
    the model coordinates) from one call of :func:`exact_products`, so its block factorization (shared with the
    state tangents) and multiplier are paid once per Hessian, with the state tangents solved ``chunk`` columns at a
    time (default :data:`EXACT_PRODUCTS_CHUNK`).  Returns ``(H, asymmetry)``: ``H`` symmetrized and the relative norm
    of its antisymmetric part (zero up to round-off for exact products)."""
    chunk = EXACT_PRODUCTS_CHUNK if chunk is None else chunk
    H = exact_products(design, x, sol, np.eye(design.model.x0.size), check=False, chunk=chunk)[0]
    asymmetry = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
    return 0.5 * (H + H.T), asymmetry


def projected_hessian(design, x, sol, chunk=None):
    """With a bootstrap current: the fit's exact Hessian with the stiff current block eliminated, as a dict.

    The current rows carry ``bootstrap_weight`` (squared in ``H``), so in one matrix with the boundary's
    relabelling valley the condition number reaches ~1e12 and the dense solve and eigenvalues are round-off
    (indefinite at the minimum). Instead: the current columns ``H[:, c]`` (one :func:`exact_products` call),
    ``K = H_cc^-1 H_cb``, then the products along the current-following directions ``[I; -K]`` give the reduced
    Hessian ``H_red = H_bb - H_bc K`` directly: along them the weighted current rows' first-order change cancels
    inside each product, not in a difference of large entries. ``H = T^T diag(H_red, H_cc) T`` with ``T = [[I,
    0], [K, I]]`` (:func:`_projected_solve`). ``leak`` is the relative size of the current rows of those
    products, zero up to round-off."""
    chunk = EXACT_PRODUCTS_CHUNK if chunk is None else chunk
    model = design.model
    n, c = model.x0.size, np.asarray(model._projected)
    bi = np.setdiff1d(np.arange(n), c)
    E = np.eye(n)
    Hc = np.asarray(exact_products(design, x, sol, E[:, c], check=False, chunk=chunk)[0])
    Hcc = 0.5 * (Hc[c] + Hc[c].T)
    K = np.linalg.solve(Hcc, Hc[bi].T)
    V = E[:, bi].copy()
    V[c] = -K
    HP = np.asarray(exact_products(design, x, sol, V, check=False, chunk=chunk)[0])
    Hred = HP[bi]
    asymmetry = float(np.linalg.norm(Hred - Hred.T) / max(np.linalg.norm(Hred), 1e-300))
    leak = float(np.linalg.norm(HP[c]) / max(np.linalg.norm(Hc[c]), 1e-300))
    return dict(Hred=0.5 * (Hred + Hred.T), Hcc=Hcc, K=K, b=bi, c=c, leak=leak), asymmetry


def _reduced_cg(design, x, rec, red, B, rtol, maxiter):
    """``Y = (Z^T S H_L S Z)^-1 B`` by block conjugate gradients on exact reduced products at the record ``rec``
    (:func:`exact_products` plus the gauge's curvature), preconditioned by ``red``'s reduced Hessian (a quasi-Newton
    or Gauss-Newton one).  Returns ``(Y, count)``, ``Y`` None when a column meets non-positive curvature or does not
    reach ``rtol`` in ``maxiter`` iterations."""
    Z, lam = red["Z"], red["lam"]
    scale = np.asarray(design.model.x_scale, dtype=float)
    Kc = np.asarray(design.model.gauge_curvature(jnp.asarray(rec["x"], dtype=float), jnp.asarray(lam)))
    evals, evecs = np.linalg.eigh(0.5 * (red["Hr"] + red["Hr"].T))
    evals = np.maximum(np.abs(evals), 1e-14 * float(np.abs(evals).max()))
    count = dict(calls=0, columns=0, seconds=0.0, per_call=[])

    def products(Y):  # Hr Y with the exact Hessian
        tp = time.perf_counter()
        count["calls"] += 1
        count["columns"] += int(Y.shape[1])
        U = scale[:, None] * (Z @ Y)
        HU = np.asarray(exact_products(design, x, rec, U, check=False, chunk=min(U.shape[1], EXACT_PRODUCTS_CHUNK))[0])
        count["seconds"] += time.perf_counter() - tp
        count["per_call"].append([int(Y.shape[1]), round(time.perf_counter() - tp, 2)])
        return Z.T @ (scale[:, None] * (HU + Kc @ U))

    def precondition(R):
        return evecs @ ((evecs.T @ R) / evals[:, None])

    Y, R = np.zeros_like(B), B.copy()
    P = precondition(R)
    D, rz = P.copy(), np.sum(R * P, axis=0)
    norms = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    active = np.linalg.norm(R, axis=0) > rtol * norms
    for _ in range(maxiter):
        if not active.any():
            break
        HD = products(D)  # every column: one shape, one compilation (each active-column count recompiled, ~145 s)
        curvature = np.sum(D * HD, axis=0)
        if np.any(curvature[active] <= 0):
            return None, count
        alpha = np.where(active, rz / np.where(active, curvature, 1.0), 0.0)
        Y, R = Y + alpha * D, R - alpha * HD
        P = precondition(R)
        rz_new = np.sum(R * P, axis=0)
        D = np.where(active, P + (rz_new / np.where(active, rz, 1.0)) * D, D)
        rz = np.where(active, rz_new, rz)
        active &= np.linalg.norm(R, axis=0) > rtol * norms
    count["converged"] = not bool(active.any())
    return (None if active.any() else Y), count


def _gauge_block_cg(design, x, sol, rows, red, G_b, G_x, direct, scale, t0, stats, rtol=1e-8, maxiter=40):
    """The gauge gradient's multipliers by :func:`_reduced_cg` on ``Z^T S H_L S Z y = Z^T S Q``, preconditioned by
    ``red``'s reduced Hessian; None when a column does not converge or the true residual exceeds 1e-6."""
    Z, lam = red["Z"], red["lam"]
    b0 = np.asarray(sol["x"], dtype=float)
    Kc = np.asarray(design.model.gauge_curvature(jnp.asarray(b0), jnp.asarray(lam)))
    Q = G_b[rows].T
    B = Z.T @ (scale[:, None] * Q)
    Y, count = _reduced_cg(design, x, sol, red, B, rtol, maxiter)
    stats["gauge_cg_last"] = dict(count, rows=int(B.shape[1]), seconds_total=time.perf_counter() - t0)
    if Y is None:
        return None
    norms = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    mu = scale[:, None] * (Z @ Y)
    H_mu, Fd_mu, _ = exact_products(design, x, sol, mu, check=False)
    residual = Z.T @ (scale[:, None] * (Q - np.asarray(H_mu) - Kc @ mu))
    true = (np.linalg.norm(residual, axis=0) / norms).tolist()
    if max(true) > 1e-6:
        return None
    evals = np.abs(np.linalg.eigvalsh(0.5 * (red["Hr"] + red["Hr"].T)))
    return dict(gradients=direct[rows] + G_x[rows] - np.asarray(Fd_mu).T, rows=rows, true_residual=true,
                asymmetry=0.0, condition=float(evals.max() / max(evals.min(), 1e-300)),
                seconds_hessian=time.perf_counter() - t0)


def _gauge_gradients(design, x, sol, G_b, G_x, direct, rows, kept, stats):
    """The exact design gradient on the exact gauge: ``dQ/dd = Q_d - F_d^T mu`` with ``mu = S Z (Z^T S H_L S Z)^-1
    Z^T S Q_b^T`` (``S`` the coordinate scale, ``Z`` the gauge's null space, ``H_L`` the Lagrangian Hessian; the
    gauge does not depend on the design).  ``G_b``, ``G_x`` and ``direct`` are the rows' gradients in the model
    coordinates, through the plasma parameters and explicit in ``x``.  The reduced Hessian is the converged
    polish's (``kept``) when it is of this point; the true residual of the reduced equations from one exact product
    call on ``mu`` must be at most 1e-6."""
    import scipy.linalg

    model = design.model
    xj = jnp.asarray(x)
    b0 = np.asarray(sol["x"], dtype=float)
    rows = list(range(G_b.shape[0]) if rows is None else rows)
    scale = np.asarray(model.x_scale, dtype=float)
    t0 = time.perf_counter()
    if (kept is not None and kept.get("projected") and kept.get("gauge_Z") is not None
            and np.array_equal(kept["x"], np.asarray(x, dtype=float))
            and float(np.linalg.norm((kept["b"] - b0) / scale)) <= 1e-6):
        return _projected_gauge_gradients(design, x, sol, rows, kept, G_b, G_x, direct, scale, t0)
    matches = (kept is not None and "gauge" in kept and np.array_equal(kept["x"], np.asarray(x, dtype=float))
               and float(np.linalg.norm((kept["b"] - b0) / scale)) <= 1e-6)
    if matches and not kept.get("exact", True) and sol.get("J") is not None:
        # block CG on exact products, preconditioned by this point's Gauss-Newton Hessian (its certified J) on
        # its gauge: within a few % of the exact one, so a few iterations; the gradient stays exact
        red = dict(kept["gauge"])
        Kc = np.asarray(model.gauge_curvature(jnp.asarray(b0), jnp.asarray(red["lam"])))
        J = np.asarray(sol["J"])[:, :b0.size]
        H_gn = J.T @ J + Kc
        red["Hr"] = red["Z"].T @ (0.5 * (H_gn + H_gn.T) * np.outer(scale, scale)) @ red["Z"]
        got = _gauge_block_cg(design, x, sol, rows, red, G_b, G_x, direct, scale, t0, stats)
        if got is not None:
            stats["gauge_block_cg"] = stats.get("gauge_block_cg", 0) + 1
            if isinstance(sol.get("anchor"), dict):  # the next trials start from this point's last model
                sol["anchor"].update(gauge_H=np.asarray(kept["H_cost"]))
            return got
        matches = False
    if matches and kept.get("exact", True):
        red = kept["gauge"]
    else:
        H, _ = dense_hessian(design, x, sol)
        g, _, _ = _cost_gradients(design, b0, design.params_at(xj), design.field_at(xj), np.asarray(x, dtype=float))
        red = _gauge_reduced(model, b0, g, H)
    Z, Hr = red["Z"], red["Hr"]
    Q = G_b[rows].T
    Kc = np.asarray(model.gauge_curvature(jnp.asarray(b0), jnp.asarray(red["lam"])))
    rhs = Z.T @ (scale[:, None] * Q)
    norms = np.maximum(np.linalg.norm(rhs, axis=0), 1e-300)
    mu = scale[:, None] * (Z @ scipy.linalg.solve(Hr, rhs, assume_a="sym"))
    # Iterative refinement on the true residual (exact products): one solve at the reduced condition (~1e8 at
    # 8 x 8) keeps ~1e-6 of the right-hand side, at the gate (QA production missed it at 2.5e-6)
    for _ in range(MULTIPLIER_REFINEMENTS + 1):
        H_mu, Fd_mu, _ = exact_products(design, x, sol, mu, check=False)
        residual = Z.T @ (scale[:, None] * (Q - np.asarray(H_mu) - Kc @ mu))
        true = (np.linalg.norm(residual, axis=0) / norms).tolist()
        if max(true) <= 1e-6:
            break
        mu = mu + scale[:, None] * (Z @ scipy.linalg.solve(Hr, residual, assume_a="sym"))
    else:
        raise RuntimeError(f"gauge multipliers missed: true residuals {true}")
    eig = np.linalg.eigvalsh(Hr)
    if isinstance(sol.get("anchor"), dict):  # the next trials' quasi-Newton start and preconditioner (exact)
        sol["anchor"].update(gauge_H=np.asarray(red["H_L"]) - np.asarray(
            model.gauge_curvature(jnp.asarray(b0), jnp.asarray(red["lam"]))))
    return dict(gradients=direct[rows] + G_x[rows] - np.asarray(Fd_mu).T, rows=rows, true_residual=true,
                asymmetry=0.0, condition=float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf"),
                seconds_hessian=time.perf_counter() - t0)


#: Iterative refinements of the dense gauge multipliers on their true residual (each one exact-product call).
MULTIPLIER_REFINEMENTS = 3
#: Conjugate-gradient iterations of the projected gauge multipliers (each one exact-product call).
PROJECTED_CG_MAXITER = 30
#: Restarts of that CG on its true defect.
PROJECTED_RESTARTS = 3


def _projected_gauge_gradients(design, x, sol, rows, kept, G_b, G_x, direct, scale, t0):
    """:func:`_gauge_gradients` with a bootstrap current: the multipliers ``mu`` of ``H_L mu = Q`` on the gauge, from
    the converged polish's projected Hessian (``H = T^T diag(H_red, H_cc) T``, ``T = [[I, 0], [K, I]]``,
    :func:`projected_hessian`) without forming the joint matrix (round-off at the bootstrap weight): the current
    block ``w_c = H_cc^-1 Q_c``, the boundary block on the gauge's null space ``Z_b`` (the gauge rows do not involve
    the current) ``Z_b^T S (H_red + d2(lam . c)) S Z_b y = Z_b^T S (Q_b - K^T Q_c)``, ``mu_b = S Z_b y``, then
    ``mu_c = w_c - K mu_b``.  Checked as the dense path: one exact product call on ``mu``, true residual <= 1e-6."""
    import scipy.linalg

    model = design.model
    P, Zb, lam = kept["H"], kept["gauge_Z"], kept["gauge_lam"]
    bi, c = np.asarray(P["b"]), np.asarray(P["c"])
    sb = scale[bi]
    b0 = np.asarray(sol["x"], dtype=float)
    curvature = np.asarray(model.gauge_curvature(jnp.asarray(b0), jnp.asarray(lam)))
    Q = G_b[rows].T
    Hs = Zb.T @ (0.5 * ((P["Hred"] + curvature[np.ix_(bi, bi)]) + (P["Hred"] + curvature[np.ix_(bi, bi)]).T)
                 * np.outer(sb, sb)) @ Zb
    sc = scale[c][:, None]  # the null space of the gauge in all coordinates: blockdiag(Z_b, I) (scaled)

    def block_solve(R):  # the projected factorization's solve on the gauge (exact up to its round-off "leak")
        w_c = np.linalg.solve(P["Hcc"], R[c])
        y = scipy.linalg.solve(Hs, Zb.T @ (sb[:, None] * (R[bi] - P["K"].T @ R[c])), assume_a="sym")
        out = np.zeros_like(R)
        out[bi] = sb[:, None] * (Zb @ y)
        out[c] = w_c - P["K"] @ out[bi]
        return out

    def reduce(R):
        return np.r_[Zb.T @ (sb[:, None] * R[bi]), sc * R[c]]

    # Block conjugate gradients on the gauge-reduced exact operator L^T (H + d2(lam . c)) L, L = blockdiag(S Z_b, S_c)
    # (symmetric positive definite at the minimum), preconditioned by the block solve: the reassembled H differs
    # from the exact one by the current rows' leak, which stalls a stationary refinement well above the gate.
    def lift(U):
        return np.r_[sb[:, None] * (Zb @ U[:Zb.shape[1]]), sc * U[Zb.shape[1]:]]

    def precondition(Rr):
        R = np.zeros_like(Q)
        R[bi] = (Zb @ Rr[:Zb.shape[1]]) / sb[:, None]
        R[c] = Rr[Zb.shape[1]:] / sc
        mu = block_solve(R)
        return np.r_[Zb.T @ (mu[bi] / sb[:, None]), mu[c] / sc]

    def full(U):  # the multipliers in model order (lift stacks the boundary block first)
        out = np.zeros_like(Q)
        lifted = lift(U)
        out[bi], out[c] = lifted[:bi.size], lifted[bi.size:]
        return out

    def cg(rhs, history):
        norms = np.maximum(np.linalg.norm(rhs, axis=0), 1e-300)
        U, Rr = np.zeros_like(rhs), rhs.copy()
        Pr = precondition(Rr)
        D, rz = Pr.copy(), np.sum(Rr * Pr, axis=0)
        active = np.ones(rhs.shape[1], dtype=bool)
        for _ in range(PROJECTED_CG_MAXITER):
            mu_d = full(D)
            AD = reduce(np.asarray(exact_products(design, x, sol, mu_d, check=False)[0]) + curvature @ mu_d)
            dAd = np.sum(D * AD, axis=0)
            if np.any(dAd[active] <= 0):
                raise RuntimeError(f"projected gauge multipliers: the reduced Hessian is not positive definite ({dAd})")
            alpha = np.where(active, rz / np.where(active, dAd, 1.0), 0.0)
            U, Rr = U + alpha * D, Rr - alpha * AD
            history.append(float(np.max(np.linalg.norm(Rr, axis=0) / norms)))
            active &= np.linalg.norm(Rr, axis=0) > 1e-8 * norms
            if not active.any():
                break
            Pr = precondition(Rr)
            rz_new = np.sum(Rr * Pr, axis=0)
            D = np.where(active, Pr + (rz_new / np.where(active, rz, 1.0)) * D, D)
            rz = np.where(active, rz_new, rz)
        return U

    # Restarted on the true defect until the residual is below 1e-6 of the right-hand side or a restart no longer moves
    # the gradient (1e-6 of it): with the current block's weight the multipliers are large (|mu| ~ 1e7 against
    # |Q| ~ 1) and the residual's round-off floor (the products repeat to ~1e-13) can sit above that gate.
    rhs = reduce(Q)
    norms = np.maximum(np.linalg.norm(rhs, axis=0), 1e-300)
    U, defect, history, gradients = np.zeros_like(rhs), rhs, [], None
    for _ in range(PROJECTED_RESTARTS):
        U = U + cg(defect, history)
        mu = full(U)
        H_mu, Fd_mu, _ = exact_products(design, x, sol, mu, check=False)
        defect = reduce(Q - np.asarray(H_mu) - curvature @ mu)
        true = (np.linalg.norm(defect, axis=0) / norms).tolist()
        history.append(max(true))
        previous, gradients = gradients, direct[rows] + G_x[rows] - np.asarray(Fd_mu).T
        moved = None if previous is None else float(np.max(
            np.linalg.norm(gradients - previous, axis=1) / np.maximum(np.linalg.norm(gradients, axis=1), 1e-300)))
        if max(true) <= 1e-6 or (moved is not None and moved <= 1e-6):
            break
    else:
        raise RuntimeError(f"projected gauge multipliers missed: true residuals {true}, last restart moved the "
                           f"gradient by {moved} (iterations: " + ", ".join(f"{h:.1e}" for h in history) + ")")
    eig = np.linalg.eigvalsh(Hs)
    return dict(gradients=direct[rows] + G_x[rows] - np.asarray(Fd_mu).T, rows=rows, true_residual=true,
                asymmetry=0.0, condition=float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf"),
                seconds_hessian=time.perf_counter() - t0)


def exact_products(design, x, sol, V, check=True, chunk=None):
    """``H V`` and ``F_d^T V`` exactly, by the second-order adjoint of the fit's cost at the record ``sol``.

    The cost ``f(z, b, x) = |r|^2 / 2`` of the interface rows is written in the projected equilibrium state
    ``z``, the model coordinates ``b`` and the design ``x`` (plasma parameters and coil field), with the
    equilibrium ``F(z, b, x) = 0`` (raw force).  With the multiplier ``F_z^T mu = f_z`` and the Lagrangian
    ``L = f - mu^T F``, ``g = L_b`` and ``c = L_x``.  Along a column ``v``: the state tangent ``F_z dz =
    -F_b v``; ``u = d/dt grad L (z + t dz, b + t v, x)`` (one forward-over-reverse pass); ``F_z^T dmu = u_z``;
    then ``H v = u_b - F_b^T dmu`` and ``F_d^T v = u_x - F_x^T dmu``.  Two linear solves on one block
    factorization per column, no re-solved equilibria and no difference step.  One jitted function (compiled
    per column count) does the whole point; with ``chunk`` the state tangents are solved that many columns at a
    time inside it (the last chunk padded with copies of the last column), else all at once.  Returns ``(HV, FdV,
    consistency)``: ``consistency`` compares ``L_b``, ``L_x`` with :func:`_cost_gradients` at the same point;
    every linear solve is certified (raises).
    """
    if design._exact_products_core is None:
        design._exact_products_core = jax.jit(
            lambda state, mask, xj, b0, V, chunk=None: _exact_products_traced(design, state, mask, xj, b0, V, chunk),
            static_argnames=("chunk",))
    model = design.model
    V = np.asarray(V, dtype=float).reshape(model.x0.size, -1)
    single = V.shape[1] == 1
    if single:  # a one-column call compiles without the product loop (one trip), with far more device memory
        V = np.repeat(V, 2, axis=1)
    # one column (repeated) is never chunked: the chunked reshape fails on an empty design (a free boundary alone)
    chunk = None if chunk is None or single or int(chunk) >= V.shape[1] else max(int(chunk), 1)
    got = design._exact_products_core(sol["state"], sol["mask"], jnp.asarray(x, dtype=float),
                                      jnp.asarray(sol["x"], dtype=float), jnp.asarray(V), chunk=chunk)
    HV, FdV, L_b, L_x, mu_ok, tangent_ok, adjoint_ok = got
    if single:
        HV, FdV = HV[:, :1], FdV[:, :1]
        tangent_ok, adjoint_ok = np.asarray(tangent_ok)[:1], np.asarray(adjoint_ok)[:1]
    if not (bool(mu_ok) and np.asarray(tangent_ok).all() and np.asarray(adjoint_ok).all()):
        raise RuntimeError(f"an exact-product linear solve missed: multiplier {bool(mu_ok)}, tangents "
                           f"{np.asarray(tangent_ok).tolist()}, adjoints {np.asarray(adjoint_ok).tolist()}")
    if not check:
        return np.asarray(HV), np.asarray(FdV), None
    xj = jnp.asarray(x, dtype=float)
    params0 = model.with_boundary(design.params_at(xj), jnp.asarray(sol["x"], dtype=float))
    g_ref, c_ref, _ = _cost_gradients(design, np.asarray(sol["x"], dtype=float), params0, design.field_at(xj),
                                      np.asarray(x, dtype=float))
    consistency = dict(g=float(np.linalg.norm(np.asarray(L_b) - g_ref) / max(np.linalg.norm(g_ref), 1e-300)),
                       c=float(np.linalg.norm(np.asarray(L_x) - c_ref) / max(np.linalg.norm(c_ref), 1e-300)))
    return np.asarray(HV), np.asarray(FdV), consistency


def _exact_products_traced(design, state, mask, xj, b0, V, chunk=None):
    """The traced body of :func:`exact_products` (the block system holds callables, so it is built here)."""
    model = design.model
    im, cfg = model._im, model.cfg
    frozen = jax.lax.stop_gradient(state)
    P = im._dof_projector(cfg, mask)
    edge = im._edge_mask(cfg)
    F_raw = im.residual_fn(cfg, frozen, mask, formulation="raw")
    z0 = P(frozen)
    params_at, field_at = design.params_at, design.field_at

    def prm(b, xx):
        return model.with_boundary(params_at(xx), b)

    def cost(z, b, xx):
        p = prm(b, xx)
        r = model.rows(im._assemble(z, im.runtime_from_params(p, cfg), frozen, P, edge), p, field_at(xx))
        return 0.5 * jnp.vdot(r, r)

    def force(z, b, xx):
        return F_raw(z, prm(b, xx))

    # The bootstrap (Redl) rows enter the curvature by Gauss-Newton, J_B^T J_B: they are a closure driven to zero
    # at every solution (residual ~1e-10), where that is the exact Hessian of the smooth model, while the
    # discrete trapped fraction's own second derivative is near-singular at grid points with |B| ~ Bmax
    # (sqrt(1 - lambda B) with both small) and, times even that residual, would dominate the product.
    def bootstrap_rows(z, b, xx):
        p = prm(b, xx)
        st = im._assemble(z, im.runtime_from_params(p, cfg), frozen, P, edge)
        return model.bootstrap.rows(st, im.runtime_from_params(p, cfg), p)

    def interface_cost(z, b, xx):
        p = prm(b, xx)
        r = model.rows_at(im._assemble(z, im.runtime_from_params(p, cfg), frozen, P, edge),
                          im.runtime_from_params(p, cfg), field_at(xx))
        return 0.5 * jnp.vdot(r, r)

    split = model.bootstrap is not None and not model.fixed_boundary

    def tree_dot(a, c):
        return sum(jnp.vdot(u, w) for u, w in zip(jax.tree.leaves(a), jax.tree.leaves(c)))

    params0 = prm(b0, xj)
    system = im._raw_block_system(params0, cfg, frozen, mask, im._active_state_fields(cfg), model.chunk)
    _, vjp_z = jax.vjp(lambda z: force(z, b0, xj), z0)

    def adjoint(rhs):
        mu, residual, tolerance = im._block_adjoint(system, lambda w: vjp_z(w)[0], system.project(rhs), cfg, cfg.adjoint_tol)
        return mu, residual <= tolerance

    mu, mu_ok = adjoint(jax.grad(cost, 0)(z0, b0, xj))

    def lagrangian(z, b, xx):
        return cost(z, b, xx) - tree_dot(mu, force(z, b, xx))

    grad_L = jax.grad(lagrangian, argnums=(0, 1, 2))
    _, L_b, L_x = grad_L(z0, b0, xj)
    if split:
        grad_second = jax.grad(lambda z, b, xx: interface_cost(z, b, xx) - tree_dot(mu, force(z, b, xx)),
                               argnums=(0, 1, 2))
        _, vjp_boot = jax.vjp(bootstrap_rows, z0, b0, xj)
    else:
        grad_second = grad_L
    _, vjp_bx = jax.vjp(lambda b, xx: force(z0, b, xx), b0, xj)

    def column(args):
        dz, v = args
        # The design is not moved: its tangent is a symbolic zero, so nothing is pushed through the coil field
        # (jax.linearize once for every column is slower and holds twice the memory).
        _, (u_z, u_b, u_x) = jax.jvp(lambda z, b: grad_second(z, b, xj), (z0, b0), (P(dz), v))
        if split:  # + J_B^T (J_B t): the bootstrap rows' Gauss-Newton curvature
            Jt = jax.jvp(lambda z, b: bootstrap_rows(z, b, xj), (z0, b0), (P(dz), v))[1]
            gz, gb, gx = vjp_boot(Jt)
            u_z = jax.tree.map(jnp.add, u_z, gz)
            u_b, u_x = u_b + gb, u_x + gx
        dmu, ok = adjoint(u_z)
        back_b, back_x = vjp_bx(dmu)
        return u_b - back_b, u_x - back_x, ok

    def columns(W):  # the columns' state tangents (on this factorization), then column by column
        dprm = jax.vmap(lambda v: jax.jvp(lambda b: prm(b, xj), (b0,), (v,))[1])(W)
        dz_batch, tangent_ok = model._tangent_solve(params0, state, mask, dprm, system=system)
        HV, FdV, adjoint_ok = jax.lax.map(column, (dz_batch, W))
        return HV, FdV, tangent_ok, adjoint_ok

    m = V.shape[1]
    if chunk is None:
        HV, FdV, tangent_ok, adjoint_ok = columns(V.T)
    else:
        pad = -m % chunk
        W = jnp.concatenate([V.T, jnp.repeat(V.T[-1:], pad, axis=0)]).reshape(-1, chunk, V.shape[0])
        HV, FdV, tangent_ok, adjoint_ok = (a.reshape((-1,) + a.shape[2:])[:m]
                                           for a in jax.lax.map(columns, W))
    return HV.T, FdV.T, L_b, L_x, mu_ok, tangent_ok, adjoint_ok


def _accept(design, x, params, field, fit, *, gate, newton=None, xtol=None, polish=True, max_residual=None,
            extra=None, values=None, stats, anchor_H=None, keep_hessian=None):
    """The acceptance rule of a boundary fit, shared by the standalone solve and the optimization: the one record
    that counts as solved, or :class:`~vmex.core.errors.TrialRejected`.

    With ``newton`` (its options: the gauge labelling) the fit's point is Newton-converged on the gauge and
    certified there (:func:`_newton_refine`); otherwise :meth:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.certify` re-solves it
    tight, polishes it with its own Jacobian and certifies its Gauss-Newton gain against ``gate``.  On the gauge
    without Newton (a restored point) the cost is stationary only within the gauge, so only the other checks apply.
    """
    model = design.model
    if newton is not None:  # converge the stationarity in every direction, certify there
        return _newton_refine(design, x, params, field, extra, fit["x"], newton, stats=stats,
                              max_residual=max_residual, values=values, anchor_H=anchor_H, keep_hessian=keep_hessian)
    # On the gauge the cost is stationary only within it: the Gauss-Newton gain of the full space does not
    # apply (a restored gauge point is then converged by refine_accepted's gauge Newton).
    gauged = model.gauge is not None
    return model.certify(params, field, fit, gate=np.inf if gauged else gate, xtol=None if gauged else xtol,
                         polish=polish and not gauged, max_residual=max_residual, extra=extra, values=values,
                         stats=stats)


def _solve_point(model, params, field, starts, *, accept, fit_options, jacobian=None, direct=None, restore=None,
                 screen=None, stats):
    """The free boundary of ``params`` and ``field`` from each start in turn: ``(record, reasons, last fit)``.

    Each start (boundary coordinates, ``None``: the model's own) is fitted (:meth:`~vmex.core.freeboundary_three_terms.ThreeTermsFreeBoundaryModel.
    solve_boundary` with ``fit_options`` and ``jacobian``) and handed to ``accept`` (:func:`_accept`); the start
    with index ``direct`` skips the fit (its evaluation goes straight to ``accept``).  ``restore()`` runs before
    every start, ``screen(i, fit)`` may name a reason to skip a fit before its acceptance.  The record is that of
    the first accepted start, else ``None``; ``reasons`` say why each start failed.
    """
    sol, why, fit = None, [], None
    for i, start in enumerate(starts):
        if restore is not None:  # every start from the same continuation
            restore()
        t0 = time.perf_counter()
        if i == direct:  # the prediction goes straight to the acceptance (no fit; the gate decides after)
            got = model.evaluate(np.asarray(start, dtype=float), params, field, tight=True)
            stats["seconds_fit"] = stats.get("seconds_fit", 0.0) + time.perf_counter() - t0
            if got is None:
                why.append(f"start {i}: the accepted boundary has no certified interior root at this design")
                continue
            fit = dict(x=np.asarray(start, dtype=float), rows=np.asarray(got[0]), aux=got[1], nfev=1, njev=0,
                       accepted=0, converged=True, reason="prediction", jacobian=jacobian)
            t0 = time.perf_counter()
            try:
                sol = accept(fit)
                stats["direct_newton_trials"] = stats.get("direct_newton_trials", 0) + 1
                break
            except (TrialRejected, VmecError, RuntimeError) as error:  # the fitted starts follow
                why.append(f"start {i} (direct Newton): {error}")
                stats["direct_newton_failed"] = stats.get("direct_newton_failed", 0) + 1
                continue
            finally:
                stats["seconds_certify"] = stats.get("seconds_certify", 0.0) + time.perf_counter() - t0
        try:
            fit = model.solve_boundary(params, field, x0=start, jacobian=jacobian, **fit_options)
        except (VmecError, RuntimeError) as error:  # an uncertified equilibrium: try the next start
            why.append(f"start {i}: {error}")
            continue
        finally:
            stats["seconds_fit"] = stats.get("seconds_fit", 0.0) + time.perf_counter() - t0
        reasons = stats.setdefault("fit_reasons", {})
        reasons[fit["reason"]] = reasons.get(fit["reason"], 0) + 1
        stats["boundary_evaluations"] = stats.get("boundary_evaluations", 0) + fit["nfev"]
        reason = None if screen is None else screen(i, fit)
        if reason:
            why.append(f"start {i}: {reason}")
            continue
        t0 = time.perf_counter()
        try:  # the certificate decides (a stall on a fresh Jacobian may be stationary)
            sol = accept(fit)
            break
        except TrialRejected as error:
            why.append(f"start {i}: {error}")
        finally:
            stats["seconds_certify"] = stats.get("seconds_certify", 0.0) + time.perf_counter() - t0
    return sol, why, fit


def _residual_multipliers(J_b, G_b):
    """Residual-space multipliers ``lambda = J_b (J_b^T J_b)^-1 G_b^T`` (a column per row of ``G_b``, the rows'
    gradients in the model coordinates): ``dh/dd = dh/dd|_b - lambda^T dr/dd`` is the Gauss-Newton design
    gradient, ``G_b`` times :func:`_least_squares_sensitivity` without curvature terms.  By a QR of ``J_b``."""
    Q, R = np.linalg.qr(J_b)
    return Q @ np.linalg.solve(R.T, np.asarray(G_b).T)


def _least_squares_sensitivity(J_b, J_d, S_bb=None, S_bd=None):
    """``db/dd`` of a least-squares stationary point, ``F(b, d) = J_b^T r = 0``, by the implicit function theorem.

    ``F_b = J_b^T J_b + S_bb`` and ``F_d = J_b^T J_d + S_bd`` with the curvature terms
    ``S_bb = sum_i r_i d^2 r_i / db^2`` and ``S_bd = sum_i r_i d^2 r_i / db dd``.  Without them it is the
    Gauss-Newton sensitivity of the optimization's gradient (:func:`_residual_multipliers`), exact only where the
    residual vanishes or is affine; with them, the exact derivative of the re-solved point.  Linear solves, no inverse.
    """
    J_b, J_d = np.asarray(J_b), np.asarray(J_d)
    F_b = J_b.T @ J_b + (0.0 if S_bb is None else np.asarray(S_bb))
    F_d = J_b.T @ J_d + (0.0 if S_bd is None else np.asarray(S_bd))
    return -np.linalg.solve(F_b, F_d)


def least_squares_sensitivity(J_y, J_d, S_yy=None, S_yd=None):
    """``dy/dd`` of a least-squares stationary point ``y(d)``, ``F(y, d) = J_y^T r = 0``: ``-F_y^-1 F_d``.

    ``F_y = J_y^T J_y + S_yy`` and ``F_d = J_y^T J_d + S_yd``, with the curvature terms ``S_yy = sum_i r_i
    d2r_i/dy2`` and the mixed ``S_yd = sum_i r_i d2r_i/dy dd``.  Without them it is the Gauss-Newton sensitivity
    ``-J_y^+ J_d`` (the Gauss-Newton design gradient); with them, the exact one (both
    terms vanish only at a zero residual).
    """
    J_y, J_d = np.atleast_2d(J_y), np.asarray(J_d, dtype=float).reshape(np.shape(J_y)[0], -1)
    if S_yy is None and S_yd is None:
        return -np.linalg.lstsq(J_y, J_d, rcond=None)[0]
    F_y = J_y.T @ J_y + (0.0 if S_yy is None else np.atleast_2d(S_yy))
    F_d = J_y.T @ J_d + (0.0 if S_yd is None else np.asarray(S_yd, dtype=float).reshape(J_y.shape[1], -1))
    return -np.linalg.solve(F_y, F_d)


def _projected_solve(P, rhs):
    """``H^-1 rhs`` for a Hessian with its current block eliminated (``projected_hessian``): with ``H = T^T diag(H_red,
    H_cc) T``, ``T = [[I, 0], [K, I]]``, each block solved on its own (no solve mixes their scales)."""
    rhs = np.asarray(rhs, dtype=float)
    b, c, K = P["b"], P["c"], P["K"]
    z_b = np.linalg.solve(P["Hred"], rhs[b] - K.T @ rhs[c])
    out = np.empty_like(rhs)
    out[b], out[c] = z_b, np.linalg.solve(P["Hcc"], rhs[c]) - K @ z_b
    return out

