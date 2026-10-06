"""Free-boundary equilibria from the three plasma-vacuum interface conditions.

At a free boundary ``S`` without a sheet current the interface conditions are
(Conlin et al. 2024, arXiv:2412.05680)

1. ``B_out . n = 0``,
2. ``|B_out|^2 = |B_in|^2 + 2 mu0 p``,
3. ``n x (B_out - B_in) = mu0 K = 0``,

with ``B_out = B_coil + B_plasma`` and ``B_plasma`` the plasma's own field from
the virtual-casing principle (:mod:`vmex.core.virtual_casing`).  NESTOR
(:mod:`vmex.core.freeboundary`) builds the vacuum field that satisfies (1) by
construction and balances only ``|B|`` in (2), so it accepts any tangential
field jump: a VMEC/NESTOR equilibrium may carry an edge sheet current.  Here
the boundary is the unknown instead.  Every trial boundary is a fixed-boundary
equilibrium, and a Gauss-Newton least-squares solve minimizes the stacked
residual of all three conditions with exact implicit derivatives
(:class:`VirtualCasingModel`).  At an exact
solution all three vanish together; at finite resolution they stop at a floor
set mainly by ``mpol`` (about 1e-3 relative at ``mpol = 5``, 5e-4 at 7), where
the relative ``weights`` decide the balance.

Condition (1) alone leaves one direction open: the flux of ``B_out`` through
the closed surface vanishes identically, and a mismatch of the net poloidal
current ``G`` adds a field tangent to ``S``.  Conditions (2) and (3) close it in
principle; numerically a row ``(G_plasma - G_coil) / G_coil`` closes it directly
(``net_current_weight``).  With ``p = 0`` on the boundary a solution satisfies
all three exactly; a nonzero edge pressure needs a sheet current and is
rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .input import VmecInput


@dataclass(frozen=True)
class BoundaryResidual:
    """Area-weighted RMS of the three interface conditions, relative to ``|B_in|``.

    ``normal`` is ``B_out . n / |B_in|``, ``pressure`` is
    ``(|B_out|^2 - |B_in|^2 - 2 mu0 p) / (2 |B_in|^2)`` and ``sheet_current``
    is ``|n x (B_out - B_in)| / |B_in| = mu0 |K| / |B_in|``.
    """

    normal: float
    pressure: float
    sheet_current: float


def _interface_terms(inp, state, external_field, *, runtime, nphi, ntheta, digits, precision, p_edge):
    """Rows and weights of :func:`boundary_residual`, plus the boundary points ``(3, nphi, ntheta)``."""
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    interface = vc.PlasmaVacuumInterface.from_surface_data(
        data, p_edge=p_edge, digits=digits, precision=precision)
    B_in = jnp.asarray(data.B_total)
    B_out = interface.total_B_out(external_field)
    B2 = interface.Bin_mag2
    scale = jnp.sqrt(B2)
    normal = jnp.sum(B_out * interface.normal, axis=0) / scale
    pressure = interface.pressure_balance_residual(external_field) / (2.0 * B2)
    sheet = jnp.cross(interface.normal, B_out - B_in, axis=0) / scale
    return jnp.concatenate([normal[None], pressure[None], sheet]), interface.weights, interface.gamma


def boundary_residual(inp: VmecInput, state, external_field: Any, *, runtime=None,
                      nphi: int, ntheta: int, digits: int = 4, precision=None,
                      p_edge: float = 0.0):
    """Pointwise interface conditions on the boundary of a live ``state``.

    Returns ``(rows, weights)``: ``rows`` has shape ``(5, nphi, ntheta)`` with
    ``B_out . n``, the pressure-balance jump and the three Cartesian components
    of ``n x (B_out - B_in)``, each scaled as in :class:`BoundaryResidual`;
    ``weights`` are the area weights (summing to one).  Traceable in ``state``
    and the external-field parameters when ``precision`` is supplied (see
    :func:`~vmex.core.virtual_casing.plan_vc_precision`).
    """
    rows, weights, _ = _interface_terms(inp, state, external_field, runtime=runtime, nphi=nphi,
                                        ntheta=ntheta, digits=digits, precision=precision, p_edge=p_edge)
    return rows, weights


def summarize_boundary_residual(rows, weights) -> BoundaryResidual:
    """RMS of each condition from :func:`boundary_residual`'s output."""
    rows, weights = np.asarray(rows), np.asarray(weights)

    def rms(value):
        return float(np.sqrt(np.sum(weights * value**2)))

    return BoundaryResidual(normal=rms(rows[0]), pressure=rms(rows[1]),
                            sheet_current=rms(np.linalg.norm(rows[2:], axis=0)))


def _edge_pressure(inp: VmecInput) -> tuple[float, float]:
    """Edge and peak pressure of ``inp`` in Pa."""
    from .profiles import pressure

    p = np.asarray(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, np.linspace(0.0, 1.0, 101),
                            pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped))
    return float(p[-1]), float(np.max(np.abs(p)))


def _axis_loop(wout):
    """Points ``(3, n)``, tangent and spacing of the magnetic axis of ``wout``: a closed curve inside the coils."""
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    harmonics = np.arange(np.asarray(wout.raxis_cc).size) * int(wout.nfp)
    R = np.cos(np.outer(phi, harmonics)) @ np.asarray(wout.raxis_cc, dtype=float)
    Z = -np.sin(np.outer(phi, harmonics)) @ np.asarray(wout.zaxis_cs, dtype=float)
    points = np.stack([R * np.cos(phi), R * np.sin(phi), Z])
    return jnp.asarray(points), jnp.asarray(np.gradient(points, phi, axis=1)), float(phi[1] - phi[0])


def _net_current(field, points, tangent, dphi):
    """Traceable ``G = (1/2 pi)`` circulation of ``field`` along a closed curve (:func:`_axis_loop`)."""
    B = vc.external_B_cartesian(field, points[:, :, None])[:, :, 0]
    return jnp.sum(B * tangent) * dphi / (2.0 * jnp.pi)


def _coil_net_current(external_field, wout) -> float:
    """``G = (1/2 pi)`` times the external field's circulation around the magnetic axis of ``wout``."""
    return float(_net_current(external_field, *_axis_loop(wout)))


def solve_free_boundary_virtual_casing(
    inp: VmecInput,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_boundary: VmecInput | None = None,
    max_mode: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
    digits: int = 4,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    net_current_weight: float = 1.0,
    label_weight: float = 1e-2,
    ftol: float = 1e-4,
    jacobian_ftol: float | None = 1e-2,
    max_nfev: int = 60,
    previous=None,
    verbose: int = 0,
):
    """Free-boundary equilibrium of ``inp`` from the three interface conditions.

    ``inp`` is a free-boundary deck (``PHIEDGE``, profiles, and the external
    field through ``external_field``, ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``);
    its boundary, or that of ``initial_boundary``, is the initial guess, so a
    fixed-boundary design and the coils fitted to it make a natural start.  The
    boundary Fourier coefficients up to ``max_mode`` (default: all) and
    ``RBC(0,0)`` are solved for at fixed toroidal flux, pressure and current or
    iota, every trial being a fixed-boundary equilibrium on the deck's
    ``NS_ARRAY`` ladder.  ``external_field`` is anything
    :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts (an
    :class:`~vmex.core.mgrid.MgridField`, a ``xyz -> B`` callable such as an
    ESSOS Biot-Savart field, or a coil field with ``b_cyl``).

    The residual is ``sqrt(area weight)`` times ``weights = (normal, pressure,
    sheet current)`` times the rows of :func:`boundary_residual` on an
    ``nphi x ntheta`` grid over one field period (default 48 x 48), with the
    virtual-casing quadrature planned once on the initial boundary to
    ``digits``.  The grid, not ``digits``, limits accuracy: for a current-free
    state virtual casing returns ``|B_plasma| / |B|`` of about 2e-4 at 48 x 48
    and 5e-5 at 64 x 64 (whose quadrature needs several times the memory).
    Rows of the boundary points' displacement along the surface in ``theta``,
    relative to the initial boundary and scaled by ``label_weight``, fix the
    boundary's poloidal labelling, which the interface conditions leave free.
    ``net_current_weight`` scales the row ``(G_plasma - G_coil) / G_coil`` of
    the net poloidal current (edge ``rbtor`` against the external field's
    circulation around the seed's magnetic axis), the direction the
    normal-field rows cannot see.

    A trust region with exact implicit Jacobians (:class:`VirtualCasingModel`)
    runs to a relative cost change of ``jacobian_ftol``; Levenberg-Marquardt
    steps with its last Jacobian then go on to ``ftol``, the floor set by the
    resolution (``None``: the trust region runs to ``ftol``).  ``previous``, an
    earlier result for the same deck (typically before a coil change), starts
    from its boundary with those steps and its Jacobian, and from its model:
    neither new coils nor a repeat solve compile anything.

    Returns a :class:`scipy.optimize.OptimizeResult` with ``x`` (the boundary
    coordinates), ``fun`` (the residual), ``jac`` (its last Jacobian), ``cost``,
    ``nfev``, ``njev``, ``input`` (the free boundary as a fixed-boundary deck),
    ``equilibrium``, ``model``, ``boundary_residual`` and
    ``initial_boundary_residual`` (:class:`BoundaryResidual` at the start).
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .optimize import solve_equilibrium, unpack_boundary

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    if previous is not None:
        model, x0, jacobian = previous.model, previous.x, previous.jac
    else:
        start = inp if initial_boundary is None else replace(
            inp, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs,
            raxis_c=initial_boundary.raxis_c, zaxis_s=initial_boundary.zaxis_s)
        model = VirtualCasingModel(start, max_mode=max_mode, nphi=int(nphi or 48), ntheta=int(ntheta or 48),
                                   digits=digits, weights=weights, net_current_weight=net_current_weight,
                                   label_weight=label_weight)
        x0 = jacobian = None
    seed_state = model.seed[0] if previous is None else previous.aux[0]
    seed_params = model.params0 if previous is None else previous.aux[2]
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, jacobian=jacobian, ftol=ftol,
                               jacobian_ftol=ftol if jacobian_ftol is None else jacobian_ftol,
                               max_nfev=max_nfev, verbose=verbose)
    state, _, params, _ = out["aux"]
    boundary = unpack_boundary(model.fixed, out["x"], model.max_mode, vary_major_radius=True)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=True, message=f"{out['accepted']} Levenberg-Marquardt steps after the trust region",
        input=boundary, equilibrium=solve_equilibrium(boundary, initial_state=state), model=model, aux=out["aux"],
        boundary_residual=model.boundary_residual(state, params, external_field), initial_boundary_residual=initial)


class VirtualCasingModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from
    :mod:`vmex.core.implicit` with the boundary, PHIEDGE and current profile
    as traced :class:`~vmex.core.implicit.ImplicitParams`, and the external
    field enters the compiled interface rows as an argument, so neither new
    coils nor new plasma parameters recompile anything.  ``x`` are the
    boundary coefficients up to ``max_mode`` and ``RBC(0,0)``
    (:func:`~vmex.core.optimize.pack_boundary`); ``params`` an
    :class:`~vmex.core.implicit.ImplicitParams` whose boundary ``x`` replaces.
    The rows are those of :func:`solve_free_boundary_virtual_casing`, with the
    quadrature plan and the labelling reference fixed on the seed boundary.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=48, ntheta=48, digits=4,
                 weights=(1.0, 1.0, 1.0), net_current_weight=1.0, label_weight=1e-2, chunk=8, device=None,
                 trial_ftol=None):
        from . import implicit as im
        from .freeboundary import _vacuum_scalars
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary, solve_equilibrium

        self.im = im
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.max_mode = int(max(fixed.mpol - 1, fixed.ntor) if max_mode is None else max_mode)
        device = jax.devices()[0] if device is None else device
        cfg = im.make_config(fixed, multigrid=True, hot_restart=True)
        self.cfg = cfg = im._canonical_config(replace(cfg, device=device))
        # Trial points (the boundary steps) may converge to a looser force residual: their rows move by far less
        # than the interface floor, and a state is solved to the deck's tolerance before it is differentiated.
        self.cfg_trial = cfg
        if trial_ftol is not None:
            ftols = np.array(fixed.ftol_array, dtype=float)
            ftols[-1] = max(float(trial_ftol), ftols[-1])
            loose = im.make_config(replace(fixed, ftol_array=ftols), multigrid=True, hot_restart=True)
            self.cfg_trial = im._canonical_config(replace(loose, device=device))
            im._template_runtime(self.cfg_trial)
        self.params0 = im.params_from_input(fixed, device=device)
        im._template_runtime(cfg)
        self.x0 = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
        self.chunk = int(chunk)

        def with_boundary(params, x):
            rbc, zbs = boundary_arrays_from_x(fixed, x, self.max_mode, vary_major_radius=True)
            return replace(params, rbc=rbc, zbs=zbs)

        self.with_boundary = with_boundary
        seed = self.solve(self.params0)
        if seed is None:
            from .errors import VmecConvergenceError
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        runtime = im.runtime_from_params(self.params0, cfg)
        surface = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        precision = vc.plan_vc_precision(surface, digits=digits)
        gamma0 = jnp.asarray(surface.gamma)
        wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
        e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma0, axis=-1), axis=-1))
        length = jnp.linalg.norm(e_theta, axis=0)
        tangent = e_theta / (length * jnp.mean(length))[None]
        scale = jnp.repeat(jnp.asarray(weights, dtype=float), jnp.asarray([1, 1, 3]))[:, None, None]
        # A closed curve inside the coils for G_coil: the seed's magnetic axis.
        axis = _axis_loop(solve_equilibrium(fixed, initial_state=state).wout)
        self.seed = (state, mask)

        def rows_at(state, runtime, field):
            values, area, gamma = _interface_terms(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                   digits=digits, precision=precision, p_edge=0.0)
            label = label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0)
            interface = (jnp.sqrt(area)[None] * jnp.concatenate([scale * values, label[None]])).ravel()
            if not net_current_weight:
                return interface
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            return jnp.concatenate([interface, (net_current_weight * (G - G_coil) / G_coil)[None]])

        def rows(state, params, field):
            return rows_at(state, im.runtime_from_params(params, cfg), field)

        self.rows_at = rows_at
        self.rows = rows
        self._rows = jax.jit(rows)
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field, runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])
        self.surface_area_rows = 6 * nphi * ntheta

        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch):
            """Projected state responses to the stacked parameter tangents: one block factorization, chunked."""
            return im._implicit_evolved_tangent_multi_rhs(
                params, cfg, jax.lax.stop_gradient(state), mask, batch, active_fields=active,
                probe_chunk_size=self.chunk, response_chunk_size=self.chunk,
                certify_rtol=float(cfg.jacobian_adjoint_tol), certify_maxiter=int(cfg.jacobian_adjoint_maxiter))[0]

        def push(fun, state, mask, params, dz_batch, params_batch):
            """Columns of ``fun(state, params)`` along the stacked responses, ``chunk`` at a time.

            As upstream's implicit Jacobian: the state of a column is assembled from its projected response and
            its parameters inside the map, so no batch of full states is ever built.
            """
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)

            def at(z, prm):
                return fun(im._assemble(z, im.runtime_from_params(prm, cfg), frozen, P, edge), prm)

            return jax.lax.map(lambda a: jax.jvp(at, (P(frozen), params), (P(a[0]), a[1]))[1],
                               (dz_batch, params_batch), batch_size=self.chunk)

        def predict(state, mask, dz_batch, dx, params):
            """First-order trial state: the anchor plus the responses times ``dx``, on the trial boundary."""
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)
            z = jax.tree.map(lambda a, d: a + jnp.tensordot(dx, d, axes=1), P(frozen), dz_batch)
            return im._assemble(z, im.runtime_from_params(params, cfg), frozen, P, edge)

        self._tangents = jax.jit(tangents)
        self._push_rows = jax.jit(lambda state, mask, params, dz, pb, field: push(
            lambda s, p: rows(s, p, field), state, mask, params, dz, pb))
        self.push = push
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)
        self._linearization = None

    def solve(self, params, seed=None, tight=True):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified.

        ``tight=False`` solves to the trial tolerance.
        """
        im, cfg = self.im, (self.cfg if tight else self.cfg_trial)
        if seed is not None:
            im._PERTURB_SEED[cfg] = seed
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), params)
        state, mask, status, _, _ = im._host_solve_and_mask_status(cfg, params_np)
        if int(status) != 0:
            return None
        place = lambda tree: jax.device_put(jax.tree.map(jnp.asarray, tree), cfg.device)  # noqa: E731
        return place(state), place(mask)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`BoundaryResidual` of a solved state."""
        return summarize_boundary_residual(*map(np.asarray, self._terms(state, params, field)))

    def evaluate(self, x, params, field, seed=None, tight=False):
        """Interface rows at boundary ``x``: ``(rows, (state, mask, params_x, tight))``, or ``None`` if the solve
        failed; ``tight`` marks a state solved to the deck's tolerance rather than the trial one."""
        params_x = self.with_boundary(params, jnp.asarray(x))
        if seed is None and self._linearization is not None:
            x_ref, state_ref, mask_ref, dz = self._linearization
            seed = self._predict(state_ref, mask_ref, dz, jnp.asarray(np.asarray(x) - x_ref), params_x)
        tight = tight or self.cfg_trial is self.cfg
        solved = self.solve(params_x, seed, tight)
        if solved is None:
            return None
        state, mask = solved
        return np.asarray(self._rows(state, params_x, field)), (state, mask, params_x, tight)

    def boundary_directions(self, params, x):
        """``ImplicitParams`` tangents of the boundary coordinates, stacked."""
        eye = jnp.eye(np.size(x))
        return jax.vmap(lambda e: jax.jvp(lambda xx: self.with_boundary(params, xx), (jnp.asarray(x),), (e,))[1])(eye)

    def linearize(self, x, aux, field, extra=None):
        """Columns of the interface rows along the boundary coordinates (and the ``extra`` parameter tangents).

        A trial state is first solved to the deck's tolerance (from itself). Returns ``(J, dz, params_batch, aux)``:
        the projected state responses of every column, for pushing other functions of the equilibrium through the
        same directions (:meth:`push`), and the state they belong to.
        """
        state, mask, params_x, tight = aux
        if not tight:
            solved = self.solve(params_x, seed=state)
            if solved is None:
                raise RuntimeError("the trial state does not converge to the deck's tolerance")
            state, mask = solved
            aux = (state, mask, params_x, True)
        batch = self.boundary_directions(params_x, x)
        dz = self._tangents(params_x, state, mask, batch)
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        columns = np.asarray(self._push_rows(state, mask, params_x, dz, batch, field)).T
        if extra is not None:  # a program of its own: one batch of every direction can exhaust the GPU
            dz_extra = self._tangents(params_x, state, mask, extra)
            columns = np.hstack([columns, np.asarray(self._push_rows(state, mask, params_x, dz_extra, extra,
                                                                     field)).T])
            join = lambda a, b: jnp.concatenate([a, b])  # noqa: E731
            dz, batch = jax.tree.map(join, dz, dz_extra), jax.tree.map(join, batch, extra)
        return columns, dz, batch, aux

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, ftol=1e-4, jacobian_ftol=1e-2, max_nfev=60,
                       target_cost=None, verbose=0):
        """K = 0 boundary for ``params`` and ``field``.

        With ``jacobian`` (from a nearby solve) Levenberg-Marquardt steps reuse it from ``x0``; otherwise, or if no
        step reduces the cost, or they end above ``target_cost``, a trust region with fresh Jacobians runs to
        ``jacobian_ftol`` and the reuse steps finish to ``ftol`` (see :func:`solve_free_boundary_virtual_casing`).
        Returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask, params, tight), ``nfev``,
        ``njev``, ``accepted`` and ``converged``.
        """
        import scipy.optimize

        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        nfev = njev = 0
        if jacobian is not None:
            first = self.evaluate(x, params, field)
            if first is None:
                raise RuntimeError("the warm-start boundary has no certified equilibrium")
            out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), x, *first, jacobian,
                                       ftol=ftol, max_nfev=max_nfev, verbose=verbose)
            close = target_cost is None or 0.5 * out["rows"] @ out["rows"] <= target_cost
            if (out["accepted"] or out["converged"]) and close:
                return dict(out, jacobian=jacobian, njev=0)
            x, nfev = out["x"], out["nfev"]
        memo = {}

        def fun(z):
            got = self.evaluate(z, params, field)
            if got is None:
                return np.full(memo.get("size", 1), 1e6)
            memo.update(key=z.tobytes(), size=got[0].size, rows=got[0], aux=got[1])
            return got[0]

        def jac(z):
            nonlocal njev
            if memo.get("key") != z.tobytes():
                fun(z)
            njev += 1
            J, _, _, memo["aux"] = self.linearize(z, memo["aux"], field)
            memo.update(jacobian=J, jkey=z.tobytes())
            return memo["jacobian"]

        fit = scipy.optimize.least_squares(fun, x, jac=jac, x_scale=self.x_scale, ftol=max(ftol, jacobian_ftol),
                                           xtol=1e-5, gtol=1e-10, max_nfev=max_nfev, verbose=verbose)
        nfev += int(fit.nfev)
        if memo.get("key") != fit.x.tobytes():
            fun(fit.x)
        J = jac(fit.x) if memo.get("jacobian") is None or memo.get("jkey") != fit.x.tobytes() else memo["jacobian"]
        out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), fit.x, memo["rows"], memo["aux"], J,
                                   ftol=ftol, max_nfev=max_nfev, verbose=verbose)
        return dict(out, jacobian=J, nfev=nfev + out["nfev"], njev=njev)


def _levenberg_marquardt(evaluate, x, r, aux, J, *, ftol, max_nfev, verbose):
    """Levenberg-Marquardt steps with a fixed Jacobian ``J`` from ``x`` (``evaluate(x) -> (r, aux)`` or ``None``).

    Each step costs one evaluation; Marquardt column scaling and one SVD serve every damping, which rises
    after a step that fails to reduce the cost and falls after one that succeeds.  Stops at a relative cost
    change of ``ftol`` or when even short steps fail.
    """
    J = np.asarray(J)
    norms = np.linalg.norm(J, axis=0)
    norms[norms == 0.0] = 1.0
    U, sv, Vt = np.linalg.svd(J / norms, full_matrices=False)
    nfev, accepted, damping, failed, converged = 1, 0, 0.0, False, False
    while nfev < max_nfev:
        g = U.T @ r
        cost = 0.5 * r @ r
        if 0.5 * g @ g <= ftol * cost:  # the Gauss-Newton step's predicted gain
            converged = True
            break
        x_new = x - (Vt.T @ (sv / (sv**2 + damping * sv[0] ** 2) * g)) / norms
        got = evaluate(x_new)
        nfev += 1
        cost_new = np.inf if got is None else 0.5 * got[0] @ got[0]
        if verbose:
            print(f"Levenberg-Marquardt step {nfev - 1}: damping {damping:.1e}, cost {cost:.6e} -> {cost_new:.6e}")
        if cost_new < cost:
            x, (r, aux), accepted = x_new, got, accepted + 1
            if cost - cost_new <= ftol * cost:
                converged = True
                break
            if not failed:  # lower the damping only after two successes in a row
                damping = 0.0 if damping < 1e-8 else damping / 4
            failed = False
        elif damping >= 1.0:
            break
        else:
            damping, failed = (1e-6 if damping == 0.0 else 4 * damping), True
    return dict(x=x, rows=r, aux=aux, nfev=nfev, accepted=accepted, converged=converged)
