"""Free-boundary equilibria by least squares of the interface conditions (``method="least_squares"``).

The second method of :func:`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms` (the first,
``method="galerkin"``, is the module :mod:`vmex.core.freeboundary_vc`'s square system on the deck's normal
displacement).  Here the unknowns are the boundary's Fourier coefficients and the solve minimizes the
area-weighted sum of squares of the interface conditions on a grid, each over ``|B_in|``: in vacuum
``B_coil . n``; at finite beta ``B_out . n``, the pressure-balance jump and ``n x (B_out - B_in)``, with the
interior edge field kept on VMEC's Galerkin band (:func:`_band_limited`).  A row ``(G_plasma - G_coil) /
G_coil`` fixes the net poloidal current.

The interface conditions fix the surface but not its poloidal labelling: at finite resolution the labelling
changes what the truncated series can represent, and the exact Hessian of the cost is flat and indefinite along
relabellings.  The labelling is therefore fixed by a canonical condition on the boundary itself, with no
reference surface: the spectral width ``F_6 = sum m^6 (R_mn^2 + Z_mn^2)`` is stationary under every ``m > 0``
relabelling and the twist ``T = sum (n nfp)^2 (R_mn^2 + Z_mn^2)`` under every shift of the angle origin
(:meth:`LeastSquaresFreeBoundaryModel._gauge_setup`).  The answer does not depend on the start or on the deck's
boundary, which is only the initial guess.

The solver (:meth:`LeastSquaresFreeBoundaryModel._newton`) is a trust-region (quasi-)Newton method on the gauge
manifold with exact second derivatives through the equilibrium (second-order adjoint,
:meth:`LeastSquaresFreeBoundaryModel.second_order`): Gauss-Newton and BFGS models first, the exact Hessian as
products in a preconditioned Lanczos subspace where they fail and to certify convergence.  Design gradients
(:class:`LeastSquaresFreeBoundaryProblem`) use the exact reduced Hessian: Gauss-Newton's would be wrong by
``sum_i r_i d^2 r_i`` (up to 21 % at finite beta).  It is reference-free and, at 8 x 8 and 12 x 12, 3-5 times
slower than the Galerkin method.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .errors import TrialRejected
from .freeboundary_vc import (BoundaryResidual, ThreeTermsFreeBoundaryModel, ThreeTermsFreeBoundaryProblem,
                              _REPORT_FTOL, _axis_loop, _edge_pressure, _fingerprint, _interface_terms,
                              _net_current, _surface_data, summarize_boundary_residual)
from .input import VmecInput


def _band_limited(data, mpol, ntor):
    """``data`` with its interior field ``B_total`` kept on the interior's own Galerkin band.

    VMEC's lambda equation zeroes the current through the boundary only on the modes it solves for (``m <= mpol -
    1``, ``|n| <= ntor``); outside that band the edge field is not determined (LP QA beta 2.5 %, mpol 8: a spurious
    3e-4 |B| just above the band, 7.5x the true content there, the same in VMEC2000).  The covariant surface
    components ``B . e_theta``, ``B . e_phi`` (periodic over a field period) are filtered to that band and the
    tangential field rebuilt from them (``B_in . n = 0``).
    """
    gamma, B = jnp.asarray(data.gamma), jnp.asarray(data.B_total)
    nphi, ntheta = gamma.shape[1], gamma.shape[2]
    phi = jnp.asarray(data.phi)[:, None]
    R, Z = jnp.hypot(gamma[0], gamma[1]), gamma[2]
    k_t = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)[None, :]
    k_p = jnp.fft.fftfreq(nphi, 1.0 / nphi)[:, None]  # field-period harmonics (n in units of nfp)
    nfp = int(data.nfp)

    def d_theta(f):
        return jnp.real(jnp.fft.ifft(1j * k_t * jnp.fft.fft(f, axis=-1), axis=-1))

    def d_phi(f):
        return jnp.real(jnp.fft.ifft(1j * nfp * k_p * jnp.fft.fft(f, axis=0), axis=0))

    c, s_ = jnp.cos(phi), jnp.sin(phi)
    Rt, Zt, Rp, Zp = d_theta(R), d_theta(Z), d_phi(R), d_phi(Z)
    e_t = jnp.stack([Rt * c, Rt * s_, Zt])
    e_p = jnp.stack([Rp * c - R * s_, Rp * s_ + R * c, Zp])
    keep = (jnp.abs(k_t) <= mpol - 1) & (jnp.abs(k_p) <= ntor)

    def band(f):
        return jnp.real(jnp.fft.ifft2(jnp.where(keep, jnp.fft.fft2(f), 0.0)))

    Bt, Bp = band(jnp.sum(B * e_t, axis=0)), band(jnp.sum(B * e_p, axis=0))
    gtt, gtp, gpp = jnp.sum(e_t * e_t, axis=0), jnp.sum(e_t * e_p, axis=0), jnp.sum(e_p * e_p, axis=0)
    det = gtt * gpp - gtp * gtp
    bt, bp = (gpp * Bt - gtp * Bp) / det, (gtt * Bp - gtp * Bt) / det
    filtered = bt[None] * e_t + bp[None] * e_p
    try:
        return replace(data, B_total=filtered)
    except TypeError:  # not a dataclass: copy the attributes
        import copy

        out = copy.copy(data)
        object.__setattr__(out, "B_total", filtered)
        return out



@dataclass(frozen=True)
class _Bundle:
    """An external field with the run's constants (seed axis, field scale) as one pytree.

    The compiled rows take these as arguments, not as constants closed over: two runs at one resolution then
    lower to the same program and share the persistent compilation cache.
    """

    field: Any
    consts: Any


jax.tree_util.register_dataclass(_Bundle, data_fields=["field", "consts"], meta_fields=[])



def _interface_terms_banded(inp, state, external_field, *, runtime, nphi, ntheta, digits, precision, p_edge,
                     band_limit=False):
    """Rows and weights of :func:`boundary_residual`, plus the boundary points ``(3, nphi, ntheta)``.

    ``band_limit`` keeps the interior edge field on VMEC's Galerkin band (:func:`_band_limited`) before the
    virtual casing, the pressure balance and the sheet current see it.
    """
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    if band_limit:
        data = _band_limited(data, int(inp.mpol), int(inp.ntor))
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



class LeastSquaresFreeBoundaryModel(ThreeTermsFreeBoundaryModel):
    """The least-squares free boundary of one deck for many external fields and plasma parameters.

    :class:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryModel` (its equilibria, tangents, pushes,
    pullbacks, warm-up and keywords) with the boundary's Fourier coefficients as coordinates (``x`` up to
    ``max_mode`` and ``RBC(0,0)``, :func:`~vmex.core.optimize.pack_boundary`; ``x0`` the deck's, ``x_scale``
    their ESS scales), the least-squares rows of the module docstring, the canonical labelling gauge
    (:meth:`_gauge_setup`, :meth:`gauge_restore`) and exact second derivatives (:meth:`second_order`,
    :meth:`gradient`).  :meth:`solve_boundary` is the trust-region (quasi-)Newton solve (:meth:`_newton`).
    ``hessian_chunk`` bounds the second-derivative columns per batch (default ``chunk``).
    """

    def __init__(self, inp: VmecInput, *, hessian_chunk=None, **options):
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

        super().__init__(inp, **options)
        fixed = self.fixed
        if not self.fixed_boundary:
            self.x0 = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
            self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
            self.n_boundary = nb = self.x0.size
            if self.bootstrap is not None:
                self.x0 = np.r_[self.x0, self.bootstrap.x0]
                self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]

            def with_boundary(params, x):
                rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
                return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

            self.with_boundary = with_boundary
        self.hessian_chunk = int(hessian_chunk or self.chunk)
        self.require_converged = False  # LeastSquaresFreeBoundaryProblem: an unconverged trial is a failed one
        self.gauge_residual = 0.0
        self._linearization = None
        self._last_linearize = None
        self._gauge_setup("canonical")
        self._second_order_setup(self.rows, self._im._active_state_fields(self.cfg), self._im._edge_mask(self.cfg))

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature):
        """The least-squares rows, with the quadrature plan of the seed ``state`` (none in vacuum)."""
        from .freeboundary import _vacuum_scalars
        from types import SimpleNamespace

        im, cfg, fixed = self._im, self.cfg, self.fixed
        nfp = int(fixed.nfp)
        surface = _surface_data(fixed, cfg, state, self.params0, nphi, ntheta)
        quad_nt, quad_np = (None, None) if quadrature is None else map(int, quadrature)
        if self.vacuum:  # the coils' field alone: no virtual-casing quadrature to plan
            precision, digits = None, None
        else:
            precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=quad_nt, quad_np=quad_np)
        self.precision = precision
        _, _, rmnc, zmns = map(np.asarray, vc._state_geometry_spectra(state, cfg.resolution))
        n_axis = int(cfg.resolution.ntor) + 1
        axis_points, axis_tangent, axis_dphi = _axis_loop(
            SimpleNamespace(raxis_cc=rmnc[0, :n_axis], zaxis_cs=zmns[0, :n_axis], nfp=nfp))
        area = np.linalg.norm(np.asarray(surface.area_vector), axis=0)
        self.B0 = float(np.sqrt(np.sum(area * np.sum(np.asarray(surface.B_total) ** 2, axis=0)) / np.sum(area)))
        # The run's constants enter the compiled rows as arguments (_Bundle), not as closed-over arrays.
        self._consts = dict(axis=(axis_points, axis_tangent, jnp.asarray(axis_dphi)), B0=jnp.asarray(self.B0))
        vacuum = self.vacuum

        def interface_rows(state, params, field, moments=False):
            """``sqrt(area)`` times the interface conditions over ``|B|`` on the grid, and the net-current row."""
            field, consts = (field.field, field.consts) if isinstance(field, _Bundle) else (field, self._consts)
            runtime = im.runtime_from_params(params, cfg)
            G_coil = jnp.abs(_net_current(field, *consts["axis"]))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            current = ((G - G_coil) / G_coil)[None]
            if vacuum:  # B_out = B_coil exactly: B_coil . n = 0 is the whole condition
                data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
                area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
                B = vc.external_B_cartesian(field, jnp.asarray(data.gamma), jnp.asarray(data.phi))
                bn = jnp.sum(B * jnp.asarray(data.normal), axis=0) / consts["B0"]
                return jnp.concatenate([(jnp.sqrt(area / jnp.sum(area)) * bn).ravel(), current])
            values, area, _ = _interface_terms_banded(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                      digits=digits, precision=precision, p_edge=0.0,
                                                      band_limit=True)
            return jnp.concatenate([(jnp.sqrt(area)[None] * values).ravel(), current])

        self.interface_rows = interface_rows
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field.field if isinstance(field, _Bundle) else field,
            runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])

    def evaluate(self, x, params, field, seed=None, tight=False):
        """:meth:`ThreeTermsFreeBoundaryModel.evaluate` with the run's constants bundled with ``field``."""
        return super().evaluate(x, params, self._bundle(field), seed, tight)

    def linearize(self, x, aux, field, extra=None):
        """:meth:`ThreeTermsFreeBoundaryModel.linearize` with the run's constants bundled with ``field``."""
        return super().linearize(x, aux, self._bundle(field), extra)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`~vmex.core.freeboundary_vc.BoundaryResidual` of a solved state (the common evaluator)."""
        return summarize_boundary_residual(*self._terms(state, params, field))

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=None, ftol=1e-10, xtol=1e-9,
                       max_iter=40, callback=None, verbose=0):
        """The free boundary for ``params`` and ``field``, from ``x0`` (default: the deck's boundary).

        :meth:`_newton`; ``max_nfev`` (the Galerkin solve's name) caps its models like ``max_iter``, and
        ``jacobian`` (a Galerkin warm start) is not used.
        """
        del jacobian
        out = self._newton(params, field, x0=x0, ftol=ftol, xtol=xtol,
                           max_iter=max_iter if max_nfev is None else int(max_nfev), callback=callback,
                           verbose=verbose)
        if self.require_converged and not out["converged"]:  # an optimizer's trial: no answer, not a value
            raise RuntimeError(f"the least-squares free boundary did not converge ({out['status']})")
        return out

    def _gauge_setup(self, gauge):
        """Equality constraints ``c(x) = 0`` that fix the boundary's poloidal labelling (``gauge``).

        A boundary's Fourier coefficients describe a surface and an angle on it; the interface conditions see
        the angle only through the truncation, too weakly to fix it (the exact Hessian of the cost is
        indefinite along relabellings).  ``gauge`` fixes it by a canonical condition on the boundary alone,
        one equation per relabelling mode ``theta -> theta + omega``, ``omega = sum sin(m theta - n nfp phi)``
        over the boundary modes (``m = 0``: ``n >= 1``), independent of any reference surface or start:

        - ``"canonical"``: reference-free, defined by the boundary itself.  For ``m > 0`` relabellings the
          spectral width ``F_6 = sum m^6 (R_mn^2 + Z_mn^2)`` is stationary; for the ``m = 0`` shifts of the
          angle origin the twist ``T = sum (n nfp)^2 (R_mn^2 + Z_mn^2)`` is stationary: two functionals on two
          disjoint blocks of relabellings, no relative weight.  Stationarity is that of the relabelled,
          truncated boundary's coefficients (a minimum exists at any resolution).  ``m^6`` keeps the relabelled
          surface
          on the original one (LP QA: 0.05 mm at 8 modes, 0.001 mm at 12; ``m^4`` moves it by 1 mm at 8);
        The model uses ``"canonical"``; ``None`` (no gauge) is the fixed-boundary model's.
        """
        self.gauge = gauge
        self.gauge_constraints = None
        if gauge is None or self.fixed_boundary:
            return
        cache = self.__dict__.setdefault("_gauge_sets", {})
        key = gauge
        if key in cache:  # a gauge used before (two-phase solves): its compiled functions, not new ones
            self.__dict__.update(cache[key])
            return
        if gauge != "canonical":
            raise ValueError(f"unknown gauge {gauge!r}")
        from .optimize import _dof_modes, boundary_arrays_from_x

        fixed, nb = self.fixed, self.n_boundary
        nfp = int(fixed.nfp)
        rbc0 = np.asarray(fixed.rbc)
        ntor_b, mpol_b = (rbc0.shape[0] - 1) // 2, rbc0.shape[1]
        m_all = np.arange(mpol_b)[None, :]
        n_all = np.arange(-ntor_b, ntor_b + 1)[:, None]
        mm = jnp.asarray(np.broadcast_to(m_all, rbc0.shape).astype(float))
        nn = jnp.asarray(np.broadcast_to(n_all, rbc0.shape).astype(float)) * nfp
        # every odd relabelling: sin(m theta - n nfp phi), m >= 1, and the phi-dependent shift of the angle origin
        # sin(-n nfp phi), m = 0, n >= 1 (it maps R(1, 0) cos(theta) onto R(1, +-n): a near-null direction)
        gm = list(_dof_modes(fixed, self.max_mode))
        g_rows = np.array([n + ntor_b for _, n in gm]); g_cols = np.array([m for m, _ in gm])
        is0 = jnp.asarray(g_cols == 0)

        def grid(nt, nz):
            """Separable Fourier factors on an ``nt x nz`` grid of one field period: (cos, sin)(m theta), (n nfp phi)."""
            theta = 2 * np.pi * np.arange(nt) / nt
            zeta = 2 * np.pi * np.arange(nz) / (nz * nfp)
            m, n = np.arange(mpol_b)[:, None], np.arange(-ntor_b, ntor_b + 1)[:, None] * nfp
            return (jnp.asarray(theta), *(jnp.asarray(f) for f in (np.cos(m * theta), np.sin(m * theta),
                                                                     np.cos(n * zeta), np.sin(n * zeta))))

        def series(c, cos_m, sin_m, cos_n, sin_n, odd):
            """``sum c_nm cos(m theta - n nfp phi)`` (``odd``: sin) on the grid; ``cos_m``/``sin_m`` may be (m, t, z)."""
            A, B = jnp.einsum("nm,nz->mz", c, cos_n), jnp.einsum("nm,nz->mz", c, sin_n)
            spec = "mt,mz->tz" if cos_m.ndim == 2 else "mtz,mz->tz"
            if odd:
                return jnp.einsum(spec, sin_m, A) - jnp.einsum(spec, cos_m, B)
            return jnp.einsum(spec, cos_m, A) + jnp.einsum(spec, sin_m, B)

        def moments(q, cos_m, sin_m, cos_n, sin_n, odd):
            """``mean q * (cos | sin)(m theta - n nfp phi)``, an (n, m) array."""
            size = q.size
            if odd:
                return (jnp.einsum("tz,mt,nz->nm", q, sin_m, cos_n) - jnp.einsum("tz,mt,nz->nm", q, cos_m, sin_n)) / size
            return (jnp.einsum("tz,mt,nz->nm", q, cos_m, cos_n) + jnp.einsum("tz,mt,nz->nm", q, sin_m, sin_n)) / size

        weights = (mm**6, nn**2)  # F_6 (m > 0 block) and the twist T (m = 0 block)

        def functionals(x):
            rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
            return tuple(jnp.sum(w * (rbc**2 + zbs**2)) for w in weights)

        # Relabelling theta -> theta + omega of the boundary x, omega = sum_k w_k sin(m_k theta - n_k nfp phi):
        # the boundary sampled at the shifted angles and projected back onto the modes (the truncation).
        modes = _dof_modes(fixed, self.max_mode)
        rows_i = np.array([n + ntor_b for _, n in modes]); cols_i = np.array([m for m, _ in modes])
        theta_f, *fine = grid(2 * (4 * mpol_b + 8), 2 * (4 * ntor_b + 8))
        m_col = jnp.arange(mpol_b, dtype=float)[:, None, None]
        norm = jnp.asarray(np.where(np.broadcast_to(m_all, rbc0.shape) == 0, 1.0, 2.0))

        def relabel(x, w):
            rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
            omega = series(jnp.zeros(rbc0.shape).at[g_rows, g_cols].set(w), *fine, odd=True)
            shifted = theta_f[:, None] + omega
            cos_s, sin_s = jnp.cos(m_col * shifted[None]), jnp.sin(m_col * shifted[None])
            R = series(rbc, cos_s, sin_s, *fine[2:], odd=False)
            Z = series(zbs, cos_s, sin_s, *fine[2:], odd=True)
            rbc_new = norm * moments(R, *fine, odd=False)
            zbs_new = norm * moments(Z, *fine, odd=True)
            # m = 0: the n and -n columns alias onto one coefficient (cos(-n nfp phi) = cos(n nfp phi))
            alias = jnp.where(jnp.asarray(cols_i) == 0, 2.0, 1.0)  # m = 0, n >= 1 (n = 0 is RBC(0,0), apart)
            x_new = jnp.concatenate([rbc_new[rows_i, cols_i] * alias, zbs_new[rows_i, cols_i] * alias,
                                     rbc_new[ntor_b, 0][None]])
            return jnp.concatenate([x_new, x[x_new.size:]])

        self._relabel = jax.jit(relabel)

        coarse = grid(4 * mpol_b + 8, 4 * ntor_b + 8)[1:]

        def constraints(x):
            """``(L R) R_theta + (L Z) Z_theta`` against every relabelling, ``L`` multiplying a harmonic by its
            functional's weight (``m^6`` for m > 0 relabellings, ``(n nfp)^2`` for the m = 0 shifts), each block
            over its functional: the first variation of ``F_6`` / ``T`` under the relabelling."""
            rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
            R_t = -series(rbc * mm, *coarse, odd=True)
            Z_t = series(zbs * mm, *coarse, odd=False)
            blocks = []
            for w in weights:
                q = R_t * series(rbc * w, *coarse, odd=False) + Z_t * series(zbs * w, *coarse, odd=True)
                blocks.append(moments(q, *coarse, odd=True)[g_rows, g_cols]
                              / (jnp.sum(w * mm * (rbc**2 + zbs**2)) + 1e-300))
            return jnp.where(is0, blocks[1], blocks[0])

        block0 = np.asarray(is0)

        def block_newton(x, w, k):
            """Gradient and Hessian of functional ``k`` of the relabelled boundary in its own block of omega."""
            mask = jnp.asarray(block0 if k else ~block0, dtype=float)

            def f(v):
                return functionals(relabel(x, w + mask * v))[k]

            z = jnp.zeros_like(w)
            return jax.grad(f)(z) * mask, jax.hessian(f)(z) * mask[:, None] * mask[None], f(z)

        self._block_newton = jax.jit(block_newton, static_argnums=2)
        self._block_value = jax.jit(lambda x, w, k: functionals(relabel(x, w))[k], static_argnums=2)
        self._gauge_blocks = block0
        self._relabel_residual = jax.jit(lambda x, w: constraints(relabel(x, w)))
        self._relabel_jacobian = jax.jit(jax.jacfwd(lambda x, w: constraints(relabel(x, w)), argnums=1))
        self.n_gauge = len(gm)
        self.gauge_constraints = jax.jit(constraints)
        self._gauge_jacobian = jax.jit(jax.jacfwd(constraints))
        names = ("gauge_constraints", "_gauge_jacobian", "_gauge_hessian", "_relabel", "_relabel_residual",
                 "_relabel_jacobian", "_block_newton", "_block_value", "_gauge_blocks", "n_gauge")
        self._gauge_hessian = jax.jit(lambda x, nu: jax.hessian(lambda z: jnp.dot(nu, constraints(z)))(x))
        cache[key] = {name: self.__dict__[name] for name in names if name in self.__dict__}

    def gauge_restore(self, x, tol=1e-12, max_iter=30):
        """The boundary ``x`` relabelled onto ``c = 0`` (the surface is kept up to the truncation of the
        relabelled boundary to the modes), by Levenberg-Marquardt on ``|c|^2`` over the relabellings; where
        ``c = 0`` is out of reach, the relabelling of least ``|c|`` (``gauge_residual`` records it)."""
        if self.gauge_constraints is None:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        zero = jnp.zeros(self.n_gauge)

        def levenberg_marquardt(residual, jacobian, step, w):
            """Levenberg-Marquardt on ``|residual(w)|^2`` with ``w <- step(w, dw)``; returns the last ``w`` and
            residual (stops at ``tol`` or when no damping decreases it)."""
            c = residual(w)
            mu = None
            for _ in range(8 * max_iter):
                if np.max(np.abs(c)) <= tol:
                    break
                A = jacobian(w)
                AtA, Atc = A.T @ A, A.T @ c
                scale = np.max(np.diag(AtA))
                mu = 1e-3 * scale if mu is None else mu  # the textbook start, adapted below
                while mu <= 1e12 * scale:
                    w_try = step(w, -np.linalg.solve(AtA + mu * np.eye(c.size), Atc))
                    c_try = residual(w_try)
                    if np.linalg.norm(c_try) < np.linalg.norm(c):
                        w, c, mu = w_try, c_try, mu / 3.0
                        break
                    mu *= 10.0
                else:  # no relabelling decreases |c|
                    break
            return w, c

        # One relabelling of x itself first (exact when the truncation lets c vanish on x's orbit), then, from
        # its best point, relabellings of the current boundary (relabelling a truncated boundary twice is not
        # one relabelling: at low resolution only these reach c = 0).
        xj = jnp.asarray(x)
        w = np.zeros(self.n_gauge)
        # Start: alternate damped-Newton minimizations of F_6 over the m > 0 relabellings and of T over the m = 0
        # shifts (each block minimizes its own functional, which is what c = 0 states to first order)
        for _ in range(8):
            moved = 0.0
            for k in (0, 1):
                active = self._gauge_blocks if k else ~self._gauge_blocks
                for _ in range(20):
                    g, H, f = map(np.asarray, self._block_newton(xj, jnp.asarray(w), k))
                    g, H = g[active], H[np.ix_(active, active)]
                    if not g.size or np.linalg.norm(g) <= 1e-13 * max(abs(float(f)), 1e-300):
                        break
                    lam, dw = 0.0, None
                    for _ in range(30):  # damping (x10 a try) until the functional decreases
                        if lam or np.all(np.linalg.eigvalsh(H) > 0):
                            trial = np.linalg.solve(H + lam * np.eye(g.size), -g)
                            w_try = w.copy()
                            w_try[active] += trial
                            if float(self._block_value(xj, jnp.asarray(w_try), k)) < float(f):
                                dw = trial
                                break
                        lam = max(10 * lam, 1e-6 * np.max(np.abs(np.diag(H))) + 1e-300)
                    if dw is None:
                        break
                    w[active] += dw
                    moved = max(moved, float(np.max(np.abs(dw))))
            if moved < 1e-12:
                break
        best = None
        for w0 in (w, np.zeros(self.n_gauge)):  # from the block minimum, and failing that from x itself
            w1, c = levenberg_marquardt(lambda w: np.asarray(self._relabel_residual(xj, jnp.asarray(w))),
                                        lambda w: np.asarray(self._relabel_jacobian(xj, jnp.asarray(w))),
                                        lambda w, dw: w + dw, w0)
            x1 = np.asarray(self._relabel(xj, jnp.asarray(w1)))
            if np.max(np.abs(c)) > tol:
                x1, c = levenberg_marquardt(lambda z: np.asarray(self.gauge_constraints(jnp.asarray(z))),
                                            lambda z: np.asarray(self._relabel_jacobian(jnp.asarray(z), zero)),
                                            lambda z, dw: np.asarray(self._relabel(jnp.asarray(z), jnp.asarray(dw))),
                                            x1)
            if best is None or np.max(np.abs(c)) < np.max(np.abs(best[1])):
                best = (x1, c)
            if np.max(np.abs(c)) <= 1e3 * tol:
                break
        x, c = best
        # Relabelling a truncated boundary is not exact: where no relabelling reaches c = 0 (a floor of ~1e-4 on
        # some strongly shaped boundaries) the gauge is the relabelling of least |c|, and the floor is recorded.
        self.gauge_residual = float(np.max(np.abs(c)))
        return x

    def _second_order_setup(self, rows, active, edge):
        """Exact second derivatives of the cost ``1/2 |rows|^2`` through the equilibrium.

        With ``F(z, p) = 0`` the raw force balance of the projected state ``z`` and ``Phi(z, p) = 1/2 |rows|^2``,
        the multiplier ``lam`` solves ``F_z^T lam = Phi_z`` (one transposed block solve) and the Lagrangian is
        ``L = Phi - lam . F``.  The gradient of the reduced cost ``f(p) = Phi(z(p), p)`` is ``L_p``; its Hessian
        along tangents ``(dz_i, dp_i)`` (``F_z dz_i = -F_p dp_i``) is ``H_ij = [dp_i; dz_i]^T grad^2 L [dp_j;
        dz_j]``, one forward-over-reverse pass of ``L`` per column.  This includes ``sum_k r_k grad^2 r_k``, the
        term Gauss-Newton drops, which is not small here: the residual stays finite at a truncated boundary.
        """
        im, cfg = self._im, self.cfg

        def tree_dot(a, b):
            return sum(jnp.sum(u * v) for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

        def lagrange_parts(state, mask, field):
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)
            F = im.residual_fn(cfg, frozen, mask, formulation="raw")

            def phi(z, prm):
                r = rows(im._assemble(z, im.runtime_from_params(prm, cfg), frozen, P, edge), prm, field)
                return 0.5 * jnp.sum(r * r)

            return frozen, P, F, P(frozen), phi

        def multiplier(state, mask, params, field):
            """``(lam, L_p, adjoint residual / tolerance)``."""
            frozen, P, F, z, phi = lagrange_parts(state, mask, field)
            system = im._raw_block_system(params, cfg, frozen, mask, active, self.chunk)
            phi_z, phi_p = jax.grad(phi, argnums=(0, 1))(z, params)
            _, vjp_z = jax.vjp(lambda zz: F(zz, params), z)
            _, vjp_p = jax.vjp(lambda prm: F(z, prm), params)
            lam, residual, tolerance = im._block_adjoint(system, lambda v: vjp_z(v)[0], P(phi_z), cfg)
            return lam, jax.tree.map(jnp.subtract, phi_p, vjp_p(lam)[0]), residual / tolerance

        def hessian_columns(state, mask, params, field, lam, dz_cols, dp_cols, dz_basis, dp_basis):
            """``[dp_i; dz_i]^T grad^2 L [dp_j; dz_j]``: a row per column ``j``, a column per basis vector ``i``."""
            _, P, F, z, phi = lagrange_parts(state, mask, field)

            def lagrangian(zz, prm):
                return phi(zz, prm) - tree_dot(lam, F(zz, prm))

            grad = jax.grad(lagrangian, argnums=(0, 1))
            basis_z = jax.vmap(P)(dz_basis)

            def column(a):
                wz, wp = jax.jvp(grad, (z, params), (P(a[0]), a[1]))[1]
                return (jax.vmap(lambda d: tree_dot(d, wz))(basis_z)
                        + jax.vmap(lambda d: tree_dot(d, wp))(dp_basis))

            return jax.lax.map(column, (dz_cols, dp_cols), batch_size=self.hessian_chunk)

        def project(dp_basis, tree):
            return jax.vmap(lambda d: tree_dot(d, tree))(dp_basis)

        self._tree_dot = tree_dot
        self._lagrange_parts = lagrange_parts
        self._multiplier = jax.jit(multiplier)
        self._hessian_columns = jax.jit(hessian_columns)
        self._project = jax.jit(project)
        self._last_second_order = None

    def second_order(self, x, aux, field, extra=None, columns=True):
        """Exact gradient and Hessian of ``1/2 |rows|^2`` in the model coordinates at ``x``.

        ``columns=False`` skips the Hessian's columns: the dict then has ``hvp`` (``v -> H v`` in the model
        coordinates, one forward-over-reverse pass each) instead of ``H`` and ``C``.

        Returns a dict with ``H`` (``n x n``, symmetrized; ``asymmetry`` its relative antisymmetric part),
        ``g`` (``J^T r``), ``g_adjoint`` (the same gradient from the multiplier: a consistency check), ``J``,
        ``r``, ``aux`` (the tight state), ``lam``, ``dz``/``batch`` (tangents and parameter directions of the
        columns, then of ``extra``), ``C`` (``n x n_extra``: the mixed second derivatives along ``extra``) and
        ``adjoint`` (multiplier residual over its tolerance).
        """
        field = self._bundle(field)
        key = _fingerprint(x, aux[2], field) + ("" if extra is None else _fingerprint(extra)) + str(columns)
        last = self._last_second_order
        if last is not None and last[0] == key:
            return last[1]
        t0 = time.perf_counter()
        J, dz, batch, aux = self.linearize(x, aux, field, extra=extra)
        t1 = time.perf_counter()
        state, mask, params_x, _ = aux
        n = self.x0.size
        with vc.remat_scans():
            lam, grad_p, adjoint = self._multiplier(state, mask, params_x, field)
        dz_b, dp_b = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
        r = np.asarray(self._rows(state, params_x, field))
        if not columns:
            def hvp(v):
                """``H v`` for every row ``v`` of ``(b, n)`` (one program call; ``b`` the block width)."""
                vj = jnp.asarray(np.atleast_2d(v))
                one = lambda a: jnp.tensordot(vj, a, axes=1)  # noqa: E731
                with vc.remat_scans():
                    return np.asarray(self._hessian_columns(state, mask, params_x, field, lam, jax.tree.map(one, dz_b),
                                                            jax.tree.map(one, dp_b), dz_b, dp_b))

            out = dict(hvp=hvp, g=J[:, :n].T @ r, g_adjoint=np.asarray(self._project(dp_b, grad_p)), J=J, r=r,
                       aux=aux, lam=lam, dz=dz, batch=batch, asymmetry=np.nan, adjoint=float(adjoint),
                       seconds=dict(jacobian=t1 - t0, multiplier=time.perf_counter() - t1))
            self._last_second_order = (key, out)
            return out
        with vc.remat_scans():  # traced (first call) with checkpointed quadrature chunks: bounded memory
            t2 = time.perf_counter()
            cols = np.asarray(self._hessian_columns(state, mask, params_x, field, lam, dz, batch, dz_b, dp_b))
            t3 = time.perf_counter()
        H, C = cols[:n].T, cols[n:].T
        asymmetry = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
        H = 0.5 * (H + H.T)
        out = dict(H=H, C=C, g=J[:, :n].T @ r, g_adjoint=np.asarray(self._project(dp_b, grad_p)), J=J, r=r,
                   aux=aux, lam=lam, dz=dz, batch=batch, asymmetry=asymmetry, adjoint=float(adjoint),
                   seconds=dict(jacobian=t1 - t0, multiplier=t2 - t1, hessian=t3 - t2))
        self._last_second_order = (key, out)
        return out

    def gradient(self, x, aux, field):
        """Exact gradient ``J^T r`` of ``1/2 |rows|^2`` at a tight state by one adjoint solve (no tangents)."""
        field = self._bundle(field)
        state, mask, params_x, _ = aux
        with vc.remat_scans():
            _, grad_p, _ = self._multiplier(state, mask, params_x, field)
        return np.asarray(self._project(self.boundary_directions(params_x, x), grad_p))

    def _krylov_model(self, so, N, GN, Hg, S, gs, gq_norm_first):
        """Preconditioned block-Krylov subspace of the exact reduced Hessian; sets ``self._krylov_T``.

        Block Lanczos with ``hessian_chunk`` vectors a block (one program call computes a block's Hessian
        products), starting from the preconditioned gradient and fixed pseudo-random directions; the subspace
        grows until the Newton step on it has a residual below ``eta |g|`` (``eta = min(1/2, sqrt(|g| /
        |g_first|))``, the Eisenstat-Walker choice for superlinear convergence).  Returns the subspace basis in the
        scaled model coordinates (``n x k``, in place of ``N``; orthonormal, so radii mean the same for every
        model) and the forcing term's scale.  ``Hg`` is the scaled gauge-constraint Hessian (``0`` without one).
        """
        Nr = N
        wg, Qg = np.linalg.eigh(Nr.T @ GN @ Nr)
        wg = np.maximum(wg, np.finfo(float).eps * wg[-1])  # J^T J is positive semidefinite: round-off floor
        Mih = (Qg / np.sqrt(wg)) @ Qg.T

        def op(V):  # V: (n_r, b) in the preconditioned reduced coordinates
            steps = Nr @ (Mih @ V)  # (n, b) scaled
            HV = (S[:, None] * so["hvp"]((S[:, None] * steps).T).T) + (Hg @ steps if np.ndim(Hg) else 0.0)
            return Mih @ (Nr.T @ HV)

        g = Mih @ (Nr.T @ gs)
        g_norm = float(np.linalg.norm(g))
        gq_norm_first = g_norm if gq_norm_first is None else gq_norm_first
        eta = min(0.5, np.sqrt(g_norm / max(gq_norm_first, 1e-300)))
        width = max(1, min(self.hessian_chunk, g.size))
        start = np.column_stack([g, np.random.default_rng(0).standard_normal((g.size, width - 1))])
        V = np.linalg.qr(start)[0]
        K, HK = np.zeros((g.size, 0)), np.zeros((g.size, 0))
        while True:
            W = op(V)
            K, HK = np.hstack([K, V]), np.hstack([HK, W])
            T = 0.5 * (K.T @ HK + HK.T @ K)
            ev = np.linalg.eigvalsh(T)
            if ev[0] > 0:  # the Newton step on the subspace and its true residual H K y - g
                y = np.linalg.solve(T, K.T @ g)
                if np.linalg.norm(HK @ y - g) <= eta * g_norm:
                    break
            if K.shape[1] >= g.size:
                break
            W = W - K @ (K.T @ W)
            W = W - K @ (K.T @ W)  # full reorthogonalization (the subspace stays small)
            Q, R = np.linalg.qr(W)
            keep = np.abs(np.diag(R)) > np.finfo(float).eps * max(np.abs(ev).max(), 1e-300) * 1e3
            if not np.any(keep):
                break
            V = Q[:, keep][:, :g.size - K.shape[1]]
        Qb, Rb = np.linalg.qr(Mih @ K)
        Rinv = np.linalg.inv(Rb)
        self._krylov_T = Rinv.T @ T @ Rinv
        return Nr @ Qb, gq_norm_first

    def _newton(self, params, field, *, x0=None, ftol=1e-10, xtol=1e-9, max_iter=40, radius=None,
                              callback=None, verbose=0):
        """Minimize ``1/2 |rows|^2`` over the boundary by a trust-region (quasi-)Newton method.

        Every iterate is a tight equilibrium.  With a ``gauge`` the iterates stay on its constraint manifold
        ``c(x) = 0`` (each trial is relabelled back onto it) and the step solves the trust-region subproblem of
        the Lagrangian Hessian ``H + sum nu_k grad^2 c_k`` in the null space of ``grad c`` (``nu`` the
        least-squares multipliers), exactly, in the eigenbasis of the ESS-scaled reduced model (indefinite ones
        included).  Models, cheapest first: Gauss-Newton ``J^T J`` from one Jacobian, then BFGS updates with
        the exact adjoint gradient (one block factorization a step); where a quasi-Newton model fails (a step
        gaining under a quarter of its prediction, or none) a fresh Gauss-Newton model, and only where that
        fails too the exact Hessian (second-order adjoint, :meth:`second_order`), which also certifies
        convergence.  A step that is not very successful is followed by a
        second-order correction ``-(J N)^+ r(x_trial)``; a very successful step at the radius is extended
        along the same model while it gains.

        Converged: at an exact, positive-definite reduced Hessian, the Newton step's predicted gain is below
        ``ftol`` times the cost or below the cost's repeatability -- measured at every exact model by solving
        the same boundary again from another state (FTOL 1e-17 equilibria repeat to 3e-11..3e-10 of the cost
        at 8 x 8 and to ~3e-6 at 12 x 12, finite beta) --, or it moves the boundary by less than ``xtol R00``,
        or no step gains although the exact model predicts one (the noise floor).  Anything else is reported as a stall
        (``converged=False``).  Returns a dict with ``x``, ``rows``, ``aux``, ``jacobian``, ``hessian``,
        ``nfev``, ``njev``, ``converged``, ``status`` and ``history`` (a row per model: cost, reduced gradient
        norm, smallest and largest eigenvalue of the reduced model and of ``J^T J``, Newton decrement, gap
        between ``J^T r`` and the adjoint gradient and Hessian asymmetry (exact models), seconds).
        ``callback(x)``, if given, is called with every accepted boundary (e.g. to checkpoint a long solve).
        """
        t_restore = time.perf_counter()
        x = self.gauge_restore(self.x0 if x0 is None else x0)
        t_eval = time.perf_counter()
        got = self.evaluate(x, params, field, tight=True)
        if verbose:
            print(f"setup {', '.join(f'{k} {v:.1f}' for k, v in getattr(self, 'setup_seconds', {}).items())} s; "
                  f"gauge restoration {t_eval - t_restore:.1f} s, first evaluation {time.perf_counter() - t_eval:.1f} s",
                  flush=True)
        if got is None:
            raise RuntimeError("the starting boundary has no certified equilibrium")
        r, aux = got
        S = np.asarray(self.x_scale, dtype=float)
        nfev, history, converged, status, started = 1, [], False, "max_iter", time.perf_counter()
        so = None
        H_full = gs_prev = x_prev = None
        need = "gauss_newton"  # the next model: "gauss_newton" (J^T J), "exact" (second order) or None (BFGS)
        noise, state_prev = 0.0, aux[0]  # the cost's measured repeatability; a state of another boundary
        gq_norm_first = None  # the first exact model's preconditioned gradient norm (the forcing term's scale)
        for it in range(int(max_iter)):
            if need is not None or H_full is None:
                if need == "exact":  # the exact Hessian as products (Lanczos below), J^T J under the next BFGS
                    so = self.second_order(x, aux, field, columns=False)
                    H_full = None
                else:  # the Jacobian alone: Gauss-Newton, the start of the quasi-Newton updates
                    J, _, _, aux = self.linearize(x, aux, field)
                    r_lin = np.asarray(self._rows(aux[0], aux[2], self._bundle(field)))
                    so = dict(J=J, r=r_lin, g=J[:, :x.size].T @ r_lin, aux=aux, asymmetry=np.nan, adjoint=np.nan,
                              g_adjoint=J[:, :x.size].T @ r_lin, seconds={})
                    H_full = None
                r, aux = so["r"], so["aux"]
                Js = so["J"][:, :x.size] * S[None]
                GN = Js.T @ Js
                if H_full is None:
                    H_full = GN
                gs = S * so["g"]
                model, need = ("exact" if need == "exact" else "gauss_newton"), None
            else:  # quasi-Newton: BFGS updates of the last model with the exact (adjoint) gradient
                gs = S * self.gradient(x, aux, field)
                step_s, y = (x - x_prev) / S, gs - gs_prev
                sy = float(step_s @ y)
                if sy > 1e-12 * np.linalg.norm(step_s) * np.linalg.norm(y):  # curvature condition (round-off)
                    Bs = H_full @ step_s
                    H_full = H_full - np.outer(Bs, Bs) / float(step_s @ Bs) + np.outer(y, y) / sy
                model = "quasi-Newton"
            fresh = model == "exact"
            if fresh:  # the cost's repeatability here: the same boundary re-solved from another seed
                again = self.evaluate(x, params, field, seed=state_prev, tight=True)
                nfev += 1
                if again is not None:
                    noise = max(noise, abs(0.5 * again[0] @ again[0] - 0.5 * r @ r))
            Hs, Hg = H_full, 0.0
            gs_prev, x_prev = gs, x.copy()
            cost = 0.5 * r @ r
            if self.gauge_constraints is not None:
                xj = jnp.asarray(x)
                A = np.asarray(self._gauge_jacobian(xj)) * S[None]
                nu = -np.linalg.lstsq(A.T, gs, rcond=None)[0]
                Hg = S[:, None] * np.asarray(self._gauge_hessian(xj, jnp.asarray(nu))) * S[None]
                Hs = Hs + Hg
                _, sv, Vt = np.linalg.svd(A)
                rank = int(np.sum(sv > 1e-10 * sv[0]))  # the constraints' rank, at round-off
                N = Vt[rank:].T
            else:
                N = np.eye(x.size)
            N_full = N
            if fresh:
                # The exact model on a Krylov subspace: preconditioned Lanczos on the reduced Hessian (one product
                # H v a vector), preconditioner the reduced Gauss-Newton matrix.  The subspace grows until the
                # Newton step's residual is below eta |g| (forcing eta = min(1/2, sqrt(|g| / |g_first|)), the
                # Eisenstat-Walker choice for superlinear convergence); the trust-region steps below are then
                # exact in it.  The eigenpairs of the tridiagonal certify convergence on the subspace.
                N, gq_norm_first = self._krylov_model(so, N, GN, Hg, S, gs, gq_norm_first)
            Hr, gr = N.T @ Hs @ N if not fresh else self._krylov_T, N.T @ gs
            w, Q = np.linalg.eigh(Hr)
            gq = Q.T @ gr
            w_gn = np.linalg.eigvalsh(N.T @ GN @ N)
            decrement = float(np.sum(gq**2 / w)) if w[0] > 0 else np.inf
            # geometric test: the Newton step moves no boundary point by more than xtol * R00 (the sum of the
            # coefficient changes bounds the displacement); below it the cost only sees the solves' noise
            moved = (float(np.sum(np.abs((S * (N @ (Q @ (-gq / w))))[:self.n_boundary])))
                     if w[0] > 0 and self.n_boundary else np.inf)  # no boundary (fixed_boundary): no geometric test
            history.append([cost, float(np.linalg.norm(gr)), w[0], w[-1], w_gn[0], w_gn[-1], decrement,
                            float(np.linalg.norm(so["g"] - so["g_adjoint"]) / max(np.linalg.norm(so["g"]), 1e-300))
                            if fresh else np.nan, so["asymmetry"] if fresh else np.nan, time.perf_counter() - started])
            if verbose:
                print(f"{model} {it}: cost {cost:.10e} |g_r| {history[-1][1]:.2e} "
                      f"eig(H_r) [{w[0]:.2e}, {w[-1]:.2e}] "
                      f"eig(JtJ_r) [{w_gn[0]:.2e}, {w_gn[-1]:.2e}] decrement/cost {decrement / (2 * cost):.2e} "
                      + (f"g-g_adj {history[-1][7]:.1e} asym {so['asymmetry']:.1e} adj {so['adjoint']:.1e} " if fresh else "")
                      + f"Newton step moves <= {moved:.1e} m "
                      + ("times " + " ".join(f"{k} {v:.1f}" for k, v in so["seconds"].items()) + " s " if fresh else "")
                      + f"({history[-1][-1]:.0f} s)", flush=True)
            if verbose > 1:  # the weakest direction of the reduced Hessian, in boundary modes
                from .optimize import _dof_modes
                labels = [f"{f}({m},{n})" for f in ("R", "Z") for m, n in _dof_modes(self.fixed, self.max_mode)] + ["R(0,0)"]
                v = S * (N @ Q[:, 0])
                v = v[:len(labels)] / np.max(np.abs(v[:len(labels)]))
                top = np.argsort(-np.abs(v))[:6]
                print("  weakest:", " ".join(f"{labels[k]}:{v[k]:+.2f}" for k in top), flush=True)
            floor = max(ftol * cost, noise)  # gains below it are not resolved by the equilibria
            if 0.5 * decrement <= floor or moved <= xtol * abs(float(x[max(self.n_boundary - 1, 0)])):
                if not fresh:  # certify with the exact Hessian
                    need = "exact"
                    continue
                converged = True
                status = ("converged" if 0.5 * decrement <= ftol * cost else
                          "converged: gain below the cost's repeatability" if 0.5 * decrement <= floor else
                          "converged: step below xtol")
                break
            newton = np.linalg.norm(gq / w) if w[0] > 0 else np.inf
            if radius is None:
                radius = min(newton, 0.1 * np.linalg.norm(x / S)) if np.isfinite(newton) else 1e-3
            if np.isfinite(newton):
                radius = max(radius, 1e-3 * newton)

            def model_step(radius):
                """The trust-region step of the reduced model for ``radius``, and its predicted gain."""
                if newton <= radius:
                    z = -gq / w
                else:  # |(w + mu)^-1 gq| = radius, mu > max(0, -w_min): bisection in log mu
                    shift = max(0.0, -w[0])
                    lo = np.log(1e-300 + 1e-14 * max(abs(w[-1]), 1e-300))
                    hi = np.log(np.linalg.norm(gq) / radius + abs(w[-1]) + 1e-300)
                    for _ in range(200):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if np.linalg.norm(gq / (w + shift + np.exp(mid))) > radius else (lo, mid)
                    z = -gq / (w + shift + np.exp(hi))
                return N @ (Q @ z), -(gq @ z + 0.5 * np.sum(w * z * z))

            def trial(x_try):
                """The relabelled trial and its rows, or ``(x_try, None)`` when it has no gauge or equilibrium."""
                try:
                    x_try = self.gauge_restore(x_try)
                except RuntimeError:  # the trial boundary has no canonical labelling nearby: a failed step
                    return x_try, None
                return x_try, self.evaluate(x_try, params, field, tight=True)

            accepted = False
            for attempt in range(12):
                step, predicted = model_step(radius)
                x_new, got = trial(x + S * step)
                nfev += 1
                cost_new = np.inf if got is None else 0.5 * got[0] @ got[0]
                rho = (cost - cost_new) / predicted if predicted > 0 else -np.inf
                length = np.linalg.norm(step)
                if verbose:
                    print(f"  step {attempt}: |s| {length:.2e} (radius {radius:.2e}, Newton {newton:.2e}) cost "
                          f"{cost_new:.10e} predicted {predicted:.2e} actual {cost - cost_new:.2e} rho {rho:.3f} "
                          f"({time.perf_counter() - started:.0f} s)", flush=True)
                if got is not None and rho <= 0.75:
                    # a step that is not "very successful" by the radius update's own test: a second-order
                    # correction from the residual at the trial point with the Jacobian at hand, x_t - (J N)^+ r_t
                    JN = so["J"][:, :x.size] @ (S[:, None] * N_full)
                    corrected, got2 = trial(x_new + S * (N_full @ -np.linalg.lstsq(JN, got[0], rcond=None)[0]))
                    nfev += 1
                    if got2 is not None and 0.5 * got2[0] @ got2[0] < cost_new:
                        x_new, got, cost_new = corrected, got2, 0.5 * got2[0] @ got2[0]
                        rho = (cost - cost_new) / predicted if predicted > 0 else -np.inf
                        if verbose:
                            print(f"  second-order correction: cost {cost_new:.10e} rho {rho:.3f}", flush=True)
                tiny = predicted <= floor  # below the cost's repeatability: accept a non-increase
                if rho < 0.25 and not tiny:
                    radius = 0.25 * length
                elif rho > 0.75 and length > 0.99 * radius:
                    radius = 2.0 * radius
                    # a very successful step at the radius: go further along the same model while it keeps
                    # gaining (each extension one equilibrium, no new derivatives) -- long valleys of the
                    # low-order shape modes at finite beta take one model instead of one per doubling
                    while newton > length:
                        step2, predicted2 = model_step(radius)
                        x_try, got_try = trial(x + S * step2)
                        nfev += 1
                        if got_try is None or 0.5 * got_try[0] @ got_try[0] >= cost_new:
                            break
                        x_new, got, cost_new, length = x_try, got_try, 0.5 * got_try[0] @ got_try[0], np.linalg.norm(step2)
                        if verbose:
                            print(f"  extended: |s| {length:.2e} cost {cost_new:.10e} predicted {predicted2:.2e}",
                                  flush=True)
                        radius = 2.0 * radius
                if rho > 0.1 or (tiny and cost_new <= cost):
                    state_prev = aux[0]
                    x, (r, aux), accepted = x_new, got, True
                    if callback is not None:
                        callback(np.asarray(x).copy())
                    if rho < 0.25 and not tiny:  # a poor model: the next better one at the new point
                        need = "exact" if model == "gauss_newton" else "gauss_newton"
                    break
                if not fresh:  # a model that fails: a fresh Jacobian (quasi-Newton), else the exact Hessian
                    need = "exact" if model == "gauss_newton" else "gauss_newton"
                    break
                if tiny:  # the exact model's gain is below the cost's repeatability: nothing left to resolve
                    break
            if not accepted and fresh:
                # no step gains although the exact model predicts one: the gain is below the cost's noise
                converged = bool(w[0] > 0)
                status = ("converged: noise floor" if converged else "stalled: no acceptable step") + \
                    f" (predicted gain {0.5 * decrement / cost:.1e} of the cost)"
                break
        if verbose:
            print(f"Newton finished: {status} after {len(history)} Hessians, {nfev} evaluations "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
        return dict(x=x, rows=r, aux=aux, jacobian=None if so is None else so["J"],
                    hessian=None if so is None else so.get("H"), nfev=nfev, njev=len(history), converged=converged,
                    status=status, accepted=len(history), history=np.asarray(history), radius=radius)

    def _bundle(self, field):
        """``field`` with the run's constants, as the compiled rows take it (see :class:`_Bundle`)."""
        if isinstance(field, _Bundle) or not hasattr(self, "_consts"):
            return field
        return _Bundle(field, self._consts)


class LeastSquaresFreeBoundaryProblem(ThreeTermsFreeBoundaryProblem):
    """:class:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryProblem` on the least-squares free boundary.

    Every trial is :meth:`LeastSquaresFreeBoundaryModel.solve_boundary`; ``b*(x)`` minimizes
    ``f(b, x) = 1/2 |r|^2`` on the gauge manifold, so a row ``h`` (loss or quantity) has the design gradient

        dh/dx = dh/dx|_b - mu^T f_bx,   (N^T W N) mu_N = N^T dh/db,   mu = N mu_N,

    with ``W`` the exact Hessian of the gauge Lagrangian and ``N`` a null-space basis of the gauge constraints
    (:meth:`_exact_gradients`).  Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition=
    "three_terms", three_terms_options=dict(method="least_squares", ...))``.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_max_nfev=40,
                  boundary_max_residual=None, **model_options):
        """Solve the seed design's free boundary (keywords as the Galerkin problem's)."""
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if plasma_from_parameters is not None:
            functions += (plasma_from_parameters,)
        if not all(map(callable, functions)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = LeastSquaresFreeBoundaryModel(inp, **model_options)
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_max_nfev=boundary_max_nfev, boundary_max_residual=boundary_max_residual)

    def __init__(self, model, loss, x0, **options):
        from .implicit import _assemble, _dof_projector, _edge_mask, runtime_from_params

        field_from_parameters = options["field_from_parameters"]
        model.require_converged = True
        edge, cfg = _edge_mask(model.cfg), model.cfg

        def coil_mixed(state, mask, params, x, dz_mu, dp_mu):
            """``d/d eps grad_x 1/2 |rows(field(x))|^2`` along the boundary response ``(dz_mu, dp_mu)``."""
            frozen = jax.lax.stop_gradient(state)
            P = _dof_projector(cfg, mask)

            def grad_x(z, prm):
                def cost(xx):
                    r = model.rows(_assemble(z, runtime_from_params(prm, cfg), frozen, P, edge), prm,
                                   field_from_parameters(xx))
                    return 0.5 * jnp.sum(r * r)
                return jax.grad(cost)(x)

            return jax.jvp(grad_x, (P(frozen), params), (P(dz_mu), dp_mu))[1]

        self._coil_mixed = jax.jit(coil_mixed)
        super().__init__(model, loss, x0, **options)

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached): :meth:`_exact_gradients`."""
        if "gradients" in sol:
            return sol["gradients"]
        return self._exact_gradients(x, sol)

    def _exact_gradients(self, x, sol):
        """Design gradients of every row with the exact Hessian of the boundary least squares.

        ``b*(x)`` minimizes ``f(b, x) = 1/2 |r|^2``; with ``H = f_bb`` (exact, :meth:`ThreeTermsFreeBoundaryModel.
        second_order`) and ``H mu = dh/db`` the row ``h`` has ``dh/dx = dh/dx|_b - mu^T f_bx``.  ``f_bx`` along
        ``mu`` is a forward-over-reverse pass through the field and, for plasma parameters, the Hessian columns
        of their directions.  (Gauss-Newton would replace ``H`` by ``J^T J`` and ``f_bx`` by ``J_b^T J_x``, wrong
        by ``sum_i r_i d^2 r_i`` when the rows do not vanish: by up to 21 % at finite beta.)
        """
        model, xj = self.model, jnp.asarray(x)
        field = self.field(xj)
        extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
        so = model.second_order(sol["x"], sol["aux"], field, extra=extra)
        state, mask, params, _ = so["aux"]
        nb = model.x0.size
        sol.update(aux=so["aux"], state=state, mask=mask, params=params, rows=so["r"],
                   values=np.asarray(self._rows(state, params, xj)))
        self.stats["linearizations"] += 1
        if self._warm_threads[1] is not None:  # the pullback and direct Jacobian
            self._warm_threads[1].join()
        state, mask, params = model._im._canonical_arrays(model.cfg, (state, mask, params))
        factors = model._im.block_factors_at(model.cfg, params, state, mask)
        params_bar = self._pullback(state, mask, params, xj, factors)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        # With a gauge the boundary solves min f s.t. c(b) = 0: the sensitivity lives in the null space N of
        # grad c, with the Lagrangian Hessian W = H + sum nu_k grad^2 c_k (ESS-scaled for conditioning).
        S = np.asarray(model.x_scale, dtype=float)
        W = S[:, None] * so["H"] * S[None]
        if model.gauge_constraints is not None:
            bj = jnp.asarray(sol["x"])
            A = np.asarray(model._gauge_jacobian(bj)) * S[None]
            nu = -np.linalg.lstsq(A.T, S * so["g"], rcond=None)[0]
            W = W + S[:, None] * np.asarray(model._gauge_hessian(bj, jnp.asarray(nu))) * S[None]
            _, sv, Vt = np.linalg.svd(A)
            N = Vt[int(np.sum(sv > 1e-10 * sv[0])):].T
        else:
            N = np.eye(nb)
        w, Q = np.linalg.eigh(N.T @ W @ N)
        if w[0] <= 0:
            raise TrialRejected(f"the reduced boundary Hessian is not positive definite (smallest eigenvalue "
                                f"{w[0]:.3e})")
        NQ = N @ Q
        mu = S[:, None] * (NQ @ ((NQ.T @ (S[:, None] * G_b.T)) / w[:, None]))  # a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        dz_b = jax.tree.map(lambda a: a[:nb], so["dz"])
        dp_b = jax.tree.map(lambda a: a[:nb], so["batch"])
        for k, m in enumerate(mu.T):
            mj = jnp.asarray(m)
            dz_mu = jax.tree.map(lambda a: jnp.tensordot(mj, a, axes=1), dz_b)
            dp_mu = jax.tree.map(lambda a: jnp.tensordot(mj, a, axes=1), dp_b)
            with vc.remat_scans():
                gradients[k] -= np.asarray(self._coil_mixed(state, mask, params, xj, dz_mu, dp_mu))
        if self._plasma.size:
            gradients[:, self._plasma] -= mu.T @ so["C"]
        J = so["J"]
        self._anchor = dict(x=self._x(x).copy(), b=sol["x"], J_b=J[:, :nb], J_p=J[:, nb:], state=state,
                            params=params, cost=0.5 * so["r"] @ so["r"])
        sol["gradients"] = gradients
        sol["hessian_eigenvalues"] = w
        return gradients



def solve_free_boundary_least_squares(inp: VmecInput, *, external_field: Any = None, mgrid_path=None,
                                      initial_boundary: VmecInput | None = None, max_mode: int | None = None,
                                      nphi: int | None = None, ntheta: int | None = None, digits: int = 4,
                                      max_nfev: int = 40, quadrature: tuple[int, int] | None = None,
                                      trial_ftol: float | None = None, chunk: int = 8, bootstrap=None,
                                      bootstrap_helicity: int = 0, bootstrap_weight: float = 100.0, previous=None,
                                      verbose: int = 0):
    """:func:`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms` with ``method="least_squares"``.

    The same keywords and result (``max_nfev`` caps the solver's models); ``initial_boundary`` is the start (its
    coefficients, relabelled onto the canonical gauge).
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .implicit import input_with_params
    from .optimize import pack_boundary, solve_equilibrium

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    if previous is not None:
        model, x0 = previous.model, previous.x
    else:
        model = LeastSquaresFreeBoundaryModel(
            inp, max_mode=max_mode, nphi=int(nphi or 48), ntheta=int(ntheta or 48), digits=digits,
            quadrature=quadrature, trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap,
            bootstrap_helicity=bootstrap_helicity, bootstrap_weight=bootstrap_weight)
        x0 = None
        if initial_boundary is not None:
            start = replace(model.fixed, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs)
            x0 = np.r_[pack_boundary(start, model.max_mode, vary_major_radius=True), model.x0[model.n_boundary:]]
    initial = model.boundary_residual(model.seed[0], model.params0, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, max_iter=max_nfev, verbose=verbose)
    state, _, params, _ = out["aux"]
    boundary = input_with_params(model.fixed, params)
    equilibrium = solve_equilibrium(
        boundary, initial_state=state, forward_ftol=min(_REPORT_FTOL, float(np.asarray(boundary.ftol_array)[-1])),
        forward_max_iterations=max(int(np.asarray(boundary.niter_array)[-1]), 200000))
    final = model.boundary_residual(equilibrium.solution, params, external_field)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=bool(out["converged"]), status=out["status"], message=out["status"],
        history=out["history"], input=boundary, equilibrium=equilibrium, model=model, aux=out["aux"],
        boundary_residual=final, normal_outside_modes=model.normal_outside_modes(equilibrium.solution, params,
                                                                                 external_field),
        initial_boundary_residual=initial)
