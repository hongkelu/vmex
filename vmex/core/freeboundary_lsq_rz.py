"""Free-boundary equilibria by least squares on the boundary's Fourier coefficients (``method="least_squares_rz"``).

The third method of :func:`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms`: the formulation of the old
campaign's least-squares solver (e3313c0f) with the exact, certified machinery of ``method="least_squares"``
(:mod:`vmex.core.freeboundary_lsq`).

Unknowns
    Every boundary coefficient ``RBC``/``ZBS`` on the band (and ``RBC(0,0)``; 255 at 8x8), as offsets from the
    deck's: any surface in any poloidal labelling, then the bootstrap current's knot values, as the other least
    squares.  ``method="least_squares"`` moves the deck's boundary along its radial lines only, so its labelling is
    the deck's; here the labelling is free, and the truncated boundary takes the labelling its rows prefer.
Rows
    The three interface conditions of :class:`~vmex.core.freeboundary_vc.BoundaryResidual` as separate
    area-weighted rows on the grid, pointwise relative to ``|B_in|``: ``B_out . n``, the pressure balance and the
    sheet current ``n x (B_out - B_in)`` (in vacuum ``B . n`` alone), with the whole interior edge field (no band
    limit: rows that see only VMEC's Galerkin band leave the error outside it, PR #94).  These are the common
    evaluator's terms, so the solve minimizes what the comparison measures.  Then the labelling gauge rows, then
    ``G_plasma = G_coil``; the net current and the bootstrap rows are equality constraints, as in the other least
    squares.
Labelling gauge
    The interface conditions see the poloidal labelling of the boundary only through the truncation of its Fourier
    series: the cost is nearly flat along relabellings ``theta -> theta + delta(theta, zeta)``, its Hessian nearly
    singular there, and the answer's labelling (and the design gradient through the inverse Hessian) would be
    undetermined.  ``gauge="deck"`` (the default, the old solver's) adds the rows ``w sqrt(area) (x - x_deck) .
    t / L``: the boundary points' displacement from the deck's along the unit poloidal tangent ``t``, over the mean
    poloidal metric ``L = <|e_theta|>`` (an angle), ``w = gauge_weight`` (default 1e-2, the old solver's).  A pure
    relabelling by ``delta`` costs ``w delta`` a point, a normal displacement nothing at first order, so the rows
    pick, among the near-equivalent labellings, the one nearest the deck's and leave the surface to the interface
    rows.  The deck is the model's: a re-referenced model (an optimizer far from its seed,
    :meth:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryModel.rereferenced`) penalizes the displacement from
    its new deck.  ``gauge="spectral"`` prices the same displacement angle by its Fourier modes over a field period:
    a mode of level ``k = max(|m|, |n|)`` has weight ``w (1 + (w_top / w - 1) (k / K)^4)``, ``K`` the band's top
    level and ``w_top = gauge_top_weight`` (capped there above it); uniform weights are the deck gauge exactly
    (Parseval).  Why: at a converged QA boundary (qass_b25, 8x8) the reduced Hessian's weak directions are the
    high-mode relabellings, where the rows' own curvature is negative (truncation; down to -4e-8, and -9e-4 far from
    the root) and a uniform weight 1e-2 only just outweighs it, while the weaker 1e-3 -- which leaves the low-mode
    labelling free -- ends nearer the resolved surface (1.34-1.42 mm against 1.70 mm) but does not converge.
    ``gauge="arc"`` asks for a labelling of the surface's own: equal arc in each toroidal plane, the rows
    ``w sqrt(area) (|e_theta| / <|e_theta|>_theta - 1)`` of the current boundary (no reference: the same rows for any
    deck or start).  Equal arc fixes the labelling up to each plane's origin (the relabellings ``delta(zeta)``,
    ``m = 0``), which the rows ``w <(x - x_deck) . t / L>_theta``, one a plane, fix to the deck's.
    ``gauge="none"`` drops the rows (to measure what they do).
Labelling pre-solve
    A deck gauge pulls the answer's labelling toward the deck's, and the 8x8 truncation prefers another: on the old
    campaign's QA seed (qass_b25, 2.5 %) the deck gauge at 1e-2 ends 1.70 mm (RMS) from the resolved (M 16) free
    boundary, at 1e-1 1.76 mm, from a relabelled deck 1.39 mm; with no gauge, or a weak one (1e-3), 1.37-1.39 mm, but
    the near-flat labelling directions then keep the solve from certifying (100 models).  ``labelling_presolve``
    (:func:`presolve_labelling`) first relaxes the labelling with a weak gauge (a bounded number of models, no
    certificate) and makes that boundary the deck; the certified solve then runs with the strong gauge from it: QA
    1.38 mm, 7 models, 146 s, a warm restart 82 s (against 1.70 mm, 71 models, 1228 s and 249 s with the deck gauge
    1e-2 alone).  The answer is then that of the pre-solved deck: deterministic for a deck and resolution.
Solver
    :meth:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryModel._newton` unchanged: SQP on the constraint
    manifold with the exact second-order adjoint Hessian, converged only at a certified root of the step test, and
    exact design gradients (:class:`LeastSquaresRZFreeBoundaryProblem`).  The step test measures the RMS
    displacement of the boundary points in metres (:meth:`LeastSquaresRZFreeBoundaryModel._boundary_rms`).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import jax.numpy as jnp

from .freeboundary_lsq import LeastSquaresFreeBoundaryModel, LeastSquaresFreeBoundaryProblem, _Bundle
from .freeboundary_vc import _interface_terms, _net_current, _surface_data
from .input import VmecInput

#: the labelling gauges of :class:`LeastSquaresRZFreeBoundaryModel` (module docstring)
GAUGES = ("deck", "spectral", "arc", "none")
#: the spectral gauge's weight at the band's top mode level (gauge_top_weight): ten times the deck gauge's 1e-2, which
#: at the converged QA boundary (qass_b25, 8x8) only just outweighs the rows' negative curvature of the high-mode
#: relabellings (provisional: the value under test, analyze_hessian / benchmark of 10-18)
SPECTRAL_TOP_WEIGHT = 1e-1
#: the solver's basis of the boundary coordinates (module docstring): the packed coefficients themselves, or the deck
#: surface's normal displacement and its relabelling (the same space, an invertible linear map)
BASES = ("rz", "normal_tangential")
#: largest condition number of the normal-tangential basis (its columns unit-scaled) accepted: round-off in the
#: coefficients stays below sqrt(eps) of a step (van der Sluis 1969 scaling, as freeboundary_vc._newton_step)
_BASIS_MAX_COND = 1.0 / np.sqrt(np.finfo(float).eps)


def _relabelling_map(inp, max_mode, ntheta, nzeta):
    """Packed-coefficient changes of the relabellings ``theta -> theta + d sin(m theta - n nfp zeta)`` of ``inp``'s
    surface, a column per mode of :func:`~vmex.core.optimize._dof_modes` (stellarator symmetric: ``d`` odd).

    The points move along the surface, ``(dR, dZ) = (R_theta, Z_theta) d``, fitted on the boundary's modes by least
    squares on a ``ntheta x nzeta`` grid of a field period (the product has harmonics beyond the band: the fit keeps
    the band's part, the relabelling the truncated series can represent).  Returns ``(M, modes)``."""
    from .optimize import _dof_modes, pack_boundary

    rbc, zbs, nfp = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float), int(inp.nfp)
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1]
    th = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ze = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    T, Zg = np.meshgrid(th, ze, indexing="ij")
    ns, ms = np.arange(-ntor, ntor + 1), np.arange(mpol)
    ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * Zg[None, None]
    cos, sin = np.cos(ph), np.sin(ph)
    Rt = np.einsum("nm,nmij->ij", -rbc * ms[None, :], sin)
    Zt = np.einsum("nm,nmij->ij", zbs * ms[None, :], cos)
    keep_z = ~((ms[None, :] == 0) & (ns[:, None] <= 0))
    keep_r = ~((ms[None, :] == 0) & (ns[:, None] < 0))
    pinv_r = np.linalg.pinv(cos[keep_r].reshape(int(keep_r.sum()), -1).T)
    pinv_z = np.linalg.pinv(sin[keep_z].reshape(int(keep_z.sum()), -1).T)
    modes = list(_dof_modes(inp, max_mode))
    zero = replace(inp, rbc=np.zeros_like(rbc), zbs=np.zeros_like(zbs))
    columns = []
    for m, n in modes:
        d = np.sin(m * T - n * nfp * Zg)
        dr, dz = np.zeros_like(rbc), np.zeros_like(zbs)
        dr[keep_r], dz[keep_z] = pinv_r @ (Rt * d).ravel(), pinv_z @ (Zt * d).ravel()
        columns.append(pack_boundary(replace(zero, rbc=dr, zbs=dz), max_mode, vary_major_radius=True))
    return np.asarray(columns).T, modes
#: the old campaign solver's weight of its labelling rows (e3313c0f, ``label_weight``)
DECK_GAUGE_WEIGHT = 1e-2


class LeastSquaresRZFreeBoundaryModel(LeastSquaresFreeBoundaryModel):
    """The least-squares free boundary on the boundary's Fourier coefficients (module docstring).

    :class:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryModel` with other coordinates (``x`` the offsets of
    the packed ``RBC``/``ZBS`` from the deck's, :func:`~vmex.core.optimize.pack_boundary` with ``RBC(0,0)``) and
    other rows (the three interface conditions, the gauge, the net current).  ``gauge`` and ``gauge_weight``: the
    labelling gauge of the module docstring.
    """

    def __init__(self, inp: VmecInput, *, gauge="deck", gauge_weight=DECK_GAUGE_WEIGHT, basis="rz",
                 gauge_top_weight=SPECTRAL_TOP_WEIGHT, **options):
        if gauge not in GAUGES:
            raise ValueError(f"gauge must be one of {GAUGES}, got {gauge!r}")
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
        self.basis = basis
        if options.get("fixed_boundary"):
            raise ValueError("the RZ least squares solves a free boundary: use LeastSquaresFreeBoundaryModel for "
                             "fixed_boundary=True")
        # read when the rows are traced (set before the base builds them)
        self.gauge, self.gauge_weight, self.gauge_top_weight = gauge, float(gauge_weight), float(gauge_top_weight)
        super().__init__(inp, **options)
        # a re-referenced model's too
        self._source[1].update(gauge=self.gauge, gauge_weight=self.gauge_weight, basis=self.basis,
                               gauge_top_weight=self.gauge_top_weight)

    def _set_band(self, max_mode):
        """The boundary coordinates on the modes up to ``max_mode``: every packed coefficient, as an offset from the
        deck's, with the spectral scales of the old solver (ESS, ``alpha = 1.2``), and the rows of that band."""
        from .optimize import _dof_modes, _ess_scale, boundary_arrays_from_x, pack_boundary

        fixed = self.fixed
        self.max_mode = int(max_mode)
        x_ref = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        modes = _dof_modes(fixed, self.max_mode)
        # the boundary's own modes (normal_outside_modes splits B.n on them), and the mean-square weight of every
        # coordinate: RBC and ZBS of a mode (m, n) != (0, 0) move the points by a harmonic of mean square 1/2
        self.boundary_modes = list(modes) + [(0, 0)]
        self._mean_square_weight = np.r_[np.full(2 * len(modes), 0.5), 1.0]
        self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
        self._basis = None  # basis "rz": the coordinates are the coefficients' offsets
        if self.basis == "normal_tangential":
            self._set_normal_tangential_basis(fixed, modes)
        self.x0 = np.zeros(x_ref.size)
        self.n_boundary = nb = x_ref.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self._linearization = None
        self._last_linearize = None
        self._x_ref = jnp.asarray(x_ref)
        self._normal_map = None

        basis = None if self._basis is None else jnp.asarray(self._basis)

        def with_boundary(params, x):
            offset = x[:nb] if basis is None else basis @ x[:nb]
            rbc, zbs = boundary_arrays_from_x(fixed, self._x_ref + offset, self.max_mode, vary_major_radius=True)
            params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        self._implicit_setup()
        self._interface_setup(self.seed[0], self.nphi, self.ntheta, self.digits)

    def _set_normal_tangential_basis(self, fixed, modes):
        """The basis ``"normal_tangential"``: ``x = (h, d)``, the deck surface's normal displacement ``h`` on the
        modes of :func:`~vmex.core.freeboundary_vc._normal_displacement_map` (metres) and its relabelling ``d`` on
        those of :func:`_relabelling_map` (radians); the packed coefficients are ``x_ref + B x``, ``B`` square and
        invertible, so the unknowns, rows and answer are those of the basis ``"rz"``.

        Why: at a converged QA boundary (qass_b25, 8x8) 137 of the 254 eigenvalues of the reduced exact Hessian in the
        ESS-scaled coefficients are below 1e-6 of the largest, every weak direction a pure relabelling; the spectral
        scales shrink the high modes, so a trust region round in those coordinates admits only tiny relabellings, and
        the solve crawled along them for most of its models.  Here the two kinds of motion have their own coordinates
        and scales in metres: ``h`` the normal displacement's ESS scales, ``d`` one scale for every mode, ``1 / L``
        with ``L = <|e_theta|>`` the mean poloidal metric (a relabelling ``d`` moves the points by about ``L d``).  No
        spectral decay on ``d``: the gauge prices a relabelling by its angle whatever its mode (curvature ``w^2 / 2``
        a unit angle), so decaying scales would only bring back the weak high-mode directions (the first try with ESS
        on ``d`` too: smallest reduced eigenvalue 8e-13 against 3e-9 in the coefficients)."""
        from .freeboundary_vc import _normal_displacement_map

        normal, normal_modes, normal_scale = _normal_displacement_map(fixed, self.max_mode)
        rbc = np.asarray(fixed.rbc, dtype=float)
        ntheta, nzeta = 8 * rbc.shape[1], 8 * ((rbc.shape[0] - 1) // 2) + 8  # twice the normal map's first grid
        tangential, tangential_modes = _relabelling_map(fixed, self.max_mode, ntheta, nzeta)
        B = np.hstack([normal, tangential])
        if B.shape[0] != B.shape[1]:
            raise ValueError(f"the normal-tangential basis is {B.shape[0]} x {B.shape[1]}, not square")
        unit = B / np.linalg.norm(B, axis=0)[None]
        cond = float(np.linalg.cond(unit))
        if cond > _BASIS_MAX_COND:
            raise ValueError(f"the normal-tangential basis is singular on this deck (condition {cond:.1e})")
        # the mean poloidal metric of the deck surface, on the same grid as the relabelling map
        nfp = int(fixed.nfp)
        th = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        ze = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
        T, Zg = np.meshgrid(th, ze, indexing="ij")
        ms = np.arange(rbc.shape[1])
        ns = np.arange(-((rbc.shape[0] - 1) // 2), (rbc.shape[0] - 1) // 2 + 1)
        ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * Zg[None, None]
        Rt = np.einsum("nm,nmij->ij", -rbc * ms[None, :], np.sin(ph))
        Zt = np.einsum("nm,nmij->ij", np.asarray(fixed.zbs, dtype=float) * ms[None, :], np.cos(ph))
        self._metric = float(np.mean(np.hypot(Rt, Zt)))
        self._basis = B
        self._normal_modes, self._tangential_modes = list(normal_modes), list(tangential_modes)
        self.x_scale = np.r_[normal_scale, np.full(len(tangential_modes), 1.0 / self._metric)]

    def _interface_setup(self, state, nphi, ntheta, digits):
        """The base least-squares setup (its constants, reports and ``B . n`` programs), then this method's rows."""
        from .freeboundary import _vacuum_scalars

        super()._interface_setup(state, nphi, ntheta, digits)
        im, cfg, fixed = self._im, self.cfg, self.fixed
        precision, digits = (None, None) if self.vacuum else (self.precision, digits)
        # the gauge's reference: the deck's boundary points on the rows' grid and their unit poloidal tangent over
        # the mean poloidal metric (the old solver's normalization: the rows are angles)
        gamma0 = jnp.asarray(_surface_data(fixed, cfg, state, self.params0, nphi, ntheta).gamma)
        wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
        e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma0, axis=-1), axis=-1))
        length = jnp.linalg.norm(e_theta, axis=0)
        self._consts = dict(self._consts, gamma0=gamma0, tangent=e_theta / (length * jnp.mean(length))[None],
                            gauge_weight=jnp.asarray(self.gauge_weight), wavenumber=wavenumber)
        gauge = self.gauge
        if gauge == "spectral":  # the weight of every grid harmonic (field periods in phi), by its mode level
            k_phi = np.abs(np.fft.fftfreq(nphi, 1.0 / nphi))[:, None]
            k_theta = np.abs(np.fft.fftfreq(ntheta, 1.0 / ntheta))[None, :]
            level = np.minimum(np.maximum(k_phi, k_theta) / max(self.max_mode, 1), 1.0)
            w = self.gauge_weight * (1.0 + (self.gauge_top_weight / self.gauge_weight - 1.0) * level**4)
            self._consts["spectral_weight"] = jnp.asarray(w)

        def interface_rows(state, params, field, moments=False):
            """``sqrt(area)`` times the interface conditions (``B . n``, pressure balance, sheet current; in vacuum
            ``B . n``), then the gauge rows, then ``(G - G_coil) / G_coil`` (an equality constraint)."""
            field, consts = (field.field, field.consts) if isinstance(field, _Bundle) else (field, self._consts)
            runtime = im.runtime_from_params(params, cfg)
            G_coil = jnp.abs(_net_current(field, *consts["axis"]))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            values, weights, gamma = _interface_terms(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                      digits=digits, precision=precision, p_edge=0.0)
            area = jnp.sqrt(weights)
            parts = [(area[None] * values).ravel()]
            if gauge == "deck":
                label = consts["gauge_weight"] * jnp.sum((gamma - consts["gamma0"]) * consts["tangent"], axis=0)
                parts.append((area * label).ravel())
            elif gauge == "arc":  # equal arc in each plane, and the planes' origins at the deck's
                e_t = jnp.real(jnp.fft.ifft(1j * consts["wavenumber"] * jnp.fft.fft(gamma, axis=-1), axis=-1))
                metric = jnp.linalg.norm(e_t, axis=0)
                arc = metric / jnp.mean(metric, axis=1, keepdims=True) - 1.0
                origin = jnp.mean(jnp.sum((gamma - consts["gamma0"]) * consts["tangent"], axis=0), axis=1)
                parts += [(consts["gauge_weight"] * area * arc).ravel(),
                          consts["gauge_weight"] * origin / np.sqrt(origin.size)]
            elif gauge == "spectral":  # the area-weighted angle's harmonics (orthonormal: Parseval), weighted
                angle = area * jnp.sum((gamma - consts["gamma0"]) * consts["tangent"], axis=0)
                harmonics = consts["spectral_weight"] * jnp.fft.fft2(angle, norm="ortho")
                parts += [jnp.real(harmonics).ravel(), jnp.imag(harmonics).ravel()]
            return jnp.concatenate(parts + [((G - G_coil) / G_coil)[None]])

        self.interface_rows = interface_rows

    def _boundary_rms(self, step):
        """RMS over the angles of the boundary points' displacement (metres) for a step of the coordinates (normal and
        tangential: the coordinates move both; basis ``"normal_tangential"``: ``h`` in metres, ``d`` times the mean
        poloidal metric)."""
        if self._basis is None:
            return float(np.sqrt(np.sum(self._mean_square_weight * np.square(step))))
        from .freeboundary_vc import displacement_rms

        k = len(self._normal_modes)
        tangential = self._metric * float(np.sqrt(0.5 * np.sum(np.square(step[k:]))))
        return float(np.hypot(displacement_rms(step[:k], self._normal_modes), tangential))

    def displacement_of(self, boundary, ntheta=None, nzeta=None):
        """Coordinates ``x`` of ``boundary``'s surface (its own coefficients, in its own labelling, on the band), with
        the deck's current values; modes outside the deck's arrays are dropped."""
        from .optimize import pack_boundary

        del ntheta, nzeta  # the other models' fit grid: here the coefficients are the coordinates
        rbc, zbs = np.zeros(np.shape(self.fixed.rbc)), np.zeros(np.shape(self.fixed.zbs))
        for mine, theirs in ((rbc, np.asarray(boundary.rbc, dtype=float)), (zbs, np.asarray(boundary.zbs, dtype=float))):
            ntor_mine, ntor_theirs = (mine.shape[0] - 1) // 2, (theirs.shape[0] - 1) // 2
            nt, mp = min(ntor_mine, ntor_theirs), min(mine.shape[1], theirs.shape[1])
            mine[ntor_mine - nt:ntor_mine + nt + 1, :mp] = theirs[ntor_theirs - nt:ntor_theirs + nt + 1, :mp]
        x = pack_boundary(replace(self.fixed, rbc=rbc, zbs=zbs), self.max_mode, vary_major_radius=True)
        offset = x - np.asarray(self._x_ref)
        if self._basis is not None:
            offset = np.linalg.solve(self._basis, offset)
        return np.r_[offset, self.x0[self.n_boundary:]]

    def _print_weakest(self, direction):
        """Print the largest coefficient components of ``direction`` (the reduced Hessian's weakest)."""
        modes = self.boundary_modes[:-1]
        labels = [f"rbc({m},{n})" for m, n in modes] + [f"zbs({m},{n})" for m, n in modes] + ["rbc(0,0)"]
        if self._basis is not None:
            labels = [f"h({m},{n})" for m, n in self._normal_modes] + [f"d({m},{n})" for m, n in self._tangential_modes]
        v = direction[:len(labels)] / np.max(np.abs(direction[:len(labels)]))
        top = np.argsort(-np.abs(v))[:6]
        print("  weakest:", " ".join(f"{labels[k]}:{v[k]:+.2f}" for k in top), flush=True)


#: the labelling pre-solve's gauge weight: weak enough that the labelling relaxes to the truncation's preferred one
#: (1e-3 ends within 0.02 mm of no gauge on qass_b25), strong enough that the solve stays well defined
PRESOLVE_GAUGE_WEIGHT = 1e-3
#: the labelling pre-solve's models: no certificate is asked for (the certified solve follows)
PRESOLVE_MODELS = 40


def presolve_labelling(inp: VmecInput, field, params=None, *, initial_boundary=None,
                       gauge_weight=PRESOLVE_GAUGE_WEIGHT, max_iter=PRESOLVE_MODELS, verbose=0,
                       **model_options) -> VmecInput:
    """``inp`` with the boundary of a weak-gauge RZ least-squares solve of ``max_iter`` models: the labelling the
    rows prefer (module docstring), as the deck of the certified, strongly gauged solve.

    ``params`` (default the deck's) are the plasma parameters of the solve and ``initial_boundary`` its start
    (default the deck's boundary); ``model_options`` are :class:`LeastSquaresRZFreeBoundaryModel` keywords other
    than the gauge's."""
    options = {k: v for k, v in model_options.items() if k not in ("gauge", "gauge_weight", "gauge_top_weight")}
    weak = LeastSquaresRZFreeBoundaryModel(replace(inp, lfreeb=False), gauge="deck", gauge_weight=gauge_weight,
                                           **options)
    x0 = None if initial_boundary is None else weak.displacement_of(initial_boundary)
    out = weak.solve_boundary(weak.params0 if params is None else params, field, x0=x0, max_iter=int(max_iter),
                              verbose=verbose)
    boundary = weak.with_boundary(weak.params0, jnp.asarray(out["x"]))
    return replace(inp, rbc=np.asarray(boundary.rbc, dtype=float), zbs=np.asarray(boundary.zbs, dtype=float))


class LeastSquaresRZFreeBoundaryProblem(LeastSquaresFreeBoundaryProblem):
    """:class:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryProblem` on the RZ least-squares free boundary
    (:class:`LeastSquaresRZFreeBoundaryModel`; the same exact design gradients).  Build with
    ``FreeBoundaryProblem.from_loss(..., three_terms_options=dict(method="least_squares_rz", ...))``;
    ``labelling_presolve`` (``True`` or :func:`presolve_labelling` keywords) relaxes the seed design's labelling
    first and makes that boundary the problem's deck."""

    model_class = LeastSquaresRZFreeBoundaryModel

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, labelling_presolve=None,
                  **options):
        """:meth:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryProblem.from_loss` (the same keywords), from the
        labelling pre-solved deck with ``labelling_presolve``."""
        if labelling_presolve:
            presolve = {} if labelling_presolve is True else dict(labelling_presolve)
            model_options = {k: v for k, v in options.items() if k not in (
                "scales", "names", "quantities", "parameter_quantities", "boundary_max_nfev", "boundary_start")}
            xj = jnp.asarray(np.asarray(x0, dtype=float))
            params = None
            if plasma_from_parameters is not None:
                from .implicit import params_from_input

                params = plasma_from_parameters(params_from_input(replace(inp, lfreeb=False)), xj)
            inp = presolve_labelling(inp, field_from_parameters(xj), params, **presolve, **model_options)
        return super().from_loss(inp, loss, x0, field_from_parameters=field_from_parameters,
                                 plasma_from_parameters=plasma_from_parameters, **options)


def solve_free_boundary_least_squares_rz(inp: VmecInput, *, gauge="deck", gauge_weight=DECK_GAUGE_WEIGHT, basis="rz",
                                         gauge_top_weight=SPECTRAL_TOP_WEIGHT, labelling_presolve=None, **keywords):
    """Solve the RZ least-squares free boundary of a deck: :func:`~vmex.core.freeboundary_lsq.
    solve_free_boundary_least_squares` (the same keywords and result) on :class:`LeastSquaresRZFreeBoundaryModel`,
    with its ``gauge``, ``gauge_weight``, ``gauge_top_weight`` and ``basis``; ``labelling_presolve`` (``True`` or
    :func:`presolve_labelling` keywords) solves from the labelling pre-solved deck (module docstring)."""
    from .freeboundary_lsq import solve_free_boundary_least_squares
    from .freeboundary import _external_field_from_input

    if labelling_presolve and keywords.get("previous") is None:
        field = keywords.get("external_field")
        if field is None:
            field = _external_field_from_input(inp, keywords.get("mgrid_path"))
        presolve = {} if labelling_presolve is True else dict(labelling_presolve)
        model_options = {k: keywords[k] for k in ("max_mode", "digits", "quadrature", "trial_ftol", "chunk",
                                                  "bootstrap", "bootstrap_helicity") if k in keywords}
        # the grid of the certified solve (solve_free_boundary_least_squares's default)
        model_options.update(nphi=int(keywords.get("nphi") or 48), ntheta=int(keywords.get("ntheta") or 48))
        inp = presolve_labelling(inp, field, initial_boundary=keywords.pop("initial_boundary", None),
                                 verbose=keywords.get("verbose", 0), **presolve, **model_options)

    return solve_free_boundary_least_squares(inp, model_class=LeastSquaresRZFreeBoundaryModel, gauge=gauge,
                                             gauge_weight=gauge_weight, basis=basis,
                                             gauge_top_weight=gauge_top_weight, **keywords)
