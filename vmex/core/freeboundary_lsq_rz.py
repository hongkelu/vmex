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
    its new deck.  ``gauge="none"`` drops the rows (to measure what they do).
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
GAUGES = ("deck", "none")
#: the old campaign solver's weight of its labelling rows (e3313c0f, ``label_weight``)
DECK_GAUGE_WEIGHT = 1e-2


class LeastSquaresRZFreeBoundaryModel(LeastSquaresFreeBoundaryModel):
    """The least-squares free boundary on the boundary's Fourier coefficients (module docstring).

    :class:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryModel` with other coordinates (``x`` the offsets of
    the packed ``RBC``/``ZBS`` from the deck's, :func:`~vmex.core.optimize.pack_boundary` with ``RBC(0,0)``) and
    other rows (the three interface conditions, the gauge, the net current).  ``gauge`` and ``gauge_weight``: the
    labelling gauge of the module docstring.
    """

    def __init__(self, inp: VmecInput, *, gauge="deck", gauge_weight=DECK_GAUGE_WEIGHT, **options):
        if gauge not in GAUGES:
            raise ValueError(f"gauge must be one of {GAUGES}, got {gauge!r}")
        if options.get("fixed_boundary"):
            raise ValueError("the RZ least squares solves a free boundary: use LeastSquaresFreeBoundaryModel for "
                             "fixed_boundary=True")
        # read when the rows are traced (set before the base builds them)
        self.gauge, self.gauge_weight = gauge, float(gauge_weight)
        super().__init__(inp, **options)
        self._source[1].update(gauge=self.gauge, gauge_weight=self.gauge_weight)  # a re-referenced model's too

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
        self.x0 = np.zeros(x_ref.size)
        self.n_boundary = nb = x_ref.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self._linearization = None
        self._last_linearize = None
        self._x_ref = jnp.asarray(x_ref)
        self._normal_map = None

        def with_boundary(params, x):
            rbc, zbs = boundary_arrays_from_x(fixed, self._x_ref + x[:nb], self.max_mode, vary_major_radius=True)
            params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        self._implicit_setup()
        self._interface_setup(self.seed[0], self.nphi, self.ntheta, self.digits)

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
                            gauge_weight=jnp.asarray(self.gauge_weight))
        gauge = self.gauge

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
            return jnp.concatenate(parts + [((G - G_coil) / G_coil)[None]])

        self.interface_rows = interface_rows

    def _boundary_rms(self, step):
        """RMS over the angles of the boundary points' displacement (metres) for a step of the packed
        coefficients (normal and tangential: the coordinates move both)."""
        return float(np.sqrt(np.sum(self._mean_square_weight * np.square(step))))

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
        return np.r_[x - np.asarray(self._x_ref), self.x0[self.n_boundary:]]

    def _print_weakest(self, direction):
        """Print the largest coefficient components of ``direction`` (the reduced Hessian's weakest)."""
        modes = self.boundary_modes[:-1]
        labels = [f"rbc({m},{n})" for m, n in modes] + [f"zbs({m},{n})" for m, n in modes] + ["rbc(0,0)"]
        v = direction[:len(labels)] / np.max(np.abs(direction[:len(labels)]))
        top = np.argsort(-np.abs(v))[:6]
        print("  weakest:", " ".join(f"{labels[k]}:{v[k]:+.2f}" for k in top), flush=True)


class LeastSquaresRZFreeBoundaryProblem(LeastSquaresFreeBoundaryProblem):
    """:class:`~vmex.core.freeboundary_lsq.LeastSquaresFreeBoundaryProblem` on the RZ least-squares free boundary
    (:class:`LeastSquaresRZFreeBoundaryModel`; the same exact design gradients).  Build with
    ``FreeBoundaryProblem.from_loss(..., three_terms_options=dict(method="least_squares_rz", ...))``."""

    model_class = LeastSquaresRZFreeBoundaryModel


def solve_free_boundary_least_squares_rz(inp: VmecInput, *, gauge="deck", gauge_weight=DECK_GAUGE_WEIGHT,
                                         **keywords):
    """Solve the RZ least-squares free boundary of a deck: :func:`~vmex.core.freeboundary_lsq.
    solve_free_boundary_least_squares` (the same keywords and result) on :class:`LeastSquaresRZFreeBoundaryModel`,
    with its ``gauge`` and ``gauge_weight``."""
    from .freeboundary_lsq import solve_free_boundary_least_squares

    return solve_free_boundary_least_squares(inp, model_class=LeastSquaresRZFreeBoundaryModel, gauge=gauge,
                                             gauge_weight=gauge_weight, **keywords)
