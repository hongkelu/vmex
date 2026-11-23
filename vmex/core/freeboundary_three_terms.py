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
(:class:`ThreeTermsModel`).  At an exact solution all three vanish together; at
finite resolution they stop at a floor set mainly by ``mpol``, the
virtual-casing grid and the singular quadrature.

Condition (1) alone leaves one direction open: the flux of ``B_out`` through
the closed surface vanishes identically, and a mismatch of the net poloidal
current ``G`` adds a field tangent to ``S``.  Conditions (2) and (3) close it in
principle; numerically a row ``(G_plasma - G_coil) / G_coil`` closes it directly
(``net_current_weight``).  With ``p = 0`` on the boundary a solution satisfies
all three exactly; a nonzero edge pressure needs a sheet current and is
rejected.  The interface conditions leave the boundary's poloidal labelling
free: ``ThreeTermsConfig.labelling`` fixes it (``"flat_arc"``, the default,
``"equal_arc"`` or ``"anchored"``).

This is an opt-in alternative to NESTOR (a VMEX extension, like position
control); nothing changes unless it is requested.  Entry points, from the most
to the least packaged:

- a free-boundary deck with ``LTHREE_TERMS = T`` (``vmex input.case``,
  :func:`~vmex.core.multigrid.solve_file`), or
  ``solve_free_boundary_multigrid(..., three_terms=ThreeTermsConfig(...))``:
  the result's ``three_terms`` field is a :class:`ThreeTermsResult`;
- :func:`solve_free_boundary_three_terms`: one solve, and ``previous=`` to
  continue from an earlier result after the coils change;
- :class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`:
  single-stage optimization, a loss and constraint rows of coils and plasma
  parameters at their free boundary, for :func:`vmex.core.optimize.minimize`;
- :class:`ThreeTermsModel`: the residual, its Jacobian, state tangents and
  pullbacks for one deck and any external field or plasma parameters, without
  recompiling;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. to
  check a NESTOR result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import time
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .errors import VmecError
from .input import VmecInput


@dataclass(frozen=True)
class BoundaryResidual:
    """Area-weighted RMS of the three interface conditions, relative to ``|B_in|``.

    ``normal`` is ``B_out . n / |B_in|``, ``pressure`` is
    ``(|B_out|^2 - |B_in|^2 - 2 mu0 p) / (2 |B_in|^2)`` and ``sheet_current``
    is ``|n x (B_out - B_in)| / |B_in| = mu0 |K| / |B_in|``.  ``normal_max`` is the largest ``|normal|`` on the grid.
    """

    normal: float
    pressure: float
    sheet_current: float
    normal_max: float = 0.0


def _fingerprint(*trees) -> str:
    """Hash of the concrete arrays of ``trees`` (cache keys of linearizations)."""
    digest = hashlib.sha1()
    for leaf in jax.tree.leaves(trees):
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


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
                            sheet_current=rms(np.linalg.norm(rows[2:], axis=0)),
                            normal_max=float(np.max(np.abs(rows[0]))))


def _edge_pressure(inp: VmecInput) -> tuple[float, float]:
    """Edge and peak pressure of ``inp`` in Pa."""
    from .profiles import pressure

    p = np.asarray(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, np.linspace(0.0, 1.0, 101),
                            pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped))
    return float(p[-1]), float(np.max(np.abs(p)))


def _edge_pressure_error(inp: VmecInput):
    """Why ``inp`` is outside the three-term problem (a nonzero edge pressure), or ``None``."""
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        return f"edge pressure is {p_edge:g} Pa: without a sheet current the boundary pressure must vanish"
    return None


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


def _poloidal_tangent(gamma):
    """``d gamma / d theta`` of boundary points ``(3, nphi, ntheta)``, spectrally on each toroidal cut."""
    ntheta = gamma.shape[-1]
    wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
    return jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma, axis=-1), axis=-1))


def _equal_arc_rows(gamma):
    """Rows that vanish when the boundary's poloidal angle is equal-arc on every toroidal cut and the m = 1 part of
    ``R + i Z`` has zero phase there: one labelling of the surface itself, the same from any start.

    ``gamma`` is ``(3, nphi, ntheta)`` Cartesian points.  Returns ``(arc, phase)``: ``|d gamma/d theta|`` relative to
    its mean on each cut, minus one, ``(nphi, ntheta)``; and ``Im c1 / |c1|`` per cut, ``(nphi,)``.
    """
    speed = jnp.linalg.norm(_poloidal_tangent(gamma), axis=0)
    arc = speed / jnp.mean(speed, axis=-1, keepdims=True) - 1.0
    c1 = jnp.fft.fft(jnp.hypot(gamma[0], gamma[1]) + 1j * gamma[2], axis=-1)[..., 1]
    return arc, jnp.imag(c1) / jnp.abs(c1)


def _jacfwd(fun, x, batch=32):
    """``jax.jacfwd(fun)(x)``, ``batch`` directions at a time: one batch of every direction of a function on the
    boundary grid holds directions x grid x modes numbers (16 x 16 modes: 18.6 GiB)."""
    columns = jax.lax.map(lambda e: jax.jvp(fun, (x,), (e,))[1], jnp.eye(x.size, dtype=x.dtype), batch_size=batch)
    return jax.tree.map(lambda a: jnp.moveaxis(a, 0, -1), columns)


def _displacement_metrics(inp, xb, max_mode):
    """``(A, B)``: the quadratic forms of a change of the boundary coordinates ``xb``
    (``pack_boundary(..., vary_major_radius=True)``) in the surface's normal and total displacement, area weighted over
    one field period.  A direction's normal share (RMS normal over RMS displacement) is ``sqrt(u A u / u B u)``."""
    from .optimize import boundary_arrays_from_x

    mpol, ntor, nfp = int(inp.mpol), int(inp.ntor), int(inp.nfp)
    theta = 2.0 * np.pi * np.arange(4 * mpol + 4) / (4 * mpol + 4)
    phi = 2.0 * np.pi * np.arange(4 * ntor + 4) / ((4 * ntor + 4) * nfp)
    m, n = np.arange(mpol), np.arange(-ntor, ntor + 1) * nfp
    angle = m * theta[:, None, None] - n[:, None] * phi[:, None, None, None]  # (phi, theta, n, m)
    cos, sin = np.cos(angle), np.sin(angle)

    def surface(z):  # R, Z and their theta and phi derivatives on the grid (VMEC's cos(m theta - n nfp phi))
        rc, zs = boundary_arrays_from_x(inp, z, max_mode, vary_major_radius=True)[:2]
        R, Z = jnp.sum(rc * cos, axis=(-2, -1)), jnp.sum(zs * sin, axis=(-2, -1))
        dR, dZ = -rc * sin, zs * cos  # per unit angle
        return (R, Z, jnp.sum(m * dR, axis=(-2, -1)), jnp.sum(m * dZ, axis=(-2, -1)),
                jnp.sum(-n[:, None] * dR, axis=(-2, -1)), jnp.sum(-n[:, None] * dZ, axis=(-2, -1)))

    xb = jnp.asarray(xb, dtype=float)
    R, Z, R_t, Z_t, R_p, Z_p = (np.asarray(a).ravel() for a in surface(xb))
    D_R, D_Z = (np.asarray(a).reshape(-1, xb.size) for a in _jacfwd(lambda z: surface(z)[:2], xb))
    # The normal e_theta x e_phi in (R, phi, Z): (-R Z_t, Z_t R_p - R_t Z_p, R R_t); dA ~ its length.
    area = np.sqrt((R * Z_t) ** 2 + (Z_t * R_p - R_t * Z_p) ** 2 + (R * R_t) ** 2)
    D_n = (-(R * Z_t)[:, None] * D_R + (R * R_t)[:, None] * D_Z) / area[:, None]
    B = D_R.T @ (area[:, None] * D_R) + D_Z.T @ (area[:, None] * D_Z)
    return D_n.T @ (area[:, None] * D_n), B


def _arc_label(inp):
    """``label(rbc, zbs)``: the equal-arc labelling's conditions on the boundary coefficients themselves, on a
    ``(4 ntor + 4) x (4 mpol + 4)`` grid of one field period: the poloidal arc speed ``|d(R, Z)/d theta|`` over its
    mean on each toroidal cut, minus one (RMS normalized), then ``Im c1 / |c1|`` of ``R + i Z``'s ``m = 1`` part per
    cut (:func:`_equal_arc_rows`).  Zero on an equal-arc labelling of zero ``m = 1`` phase."""
    mpol, ntor, nfp = int(inp.mpol), int(inp.ntor), int(inp.nfp)
    theta = 2.0 * np.pi * np.arange(4 * mpol + 4) / (4 * mpol + 4)
    phi = 2.0 * np.pi * np.arange(4 * ntor + 4) / ((4 * ntor + 4) * nfp)

    def label(rbc, zbs):
        m = jnp.arange(rbc.shape[1])
        n = (jnp.arange(rbc.shape[0]) - (rbc.shape[0] - 1) // 2) * nfp
        angle = m * theta[:, None, None] - n[:, None] * phi[:, None, None, None]  # (phi, theta, n, m)
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        R, Z = jnp.sum(rbc * cos, axis=(-2, -1)), jnp.sum(zbs * sin, axis=(-2, -1))
        speed = jnp.hypot(jnp.sum(-m * rbc * sin, axis=(-2, -1)), jnp.sum(m * zbs * cos, axis=(-2, -1)))
        arc = speed / jnp.mean(speed, axis=-1, keepdims=True) - 1.0
        c1 = jnp.fft.fft(R + 1j * Z, axis=-1)[..., 1]
        return jnp.concatenate([arc.ravel() / np.sqrt(arc.size), jnp.imag(c1) / jnp.abs(c1) / np.sqrt(phi.size)])

    return label


#: The flat-arc label rows (``labelling="flat_arc"``) hold the tangential singular directions of the fit's Jacobian,
#: taken in column-scaled coordinates: those whose normal share is below :data:`LABEL_NORMAL_SHARE`.  The normal share
#: is the RMS normal displacement of the surface over its RMS displacement.  The label rows are soft and zero at the
#: reference fit, so the interface rows still set any direction they determine.  The label weight decides.
LABEL_NORMAL_SHARE = 0.2


#: Largest RMS interface residual (relative to ``|B|``) of a three-term free boundary that counts as solved: a guard
#: against wrong stationary points, not an accuracy standard (about upstream's DEL-BSQ gate of 2e-2 on B^2).
ACCEPTED_RESIDUAL = 1e-2

#: Default labelling-row weight of each labelling: the smallest that keeps the fit well posed for the anchored
#: labelling, and for equal arc the weight that biases the interface rows least while it still pins the labelling
#: on the benchmark designs; for flat arc the same weak weight: its rows must pin the labelling only, since the
#: directions they hold are nearly flat, not free (at 1e-1 a single-stage fit that follows them ends 20 times
#: further from the free boundary in B.n).  These are the accepted labellings.  A labelling at weight 0 has no
#: labelling rows: the model names it ``"none"``.
LABEL_WEIGHTS = {"flat_arc": 1e-3, "equal_arc": 1e-3, "anchored": 1e-2}


class _LabelledField(NamedTuple):
    """An external field with the flat-arc label rows' arrays: arguments of the compiled rows, so new label rows of the
    same shape do not recompile them."""

    field: Any
    weights: Any
    reference: Any


def _external_field(field):
    """The external field of ``field``, with label rows or not."""
    return field.field if isinstance(field, _LabelledField) else field


@dataclass(frozen=True)
class ThreeTermsConfig:
    """Settings of a three-term free-boundary solve (``three_terms=`` of the free-boundary solvers).

    ``labelling``
        Fixes the boundary's poloidal labelling, which the interface conditions leave free.

        - ``"flat_arc"`` (the default) holds the arc-length distribution of a reference fit along the fit's flat,
          tangential directions on the current surface (:meth:`ThreeTermsModel.set_label_rows`): the labelling of the
          fit without label rows, held along the directions that fit leaves nearly free.
          :func:`solve_free_boundary_three_terms` sets the rows at its fit without them.  The single-stage problem
          sets them at every accepted step.
        - ``"equal_arc"`` asks for equal arc length on every toroidal cut of the current surface, and a zero phase of
          its ``m = 1`` part.
        - ``"anchored"`` penalizes the tangential offset of the boundary points from a reference boundary, the
          initial one.  Each labelled point is held near its position there, and the surface is free to move.
    ``label_weight``
        Scales the labelling rows against the interface rows (default :data:`LABEL_WEIGHTS`).  At weight 0 a
        labelling has no rows.
    ``start``
        The initial boundary: ``"deck"``, ``"nestor"`` (this ladder's NESTOR free boundary) or ``"best"``.  ``"best"``
        tries both and keeps the lower cost, because the interface least squares has several minima.
    ``weights``, ``nphi``, ``ntheta``
        The residual is ``sqrt(area weight)`` times ``weights = (normal, pressure, sheet current)`` times the rows of
        :func:`boundary_residual` on an ``nphi x ntheta`` grid over one field period.
    ``digits``, ``quadrature``
        The virtual-casing quadrature is planned on the initial boundary to ``digits``, or fixed by
        ``quadrature = (quad_nt, quad_np)``.  Use ``(4 nfp nphi, 2 ntheta)``.
    ``net_current_weight``
        Scales the net-poloidal-current row.
    ``max_mode``
        The unknowns are the boundary Fourier coefficients up to ``max_mode`` (default: all) and ``RBC(0,0)``.
    ``jacobian_ftol``, ``ftol``, ``max_nfev``
        The fresh-Jacobian fit, a trust region with exact implicit Jacobians, runs to a relative cost change of
        ``jacobian_ftol``.  The reused-Jacobian steps, Levenberg-Marquardt steps with its last Jacobian, then go on
        to ``ftol``, in ``max_nfev`` evaluations at most.
    ``chunk``
        Bounds the Jacobian columns per batch.
    ``max_residual``
        ``success`` also requires every RMS interface residual at or below it.
    ``bootstrap``, ``bootstrap_helicity``, ``bootstrap_weight``
        ``bootstrap``, a :class:`~vmex.core.bootstrap.KineticProfiles`, solves a Redl-self-consistent current profile
        together with the boundary.  It is opt-in.

    The default, as in VMEC, holds the deck's pressure and current as inputs.  In single-stage optimization the
    current is then a design variable held self-consistent by a constraint row
    (:class:`~vmex.optimize.RedlBootstrapMismatch`).  Prefer that.  With the current in the fit, its rows dominate
    the fit's cost and its stopping test, and the interface residuals end where the path leaves them (QA at
    2.5 % beta).

    :class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem` ignores ``ftol``,
    ``jacobian_ftol``, ``max_nfev``, ``max_residual`` and ``start``.  It takes its own ``boundary_ftol``,
    ``boundary_max_nfev`` and ``boundary_max_residual``.
    """

    labelling: str = "flat_arc"
    label_weight: float | None = None
    start: str = "best"
    max_mode: int | None = None
    nphi: int = 48
    ntheta: int = 48
    digits: int = 4
    quadrature: tuple[int, int] | None = None
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    net_current_weight: float = 1.0
    ftol: float = 1e-4
    jacobian_ftol: float | None = 1e-2
    max_nfev: int = 60
    chunk: int = 8
    max_residual: float = ACCEPTED_RESIDUAL
    bootstrap: Any = field(default=None, repr=False)
    bootstrap_helicity: int = 0
    bootstrap_weight: float = 100.0

    def __post_init__(self) -> None:
        if self.labelling not in LABEL_WEIGHTS:
            raise ValueError(f"labelling must be one of {list(LABEL_WEIGHTS)}, got {self.labelling!r}")
        if self.start not in ("best", "deck", "nestor"):
            raise ValueError(f"start must be 'best', 'deck' or 'nestor', got {self.start!r}")

    @classmethod
    def from_input(cls, inp: Any) -> "ThreeTermsConfig | None":
        """The solve a deck requests (``LTHREE_TERMS``, ``THREE_TERMS_LABELLING``, ``THREE_TERMS_LABEL_WEIGHT``),
        else ``None``.  A negative weight means the labelling's default."""
        if not bool(getattr(inp, "lthree_terms", False)):
            return None
        weight = float(inp.three_terms_label_weight)
        return cls(labelling=str(inp.three_terms_labelling).strip().lower(),
                   label_weight=None if weight < 0.0 else weight)

    @property
    def resolved_label_weight(self) -> float:
        """The labelling-row weight in use."""
        return LABEL_WEIGHTS[self.labelling] if self.label_weight is None else float(self.label_weight)

    def model_options(self) -> dict:
        """:class:`ThreeTermsModel` keywords of this configuration."""
        return dict(max_mode=self.max_mode, nphi=int(self.nphi), ntheta=int(self.ntheta), digits=self.digits,
                    weights=tuple(self.weights), net_current_weight=self.net_current_weight,
                    label_weight=self.resolved_label_weight,
                    labelling=self.labelling if self.resolved_label_weight else "none",
                    quadrature=self.quadrature, chunk=self.chunk,
                    bootstrap=self.bootstrap, bootstrap_helicity=self.bootstrap_helicity,
                    bootstrap_weight=self.bootstrap_weight)


def resolve_three_terms(requested: Any, inp: Any) -> "ThreeTermsConfig | None":
    """An explicit configuration wins (``True``: the defaults); ``None`` follows the deck; ``False`` disables it."""
    if requested is None:
        return ThreeTermsConfig.from_input(inp)
    if requested is True:
        return ThreeTermsConfig()
    return None if requested is False else requested


@dataclass(frozen=True)
class ThreeTermsResult:
    """A three-term solve, returned as ``result.three_terms`` (and by :func:`solve_free_boundary_three_terms`).

    ``boundary_residual`` and ``initial_boundary_residual`` are the :class:`BoundaryResidual` of the solution and
    of the initial boundary, ``start`` the initial boundary kept (``"deck"`` or ``"nestor"``).  As DESC's least
    squares, the fit stops where it cannot improve any more: ``converged`` is True when the Levenberg-Marquardt
    finish met its relative cost-change test, or its steps still gained under ``1 - LM_SLOW_COST`` each after a
    fresh Jacobian; False when it ran out of evaluations or its trust radius collapsed.  A stationary point need not
    be the free boundary, and the interface rows cannot reach zero at a finite resolution, so ``success`` also
    requires every RMS interface residual at or below the configuration's ``max_residual``.  ``x`` are the boundary
    coordinates (then any bootstrap-current values) and ``cost`` the fit's; ``input`` is the free boundary (and
    current) as a fixed-boundary deck, ``equilibrium`` its equilibrium; ``jacobian``, ``model`` and ``aux``
    continue the solve after a coil change (``previous=``).
    """

    boundary_residual: BoundaryResidual
    initial_boundary_residual: BoundaryResidual
    config: ThreeTermsConfig
    start: str
    converged: bool
    success: bool
    message: str
    cost: float
    nfev: int
    njev: int
    x: np.ndarray = field(repr=False)
    jacobian: Any = field(repr=False, default=None)
    input: Any = field(repr=False, default=None)
    equilibrium: Any = field(repr=False, default=None)
    model: Any = field(repr=False, default=None)
    aux: Any = field(repr=False, default=None)


def solve_free_boundary_three_terms(
    inp: VmecInput,
    config: ThreeTermsConfig | None = None,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_boundary: VmecInput | None = None,
    previous: ThreeTermsResult | None = None,
    verbose: int = 0,
) -> ThreeTermsResult:
    """Free-boundary equilibrium of ``inp`` from the three interface conditions, from one initial boundary.

    ``inp`` is a free-boundary deck (``PHIEDGE``, profiles, and the external field through ``external_field``,
    ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``); its boundary, or that of ``initial_boundary``, is the initial
    guess, so a fixed-boundary design and the coils fitted to it make a natural start.  Every trial is a
    fixed-boundary equilibrium on the deck's ``NS_ARRAY`` ladder at fixed toroidal flux, pressure and current or
    iota.  ``external_field`` is anything :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts (an
    :class:`~vmex.core.mgrid.MgridField`, a ``xyz -> B`` callable such as an ESSOS Biot-Savart field, or a coil
    field with ``b_cyl``).  ``config`` is a :class:`ThreeTermsConfig` (its ``start`` is
    :func:`~vmex.core.multigrid.solve_free_boundary_multigrid`'s).  ``previous``, an earlier result for the same
    deck (typically before a coil change), starts from its boundary, Jacobian and model: neither new coils nor a
    repeat solve compile anything.

    With the labelling ``"flat_arc"`` the boundary is first fitted without label rows.  The rows are then set at
    that fit (:meth:`ThreeTermsModel.set_label_rows`), where they vanish, and the fit goes on from there under
    them.  A ``previous`` result keeps its rows.
    """
    from .freeboundary import _resolve_mgrid
    from .mgrid import MgridField
    from .optimize import solve_equilibrium

    config = ThreeTermsConfig() if config is None else config
    if bool(inp.lasym):
        raise NotImplementedError("the three-term free boundary supports lasym = False only")
    if _edge_pressure_error(inp):
        raise ValueError(_edge_pressure_error(inp))
    if external_field is None:  # tricubic: virtual casing samples between the mgrid planes, where the trilinear
        # interpolant has kinks that NESTOR (on the planes) never sees; on the CTH-like deck they cost 0.4 mm
        external_field = MgridField.from_input(inp, _resolve_mgrid(inp, mgrid_path), order=3)
    if previous is not None:
        model, x0, jacobian = previous.model, previous.x, previous.jacobian
    else:
        start = inp if initial_boundary is None else replace(
            inp, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs,
            raxis_c=initial_boundary.raxis_c, zaxis_s=initial_boundary.zaxis_s)
        model = ThreeTermsModel(start, **config.model_options())
        x0 = jacobian = None
    seed_state = model.seed[0] if previous is None else previous.aux[0]
    seed_params = model.params0 if previous is None else previous.aux[2]
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    fit = dict(ftol=config.ftol, max_nfev=config.max_nfev, verbose=verbose)
    out = model.solve_boundary(model.params0, model.labelled(external_field), x0=x0, jacobian=jacobian,
                               jacobian_ftol=config.ftol if config.jacobian_ftol is None else config.jacobian_ftol,
                               refresh_on_stall=True, **fit)
    if model.labelling == "flat_arc" and model._labels is None:  # flat arc's rows, set at the fit without them
        J = model.linearize(out["x"], out["aux"], external_field)[0]
        fresh = J is not out["jacobian"]
        if model.set_label_rows(out["x"], J):
            field = model.labelled(external_field)
            J = model.linearize(out["x"], out["aux"], field)[0]  # the same linearization, with the new rows
            # The rows vanish at the fit, so its cost and gradient are unchanged: the steps start where it ended.
            held = model.solve_boundary(model.params0, field, x0=out["x"], jacobian=J, **fit)
            out = dict(held, nfev=out["nfev"] + held["nfev"], njev=out["njev"] + int(fresh),
                       accepted=out["accepted"] + held["accepted"],
                       converged=held["converged"] or held["reason"] == "stalled")  # as refresh_on_stall counts
    state, _, params = out["aux"]
    boundary = model._im.input_with_params(model.fixed, params)
    residual = model.boundary_residual(state, params, external_field)
    converged = bool(out["converged"])
    largest = max(residual.normal, residual.pressure, residual.sheet_current)
    return ThreeTermsResult(
        boundary_residual=residual, initial_boundary_residual=initial, config=config,
        start="nestor" if initial_boundary is not None else "deck", converged=converged,
        success=converged and largest <= config.max_residual,
        message=(f"{out['accepted']} Levenberg-Marquardt steps after the trust region; "
                 + ("the cost stopped improving" if converged else "stopped before the cost stopped improving")
                 + f"; interface residuals B.n {residual.normal:.2e}, pressure {residual.pressure:.2e}, "
                 f"sheet current {residual.sheet_current:.2e}"
                 + ("" if largest <= config.max_residual else f" (above the accepted {config.max_residual:.0e})")),
        cost=float(0.5 * out["rows"] @ out["rows"]), nfev=int(out["nfev"]), njev=int(out["njev"]), x=out["x"],
        jacobian=out["jacobian"], input=boundary, equilibrium=solve_equilibrium(boundary, initial_state=state),
        model=model, aux=out["aux"])


class ThreeTermsModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from :mod:`vmex.core.implicit`, with the boundary,
    PHIEDGE and current profile as traced :class:`~vmex.core.implicit.ImplicitParams`.  The external field enters the
    compiled interface rows as an argument.  So neither new coils nor new plasma parameters recompile anything, which
    is what an optimizer that solves the free boundary at every step needs.

    Coordinates:

    - ``x``: the boundary coefficients up to ``max_mode`` and ``RBC(0,0)``
      (:func:`~vmex.core.optimize.pack_boundary`).  ``x0`` are the deck's and ``x_scale`` their ESS scales.
    - ``params``: an ``ImplicitParams`` (``params0`` the deck's).  ``with_boundary(params, x)`` replaces its boundary.

    The rows are those of :func:`solve_free_boundary_three_terms`, with the quadrature plan and the labelling
    reference fixed on the deck's boundary.  ``seed`` is that boundary's ``(state, mask)``.  ``labelling`` and
    ``label_weight`` are those of :class:`ThreeTermsConfig` (default flat arc at :data:`LABEL_WEIGHTS`).  Flat arc
    has no rows until :meth:`set_label_rows` sets them.

    Methods:

    - :meth:`solve_boundary`: the K = 0 boundary, cold or warm;
    - :meth:`evaluate`: rows at a boundary;
    - :meth:`linearize`: their Jacobian and the state responses;
    - :meth:`pullback`: reverse-mode gradients of other functions of the equilibrium;
    - :meth:`boundary_residual` and :meth:`solve`;
    - ``rows(state, params, field)`` and ``rows_at(state, runtime, field)``: the traceable rows themselves.

    ``quadrature = (quad_nt, quad_np)``
        Fixes the singular quadrature instead of planning it to ``digits``.  ``quad_nt`` is a multiple of
        ``nfp * nphi``.  The plan's error estimate alone can exhaust a GPU on a strongly shaped boundary.  Use
        ``(4 nfp nphi, 2 ntheta)``, at no extra memory.  With ``quad_np = ntheta`` the tangential plasma field on a
        vacuum surface is off by ~3e-4 ``|B|``, the fit absorbs it in the boundary, and its near-resonant B.n
        harmonics grow several-fold.  On the 8 x 8 vacuum benchmark, field lines then leave the LCFS by 7 mm instead
        of 1 mm.
    ``chunk``
        Bounds the Jacobian columns per batch.
    ``bootstrap``, ``bootstrap_helicity``, ``bootstrap_weight``
        ``bootstrap``, a :class:`~vmex.core.bootstrap.KineticProfiles`, makes the current profile Redl's bootstrap
        current in the same solve.  ``bootstrap_helicity`` is the quasisymmetry ``N`` of the Redl model.  ``x`` gains
        the current's knot values after the ``n_boundary`` boundary coordinates.  The knots are three inside the
        first surface, then one on every half-mesh surface.  The rows gain one self-consistency row per knot, scaled
        by ``bootstrap_weight``.  The weight is large by default: the current is no interface condition to trade
        against the others, so its block is solved far below their floor.  See
        :class:`~vmex.core.bootstrap.HalfMeshCurrent`, which needs ``ns <= 97`` and a prescribed-current deck.  The
        current then follows the equilibrium instead of being an input, and :meth:`bootstrap_residual` reports how
        far it is from Redl's.
    ``fixed_boundary=True``
        With ``bootstrap``, holds the boundary at that of ``params`` and solves only the current.  ``x`` is the
        current values alone, the rows are the bootstrap ones, and no external field or virtual casing enters.  This
        is the fixed-boundary single stage's equilibrium, with the same Jacobians and pullbacks.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=48, ntheta=48, digits=4,
                 weights=(1.0, 1.0, 1.0), net_current_weight=1.0, label_weight=None, chunk=8,
                 quadrature=None, bootstrap=None, bootstrap_helicity=0, bootstrap_weight=100.0,
                 fixed_boundary=False, labelling="flat_arc"):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

        self._im = im
        if labelling not in LABEL_WEIGHTS and (labelling, label_weight) != ("none", 0.0):  # "none": at weight 0
            raise ValueError(f"labelling must be one of {list(LABEL_WEIGHTS)}, got {labelling!r}")
        if label_weight is None:
            label_weight = LABEL_WEIGHTS[labelling]
        if not label_weight:  # no labelling rows: the same problem whichever labelling was named
            labelling, label_weight = "none", 0.0
        if _edge_pressure_error(inp):
            raise ValueError(_edge_pressure_error(inp))
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.bootstrap = None if bootstrap is None else HalfMeshCurrent(fixed, bootstrap, bootstrap_helicity,
                                                                       bootstrap_weight)
        self.fixed_boundary = bool(fixed_boundary)
        self.labelling, self.label_weight = labelling, float(label_weight)
        self._labels = None  # flat arc's rows, once set: (label_weight * M^T, reference arc distribution)
        self._arc_label = _arc_label(fixed)
        if self.fixed_boundary and self.bootstrap is None:
            raise ValueError("fixed_boundary=True solves only the bootstrap current: pass bootstrap=")
        if self.bootstrap is not None:
            self.fixed = fixed = self.bootstrap.deck
        self.max_mode = int(max(fixed.mpol - 1, fixed.ntor) if max_mode is None else max_mode)
        device = jax.devices()[0]

        def configure(deck):
            cfg = im.make_config(deck, multigrid=True, hot_restart=True)
            self.cfg = cfg = im._canonical_config(replace(cfg, device=device))
            self.params0 = im.params_from_input(deck, device=device)
            im._template_runtime(cfg)

        configure(fixed)
        self.x0 = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
        if self.fixed_boundary:
            self.x0, self.x_scale = self.x0[:0], self.x_scale[:0]
        self.n_boundary = nb = self.x0.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self.chunk = int(chunk)

        def with_boundary(params, x):
            if not self.fixed_boundary:
                rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        seed = self.solve(self.params0)
        if seed is None:
            from .errors import VmecConvergenceError
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        if np.size(fixed.ns_array) > 1:
            # A radial ladder is the cold start's: every later solve starts from a state on the final grid, which a
            # ladder would first interpolate down to its coarsest grid.
            last = lambda a: np.atleast_1d(np.asarray(a))[-1:]  # noqa: E731
            self.fixed = replace(self.fixed, ns_array=last(fixed.ns_array), ftol_array=last(fixed.ftol_array),
                                 niter_array=last(fixed.niter_array))
            configure(self.fixed)
            im._HOT_CACHE[self.cfg] = state
        self.seed = (state, mask)
        if not self.fixed_boundary:
            self._interface_setup(state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight,
                                  labelling)
        self._implicit_setup()

    def coordinates(self, inp: VmecInput) -> np.ndarray:
        """The model coordinates of another deck's boundary (and, with ``bootstrap``, its current), as ``x0`` of
        :meth:`solve_boundary`: a start that keeps this model's labelling reference and quadrature plan."""
        from .bootstrap import HalfMeshCurrent
        from .optimize import pack_boundary

        x = pack_boundary(inp, self.max_mode, vary_major_radius=True)[:self.n_boundary]
        if self.bootstrap is None:
            return x
        b, n = self.bootstrap, self.bootstrap.knots.size
        if inp.pcurr_type == "line_segment_ip":  # this model's own profile: its knot values (a WOUT's CURTOR is solved)
            knots = np.asarray(inp.ac_aux_s, dtype=float)
            if knots.size >= n and np.allclose(knots[:n], b.knots, rtol=0, atol=1e-12):
                return np.r_[x, np.asarray(inp.ac_aux_f, dtype=float)[1:n - 1]]
        return np.r_[x, HalfMeshCurrent(inp, b.profiles, b.helicity_n, b.weight).x0]

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight,
                         labelling="flat_arc"):
        """The interface rows, with the quadrature plan and labelling reference of the seed ``state``."""
        from .freeboundary import _vacuum_scalars
        from .optimize import solve_equilibrium

        im, cfg, fixed = self._im, self.cfg, self.fixed
        runtime = im.runtime_from_params(self.params0, cfg)
        surface = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        quad_nt, quad_np = (None, None) if quadrature is None else map(int, quadrature)
        precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=quad_nt, quad_np=quad_np)
        if labelling == "anchored":  # the reference: the seed's points and poloidal tangents of unit mean length
            gamma0 = jnp.asarray(surface.gamma)
            e_theta = _poloidal_tangent(gamma0)
            length = jnp.linalg.norm(e_theta, axis=0)
            tangent = e_theta / (length * jnp.mean(length))[None]
        scale = jnp.repeat(jnp.asarray(weights, dtype=float), jnp.asarray([1, 1, 3]))[:, None, None]
        # A closed curve inside the coils for G_coil: the seed's magnetic axis.
        axis = _axis_loop(solve_equilibrium(fixed, initial_state=state).wout)

        def rows_at(state, runtime, field):
            values, area, gamma = _interface_terms(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                   digits=digits, precision=precision, p_edge=0.0)
            rows, phase = scale * values, jnp.zeros(0)
            if labelling == "equal_arc":  # a labelling of the current surface alone: no reference to the start
                label, phase = _equal_arc_rows(gamma)
                rows, phase = jnp.concatenate([rows, label_weight * label[None]]), label_weight * phase / np.sqrt(nphi)
            elif labelling == "anchored":
                rows = jnp.concatenate([rows, label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0)[None]])
            interface = jnp.concatenate([(jnp.sqrt(area)[None] * rows).ravel(), phase])
            if not net_current_weight:
                return interface
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            return jnp.concatenate([interface, (net_current_weight * (G - G_coil) / G_coil)[None]])

        self.rows_at = rows_at
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field, runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])

    def _label_rows(self, params, weights, reference):
        """The label rows at ``params`` (they depend on the boundary alone)."""
        return weights @ (self._arc_label(params.rbc, params.zbs) - reference)

    def labelled(self, field, labels=None):
        """``field`` with the label rows' arrays ``labels`` (default: the model's current ones)."""
        labels = self._labels if labels is None else labels
        return field if labels is None else _LabelledField(field, *labels)

    def _implicit_setup(self):
        """The rows (interface, then bootstrap) and their implicit tangents, pushes and pullbacks."""
        im, cfg = self._im, self.cfg

        def rows(state, params, field):
            runtime = im.runtime_from_params(params, cfg)
            labels = field if isinstance(field, _LabelledField) else None
            parts = [] if self.fixed_boundary else [self.rows_at(state, runtime, _external_field(field))]
            self._n_interface = sum(part.shape[0] for part in parts)  # static: where the label rows start
            if labels is not None:
                parts.append(self._label_rows(params, labels.weights, labels.reference))
            if self.bootstrap is not None:
                parts.append(self.bootstrap.rows(state, runtime, params))
            return jnp.concatenate(parts)

        self.rows = rows
        self._rows = jax.jit(rows)
        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch, bands):
            """Projected state responses to the stacked parameter tangents, chunked, on the config's block
            factorization at ``state`` (``bands``, as upstream's block Jacobian: the program builds no second one)."""
            return im._implicit_evolved_tangent_multi_rhs(
                params, cfg, jax.lax.stop_gradient(state), mask, batch, active_fields=active,
                probe_chunk_size=self.chunk, response_chunk_size=self.chunk,
                certify_rtol=float(cfg.jacobian_adjoint_tol), certify_maxiter=int(cfg.jacobian_adjoint_maxiter),
                bands=bands)[0]

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
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)

        def pullback(fun, state, mask, params, cotangents, *args, bands=None):
            """``ImplicitParams`` gradient of ``cotangents[i] . fun(state, params, *args)`` through the equilibrium.

            Reverse mode with one block factorization shared by every row (upstream's implicit pullback): for a few
            scalar rows of an expensive function this costs a few reverse passes, not one forward pass per
            parameter direction.  ``fun`` may be a tuple of functions with ``cotangents`` a tuple of their rows'
            cotangents: each row then runs only its own function's reverse pass, and the rows come out in order.
            ``bands`` (:meth:`stored_bands`) is that factorization where the program cannot build its own.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk, bands=bands)
            funs, blocks = (fun, cotangents) if isinstance(fun, tuple) else ((fun,), (cotangents,))
            parts = []
            for f, c in zip(funs, blocks):
                _, vjp = jax.vjp(lambda s, p, f=f: f(s, p, *args), frozen, params)

                def row(c, vjp=vjp):
                    state_bar, params_bar = vjp(c)
                    return jax.tree.map(jnp.add, params_bar, back(state_bar)[0])

                parts.append(jax.lax.map(row, c))
            return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)

        self.pullback = pullback
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._radius = None  # the boundary steps' trust radius, carried from one solve to the next

    def stored_bands(self, params, state, mask):
        """The config's block factorization at ``state`` for a pullback of a system too large to build one in its
        own program (``implicit._large_bands``); ``None`` otherwise.  Factored at ``state`` itself: the adjoint's
        one refinement pass does not absorb a reused factorization's distance."""
        im, cfg = self._im, self.cfg
        large = im._large_bands(cfg.resolution.ns, len(im._active_state_fields(cfg)) * int(mask.R_cos.shape[1]),
                                state.R_cos.dtype)
        return im._block_bands(cfg, params, state, mask, z=state) if large else None

    def solve(self, params, seed=None):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified."""
        im, cfg = self._im, self.cfg
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
        return summarize_boundary_residual(*map(np.asarray, self._terms(state, params, _external_field(field))))

    def bootstrap_residual(self, state, params) -> float:
        """``max_j |I'(s_j) - I'_Redl(s_j)| / max_j |I'_Redl(s_j)|`` of a solved state (with ``bootstrap``)."""
        difference, target = map(np.asarray, self.bootstrap.mismatch(
            state, self._im.runtime_from_params(params, self.cfg), params))
        return float(np.max(np.abs(difference)) / np.max(np.abs(target)))

    def evaluate(self, x, params, field, seed=None):
        """Interface rows at boundary ``x``: ``(rows, (state, mask, params_x))``, or ``None`` if the solve failed."""
        params_x = self.with_boundary(params, jnp.asarray(x))
        if seed is None and self._linearization is not None:
            x_ref, state_ref, mask_ref, dz = self._linearization
            seed = self._predict(state_ref, mask_ref, dz, jnp.asarray(np.asarray(x) - x_ref), params_x)
        solved = self.solve(params_x, seed)
        if solved is None:
            return None
        state, mask = solved
        return np.asarray(self._rows(state, params_x, field)), (state, mask, params_x)

    def boundary_directions(self, params, x):
        """``ImplicitParams`` tangents of the boundary coordinates, stacked."""
        eye = jnp.eye(np.size(x))
        return jax.vmap(lambda e: jax.jvp(lambda xx: self.with_boundary(params, xx), (jnp.asarray(x),), (e,))[1])(eye)

    def set_label_rows(self, x, jacobian):
        """Hold the fit's flat, tangential directions at boundary ``x``; returns their number.

        The interface rows leave the boundary's labelling free: a few directions of the boundary coordinates are
        nearly flat for them, and a fit stops anywhere along those.  With ``jacobian`` the fit's rows' Jacobian at
        ``x`` (:meth:`linearize`), the held directions are the right singular vectors of its boundary columns
        (scaled to unit norm), the largest left out, with a surface displacement of normal share below
        :data:`LABEL_NORMAL_SHARE` (a relabelling, not a shape change).  The rows are
        ``label_weight * M^T (L - L_x)``, ``L`` the arc-length distribution of the current boundary
        (:func:`_arc_label`), ``L_x`` that of the fit at ``x`` and ``M`` an orthonormal basis of its change along the
        held directions: the labelling's arc distribution held at ``x``'s, as far as those directions reach.  No
        point positions are held, so the surface moves freely and its labelling cannot degenerate.  Zero at ``x`` (an
        absolute equal-arc target costs the interface accuracy: the flat directions are weak, not free).
        The rows keep their number when fewer directions are held (zero rows), so a refresh recompiles nothing.
        """
        if self.labelling != "flat_arc":
            raise ValueError("set_label_rows needs labelling='flat_arc'")
        nb, last = self.n_boundary, self._last_linearize
        self._radius = None  # the next fit is of a new problem
        old = 0 if self._labels is None else self._labels[0].shape[0]
        xb = np.asarray(x, dtype=float)[:nb]
        J = np.asarray(jacobian)[:, :nb]
        if self._labels is not None:  # a refresh: the interface (and bootstrap) rows' Jacobian alone
            J = np.delete(J, np.s_[self._n_interface:self._n_interface + self._labels[0].shape[0]], axis=0)
        norms = np.linalg.norm(J, axis=0)
        norms[norms == 0.0] = 1.0
        _, sv, Vt = np.linalg.svd(J / norms, full_matrices=False)
        A, B = _displacement_metrics(self.fixed, xb, self.max_mode)
        candidates = Vt[1:]  # every direction but the stiffest; the normal share picks the tangential ones
        tangential = lambda u: u @ A @ u < LABEL_NORMAL_SHARE ** 2 * (u @ B @ u)  # noqa: E731
        held = [u for u in candidates if tangential(u / norms)]
        if held:
            L = _jacfwd(self._boundary_arc, jnp.asarray(xb, dtype=float))
            M = np.linalg.qr(np.asarray(L) @ (np.array(held).reshape(-1, nb) / norms).T)[0]
            weights = self.label_weight * M.T
            weights = np.vstack([weights, np.zeros((max(old - weights.shape[0], 0), weights.shape[1]))])
            self._labels = (jnp.asarray(weights), self._boundary_arc(jnp.asarray(xb)))
        elif self._labels is not None:
            self._labels = (jnp.zeros_like(self._labels[0]), self._boundary_arc(jnp.asarray(xb)))
        if last is not None and last[2][0] is jacobian:  # keep that linearization, its label rows swapped: they
            # depend on the boundary alone, so their columns are the new rows' derivatives along its tangents
            key, extra_key, (columns, dz, batch, aux), linearization = last
            new = np.zeros((0, columns.shape[1])) if self._labels is None else np.asarray(jax.lax.map(
                lambda t: jax.jvp(lambda p: self._label_rows(p, *self._labels), (aux[2],), (t,))[1], batch,
                batch_size=32)).T  # in batches, as _jacfwd
            i = self._n_interface
            columns = np.vstack([columns[:i], new, columns[i + old:]])
            self._last_linearize = (key, extra_key, (columns, dz, batch, aux), linearization)
            self._linearization = linearization
        else:  # a cached linearization of other label rows must not answer for the new ones
            self._last_linearize = None
        return len(held)

    def _boundary_arc(self, z):
        """The arc-length distribution (:func:`_arc_label`) of the boundary coordinates ``z``."""
        from .optimize import boundary_arrays_from_x

        return self._arc_label(*boundary_arrays_from_x(self.fixed, z, self.max_mode, vary_major_radius=True)[:2])

    def label_rows_curvature(self, x, params):
        """``sum_i g_i(x) Hess g_i(x)`` of the flat-arc label rows ``g`` (nonlinear in the boundary) at the fit
        ``x`` with parameters ``params``, over all fit coordinates: the second-order term of the fit's stationarity
        (Gauss-Newton's ``J^T J`` misses it while ``g`` is not zero).  ``None`` without label rows."""
        if self._labels is None:
            return None
        (M, reference), nb = self._labels, self.n_boundary
        xb = jnp.asarray(x, dtype=float)[:nb]
        g = M @ (self._arc_label(params.rbc, params.zbs) - reference)
        S = np.zeros((np.size(x), np.size(x)))
        S[:nb, :nb] = np.asarray(_jacfwd(jax.grad(lambda z: g @ (M @ self._boundary_arc(z))), xb))
        return S

    def linearize(self, x, aux, field, extra=None):
        """Columns of the interface rows along the boundary coordinates (and the ``extra`` parameter tangents).

        Returns ``(J, dz, params_batch, aux)``: the Jacobian, the projected state responses and parameter tangents of
        its columns, and the state they belong to.  The responses also seed the next trials (first-order predicted
        states).  A call at the boundary, plasma parameters and field of the latest one returns its result (without
        ``extra``'s columns when not asked for).
        """
        state, mask, params_x = aux
        key = _fingerprint(x, params_x, _external_field(field))
        extra_key = None if extra is None else _fingerprint(extra)
        last = self._last_linearize
        if last is not None and last[0] == key and extra_key in (None, last[1]):
            J, dz, batch, aux = last[2]
            self._linearization = last[3]
            if extra_key is None and last[1] is not None:  # drop the extra columns
                n = self.x0.size
                J = J[:, :n]
                dz, batch = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
            return J, dz, batch, aux
        directions = self.boundary_directions(params_x, x)
        nb = self.n_boundary
        # Boundary, bootstrap-current and extra directions each in a program of its own: one batch of every
        # direction can exhaust the GPU.
        groups = [jax.tree.map(lambda a: a[:nb], directions)] if nb else []
        if self.bootstrap is not None:
            groups.append(jax.tree.map(lambda a: a[nb:], directions))
        n_coordinates = len(groups)
        if extra is not None:
            groups.append(extra)
        join = lambda *a: jnp.concatenate(a)  # noqa: E731
        # One tangent solve (one block factorization) for every group; the pushes stay per group.
        self._last_linearize = self._linearization = None  # dropped first: two sets of responses need not fit
        bands = self._im._block_bands(self.cfg, params_x, state, mask)
        dz_all = self._tangents(params_x, state, mask, jax.tree.map(join, *groups), bands)
        ends = np.cumsum([jax.tree.leaves(batch)[0].shape[0] for batch in groups])
        parts = [(jax.tree.map(lambda a: a[lo:hi], dz_all), batch)
                 for lo, hi, batch in zip(np.r_[0, ends[:-1]], ends, groups)]
        columns = np.hstack([np.asarray(self._push_rows(state, mask, params_x, dz, batch, field)).T
                             for dz, batch in parts])
        del bands, dz_all
        if self._im._large_bands(self.cfg.resolution.ns, len(self._im._active_state_fields(self.cfg))
                                 * int(mask.R_cos.shape[1]), state.R_cos.dtype):
            # a large system keeps its responses (the trials' first-order seeds) in host memory: the GPU then
            # holds one block factorization at a time beside them (16 x 16 on 201 surfaces: 4.7 GiB)
            parts = [(jax.tree.map(np.asarray, dz), batch) for dz, batch in parts]
            join = lambda *a: np.concatenate(a)  # noqa: E731
        dz = jax.tree.map(join, *[dz for dz, _ in parts[:n_coordinates]])
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        if extra is not None:
            dz = jax.tree.map(join, dz, parts[-1][0])
        batch = jax.tree.map(join, *[batch for _, batch in parts])
        self._last_linearize = (key, extra_key, (columns, dz, batch, aux), self._linearization)
        return columns, dz, batch, aux

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, ftol=1e-4, jacobian_ftol=1e-2, max_nfev=60,
                       target_cost=None, refresh_on_stall=False, slow_cost=None, verbose=0):
        """K = 0 boundary for ``params`` and ``field``.

        The solve has two phases.  The fresh-Jacobian fit is a trust region that takes a new Jacobian at every step
        and runs to ``jacobian_ftol``.  The reused-Jacobian steps are Levenberg-Marquardt steps on one Jacobian
        (:func:`_levenberg_marquardt`) and run to ``ftol``.

        With ``jacobian`` (from a nearby solve) the reused-Jacobian steps start from ``x0``, and the result is theirs.
        If they neither reduce the cost, converge nor stall, or end above ``target_cost``, they run again on one
        fresh Jacobian taken where they stopped.  If that still ends above ``target_cost``, both phases run from there.

        Without ``jacobian`` both phases run: the fresh-Jacobian fit, then the reused-Jacobian steps on its last
        Jacobian (see :func:`solve_free_boundary_three_terms`).  With ``refresh_on_stall``, reused-Jacobian steps that
        stalled on a stale Jacobian retry once on a fresh one there.  ``slow_cost`` is the reused-Jacobian steps'
        threshold (:func:`_levenberg_marquardt`, default ``LM_SLOW_COST``): a step is taken only if it leaves the cost
        below this fraction.

        Returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask, params), ``nfev``, ``njev``,
        ``accepted``, ``converged``, ``reason`` and ``radius`` (of the reused-Jacobian steps).
        """
        import scipy.optimize

        slow_cost = LM_SLOW_COST if slow_cost is None else slow_cost
        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        nfev = njev = 0
        if jacobian is not None:
            first = self.evaluate(x, params, field)
            if first is None:
                raise RuntimeError("the warm-start boundary has no certified equilibrium")
            out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), x, *first, jacobian,
                                       ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius,
                                       slow_cost=slow_cost)
            self._radius = out["radius"]
            close = target_cost is None or 0.5 * out["rows"] @ out["rows"] <= target_cost
            if (out["accepted"] or out["converged"] or out["reason"] == "stalled") and close:
                return dict(out, jacobian=jacobian, njev=0)  # stalled at a close fit: nothing left on this Jacobian
            # One fresh Jacobian where the steps stalled, and the same steps again, before a full trust region.
            J1, _, _, aux1 = self.linearize(out["x"], out["aux"], field)
            again = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), out["x"], out["rows"], aux1, J1,
                                         ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius,
                                         fresh=True, slow_cost=slow_cost)
            self._radius = again["radius"]
            again["nfev"] += out["nfev"] - 1  # its first evaluation is the last of ``out``
            if target_cost is None or 0.5 * again["rows"] @ again["rows"] <= target_cost:
                return dict(again, jacobian=J1, njev=1)
            x, nfev, njev = again["x"], again["nfev"], 1
        memo = {}

        def fun(z):
            got = self.evaluate(z, params, field)
            if got is None:  # no state here: a Jacobian asked at z must not linearize another point's state
                memo.update(key=z.tobytes(), rows=None, aux=None)
                return np.full(memo.get("size", 1), 1e6)
            memo.update(key=z.tobytes(), size=got[0].size, rows=got[0], aux=got[1])
            return got[0]

        def jac(z):
            nonlocal njev
            if memo.get("key") != z.tobytes():
                fun(z)
            if memo["aux"] is None:
                raise RuntimeError("the boundary where the fit needs a Jacobian has no certified equilibrium")
            njev += 1
            J, _, _, memo["aux"] = self.linearize(z, memo["aux"], field)
            memo.update(jacobian=J, jkey=z.tobytes())
            return memo["jacobian"]

        fit = scipy.optimize.least_squares(fun, x, jac=jac, x_scale=self.x_scale, ftol=max(ftol, jacobian_ftol),
                                           xtol=1e-5, gtol=1e-10, max_nfev=max_nfev, verbose=verbose)
        nfev += int(fit.nfev)
        if memo.get("key") != fit.x.tobytes():
            fun(fit.x)
            nfev += 1
        J = jac(fit.x) if memo.get("jacobian") is None or memo.get("jkey") != fit.x.tobytes() else memo["jacobian"]
        out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), fit.x, memo["rows"], memo["aux"], J,
                                   ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius,
                                   slow_cost=slow_cost)
        self._radius = out["radius"]
        if refresh_on_stall and not out["converged"]:  # stalled on a stale Jacobian: one fresh one there
            J, _, _, aux = self.linearize(out["x"], out["aux"], field)
            again = _levenberg_marquardt(lambda z: self.evaluate(z, params, field), out["x"], out["rows"], aux, J,
                                         ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius,
                                         fresh=True, slow_cost=slow_cost)
            self._radius, njev = again["radius"], njev + 1
            # still stalled on a fresh Jacobian: the steps gain nothing the rows can resolve, the point is
            # stationary (DESC's least squares counts relative gains below its ftol, 1e-2, as converged)
            out = dict(again, nfev=out["nfev"] + again["nfev"] - 1,
                       converged=again["converged"] or again["reason"] == "stalled")
        return dict(out, jacobian=J, nfev=nfev + out["nfev"] - 1, njev=njev)  # the steps start at an evaluated point


#: Constants of :func:`_levenberg_marquardt`.
LM_RADIUS_FLOOR = 0.05   # a carried radius starts at no less than this fraction of the first Gauss-Newton step
LM_RADIUS_STOP = 1e-4    # the steps stop when the radius is below this fraction of the Gauss-Newton step
LM_POOR_GAIN = 0.25      # a step that gained less than this fraction of its predicted reduction shrinks the radius
LM_SHRINK = 0.25         # the shrunk radius, as a fraction of the poor step's length
LM_GOOD_GAIN = 0.75      # a step that gained more than this fraction of its predicted reduction may grow the radius
LM_AT_RADIUS = 0.99      # a step longer than this fraction of the radius counts as a step at the radius
LM_GROW = 2.0            # the radius' growth factor after a good step at the radius
LM_SLOW_COST = 0.98      # a step that leaves the cost above this fraction of it is slow (it gained under 2%)
#                          (5 % left QH's interface residuals 15 % higher, 1 % cost 70-85 % more solves)
LM_SLOW_STEPS = 3        # this many slow steps in a row stop the fit as stalled: the Jacobian is stale there
LM_BISECTIONS = 60       # bisections of log(damping) for the step of the radius' length: 2^-60 of the bracket


def _levenberg_marquardt(evaluate, x, r, aux, J, *, ftol, max_nfev, verbose, radius=None, fresh=False,
                         slow_cost=LM_SLOW_COST):
    """Reused-Jacobian steps: trust-region Gauss-Newton steps on one fixed Jacobian ``J`` from ``x``.

    ``evaluate(x)`` returns ``(r, aux)``, or ``None`` where the solve fails.  Each step costs one evaluation.  These
    steps follow the fresh-Jacobian fit of :meth:`ThreeTermsModel.solve_boundary`, or replace it when a nearby
    Jacobian is given.  The ``LM_*`` constants hold the numbers below.

    The step.  In Marquardt-scaled coordinates it is the Levenberg-Marquardt step whose length is the trust radius,
    or the Gauss-Newton step if that is shorter.  Both come from one SVD of ``J``.

    The radius.  It shrinks to ``LM_SHRINK`` of the length of a step that gained less than ``LM_POOR_GAIN`` of its
    predicted reduction, or that failed to solve.  It grows by ``LM_GROW`` after a step at the radius that gained
    more than ``LM_GOOD_GAIN`` of its predicted reduction.  ``radius`` carries a radius over, raised to at least
    ``LM_RADIUS_FLOOR`` of the first Gauss-Newton step.  The default is that step's length.

    The stop.  ``reason`` is one of:

    - ``"ftol"``: a relative cost change of ``ftol`` that the linear model predicted;
    - ``"stalled"``: ``LM_SLOW_STEPS`` steps in a row each gained under ``1 - LM_SLOW_COST`` of the cost (the
      Jacobian is stale there), or a slow step left the radius as it was (the next step would be the same one).
      Slow steps are not taken, so the fit ends where its last real gain left it and a fit started there with the
      same Jacobian and radius returns the same point;
    - ``"radius"``: the radius fell below ``LM_RADIUS_STOP`` of the Gauss-Newton step;
    - ``"max_nfev"``: the evaluations ran out.

    ``fresh`` marks ``J`` as taken at ``x`` where the last steps stalled.  Until a step is accepted, a step that
    fails is then the radius' fault, not the Jacobian's.  The carried radius is kept as it is.  Failed steps only
    shrink it, down to a predicted relative gain of ``ftol`` (``"radius"``).  The Gauss-Newton step of a nearly
    singular ``J`` is long in directions the rows barely see, so its length is no yardstick.  A short step that gains
    what was predicted is not a slow one.  The radius returned is the last accepted step's.
    """
    J = np.asarray(J)
    if not np.all(np.isfinite(J)):
        raise RuntimeError("nonfinite boundary Jacobian")
    norms = np.linalg.norm(J, axis=0)
    norms[norms == 0.0] = 1.0
    try:
        U, sv, Vt = np.linalg.svd(J / norms, full_matrices=False)
    except np.linalg.LinAlgError as error:  # a rejected trial, not the end of the run
        raise RuntimeError(f"boundary Jacobian: {error}") from error
    nfev, accepted, converged, slow, reason, last = 1, 0, False, 0, "max_nfev", None
    started = time.perf_counter()

    def step(g, lam):
        return -(sv / (sv**2 + lam)) * g  # in the singular basis of the scaled Jacobian

    while nfev < max_nfev:
        g = U.T @ r
        cost = 0.5 * r @ r
        if 0.5 * g @ g <= ftol * cost:  # the Gauss-Newton step's predicted gain
            converged, reason = True, "ftol"
            break
        full = np.linalg.norm(step(g, 0.0))
        if nfev == 1:  # a carried radius, but not far below this Jacobian's own step
            radius = full if radius is None else radius if fresh else max(radius, LM_RADIUS_FLOOR * full)
        if not fresh and radius < LM_RADIUS_STOP * full:
            reason = "radius"
            break
        lam = 0.0
        if full > radius:  # the damping whose step has the radius' length (bisection in log lam)
            lo, hi = np.log(sv[-1] ** 2 * 1e-12 + 1e-300), np.log(sv[0] ** 2 * (full / radius) + 1e-300)
            for _ in range(LM_BISECTIONS):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if np.linalg.norm(step(g, np.exp(mid))) > radius else (lo, mid)
            lam = np.exp(hi)
        z = step(g, lam)
        length = np.linalg.norm(z)
        predicted = cost - 0.5 * (r @ r - g @ g + np.sum((g + sv * z) ** 2))
        if fresh and predicted <= ftol * cost:
            reason = "radius"
            break
        x_new = x + (Vt.T @ z) / norms
        got = evaluate(x_new)
        nfev += 1
        cost_new = np.inf if got is None else 0.5 * got[0] @ got[0]
        rho = (cost - cost_new) / predicted if predicted > 0 else -np.inf
        if verbose:
            print(f"Levenberg-Marquardt step {nfev - 1}: |step| {length:.2e} (radius {radius:.2e}), cost "
                  f"{cost:.6e} -> {cost_new:.6e}, gain ratio {rho:.2f} ({time.perf_counter() - started:.1f} s)",
                  flush=True)
        radius_before = radius
        if rho < LM_POOR_GAIN:
            radius = LM_SHRINK * length
        elif rho > LM_GOOD_GAIN and length > LM_AT_RADIUS * radius:
            radius = LM_GROW * radius
        at_x = fresh and not accepted  # J is still the Jacobian at x
        small_gain = cost_new > slow_cost * cost
        # not the Jacobian's fault: a failed step at x itself, or a step on a fresh J that gained what was predicted
        excused = (at_x and cost_new >= cost) or (fresh and rho > LM_GOOD_GAIN)
        slow = slow + 1 if small_gain and not excused else 0
        if slow and radius == radius_before:  # not taken, and the next step would be this one again
            reason = "stalled"
            break
        if cost_new < cost and not slow:  # a slow step is not taken: a fit started where this one ends, ends there
            x, (r, aux), accepted, last = x_new, got, accepted + 1, length
            if cost - cost_new <= ftol * cost and rho > LM_POOR_GAIN:  # a small gain the linear model foresaw
                converged, reason = True, "ftol"
                break
        if slow == LM_SLOW_STEPS:  # slow steps in a row: the Jacobian no longer describes the rows here
            reason = "stalled"
            break
    return dict(x=x, rows=r, aux=aux, nfev=nfev, accepted=accepted, converged=converged,
                radius=last if fresh and last is not None else radius, reason=reason)
