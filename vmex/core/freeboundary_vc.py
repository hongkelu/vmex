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
(:class:`ThreeTermFreeBoundaryModel`).  At an exact
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

Entry points, from the most to the least packaged:

- a free-boundary deck with ``!@VMEX BOUNDARY_CONDITION = THREE_TERM``
  (``vmex input.case``, :func:`~vmex.core.multigrid.solve_file`), the CLI flag
  ``--boundary-condition three-term``, or
  ``solve_free_boundary_multigrid(..., boundary_condition="three_term")``;
- :func:`solve_free_boundary_three_term`: one solve with every control, and
  ``previous=`` to restart from an earlier result after the coils change;
- ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
  (:class:`ThreeTermFreeBoundaryProblem`): single-stage optimization, a loss
  and constraint rows of coils and plasma parameters at their free boundary,
  for :func:`vmex.core.optimize.minimize`;
- :class:`ThreeTermFreeBoundaryModel`: the residual, its Jacobian, state tangents and
  pullbacks for one deck and any external field or plasma parameters, without
  recompiling -- the pieces an optimizer needs;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. to
  check a NESTOR result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .bootstrap import CLOSURE_ATOL, CLOSURE_RTOL
from .errors import TrialRejected, VmecConvergenceError, VmecError
from .input import VmecInput
from .problem import FunctionProblem


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
                            sheet_current=rms(np.linalg.norm(rows[2:], axis=0)))


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


def _equal_arc_rows(gamma):
    """Rows that vanish when the boundary's poloidal angle is equal-arc on every toroidal cut and the m = 1 part of
    ``R + i Z`` has zero phase there: one labelling of the surface itself, the same from any start.  The phase is
    the angle of that part, in (-pi, pi]: zero on one branch only (``Im c1 / |c1|`` also vanished for the angle
    shifted by pi), and not finite where the part vanishes (a rejected trial).

    ``gamma`` is ``(3, nphi, ntheta)`` Cartesian points.  Returns ``(arc, phase)``: ``|d gamma/d theta|`` relative to
    its mean on each cut, minus one, ``(nphi, ntheta)``; and the phase of ``c1`` per cut, ``(nphi,)``.
    """
    ntheta = gamma.shape[-1]
    wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
    e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma, axis=-1), axis=-1))
    speed = jnp.linalg.norm(e_theta, axis=0)
    arc = speed / jnp.mean(speed, axis=-1, keepdims=True) - 1.0
    c1 = jnp.fft.fft(jnp.hypot(gamma[0], gamma[1]) + 1j * gamma[2], axis=-1)[..., 1]
    return arc, jnp.arctan2(jnp.imag(c1), jnp.real(c1))


#: Stationarity (the gain a Gauss-Newton step predicts) a trial's polish rounds aim at.
POLISH_FTOL = 1e-4
#: Most polish rounds (each with a fresh Jacobian) a trial gets to reach its certificate.
POLISH_ROUNDS = 5
#: Most evaluations of one polish round (Levenberg-Marquardt steps with that round's Jacobian).
POLISH_MAX_NFEV = 20
#: Levenberg-Marquardt stops as stalled after this many steps in a row that each gained less than STALL_GAIN of the
#: cost (the Jacobian no longer describes the rows there).
STALL_STEPS = 3
STALL_GAIN = 0.05
#: What counts as a stalled step: ``"gain"`` (it gained under STALL_GAIN of the cost, the default) or ``"model"`` (its
#: gain ratio, actual over predicted reduction, is under 0.25: the Jacobian no longer describes the rows; a step the
#: model predicted well continues however small its gain, the trust radius growing after full steps).
STALL_RULE = "gain"
#: Jacobian of the Levenberg-Marquardt steps between fresh ones: ``"none"`` (fixed, the default) or ``"broyden"``
#: (a rank-one secant update after every accepted step; every certificate still uses a fresh Jacobian).
JACOBIAN_UPDATE = "none"

#: Extrapolate an accepted Gauss-Newton step: after an undamped step that gained over three quarters of its
#: prediction, try 2, 4, 8, ... times it (one evaluation each, current coordinates re-solved), keeping each that
#: lowers the cost.  For a nonzero-residual fit whose Gauss-Newton steps point the same way round after round
#: (the model's curvature along the step too high), this covers in a few evaluations what fresh Jacobians do in
#: as many rounds.
GN_EXTRAPOLATE = False
GN_EXTRAPOLATE_MAX = 6  # doublings at most

#: Geodesic acceleration (Transtrum & Sethna 2012) of each Levenberg-Marquardt step: one extra evaluation at
#: ``x + GEODESIC_H dx`` estimates the rows' second directional derivative ``r''(dx, dx)``; the correction ``a``
#: solves the same damped (reduced) model for it, and the trial is ``x + dx + a / 2`` when ``|a| <= GEODESIC_ALPHA
#: |dx|`` (Marquardt-scaled).  For rows that curve at the step's own scale (the Redl bootstrap rows at 2.5%: the
#: linear model misses by 4-60% at a 1e-2 step) the second-order step follows the curved valley.
GEODESIC = False
GEODESIC_H = 0.1
GEODESIC_ALPHA = 0.75

#: Columns per state-tangent solve of :meth:`ThreeTermFreeBoundaryProblem.dense_hessian`: every column's tangent at
#: once holds a raw-force linearization per column (64 exceed an 80 GB GPU on the QH 2.5% case at 8 x 8).
EXACT_PRODUCTS_CHUNK = 16

#: Version of :meth:`ThreeTermFreeBoundaryProblem.save_checkpoint`'s file.
CHECKPOINT_SCHEMA = "vmex-three-term-checkpoint/4"

#: Largest RMS interface residual (relative to ``|B|``) of a three-term free boundary that counts as solved: a guard
#: against wrong stationary points, not an accuracy standard (about upstream #614's DEL-BSQ gate of 2e-2 on B^2).
ACCEPTED_RESIDUAL = 1e-2


def solve_free_boundary_three_term(
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
    labelling: str = "start",
    m1_complement: str = "solved",
    ftol: float = 1e-4,
    jacobian_ftol: float | None = 1e-2,
    max_nfev: int = 60,
    quadrature: tuple[int, int] | None = None,
    trial_ftol: float | None = None,
    chunk: int = 8,
    bootstrap=None,
    bootstrap_helicity: int = 0,
    bootstrap_weight: float = 100.0,
    previous=None,
    max_residual: float = ACCEPTED_RESIDUAL,
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
    boundary's poloidal labelling, which the interface conditions leave free;
    ``labelling="equal_arc"`` replaces them by equal arc length of the current surface.
    ``m1_complement`` is the interior's m = 1 complement policy (``ImplicitConfig.m1_complement``):
    ``"solved"`` (default) refines it with the evolved dofs so the interior root is unique; ``"frozen"`` is
    VMEC2000's released complement.
    ``net_current_weight`` scales the row ``(G_plasma - G_coil) / G_coil`` of
    the net poloidal current (edge ``rbtor`` against the external field's
    circulation around the seed's magnetic axis), the direction the
    normal-field rows cannot see.

    A trust region with exact implicit Jacobians (:class:`ThreeTermFreeBoundaryModel`)
    runs to a relative cost change of ``jacobian_ftol``; Levenberg-Marquardt
    steps with its last Jacobian then go on to ``ftol``, the floor set by the
    resolution (``None``: the trust region runs to ``ftol``).  ``previous``, an
    earlier result for the same deck (typically before a coil change), starts
    from its boundary with those steps and its Jacobian, and from its model:
    neither new coils nor a repeat solve compile anything.  ``quadrature``,
    ``trial_ftol`` and ``chunk`` are :class:`ThreeTermFreeBoundaryModel`'s (fixed
    singular quadrature, looser trial equilibria, Jacobian columns per batch).
    ``bootstrap`` (kinetic profiles), ``bootstrap_helicity`` and
    ``bootstrap_weight`` solve for a Redl-self-consistent current profile
    together with the boundary (:class:`ThreeTermFreeBoundaryModel`).

    ``converged`` means certified, by the same rule as every trial of the optimization
    (:meth:`ThreeTermFreeBoundaryModel.certify`): the fit's final point, re-solved to the deck's tolerance and
    polished with fresh Jacobians, predicts a Gauss-Newton gain of at most ``max(ftol, POLISH_FTOL)`` of its cost
    (``certificate``; ``message`` says why not).  A stalled finish is not convergence by itself.  A stationary
    point need not be the free boundary (a wrong local minimum is one too), and the interface rows cannot reach
    zero at a finite resolution (spectral truncation, the virtual-casing grid: VMEC2000's own CTH-like free
    boundary scores 6e-4 in vacuum and 2.4e-3 at finite beta), so ``success`` also requires every RMS interface
    residual (:class:`BoundaryResidual`) at or below ``max_residual``, a stated acceptance tolerance above those
    floors; ``boundary_residual`` is always reported.

    Returns a :class:`scipy.optimize.OptimizeResult` with ``x`` (the boundary
    coordinates, then any bootstrap-current values), ``fun`` (the residual), ``jac`` (its last Jacobian), ``cost``,
    ``nfev``, ``njev``, ``input`` (the free boundary, and the current, as a fixed-boundary deck),
    ``equilibrium``, ``model``, ``boundary_residual``, ``converged``, ``certificate`` (``None`` when the fit was
    not certified) and ``initial_boundary_residual`` (:class:`BoundaryResidual` at the start).
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _resolve_mgrid
    from .mgrid import MgridField
    from .optimize import solve_equilibrium

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    if _edge_pressure_error(inp):
        raise ValueError(_edge_pressure_error(inp))
    if external_field is None:  # tricubic: virtual casing samples between the mgrid planes, where the trilinear
        # interpolant has kinks that NESTOR (on the planes) never sees; on the CTH-like deck they cost 0.4 mm
        external_field = MgridField.from_input(inp, _resolve_mgrid(inp, mgrid_path), order=3)
    if previous is not None:
        model, x0, jacobian = previous.model, previous.x, previous.jac
    else:
        start = inp if initial_boundary is None else replace(
            inp, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs,
            raxis_c=initial_boundary.raxis_c, zaxis_s=initial_boundary.zaxis_s)
        model = ThreeTermFreeBoundaryModel(start, max_mode=max_mode, nphi=int(nphi or 48), ntheta=int(ntheta or 48),
                                   digits=digits, weights=weights, net_current_weight=net_current_weight,
                                   label_weight=label_weight, labelling=labelling, quadrature=quadrature,
                                   trial_ftol=trial_ftol,
                                   chunk=chunk, bootstrap=bootstrap, bootstrap_helicity=bootstrap_helicity,
                                   bootstrap_weight=bootstrap_weight, m1_complement=m1_complement)
        x0 = jacobian = None
    seed_state = model.seed[0] if previous is None else previous.aux[0]
    seed_params = model.params0 if previous is None else previous.aux[2]
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, jacobian=jacobian, ftol=ftol,
                               jacobian_ftol=ftol if jacobian_ftol is None else jacobian_ftol,
                               max_nfev=max_nfev, refresh_on_stall=True, verbose=verbose)
    stats: dict[str, Any] = {}
    try:  # the optimization's acceptance rule: one definition of a solved free boundary
        out = model.certify(model.params0, external_field, out, gate=max(ftol, POLISH_FTOL),
                            stats=stats)
        converged, why = True, f"certified (Gauss-Newton gain {out['certificate']['stationarity']:.1e})"
        jacobian_out = out["J"][:, :model.x0.size]
    except TrialRejected as error:
        converged, why, jacobian_out = False, f"not certified: {error}", out["jacobian"]
    state, _, params, _ = out["aux"]
    boundary = model._im.input_with_params(model.fixed, params)
    residual = model.boundary_residual(state, params, external_field)
    largest = max(residual.normal, residual.pressure, residual.sheet_current)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=jacobian_out, cost=0.5 * out["rows"] @ out["rows"],
        nfev=out["nfev"] + stats.get("boundary_evaluations", 0), njev=out["njev"] + stats.get("linearizations", 0),
        success=converged and largest <= max_residual, converged=converged,
        certificate=out.get("certificate") if converged else None,
        message=(f"{out['accepted']} Levenberg-Marquardt steps after the trust region (finish: {out['reason']}); "
                 + why + f"; interface residuals B.n {residual.normal:.2e}, pressure {residual.pressure:.2e}, "
                 f"sheet current {residual.sheet_current:.2e}"
                 + ("" if largest <= max_residual else f" (above the accepted {max_residual:.0e})")),
        input=boundary, equilibrium=solve_equilibrium(boundary, initial_state=state), model=model, aux=out["aux"],
        boundary_residual=residual, initial_boundary_residual=initial)


class ThreeTermFreeBoundaryModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from
    :mod:`vmex.core.implicit`, with the boundary, PHIEDGE and current profile
    as traced :class:`~vmex.core.implicit.ImplicitParams`, and the external
    field enters the compiled interface rows as an argument: neither new coils
    nor new plasma parameters recompile anything, which is what an optimizer
    that solves the free boundary at every step needs.

    Coordinates: ``x`` are the boundary coefficients up to ``max_mode`` and
    ``RBC(0,0)`` (:func:`~vmex.core.optimize.pack_boundary`; ``x0`` the
    deck's, ``x_scale`` their ESS scales), ``params`` an ``ImplicitParams``
    (``params0`` the deck's) whose boundary ``with_boundary(params, x)``
    replaces.  The rows are those of :func:`solve_free_boundary_three_term`,
    with the quadrature plan and the labelling reference fixed on the deck's
    boundary (``seed``: its ``(state, mask)``).

    Methods: :meth:`solve_boundary` (the K = 0 boundary, cold or warm),
    :meth:`evaluate` (rows at a boundary), :meth:`linearize` (their Jacobian
    and the state responses), :meth:`pullback` (reverse-mode gradients of other
    functions of the equilibrium), :meth:`boundary_residual` and
    :meth:`solve`; ``rows(state, params, field)`` and ``rows_at(state,
    runtime, field)`` are the traceable rows themselves.

    ``trial_ftol`` solves the trial equilibria of the boundary steps to a looser
    force residual (each is re-solved to the deck's before it is
    differentiated); ``quadrature = (quad_nt, quad_np)`` fixes the singular
    quadrature instead of planning it to ``digits`` (``quad_nt`` a multiple of
    ``nfp * nphi``), whose error estimate alone can exhaust a GPU on a strongly
    shaped boundary.  Use ``(4 nfp nphi, 2 ntheta)``: with ``quad_np = ntheta``
    the tangential plasma field on a vacuum surface is off by ~3e-4 ``|B|``, the
    fit absorbs it in the boundary, and its near-resonant B.n harmonics grow
    several-fold (8 x 8 vacuum benchmark: field lines leave the LCFS by 7 mm
    instead of 1 mm), at no extra memory.  ``chunk`` bounds the Jacobian
    columns per batch.

    ``bootstrap`` (:class:`~vmex.core.bootstrap.KineticProfiles`, with
    ``bootstrap_helicity`` the quasisymmetry ``N`` of the Redl model) makes the
    current profile Redl's bootstrap current in the same solve: ``x`` gains its
    knot values (three inside the first surface, then one on every half-mesh
    surface) after the boundary coordinates (``n_boundary`` of them), and the
    rows gain one self-consistency row per knot, scaled by ``bootstrap_weight`` (large by default: the current is
    no interface condition to trade against the others, so its block is solved
    far below their floor; see
    :class:`~vmex.core.bootstrap.HalfMeshCurrent`;
    ``ns <= 97``, a prescribed-current deck).  The current then follows the
    equilibrium instead of being an input, and :meth:`bootstrap_residual`
    reports how far it is from Redl's.  ``fixed_boundary=True`` (with
    ``bootstrap``) holds the boundary at that of ``params`` and solves only the
    current: ``x`` is the current values alone, the rows are the bootstrap ones
    and no external field or virtual casing enters -- the fixed-boundary
    single stage's equilibrium, with the same Jacobians and pullbacks.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=48, ntheta=48, digits=4,
                 weights=(1.0, 1.0, 1.0), net_current_weight=1.0, label_weight=1e-2, chunk=8, device=None,
                 trial_ftol=None, quadrature=None, bootstrap=None, bootstrap_helicity=0, bootstrap_weight=100.0,
                 fixed_boundary=False, labelling="start", bootstrap_g_convention="simsopt", m1_complement="solved"):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

        self._im = im
        if _edge_pressure_error(inp):
            raise ValueError(_edge_pressure_error(inp))
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.bootstrap = None if bootstrap is None else HalfMeshCurrent(fixed, bootstrap, bootstrap_helicity,
                                                                       bootstrap_weight, bootstrap_g_convention)
        #: The settings that define the discrete problem (a checkpoint must match them to restore), the bootstrap
        #: closure by its kinetic profiles and conventions (:meth:`~vmex.core.bootstrap.HalfMeshCurrent.identity`).
        self.options = dict(max_mode=max_mode, nphi=nphi, ntheta=ntheta, digits=digits, weights=list(weights),
                            net_current_weight=net_current_weight, label_weight=label_weight, trial_ftol=trial_ftol,
                            quadrature=None if quadrature is None else list(quadrature),
                            bootstrap=None if self.bootstrap is None else self.bootstrap.identity(),
                            bootstrap_helicity=bootstrap_helicity, bootstrap_weight=bootstrap_weight,
                            fixed_boundary=bool(fixed_boundary), labelling=labelling, m1_complement=m1_complement)
        self.fixed_boundary = bool(fixed_boundary)
        if labelling not in ("start", "equal_arc"):
            raise ValueError(f"labelling must be 'start' or 'equal_arc', got {labelling!r}")
        if self.fixed_boundary and self.bootstrap is None:
            raise ValueError("fixed_boundary=True solves only the bootstrap current: pass bootstrap=")
        if self.bootstrap is not None:
            self.fixed = fixed = self.bootstrap.deck
        self.max_mode = int(max(fixed.mpol - 1, fixed.ntor) if max_mode is None else max_mode)
        device = jax.devices()[0] if device is None else device
        cfg = im.make_config(fixed, multigrid=True, hot_restart=True, m1_complement=m1_complement)
        self.cfg = cfg = im._canonical_config(replace(cfg, device=device))
        # Trial points (the boundary steps) may converge to a looser force residual: their rows move by far less
        # than the interface floor, and a state is solved to the deck's tolerance before it is differentiated.
        self.cfg_trial = cfg
        if trial_ftol is not None:
            ftols = np.array(fixed.ftol_array, dtype=float)
            ftols[-1] = max(float(trial_ftol), ftols[-1])
            loose = im.make_config(replace(fixed, ftol_array=ftols), multigrid=True, hot_restart=True,
                                   m1_complement=m1_complement)
            self.cfg_trial = im._canonical_config(replace(loose, device=device))
            im._template_runtime(self.cfg_trial)
        self.params0 = im.params_from_input(fixed, device=device)
        im._template_runtime(cfg)
        self.x0 = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        self.x_scale = _ess_scale(fixed, self.max_mode, 1.2, vary_major_radius=True)
        if self.fixed_boundary:
            self.x0, self.x_scale = self.x0[:0], self.x_scale[:0]
        self.n_boundary = nb = self.x0.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        # The current values are re-solved after every boundary step (variable projection): their rows, scaled by
        # bootstrap_weight, leave their linear model long before the interface rows do (QH 2.5%: a step that halves
        # the interface cost takes them from 1e-7 to order one, and the fit stalls).
        self._projected = np.arange(nb, self.x0.size) if self.bootstrap is not None and nb else None
        self.chunk = int(chunk)

        def with_boundary(params, x):
            if not self.fixed_boundary:
                rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary

        def surface(x):
            """``(R, Z)`` of the boundary of coordinates ``x`` on a 48 x 48 grid over one field period."""
            rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
            th = jnp.linspace(0.0, 2 * jnp.pi, 48, endpoint=False)
            ph = jnp.linspace(0.0, 2 * jnp.pi / int(fixed.nfp), 48, endpoint=False)
            n = (jnp.arange(rbc.shape[0]) - (rbc.shape[0] - 1) // 2) * int(fixed.nfp)
            m = jnp.arange(rbc.shape[1])
            angle = m[None, None, None, :] * th[:, None, None, None] - n[None, None, :, None] * ph[None, :, None, None]
            return (jnp.einsum("tpnm,nm->tp", jnp.cos(angle), rbc), jnp.einsum("tpnm,nm->tp", jnp.sin(angle), zbs))

        def normal_operator(x):
            """``(N, a)``: ``N @ dx`` is the boundary's normal displacement [m] on the grid for a change ``dx`` of
            the boundary coordinates, ``a`` the RMS minor radius."""
            R, Z = surface(x)
            dR, dZ = jax.jacfwd(surface)(x)
            th_axis, ph_axis = 0, 1
            k = jnp.fft.fftfreq(48, 1.0 / 48)
            def d(f, ax):  # spectral derivative along axis ax (the wavenumbers broadcast along that axis)
                kk = k[:, None] if ax == 0 else k[None, :]
                return jnp.real(jnp.fft.ifft(1j * kk * jnp.fft.fft(f, axis=ax), axis=ax))

            R_t, Z_t = d(R, th_axis), d(Z, th_axis)
            R_p, Z_p = d(R, ph_axis) * int(fixed.nfp), d(Z, ph_axis) * int(fixed.nfp)
            cross = jnp.stack([-R * Z_t, Z_t * R_p - R_t * Z_p, R * R_t])  # e_theta x e_phi, cylindrical
            length = jnp.linalg.norm(cross, axis=0)
            n_R, n_Z = cross[0] / length, cross[2] / length
            N = (dR * n_R[..., None] + dZ * n_Z[..., None]).reshape(-1, x.size)
            a = jnp.sqrt(2 * jnp.mean((R - jnp.mean(R)) ** 2 + Z**2))
            return N, a

        self._normal_operator = jax.jit(normal_operator) if nb else None
        seed = self.solve(self.params0)
        if seed is None:
            from .errors import VmecConvergenceError
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        self.seed = (state, mask)
        if not self.fixed_boundary:
            self._interface_setup(state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight,
                                  labelling)
        self._implicit_setup()
        self.row_blocks, self.column_blocks = self._layout(nphi, ntheta, labelling, net_current_weight)
        #: Called with a complete record at every certificate round and at the outcome of :meth:`certify` (``None``:
        #: nothing recorded).  See :meth:`certify`.
        self.recorder = None

    def _layout(self, nphi, ntheta, labelling, net_current_weight):
        """``(row_blocks, column_blocks)``: ``(name, start, stop)`` of each block of the rows and of the model
        coordinates.  Interface rows are component-major: ``B.n``, ``pressure``, ``K_x``, ``K_y``, ``K_z``, ``label``,
        each ``nphi * ntheta`` (``(nphi, ntheta)`` row-major), then the labelling phase rows, the net-current row and
        the bootstrap rows."""
        sizes = []
        if not self.fixed_boundary:
            grid = int(nphi) * int(ntheta)
            sizes += [(name, grid) for name in ("B.n", "pressure", "K_x", "K_y", "K_z", "label")]
            if labelling == "equal_arc":
                sizes.append(("phase", int(nphi)))
            if net_current_weight:
                sizes.append(("net_current", 1))
        if self.bootstrap is not None:
            sizes.append(("bootstrap", int(self.bootstrap.x0.size)))
        rows, start = [], 0
        for name, size in sizes:
            rows.append((name, start, start + size))
            start += size
        nb = self.n_boundary
        columns = [("boundary", 0, nb)] + ([("current", nb, int(self.x0.size))] if self.x0.size > nb else [])
        return rows, columns

    def resolution(self):
        """The effective discretization of this model (what a record or a run manifest reports)."""
        fixed = self.fixed
        return dict(mpol=int(fixed.mpol), ntor=int(fixed.ntor), nfp=int(fixed.nfp),
                    ns=[int(v) for v in np.atleast_1d(fixed.ns_array)],
                    ftol=[float(v) for v in np.atleast_1d(fixed.ftol_array)], max_mode=int(self.max_mode),
                    nphi=self.options["nphi"], ntheta=self.options["ntheta"], quadrature=self.options["quadrature"],
                    digits=self.options["digits"], trial_ftol=self.options["trial_ftol"],
                    n_boundary=int(self.n_boundary), n_coordinates=int(self.x0.size))

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight,
                         labelling="start"):
        """The interface rows, with the quadrature plan and labelling reference of the seed ``state``."""
        from .freeboundary import _vacuum_scalars
        from .optimize import solve_equilibrium

        im, cfg, fixed = self._im, self.cfg, self.fixed
        runtime = im.runtime_from_params(self.params0, cfg)
        surface = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        quad_nt, quad_np = (None, None) if quadrature is None else map(int, quadrature)
        precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=quad_nt, quad_np=quad_np)
        gamma0 = jnp.asarray(surface.gamma)
        wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
        e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma0, axis=-1), axis=-1))
        length = jnp.linalg.norm(e_theta, axis=0)
        tangent = e_theta / (length * jnp.mean(length))[None]
        scale = jnp.repeat(jnp.asarray(weights, dtype=float), jnp.asarray([1, 1, 3]))[:, None, None]
        # A closed curve inside the coils for G_coil: the seed's magnetic axis.
        axis = _axis_loop(solve_equilibrium(fixed, initial_state=state).wout)

        def rows_at(state, runtime, field):
            values, area, gamma = _interface_terms(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                                   digits=digits, precision=precision, p_edge=0.0)
            if labelling == "equal_arc":  # a labelling of the current surface alone: no reference to the start
                label, phase = _equal_arc_rows(gamma)
                label, phase = label_weight * label, label_weight * phase / np.sqrt(nphi)
            else:
                label, phase = label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0), jnp.zeros(0)
            interface = jnp.concatenate([(jnp.sqrt(area)[None] * jnp.concatenate([scale * values, label[None]])).ravel(),
                                         phase])
            if not net_current_weight:
                return interface
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            return jnp.concatenate([interface, (net_current_weight * (G - G_coil) / G_coil)[None]])

        self.rows_at = rows_at
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field, runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])

    def _implicit_setup(self):
        """The rows (interface, then bootstrap) and their implicit tangents, pushes and pullbacks."""
        im, cfg = self._im, self.cfg

        def rows(state, params, field):
            runtime = im.runtime_from_params(params, cfg)
            parts = [] if self.fixed_boundary else [self.rows_at(state, runtime, field)]
            if self.bootstrap is not None:
                parts.append(self.bootstrap.rows(state, runtime, params))
            return jnp.concatenate(parts)

        self.rows = rows
        self._rows = jax.jit(rows)
        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch, system=None):
            """Projected state responses to the stacked parameter tangents (one block factorization, chunked), and
            whether each column's linear solve is certified.  ``system``: the factorization of this linearization
            (``im._raw_block_system`` with ``probe_chunk_size = self.chunk``) when the caller holds it."""
            solution, report = im._implicit_evolved_tangent_multi_rhs(
                params, cfg, jax.lax.stop_gradient(state), mask, batch, active_fields=active,
                probe_chunk_size=self.chunk, response_chunk_size=self.chunk,
                certify_rtol=float(cfg.jacobian_adjoint_tol), certify_maxiter=int(cfg.jacobian_adjoint_maxiter),
                system=system)
            return solution, report.converged

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

        self._tangent_solve = tangents  # traceable, for callers inside a jit holding the factorization
        self._tangents = jax.jit(tangents)
        self.push = push
        self._push_rows = jax.jit(lambda state, mask, params, dz, pb, field: push(
            lambda s, p: rows(s, p, field), state, mask, params, dz, pb))
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)

        def pullback(fun, state, mask, params, cotangents, *args):
            """``ImplicitParams`` gradient of ``cotangents[i] . fun(state, params, *args)`` through the equilibrium, per i.

            Reverse mode with one block factorization shared by every row (upstream's implicit pullback): for a few
            scalar rows of an expensive function this costs a few reverse passes, not one forward pass per
            parameter direction.  ``fun`` may be a tuple of functions with ``cotangents`` a tuple of their rows'
            cotangents: each row then runs only its own function's reverse pass, and the rows come out in order.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk)
            funs, blocks = (fun, cotangents) if isinstance(fun, tuple) else ((fun,), (cotangents,))
            parts = []
            for f, c in zip(funs, blocks):
                _, vjp = jax.vjp(lambda s, p, f=f: f(s, p, *args), frozen, params)

                def row(c, vjp=vjp):
                    state_bar, params_bar = vjp(c)
                    return jax.tree.map(jnp.add, params_bar, back(state_bar)[0])

                parts.append(jax.lax.map(row, c))
            return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)

        def primed_pullback(*args, **kwargs):
            self._prime()
            return pullback(*args, **kwargs)

        self.pullback = primed_pullback
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._radius = None  # the boundary steps' trust radius, carried from one solve to the next

    def _prime(self):
        """(Re)build the configs' template runtimes and boundary tables concretely, before any jitted function of
        this model reads them under a trace.  They are cached (bounded caches): another config's work in the same
        process -- a Picard loop's per-iterate configs, say -- can evict them, and a traced function cannot rebuild
        them (their host-side setup is not traceable)."""
        for cfg in {self.cfg, self.cfg_trial}:
            self._im._template_runtime(cfg)
            self._im._boundary_pack_tables(cfg)

    def solve(self, params, seed=None, tight=True):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified.

        ``tight=False`` solves to the trial tolerance, without the Newton refinement that certifies a state for
        differentiation (:meth:`linearize` re-solves tight first).
        """
        self._prime()
        im, cfg = self._im, (self.cfg if tight else self.cfg_trial)
        if seed is not None:
            im._PERTURB_SEED[cfg] = seed
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), params)
        if tight:
            state, mask, status, _, _ = im._host_solve_and_mask_status(cfg, params_np)
            if int(status) != 0:
                return None
        else:  # a trial point is not differentiated: the forward solve alone, without the Newton refinement
            from .errors import VmecError

            with im._device_context(cfg):
                try:
                    state, mask = im._host_solve_and_mask_impl(cfg, params_np, refine=False)
                except (VmecError, RuntimeError):  # RuntimeError: the relayed sentinel of a VmecError
                    return None
            result = im._LAST_SOLVE[cfg][1]
            fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
            if not (bool(result.converged) or fsq / cfg.ftol <= cfg.max_fsq_ratio):
                return None
        place = lambda tree: jax.device_put(jax.tree.map(jnp.asarray, tree), cfg.device)  # noqa: E731
        return place(state), place(mask)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`BoundaryResidual` of a solved state."""
        return summarize_boundary_residual(*map(np.asarray, self._terms(state, params, field)))

    def bootstrap_residual(self, state, params) -> float:
        """``max_j |I'(s_j) - I'_Redl(s_j)|`` over ``max_j |I'_Redl(s_j)|`` (floored: finite for a zero drive,
        :func:`~vmex.core.bootstrap.closure_scale`) of a solved state (with ``bootstrap``)."""
        return self.bootstrap_closure(state, params)["relative"]

    def bootstrap_closure(self, state, params) -> dict:
        """The physical closure of the bootstrap current at a solved state
        (:meth:`~vmex.core.bootstrap.HalfMeshCurrent.closure`)."""
        return self.bootstrap.closure(state, self._im.runtime_from_params(params, self.cfg), params)

    def evaluate(self, x, params, field, seed=None, tight=False):
        """Interface rows at boundary ``x``: ``(rows, (state, mask, params_x, tight))``, or ``None`` if the solve
        failed; ``tight`` marks a state solved to the deck's tolerance rather than the trial one."""
        self._prime()
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

    def current_columns(self, x, aux, field):
        """The bootstrap-current columns of the rows at ``x`` (a fresh Jacobian block, at the deck's tolerance):
        ``(J_c, aux)``, ``J_c`` of shape ``(rows, n_current)``.  Used to certify the conditional stationarity of the
        current that the Levenberg-Marquardt steps reach with a nearby Jacobian's columns."""
        self._prime()
        if self.bootstrap is None:
            raise ValueError("current_columns needs a bootstrap current")
        state, mask, params_x, tight = aux
        if not tight:
            solved = self.solve(params_x, seed=state)
            if solved is None:
                raise RuntimeError("the state does not converge to the deck's tolerance")
            state, mask = solved
            aux = (state, mask, params_x, True)
        nb = self.n_boundary
        directions = jax.tree.map(lambda a: a[nb:], self.boundary_directions(params_x, x))
        dz, certified = self._tangents(params_x, state, mask, directions)
        if not np.asarray(certified).all():
            raise RuntimeError("the current block's tangent solves are not certified")
        return np.asarray(self._push_rows(state, mask, params_x, dz, directions, field)).T, aux

    def linearize(self, x, aux, field, extra=None):
        """Columns of the interface rows along the boundary coordinates (and the ``extra`` parameter tangents).

        A trial state is first solved to the deck's tolerance (from itself). Returns ``(J, dz, params_batch, aux)``:
        the Jacobian, the projected state responses and parameter tangents of its columns, and the (tight) state
        they belong to.  The responses also seed the next trials (first-order predicted states).  A call at the
        boundary, plasma parameters and field of the latest one returns its result (without ``extra``'s columns
        when not asked for): a polish that stalls asks again for the Jacobian it was given.
        """
        self._prime()
        state, mask, params_x, tight = aux
        key, extra_key = _fingerprint(x, params_x, field), None if extra is None else _fingerprint(extra)
        last = self._last_linearize
        if last is not None and last[0] == key and extra_key in (None, last[1]):
            J, dz, batch, aux = last[2]
            self._linearization = last[3]
            if extra_key is None and last[1] is not None:  # drop the extra columns
                n = self.x0.size
                J = J[:, :n]
                dz, batch = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
            return J, dz, batch, aux
        if not tight:
            solved = self.solve(params_x, seed=state)
            if solved is None:
                raise RuntimeError("the trial state does not converge to the deck's tolerance")
            state, mask = solved
            aux = (state, mask, params_x, True)
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
        dz_all, certified = self._tangents(params_x, state, mask, jax.tree.map(join, *groups))
        certified = np.asarray(certified)
        if not certified.all():  # an uncertified response is no Jacobian column: reject rather than use it
            raise RuntimeError(f"{int((~certified).sum())} of {certified.size} tangent solves are not certified")
        ends = np.cumsum([jax.tree.leaves(batch)[0].shape[0] for batch in groups])
        parts = [(jax.tree.map(lambda a: a[lo:hi], dz_all), batch)
                 for lo, hi, batch in zip(np.r_[0, ends[:-1]], ends, groups)]
        columns = np.hstack([np.asarray(self._push_rows(state, mask, params_x, dz, batch, field)).T
                             for dz, batch in parts])
        dz = jax.tree.map(join, *[dz for dz, _ in parts[:n_coordinates]])
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        if extra is not None:
            dz = jax.tree.map(join, dz, parts[-1][0])
        batch = jax.tree.map(join, *[batch for _, batch in parts])
        self._last_linearize = (key, extra_key, (columns, dz, batch, aux), self._linearization)
        return columns, dz, batch, aux

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, ftol=1e-4, jacobian_ftol=1e-2, max_nfev=60,
                       target_cost=None, refresh_on_stall=False, tight=False, fresh_current=False, verbose=0):
        """K = 0 boundary for ``params`` and ``field``.

        With ``jacobian`` (from a nearby solve) Levenberg-Marquardt steps reuse it from ``x0``; otherwise, or if no
        step reduces the cost, or they end above ``target_cost``, a trust region with fresh Jacobians runs to
        ``jacobian_ftol`` and the reuse steps finish to ``ftol`` (see :func:`solve_free_boundary_three_term`);
        with ``refresh_on_stall`` a finish that stalled on a stale Jacobian retries once with a fresh one there.
        ``tight`` evaluates every point at the deck's tolerance instead of the trial one (:meth:`certify`'s polish:
        the stiff bootstrap rows amplify a trial state's error far above a certificate's gate).  ``fresh_current``
        certifies each finish's conditional current stationarity with a fresh current block
        (:meth:`current_columns`; see :func:`_levenberg_marquardt`).
        Returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask, params, tight), ``nfev``,
        ``njev``, ``accepted`` and ``converged``.
        """
        import scipy.optimize

        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        nfev = njev = 0
        columns = ((lambda z, aux: self.current_columns(z, aux, field))
                   if fresh_current and self._projected is not None else None)
        if jacobian is not None:
            first = self.evaluate(x, params, field, tight=tight)
            if first is None:
                raise RuntimeError("the warm-start boundary has no certified equilibrium")
            out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field, tight=tight), x, *first, jacobian,
                                       ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius, project=self._projected,
                                   current_columns=columns)
            self._radius = out["radius"]
            close = target_cost is None or 0.5 * out["rows"] @ out["rows"] <= target_cost
            if (out["accepted"] or out["converged"]) and close:
                return dict(out, jacobian=jacobian, njev=0)
            # One fresh Jacobian where the steps stalled, and the same steps again, before a full trust region.
            J1, _, _, aux1 = self.linearize(out["x"], out["aux"], field)
            again = _levenberg_marquardt(lambda z: self.evaluate(z, params, field, tight=tight), out["x"], out["rows"], aux1, J1,
                                         ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius, project=self._projected,
                                   current_columns=columns)
            self._radius = again["radius"]
            again["nfev"] += out["nfev"]
            if target_cost is None or 0.5 * again["rows"] @ again["rows"] <= target_cost:
                return dict(again, jacobian=J1, njev=1)
            x, nfev = again["x"], again["nfev"]
        memo = {}

        def fun(z):
            got = self.evaluate(z, params, field, tight=tight)
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
        J = jac(fit.x) if memo.get("jacobian") is None or memo.get("jkey") != fit.x.tobytes() else memo["jacobian"]
        out = _levenberg_marquardt(lambda z: self.evaluate(z, params, field, tight=tight), fit.x, memo["rows"], memo["aux"], J,
                                   ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius, project=self._projected,
                                   current_columns=columns)
        self._radius = out["radius"]
        if refresh_on_stall and not out["converged"]:  # stalled on a stale Jacobian: one fresh one there
            J, _, _, aux = self.linearize(out["x"], out["aux"], field)
            again = _levenberg_marquardt(lambda z: self.evaluate(z, params, field, tight=tight), out["x"], out["rows"], aux, J,
                                         ftol=ftol, max_nfev=max_nfev, verbose=verbose, radius=self._radius, project=self._projected,
                                   current_columns=columns)
            self._radius, njev = again["radius"], njev + 1
            # Still stalled on a fresh Jacobian: possibly stationary, but only a certificate at the final state
            # says so (:meth:`certify`); ``converged`` stays the finish's own cost-change test.
            out = dict(again, nfev=out["nfev"] + again["nfev"])
        return dict(out, jacobian=J, nfev=nfev + out["nfev"], njev=njev)

    def certify(self, params, field, fit, *, gate, xtol=None, polish=True, max_residual=None,
                extra=None, values=None, stats=None):
        """The one record of a boundary fit that counts as solved, or :class:`~vmex.core.errors.TrialRejected`.

        The fit's point is re-solved to the deck's tolerance and linearized there (with the ``extra`` parameter
        tangents' columns); with ``polish`` it is polished with that fresh Jacobian, up to ``POLISH_ROUNDS`` times,
        until its certificate (:func:`_fit_certificate`, on the model coordinates: boundary and any bootstrap
        current together) predicts a Gauss-Newton gain of at most ``gate`` of the cost, and a Gauss-Newton step of
        at most ``xtol`` (scaled) if given.  The certificate also reports the step's normal displacement of the
        boundary (``normal_step``, a diagnostic only: a small normal step does not bound the outputs' error, which
        keep moving along weak directions while it is small).  Every finite rows/Jacobian check, the certificate and
        ``max_residual`` (each :class:`BoundaryResidual` term) apply to the final state, whose ``values(state,
        params)`` are computed only then.  The standalone solve (:func:`solve_free_boundary_three_term`) and the
        optimization (:class:`ThreeTermFreeBoundaryProblem`) accept a state through this one method.  ``stats``
        (a dict) gains ``linearizations``, ``max_condition``, ``max_step``, ``polish_rounds`` and
        ``failed_polishes``.
        """
        stats = {} if stats is None else stats
        n, scale = self.x0.size, np.asarray(self.x_scale)[None]
        rounds = [0]

        def record(kind, sol, certificate=None, normal=None, reason=None):
            """Hand the recorder (if any) a complete record of this certificate round or of the outcome."""
            if self.recorder is None:
                return
            entry = dict(kind=kind, round=rounds[0], reason=reason, gate=float(gate), polish_target=float(
                min(gate, POLISH_FTOL)), xtol=xtol, certificate=certificate,
                resolution=self.resolution(), row_blocks=self.row_blocks, column_blocks=self.column_blocks,
                options=self.options)
            if sol is not None:  # the fit (Levenberg-Marquardt) that produced this point: diagnostics only
                entry["fit"] = {k: (v if isinstance(v, (str, bool, type(None))) else float(v))
                                for k, v in sol.items() if k in ("nfev", "accepted", "reason", "projections",
                                                                  "projection_certified", "jacobian_updates",
                                                                  "extrapolated", "accelerated")}
            if sol is not None:
                entry.update(x=np.asarray(sol["x"]), rows=np.asarray(sol["rows"]), J=np.asarray(sol["J"]),
                             x_scale=np.asarray(self.x_scale), state=[np.asarray(a) for a in jax.tree.leaves(sol["state"])],
                             rbc=np.asarray(sol["params"].rbc), zbs=np.asarray(sol["params"].zbs),
                             curtor=float(np.asarray(sol["params"].curtor)),
                             ac_aux_f=np.asarray(sol["params"].ac_aux_f), closure=sol.get("closure"))
                if normal is not None:
                    entry["normal"] = normal
                if values is not None and kind == "round":  # a diagnostic only: the record's values come last
                    try:
                        entry["values"] = np.asarray(values(sol["state"], sol["params"]))
                    except (VmecError, RuntimeError):
                        entry["values"] = None
            self.recorder(entry)

        def tight(b, aux):
            try:  # the state may not tighten to the deck's tolerance, or a row's own solve may fail
                J, _, _, aux = self.linearize(b, aux, field, extra=extra)
                state, mask, params_b, _ = aux
                rows = np.asarray(self._rows(state, params_b, field))
            except (VmecError, RuntimeError) as error:  # RuntimeError: the relayed sentinel of a VmecError
                raise TrialRejected(str(error)) from error
            J = np.asarray(J)
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(rows))):
                raise TrialRejected("nonfinite rows or boundary Jacobian at the tightened state")
            stats["linearizations"] = stats.get("linearizations", 0) + 1
            return dict(x=np.asarray(b), aux=aux, state=state, mask=mask, params=params_b, rows=rows, J=J)

        sol = certificate = None
        try:
            sol = dict(fit, **tight(fit["x"], fit["aux"]))
            out_of_rounds = False  # the loop ended on the polish budget, not on a stalled or failed polish
            for round_ in range(POLISH_ROUNDS + 1):
                normal = None
                if self._normal_operator is not None:  # the remaining step's motion of the surface itself
                    N, a = map(np.asarray, self._normal_operator(jnp.asarray(sol["x"])))
                    normal = np.hstack([N * scale[:, :N.shape[1]], np.zeros((N.shape[0], n - N.shape[1]))]) / a
                Js = sol["J"][:, :n] * scale
                sol["certificate"] = certificate = _fit_certificate(Js, sol["rows"], normal)
                record("round", sol, certificate, normal)
                rounds[0] += 1
                stats["max_condition"] = max(stats.get("max_condition", 0.0), certificate["condition"])
                if certificate["rank_deficient"]:
                    raise TrialRejected(f"rank-deficient boundary Jacobian (condition {certificate['condition']:.2e})")
                settled = certificate["stationarity"] <= gate
                # The polish aims at POLISH_FTOL; the gate only decides rejection below (#121 7d73da7: stopping at
                # the gate left most trials at 1e-3 and the plasma-map derivative test 1.8% off).
                done = certificate["stationarity"] <= min(gate, POLISH_FTOL) and (
                    xtol is None or certificate["step"] <= xtol)
                if done or not polish or round_ == POLISH_ROUNDS:
                    out_of_rounds = polish and not done and round_ == POLISH_ROUNDS
                    break
                # Short of the polish target: polish with this (fresh, tight) Jacobian and certify again.
                try:
                    # Tight evaluations: the polish makes the final record, judged on tight states.  On trial ones
                    # (QH 2.5%: trial FTOL 1e-11 against the deck's 1e-15) Redl's target differs enough that the
                    # current solved there leaves 98% of the tight cost in the bootstrap rows, and the polish cycles.
                    polished = self.solve_boundary(params, field, x0=sol["x"], jacobian=sol["J"][:, :n],
                                                   ftol=0.1 * min(gate, POLISH_FTOL), max_nfev=POLISH_MAX_NFEV,
                                                   tight=True, fresh_current=True)
                except (VmecError, RuntimeError):
                    stats["failed_polishes"] = stats.get("failed_polishes", 0) + 1
                    break
                stats["boundary_evaluations"] = stats.get("boundary_evaluations", 0) + polished["nfev"]
                if not polished["rows"] @ polished["rows"] < sol["rows"] @ sol["rows"]:
                    break  # the polish gains nothing: the certificate below decides
                sol.update(tight(polished["x"], polished["aux"]))  # the same fit, nearer its stationary point
                sol.update({k: polished[k] for k in ("nfev", "accepted", "reason", "projections", "projection_certified",
                                                      "jacobian_updates", "extrapolated", "accelerated") if k in polished})
                stats["polish_rounds"] = stats.get("polish_rounds", 0) + 1
            if not settled:
                normal_note = "" if certificate["normal_step"] is None else (
                    f" and moves the boundary normally by {certificate['normal_step']:.2e} of the minor radius")
                budget_note = f" when the polish budget of {POLISH_ROUNDS} rounds ran out" if out_of_rounds else ""
                raise TrialRejected(f"boundary fit not stationary: a Gauss-Newton step with its own Jacobian gains "
                                    f"{certificate['stationarity']:.2e} of the cost{normal_note}{budget_note}")
            if certificate["normal_step"] is not None:
                stats["max_normal_step"] = max(stats.get("max_normal_step", 0.0), certificate["normal_step"])
            stats["max_step"] = max(stats.get("max_step", 0.0), certificate["step"])
            if xtol is not None and certificate["step"] > xtol:
                raise TrialRejected(f"boundary fit not stationary: its Gauss-Newton step is {certificate['step']:.2e} "
                                    f"(scaled) long, above boundary_xtol")
            sol["boundary_residual"] = residual = self.boundary_residual(sol["state"], sol["params"], field) \
                if not self.fixed_boundary else None
            if max_residual is not None and residual is not None and max(
                    residual.normal, residual.pressure, residual.sheet_current) > max_residual:
                raise TrialRejected("the certified fit is not a free boundary: an interface residual is above "
                                    "boundary_max_residual")
            if self.bootstrap is not None:  # a separate gate: a stationary finite-weight fit need not close
                sol["closure"] = closure = self.bootstrap_closure(sol["state"], sol["params"])
                if not closure["passed"]:
                    raise TrialRejected(f"the bootstrap current does not close: max |I' - I'_Redl| "
                                        f"{closure['max_abs']:.3e} A above {CLOSURE_ATOL:g} + {CLOSURE_RTOL:g} "
                                        f"x {closure['target_inf']:.3e}")
            if values is not None:
                try:
                    sol["values"] = np.asarray(values(sol["state"], sol["params"]))
                except (VmecError, RuntimeError) as error:
                    raise TrialRejected(str(error)) from error
                if not np.all(np.isfinite(sol["values"])):
                    raise TrialRejected("nonfinite values at the certified state")
        except TrialRejected as error:
            record("rejected", sol, certificate, reason=str(error))
            raise
        record("certified", sol, certificate)
        return sol


class ThreeTermFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-term free boundary, for SLSQP.

    Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
    (:meth:`~vmex.core.freeboundary_problem.FreeBoundaryProblem.from_loss`); pass it to
    :func:`vmex.core.optimize.minimize`.  ``loss``, ``quantities``, ``parameter_quantities``,
    ``field_from_parameters`` and ``plasma_from_parameters`` are as there.  Every trial is the K = 0 boundary
    of the field and plasma parameters of ``x`` (:class:`ThreeTermFreeBoundaryModel`), warm-started from the
    first-order prediction of the latest linearized solution; a trial without a certified solution raises
    :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design gradient of the
    implicit function theorem of the boundary least squares,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b (J_b^T J_b)^-1 dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.  ``model`` is the :class:`ThreeTermFreeBoundaryModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures, boundary evaluations and linearizations) report progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_ftol=1e-3, boundary_max_nfev=40,
                  boundary_max_residual=None, boundary_xtol=None, checkpoint=None, semantic_identity=None,
                  seed_start=None, newton_polish=None, exact_gradient=None, **model_options):
        """Solve the seed design's free boundary (or, with ``checkpoint``, restore a saved accepted point instead:
        :meth:`restore_checkpoint`; ``x0`` then only fixes the design's layout).  ``semantic_identity`` (any
        JSON-serializable value, e.g. a version and the names of the maps and objectives) is the caller's declaration
        of what the callbacks compute: a checkpoint restores only into a problem with the same declaration, since
        callbacks that agree at one design can still differ in their derivatives.  ``seed_start`` (start-dependence
        qualification) fits the seed design's boundary from another start than the deck's boundary, the deck staying
        the reference: ``("random", eps, seed)`` adds ``eps`` times a standard normal vector in the scaled boundary
        coordinates; ``("scale", eps)`` scales every boundary coefficient but the major radius by ``1 + eps``.

        ``model_options`` are :class:`ThreeTermFreeBoundaryModel` keywords (``nphi``, ``ntheta``,
        ``quadrature``, ``chunk``, ``trial_ftol``, ``bootstrap``, ...).  A trial's boundary fit stops at a
        relative cost change of ``boundary_ftol`` or after ``boundary_max_nfev`` evaluations; every trial is then
        re-solved tight, polished with its own Jacobian and certified before it has a value (one record per
        design: a gradient never moves it).  With ``boundary_max_residual`` a fit that ends with
        any :class:`BoundaryResidual` term above it is no solution: the trial tries its next start, or is
        rejected.  With ``boundary_xtol`` a trial's certified fit must also be within that (scaled) length of
        its stationary point (the certificate's ``step``; it is always recorded, ``stats["max_step"]``).  The
        certificate's normal displacement of that step is recorded too (``stats["max_normal_step"]``).
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if not all(callable(f) for f in functions) or not (plasma_from_parameters is None
                                                           or callable(plasma_from_parameters)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = ThreeTermFreeBoundaryModel(inp, **model_options)
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_ftol=boundary_ftol, boundary_max_nfev=boundary_max_nfev,
                   boundary_max_residual=boundary_max_residual, boundary_xtol=boundary_xtol, checkpoint=checkpoint,
                   semantic_identity=semantic_identity, seed_start=seed_start, newton_polish=newton_polish,
                   exact_gradient=exact_gradient)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_ftol, boundary_max_nfev, boundary_max_residual=None,
                 boundary_xtol=None, checkpoint=None, semantic_identity=None, seed_start=None, newton_polish=None,
                 exact_gradient=None):
        im = model._im
        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        self.boundary_xtol = None if boundary_xtol is None else float(boundary_xtol)
        #: Newton refinement of every certified fit on its cost with the exact Hessian (:meth:`_newton_refine`):
        #: None, or a dict of its options (step_tol, max_steps, quasi, refresh_every, warm, warm_max_step, floor,
        #: radius, track, verbose).  It makes the fit's stationary point, and so its response, determined.
        self.newton_polish = None if newton_polish is None else dict(newton_polish)
        #: The optimizer's design gradient: falsy for Gauss-Newton (no curvature terms: exact only at a zero residual),
        #: truthy for the exact derivative of the converged stationarity (:meth:`_exact_gradients`; needs
        #: ``newton_polish``, which converges the point it differentiates).
        self.exact_gradient = bool(exact_gradient)
        if self.exact_gradient and self.newton_polish is None:
            raise ValueError("the exact gradient differentiates a converged stationary point: pass newton_polish")
        self.semantic_identity = json.loads(json.dumps(semantic_identity))
        self._n_rows = 1 + len(quantities) + len(parameter_quantities)
        x0 = np.asarray(x0, dtype=float)
        if checkpoint is not None:  # the saved design is the accepted point; no seed solve
            saved = np.asarray(np.load(checkpoint, allow_pickle=False)["x"], dtype=float)
            if saved.shape != x0.shape:
                raise ValueError(f"the checkpoint's design has {saved.size} coordinates, this problem {x0.size}")
            x0 = saved
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters
        self._params_at = jax.jit(lambda x: params_at(model.params0, x))

        def part(f, with_x):
            def value(state, params, x):
                runtime = im.runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            return value

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities))

        def rows(state, params, x):
            return jnp.concatenate([f(state, params, x) for f in parts])

        self._rows = jax.jit(rows)
        # The coordinates of x that move plasma parameters: their columns of the interface rows enter the gradient.
        # Structural dependence (a NaN tangent reaches every output that depends on the coordinate, even where its
        # derivative is zero, as x^2 at 0), not a numerical Jacobian at x0.
        base = model.params0
        probes = jnp.where(jnp.eye(x0.size, dtype=bool), jnp.nan, 0.0)
        tangent = jax.vmap(lambda e: jax.jvp(lambda x: params_at(base, x), (jnp.asarray(x0),), (e,))[1])(probes)
        moved = sum(np.any(np.asarray(a).reshape(x0.size, -1) != 0, axis=1) for a in jax.tree.leaves(tangent)
                    if jnp.issubdtype(a.dtype, jnp.inexact))
        self._plasma = np.flatnonzero(moved)
        eye = jnp.eye(x0.size)[self._plasma]
        # The derivatives are those of the forward map, params_at(params0, x), at every x.
        self._plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
            lambda z: params_at(base, z), (x,), (e,))[1])(eye))
        def pullback(state, mask, params, x):
            """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only."""
            sizes = [jax.eval_shape(f, state, params, x).size for f in parts]
            return model.pullback(parts, state, mask, params, tuple(jnp.eye(k) for k in sizes), x)

        self._pullback = jax.jit(pullback)

        def cost_pullback(state, mask, params, r, x):
            """``ImplicitParams`` gradient of ``r . rows(state, params, field(x))`` through the equilibrium (``r``: the
            rows themselves, so this is the gradient of ``|r|^2 / 2`` at a fixed boundary); jitted once."""
            return model.pullback(model.rows, state, mask, params, r[None, :], field_from_parameters(x))

        self._cost_pullback = jax.jit(cost_pullback)

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(lambda z: params_at(model.params0, z), x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self._coordinates = coordinates
        self._direct = jax.jit(jax.jacrev(rows, argnums=2))
        self._field_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(
            lambda z: model.rows(state, params, field_from_parameters(z)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(
            lambda z: model.rows(state, params, field_from_parameters(z)), (x,), (dx,))[1])
        self._cache, self._rejections, self._anchor = {}, {}, {}
        # seconds_*: wall time per phase (trial fits; certificates with their polish; design gradients)
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, fit_reasons={},
                          seconds_fit=0.0, seconds_certify=0.0, seconds_gradient=0.0,
                          max_condition=0.0)
        self._continuation = None
        # The base problem only stores the layout (names, scales), which the checkpoint definition includes.
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})
        if checkpoint is not None:
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0  # the layout restore_checkpoint checks
            self.restore_checkpoint(checkpoint)
        else:
            seed = self._solve(x0, starts=None if seed_start is None else [self._seed_start(seed_start)])
            if seed is None:
                raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
            self._promote(seed)
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0

    def _seed_start(self, spec):
        """The model coordinates of a perturbed seed start (``from_loss(seed_start=...)``)."""
        model = self.model
        b = np.array(model.x0, dtype=float)
        nb = int(model.n_boundary)
        kind = spec[0]
        if kind == "random":
            eps, seed = float(spec[1]), int(spec[2])
            b[:nb] += eps * np.asarray(model.x_scale, dtype=float)[:nb] * np.random.default_rng(seed).standard_normal(nb)
        elif kind == "scale":
            r00 = int(np.argmax(np.abs(b[:nb])))  # the major radius: the largest coordinate
            keep = b[r00]
            b[:nb] *= 1.0 + float(spec[1])
            b[r00] = keep
        else:
            raise ValueError(f"seed_start kind must be 'random' or 'scale', got {kind!r}")
        return b

    def _checkpoint(self):
        """The continuation a trial starts from: the boundary fit's latest linearization and trust radius, and the
        equilibrium solver's hot-restart state."""
        im, model = self.model._im, self.model
        return dict(linearization=model._linearization, radius=model._radius,
                    hot={cfg: im._HOT_CACHE.get(cfg) for cfg in {model.cfg, model.cfg_trial}})

    def _restore(self, checkpoint):
        im, model = self.model._im, self.model
        model._linearization, model._radius = checkpoint["linearization"], checkpoint["radius"]
        for cfg, state in checkpoint["hot"].items():
            im._PERTURB_SEED.pop(cfg, None)
            if state is None:
                im._HOT_CACHE.pop(cfg, None)
            else:
                im._HOT_CACHE[cfg] = state

    def _solve(self, x, starts=None, polish=True):
        """The trial at ``x`` (cached): its final record (:meth:`_finalize`), or ``None`` (the reason in
        ``_rejections``).

        Every trial starts from the continuation of the accepted point, so neither a rejected trial nor the order
        of trials changes what a later trial starts from.
        """
        key = np.asarray(x, dtype=float).tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        x = np.asarray(x, dtype=float)
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        if self._plasma.size:  # the plasma map may move the pressure: the problem needs it zero at the edge
            error = _edge_pressure_error(self.model._im.input_with_params(self.model.fixed, params))
            if error:
                return self._reject(key, error)
        anchor, J = self._anchor, None
        if (starts is None and anchor and isinstance(anchor.get("hessian"), np.ndarray) and self.newton_polish is not None
                and self.newton_polish.get("warm")):
            # From the accepted point's converged boundary, its Hessian's Newton step is the exact first-order
            # prediction of this trial's boundary; Newton (fresh Hessians) converges it, no fit needed.
            if self._continuation is not None:
                self._restore(self._continuation)
            t0 = time.perf_counter()
            try:
                opt = dict(step_tol=1e-8, max_steps=10)
                opt.update(self.newton_polish or {})
                opt["H0"] = anchor["hessian"]
                extra = self._plasma_directions(params, jnp.asarray(x)) if self._plasma.size else None
                sol = self._newton_refine_exact(x, params, field, extra, np.asarray(anchor["b"], dtype=float), opt, t0)
                if sol["newton"]["converged"]:
                    nb = self.model.x0.size
                    sol.update(continuation=self._checkpoint(), anchor=dict(
                        x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=sol["J"][:, :nb], J_p=sol["J"][:, nb:],
                        state=sol["state"], params=sol["params"], cost=0.5 * sol["rows"] @ sol["rows"],
                        hessian=sol.get("hessian")))
                    self.stats["warm_newton_trials"] = self.stats.get("warm_newton_trials", 0) + 1
                    self._cache[key] = sol
                    return sol
                self.stats["warm_newton_unconverged"] = self.stats.get("warm_newton_unconverged", 0) + 1
            except (TrialRejected, VmecError, RuntimeError):  # fall back to the fit from the anchor's prediction
                self.stats["warm_newton_failed"] = self.stats.get("warm_newton_failed", 0) + 1
            finally:
                self.stats["seconds_certify"] += time.perf_counter() - t0
        if starts is not None:  # given starts (a restore: the saved boundary)
            anchor = {}
        elif anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
            exact = self._exact_prediction(x, params, field, anchor)
            if exact is not None:  # first: along the flat directions the Gauss-Newton prediction is the wrong one
                starts.insert(0, exact)
        else:
            starts = [None]
        sol, why = None, []
        for i, start in enumerate(starts):
            if self._continuation is not None:  # every start from the accepted point's continuation
                self._restore(self._continuation)
            t0 = time.perf_counter()
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev, refresh_on_stall=True,
                                                target_cost=4 * anchor["cost"] if anchor else None)
            except (VmecError, RuntimeError) as error:  # an uncertified equilibrium: try the next start
                why.append(f"start {i}: {error}")
                continue
            finally:
                self.stats["seconds_fit"] += time.perf_counter() - t0
            reasons = self.stats["fit_reasons"]
            reasons[fit["reason"]] = reasons.get(fit["reason"], 0) + 1
            self.stats["boundary_evaluations"] += fit["nfev"]
            if not (fit["converged"] or fit["reason"] == "stalled"):  # out of budget, or its trust radius collapsed
                why.append(f"start {i}: the fit stopped ({fit['reason']})")
                continue
            if self.boundary_max_residual is not None:  # a screen on the fit's trial state: twice the final bound
                state, _, params_b, _ = fit["aux"]
                residual = self.model.boundary_residual(state, params_b, field)
                if max(residual.normal, residual.pressure, residual.sheet_current) > 2 * self.boundary_max_residual:
                    why.append(f"start {i}: the fit stopped away from the free boundary")
                    continue
            t0 = time.perf_counter()
            try:  # the certificate decides (a stall on a fresh Jacobian may be stationary)
                sol = self._finalize(x, params, field, fit, polish)
                break
            except TrialRejected as error:
                why.append(f"start {i}: {error}")
            finally:
                self.stats["seconds_certify"] += time.perf_counter() - t0
        if sol is None:
            return self._reject(key, "; ".join(why) or "no three-term free boundary at this design point")
        self._cache[key] = sol
        return sol

    def _exact_prediction(self, x, params, field, anchor):
        """The trial's boundary to first order by the exact sensitivity at the accepted point: ``b - H^-1 g``, ``g`` the
        fit's cost gradient at the accepted boundary under the trial's design (one tight solve and one reverse pass)
        and ``H`` the accepted point's exact Hessian (kept when its exact gradient was taken).  The Gauss-Newton
        prediction drops the curvature, which dominates along the fit's flat (relabelling) directions, so its start
        leaves the polish a long crawl there.  None without a kept Hessian or when the solve fails."""
        H = anchor.get("hessian")
        if H is None:
            return None
        if self._continuation is not None:
            self._restore(self._continuation)
        t0 = time.perf_counter()
        try:
            g, _, _ = self._cost_gradients(np.asarray(anchor["b"], dtype=float), params, field, np.asarray(x, dtype=float))
        except (TrialRejected, VmecError, RuntimeError):
            return None
        finally:
            self.stats["seconds_fit"] += time.perf_counter() - t0
        step = _projected_solve(H, g) if isinstance(H, dict) else np.linalg.solve(H, g)
        if not np.all(np.isfinite(step)):
            return None
        self.stats["exact_prediction_starts"] = self.stats.get("exact_prediction_starts", 0) + 1
        return np.asarray(anchor["b"], dtype=float) - step

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        self.stats["last_rejection"] = str(reason)[:400]
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _finalize(self, x, params, field, fit, polish=True, newton=True):
        """A trial's one record: its fit certified by :meth:`ThreeTermFreeBoundaryModel.certify` (re-solved to the
        deck's tolerance, polished with its own Jacobian when ``polish``, certified, and only then valued).  Every
        value, gradient, acceptance and checkpoint of ``x`` uses this record; differentiating it adds gradients and
        never moves its boundary.  Raises :class:`~vmex.core.errors.TrialRejected`."""
        model, xj, nb = self.model, jnp.asarray(x), self.model.x0.size
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        # With the Newton polish, the fit's own certificate does not decide: Newton converges the point and the gate
        # applies to that (the final certify of the polish); a Levenberg-Marquardt stall is only where Newton starts.
        newton_decides = newton and self.newton_polish is not None
        sol = model.certify(params, field, fit, gate=np.inf if newton_decides else max(self.boundary_ftol, POLISH_FTOL),
                            xtol=None if newton_decides else self.boundary_xtol, polish=polish,
                            max_residual=self.boundary_max_residual, extra=extra,
                            values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats)
        if newton and self.newton_polish is not None:  # converge the stationarity in every direction, certify there
            sol = self._newton_refine(x, params, field, sol, extra)
        J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
        # What later trials start from if the optimizer accepts this one (accept_x): its linearization now.
        sol.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=sol["state"],
            params=sol["params"], cost=0.5 * sol["rows"] @ sol["rows"], hessian=sol.get("hessian")))
        return sol

    def _newton_refine(self, x, params, field, sol, extra):
        """Newton steps on the fit's cost ``|r|^2 / 2`` from a certified fit, with the exact Hessian, then its
        certificate.

        The Levenberg-Marquardt polish (Gauss-Newton) converges linearly along the weak (relabelling) directions, so a
        fit certified by its gain can still be far, in the scaled step, from its stationary point there; the response
        of the stationary point is then not determined by the fit.  :meth:`_newton_refine_exact` (or, with a
        bootstrap current, :meth:`_newton_refine_projected`) converges it; the final point is certified by
        :meth:`ThreeTermFreeBoundaryModel.certify` (re-solved, linearized, no further polish), so the record, its
        Jacobian and its values come from the same code as every other trial.  ``sol["newton"]`` records the steps.
        """
        opt = dict(step_tol=1e-8, max_steps=10)
        opt.update(self.newton_polish or {})
        t0 = time.perf_counter()
        b = np.asarray(sol["x"], dtype=float).copy()
        if self.model._projected is not None:
            return self._newton_refine_projected(x, params, field, extra, b, opt, t0)
        M = (self._anchor or {}).get("hessian")
        if opt.get("krylov", True) and isinstance(M, np.ndarray) and M.shape == (b.size, b.size):
            return self._newton_refine_krylov(x, params, field, extra, b, opt, t0, M)
        return self._newton_refine_exact(x, params, field, extra, b, opt, t0)

    def _newton_refine_exact(self, x, params, field, extra, b, opt, t0):
        """:meth:`_newton_refine` by full Newton steps on the fit's cost with the exact Hessian, in a trust region.

        At each iterate ``b``: the tight interior solve, ``g`` (:meth:`_cost_gradients`) and the dense exact Hessian
        ``H`` (:meth:`dense_hessian`, refreshed every step: along the fit's flat, curved directions it changes over
        one step).  The step solves ``(H + lam D^2) d = -g`` (``D = diag(1/scale)``; ``lam = 0`` when the Newton step
        fits the radius and ``H`` is positive definite) and is accepted by the ratio of the actual to the predicted
        decrease of the cost ``|r|^2 / 2`` (not a stationarity norm, which can fall while the cost rises).  Stops
        when the accepted Newton step is at most ``step_tol`` (scaled), or when its predicted decrease is below the
        cost's rounding (``floor``).  With ``quasi`` the first step uses the exact Hessian and later ones a damped
        (Powell) BFGS update on the exact gradient's secant, refreshed exactly every ``refresh_every`` steps and when
        the model stalls; the record's gradient still builds the exact Hessian.  With ``H0`` (a warm trial: the
        accepted point's Hessian) the first step is its prediction.  The last exact Hessian is kept for the record's
        exact gradient (:meth:`_exact_dense_gradients`).  Then certified as every other record.
        """
        import scipy.linalg

        model, nb = self.model, self.model.x0.size
        scale = np.asarray(model.x_scale, dtype=float)

        def point(bb):  # (cost, g, sol-like record) at bb, or None
            got = model.evaluate(bb, params, field, tight=True)
            if got is None:
                return None
            rows, aux = got
            state, mask, params_b, _ = aux
            r = jnp.asarray(rows)
            params_bar = self._cost_pullback(state, mask, params_b, r, jnp.asarray(x))
            b_bar, _ = self._coordinates(params_b, jnp.asarray(bb), jnp.asarray(x), params_bar)
            return (0.5 * float(np.asarray(rows) @ np.asarray(rows)), np.asarray(b_bar)[0],
                    dict(state=state, mask=mask, x=np.asarray(bb, dtype=float), params=params_b))

        here = point(b)
        if here is None:
            raise TrialRejected("the certified fit has no tight interior root for its Newton polish")
        cost, g, rec = here
        radius = float(opt.get("radius", 1e-1))
        history, evaluations, converged, seconds_hessian, H = [], 1, False, 0.0, None
        quasi, secant, refresh, since_exact = bool(opt.get("quasi")), None, False, 0
        for _ in range(int(opt["max_steps"])):
            th = time.perf_counter()
            since_exact += 1
            if H is None and opt.get("H0") is not None:  # the accepted point's Hessian: the predicted first step
                H, asymmetry = np.asarray(opt["H0"]), 0.0
            elif quasi and H is not None and (refresh or since_exact > int(opt.get("refresh_every", 10))):
                H, asymmetry = self.dense_hessian(x, rec)  # BFGS stalled, or due: the exact Hessian again
                since_exact, secant = 0, None
            elif quasi and H is not None:  # damped BFGS (Powell) from the exact start, on the exact g's secant
                Hs_old = H * np.outer(scale, scale)
                if secant is not None:
                    s_, y_ = secant
                    Bs = Hs_old @ s_
                    sBs, sy = float(s_ @ Bs), float(s_ @ y_)
                    if sBs > 0:
                        if sy < 0.2 * sBs:
                            theta = 0.8 * sBs / (sBs - sy)
                            y_ = theta * y_ + (1 - theta) * Bs
                            sy = float(s_ @ y_)
                        Hs_old = Hs_old + np.outer(y_, y_) / sy - np.outer(Bs, Bs) / sBs
                H, asymmetry = 0.5 * (Hs_old + Hs_old.T) / np.outer(scale, scale), 0.0
            else:
                H, asymmetry = self.dense_hessian(x, rec)
            seconds_hessian += time.perf_counter() - th
            Hs = H * (scale[:, None] * scale[None, :])  # scaled coordinates
            gs = g * scale
            try:
                newton = -scipy.linalg.solve(Hs, gs, assume_a="sym")
                eigenvalues, eigenvectors = np.linalg.eigh(Hs)
                definite = bool(np.all(eigenvalues > 0))
            except np.linalg.LinAlgError:
                newton, definite, eigenvalues, eigenvectors = None, False, None, None
            step_newton = float(np.linalg.norm(newton)) if newton is not None else np.inf
            history.append(dict(cost=cost, step=step_newton, radius=radius, definite=definite,
                                stationarity=float(abs(gs @ newton)) if newton is not None else np.inf,
                                evaluations=evaluations, asymmetry=asymmetry))
            if eigenvalues is not None and newton is not None:  # where the Newton step lives (degenerate minimum?)
                share = (eigenvectors[:, :5].T @ newton) / max(float(np.linalg.norm(newton)), 1e-300)
                history[-1].update(smallest_eigenvalues=eigenvalues[:5].tolist(),
                                   largest_eigenvalue=float(eigenvalues[-1]), step_share_weakest=share.tolist())
            if opt.get("track"):  # the exact gradient at every iterate: does it settle as the point converges?
                history[-1]["gradients"] = self._exact_dense_gradients(x, rec, H=H)["gradients"].tolist()
            if opt.get("verbose"):
                drift = ""
                if opt.get("track") and len(history) > 1:
                    G0, G1 = np.array(history[-2]["gradients"]), np.array(history[-1]["gradients"])
                    drift = (" gradient change vs previous iterate max "
                             f"{float(np.max(np.linalg.norm(G1 - G0, axis=1) / np.linalg.norm(G1, axis=1))):.1e}")
                eig = ""
                if "smallest_eigenvalues" in history[-1]:
                    eig = (" smallest eigenvalues " + " ".join(f"{e:.2e}" for e in history[-1]["smallest_eigenvalues"])
                           + " step share on them " + " ".join(f"{abs(c):.2f}" for c in history[-1]["step_share_weakest"]))
                print(f"[newton] iterate {len(history) - 1}: cost {cost:.12e}, Newton step {step_newton:.2e}, "
                      f"definite {definite}, radius {radius:.1e}, Hessian {seconds_hessian:.0f} s total{drift}{eig}",
                      flush=True)
            if definite and step_newton <= opt["step_tol"]:
                converged = True
                break
            if len(history) == 1 and opt.get("H0") is not None and step_newton > opt.get("warm_max_step", 0.05):
                # the warm prediction is outside Newton's fast region (a large design step): the caller fits instead
                history[-1]["warm_too_far"] = True
                break
            # The cost's own floor: a Newton step whose predicted decrease is below the cost's rounding cannot be
            # told from noise by the actual/predicted ratio; the point is stationary to the cost's precision.
            if definite and (0.5 * float(gs @ -newton) <= opt.get("floor", 1e-12) * max(cost, 1e-300)
                             or (radius < 1e-6 and step_newton < 1e-5)):
                converged = True
                history[-1]["floor"] = True
                break
            accepted = False
            for _ in range(8):
                if definite and step_newton <= radius:
                    ds = newton
                else:  # the trust-region step: raise lam until the step fits
                    lam_lo, lam_hi = 0.0, max(1.0, float(np.abs(np.diag(Hs)).max()))
                    while np.linalg.norm(scipy.linalg.solve(Hs + lam_hi * np.eye(nb), -gs, assume_a="sym")) > radius:
                        lam_hi *= 10
                    for _ in range(40):
                        lam = 0.5 * (lam_lo + lam_hi)
                        ds = scipy.linalg.solve(Hs + lam * np.eye(nb), -gs, assume_a="sym")
                        if np.linalg.norm(ds) > radius or not np.all(np.isfinite(ds)):
                            lam_lo = lam
                        else:
                            lam_hi = lam
                    ds = scipy.linalg.solve(Hs + lam_hi * np.eye(nb), -gs, assume_a="sym")
                predicted = -(gs @ ds + 0.5 * ds @ Hs @ ds)
                trial = point(b + ds * scale)
                evaluations += 1
                rho = (cost - trial[0]) / predicted if trial is not None and predicted > 0 else -1.0
                if rho > 0.1:
                    secant = (np.asarray(ds, dtype=float).copy(), (trial[1] - g) * scale)  # scaled s and y
                    b, (cost, g, rec) = b + ds * scale, trial
                    if rho > 0.75 and np.linalg.norm(ds) > 0.9 * radius:
                        radius *= 2
                    accepted = True
                    history[-1].update(taken=float(np.linalg.norm(ds)), rho=float(rho))
                    break
                radius = 0.25 * min(radius, float(np.linalg.norm(ds)))
            if not accepted:
                history[-1]["stalled"] = True
                if quasi and since_exact > 1:  # a stale quasi-Newton model: refresh it exactly before giving up
                    refresh = True
                    radius = max(radius, float(opt.get("radius", 1e-1)) * 1e-2)
                    continue
                break
            refresh = False
        self._hessian = dict(b=np.asarray(rec["x"]).copy(), x=np.asarray(x, dtype=float).copy(), H=H) \
            if converged else None
        got = model.evaluate(b, params, field, tight=True)
        if got is None:
            raise TrialRejected("the Newton-refined boundary has no certified interior root")
        rows, aux = got
        fit = dict(x=b, aux=aux, rows=np.asarray(rows), nfev=evaluations, converged=True, reason="newton",
                   accepted=len(history))
        xj = jnp.asarray(x)
        out = model.certify(params, field, fit, gate=max(self.boundary_ftol, POLISH_FTOL), xtol=self.boundary_xtol,
                            polish=False, max_residual=self.boundary_max_residual, extra=extra,
                            values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats)
        out["newton"] = dict(steps=history, converged=converged, evaluations=evaluations, merit=cost,
                             subspace=nb, products="exact", seconds_hessian=seconds_hessian,
                             seconds=time.perf_counter() - t0, warm=opt.get("H0") is not None)
        if converged and H is not None and not quasi:  # carried by the anchor: the next trials' first Newton step
            out["hessian"] = np.asarray(H)
        if quasi:  # a quasi-Newton matrix is not the exact Hessian: the record's gradient builds that afresh
            self._hessian = None
        self.stats["newton_evaluations"] = self.stats.get("newton_evaluations", 0) + evaluations
        self.stats["seconds_newton"] = self.stats.get("seconds_newton", 0.0) + out["newton"]["seconds"]
        return out

    def _newton_refine_krylov(self, x, params, field, extra, b, opt, t0, M):
        """:meth:`_newton_refine_exact` without dense Hessians: each step by truncated conjugate gradients (Steihaug)
        on exact Hessian products (:meth:`exact_products`, one column per product), preconditioned by ``M``, the
        accepted point's exact Hessian (a trial is a small design step from it, so few products are needed).

        The step is the exact Newton step to ``cg_rtol`` (or the trust-region boundary, or a direction of negative
        curvature) and the trust region judges it on the actual cost, as the dense polish; it converges on the same
        tests (scaled step at most ``step_tol``, or the cost's floor).  No Hessian is kept: the record's exact
        gradient builds the dense one only for a point the optimizer differentiates."""
        model, nb = self.model, self.model.x0.size
        scale = np.asarray(model.x_scale, dtype=float)
        rtol, maxiter = float(opt.get("cg_rtol", 1e-3)), int(opt.get("cg_maxiter", 40))
        Ms = np.asarray(M) * np.outer(scale, scale)
        lam, Q = np.linalg.eigh(0.5 * (Ms + Ms.T))
        lam = np.maximum(np.abs(lam), 1e-14 * float(np.abs(lam).max()))

        def precondition(r):
            return Q @ ((Q.T @ r) / lam)

        def point(bb):  # (cost, g, sol-like record) at bb, or None
            got = model.evaluate(bb, params, field, tight=True)
            if got is None:
                return None
            rows, aux = got
            state, mask, params_b, _ = aux
            r = jnp.asarray(rows)
            params_bar = self._cost_pullback(state, mask, params_b, r, jnp.asarray(x))
            b_bar, _ = self._coordinates(params_b, jnp.asarray(bb), jnp.asarray(x), params_bar)
            return (0.5 * float(np.asarray(rows) @ np.asarray(rows)), np.asarray(b_bar)[0],
                    dict(state=state, mask=mask, x=np.asarray(bb, dtype=float), params=params_b))

        def steihaug(rec, gs, radius):
            """``(ds, Hs ds, products, interior)``: truncated preconditioned CG on ``Hs ds = -gs`` within ``radius``."""
            def hs(v):
                return scale * np.asarray(self.exact_products(x, rec, (v * scale)[:, None], check=False)[0])[:, 0]

            z, Hz = np.zeros(nb), np.zeros(nb)
            r = gs.copy()
            y = precondition(r)
            d = -y
            ry, norm0, used = float(r @ y), float(np.linalg.norm(gs)), 0
            for _ in range(maxiter):
                Hd = hs(d)
                used += 1
                dHd = float(d @ Hd)
                if dHd <= 0 or np.linalg.norm(z + (ry / dHd) * d) >= radius:  # to the boundary along d
                    a, bq, cq = float(d @ d), 2 * float(z @ d), float(z @ z) - radius**2
                    tau = (-bq + np.sqrt(max(bq * bq - 4 * a * cq, 0.0))) / (2 * a)
                    return z + tau * d, Hz + tau * Hd, used, False
                alpha = ry / dHd
                z, Hz, r = z + alpha * d, Hz + alpha * Hd, r + alpha * Hd
                if float(np.linalg.norm(r)) <= rtol * norm0:
                    return z, Hz, used, True
                y = precondition(r)
                ry_new = float(r @ y)
                d, ry = -y + (ry_new / ry) * d, ry_new
            return z, Hz, used, True

        here = point(b)
        if here is None:
            raise TrialRejected("the certified fit has no tight interior root for its Newton polish")
        cost, g, rec = here
        radius = float(opt.get("radius", 1e-1))
        history, evaluations, converged, products, seconds_products = [], 1, False, 0, 0.0
        for _ in range(int(opt["max_steps"])):
            gs = g * scale
            th = time.perf_counter()
            ds, Hds, used, interior = steihaug(rec, gs, radius)
            seconds_products += time.perf_counter() - th
            products += used
            step = float(np.linalg.norm(ds))
            predicted = -(gs @ ds + 0.5 * ds @ Hds)
            history.append(dict(cost=cost, step=step, radius=radius, interior=interior, products=used,
                                evaluations=evaluations))
            if opt.get("verbose"):
                print(f"[newton krylov] iterate {len(history) - 1}: cost {cost:.12e}, step {step:.2e} "
                      f"({'interior' if interior else 'boundary'}), radius {radius:.1e}, {used} products, "
                      f"{seconds_products:.0f} s in products", flush=True)
            if interior and step <= opt["step_tol"]:
                converged = True
                break
            if interior and (predicted <= opt.get("floor", 1e-12) * max(cost, 1e-300)
                             or (radius < 1e-6 and step < 1e-5)):
                converged = True
                history[-1]["floor"] = True
                break
            trial = point(b + ds * scale)
            evaluations += 1
            rho = (cost - trial[0]) / predicted if trial is not None and predicted > 0 else -1.0
            history[-1]["rho"] = float(rho)
            if rho > 0.1:
                b, (cost, g, rec) = b + ds * scale, trial
                if rho > 0.75 and not interior:
                    radius *= 2
            else:
                radius = 0.25 * min(radius, step)
                if radius < 1e-10:
                    history[-1]["stalled"] = True
                    break
        self._hessian = None
        got = model.evaluate(b, params, field, tight=True)
        if got is None:
            raise TrialRejected("the Newton-refined boundary has no certified interior root")
        rows, aux = got
        fit = dict(x=b, aux=aux, rows=np.asarray(rows), nfev=evaluations, converged=True, reason="newton",
                   accepted=len(history))
        xj = jnp.asarray(x)
        out = model.certify(params, field, fit, gate=max(self.boundary_ftol, POLISH_FTOL), xtol=self.boundary_xtol,
                            polish=False, max_residual=self.boundary_max_residual, extra=extra,
                            values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats)
        out["newton"] = dict(steps=history, converged=converged, evaluations=evaluations, merit=cost, subspace=nb,
                             products="krylov", product_calls=products, seconds_hessian=seconds_products,
                             seconds=time.perf_counter() - t0, warm=False)
        self.stats["newton_evaluations"] = self.stats.get("newton_evaluations", 0) + evaluations
        self.stats["newton_products"] = self.stats.get("newton_products", 0) + products
        self.stats["seconds_newton"] = self.stats.get("seconds_newton", 0.0) + out["newton"]["seconds"]
        return out

    def _newton_refine_projected(self, x, params, field, extra, b, opt, t0):
        """:meth:`_newton_refine_exact` with a bootstrap current: Newton on the boundary with the current projected.

        The joint Hessian is round-off at the bootstrap weight (:meth:`projected_hessian`), so each step is taken on
        the reduced cost ``min_c f(b, c)``: its gradient ``g_b - K^T g_c`` and Hessian ``H_red`` (exact at every
        iterate), in the same trust region on the cost. The current follows with ``-K d_b - H_cc^-1 g_c`` and is then
        re-solved at the trial boundary by Newton steps on its own block (``H_cc``, at most
        :data:`PROJECTION_CORRECTIONS`), so the trust region judges the reduced cost. Converged as the exact polish
        (Newton step or the cost's floor); the last projected Hessian is kept for the record's exact gradient."""
        import scipy.linalg

        model = self.model
        scale = np.asarray(model.x_scale, dtype=float)
        c = np.asarray(model._projected)
        bi = np.setdiff1d(np.arange(model.x0.size), c)
        sb = scale[bi]

        def point(bb):  # (cost, g, sol-like record) at bb, or None
            got = model.evaluate(bb, params, field, tight=True)
            if got is None:
                return None
            rows, aux = got
            state, mask, params_b, _ = aux
            r = jnp.asarray(rows)
            params_bar = self._cost_pullback(state, mask, params_b, r, jnp.asarray(x))
            b_bar, _ = self._coordinates(params_b, jnp.asarray(bb), jnp.asarray(x), params_bar)
            return (0.5 * float(np.asarray(rows) @ np.asarray(rows)), np.asarray(b_bar)[0],
                    dict(state=state, mask=mask, x=np.asarray(bb, dtype=float), params=params_b))

        def project(bb, here, Hcc):  # the current re-solved at this boundary: Newton on its block
            used = 0
            for _ in range(PROJECTION_CORRECTIONS):
                dc = -np.linalg.solve(Hcc, here[1][c])
                if -0.5 * float(here[1][c] @ dc) <= 1e-14 * max(here[0], 1e-300):
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
        radius = float(opt.get("radius", 1e-1))
        history, evaluations, converged, seconds_hessian, P = [], 1, False, 0.0, None
        for _ in range(int(opt["max_steps"])):
            th = time.perf_counter()
            P, asymmetry = self.projected_hessian(x, rec)
            seconds_hessian += time.perf_counter() - th
            Hs = P["Hred"] * np.outer(sb, sb)
            gs = (g[bi] - P["K"].T @ g[c]) * sb
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
            if opt.get("track"):
                history[-1]["gradients"] = self._exact_dense_gradients(x, rec, H=P)["gradients"].tolist()
            if opt.get("verbose"):
                drift = ""
                if opt.get("track") and len(history) > 1:
                    G0, G1 = np.array(history[-2]["gradients"]), np.array(history[-1]["gradients"])
                    drift = (" gradient change vs previous iterate max "
                             f"{float(np.max(np.linalg.norm(G1 - G0, axis=1) / np.linalg.norm(G1, axis=1))):.1e}")
                eig = "" if eigenvalues is None else (" smallest eigenvalues " + " ".join(
                    f"{e:.2e}" for e in eigenvalues[:5]) + f" largest {eigenvalues[-1]:.2e}")
                print(f"[newton projected] iterate {len(history) - 1}: cost {cost:.12e}, Newton step "
                      f"{step_newton:.2e}, definite {definite}, radius {radius:.1e}, current gain {current_gain:.1e}, "
                      f"leak {P['leak']:.1e}, Hessian {seconds_hessian:.0f} s total{drift}{eig}", flush=True)
            if definite and step_newton <= opt["step_tol"]:
                converged = True
                break
            if definite and (0.5 * float(gs @ -newton) + current_gain <= opt.get("floor", 1e-12) * max(cost, 1e-300)
                             or (radius < 1e-6 and step_newton < 1e-5)):
                converged = True
                history[-1]["floor"] = True
                break
            accepted = False
            for _ in range(8):
                if definite and step_newton <= radius:
                    ds = newton
                else:  # the trust-region step: raise lam until the step fits
                    eye = np.eye(bi.size)
                    lam_lo, lam_hi = 0.0, max(1.0, float(np.abs(np.diag(Hs)).max()))
                    while np.linalg.norm(scipy.linalg.solve(Hs + lam_hi * eye, -gs, assume_a="sym")) > radius:
                        lam_hi *= 10
                    for _ in range(40):
                        lam = 0.5 * (lam_lo + lam_hi)
                        ds = scipy.linalg.solve(Hs + lam * eye, -gs, assume_a="sym")
                        if np.linalg.norm(ds) > radius or not np.all(np.isfinite(ds)):
                            lam_lo = lam
                        else:
                            lam_hi = lam
                    ds = scipy.linalg.solve(Hs + lam_hi * eye, -gs, assume_a="sym")
                predicted = -(gs @ ds + 0.5 * ds @ Hs @ ds) + current_gain
                d = np.zeros_like(b)
                d[bi] = ds * sb
                d[c] = -P["K"] @ d[bi] - np.linalg.solve(P["Hcc"], g[c])
                trial = point(b + d)
                evaluations += 1
                b_try = b + d
                if trial is not None:
                    b_try, trial, used = project(b_try, trial, P["Hcc"])
                    evaluations += used
                rho = (cost - trial[0]) / predicted if trial is not None and predicted > 0 else -1.0
                if rho > 0.1:
                    b, (cost, g, rec) = b_try, trial
                    if rho > 0.75 and np.linalg.norm(ds) > 0.9 * radius:
                        radius *= 2
                    accepted = True
                    history[-1].update(taken=float(np.linalg.norm(ds)), rho=float(rho))
                    break
                radius = 0.25 * min(radius, float(np.linalg.norm(ds)))
            if not accepted:
                history[-1]["stalled"] = True
                break
        self._hessian = dict(b=np.asarray(rec["x"]).copy(), x=np.asarray(x, dtype=float).copy(), H=P) \
            if converged else None
        got = model.evaluate(b, params, field, tight=True)
        if got is None:
            raise TrialRejected("the Newton-refined boundary has no certified interior root")
        rows, aux = got
        fit = dict(x=b, aux=aux, rows=np.asarray(rows), nfev=evaluations, converged=True, reason="newton",
                   accepted=len(history))
        xj = jnp.asarray(x)
        out = model.certify(params, field, fit, gate=max(self.boundary_ftol, POLISH_FTOL), xtol=self.boundary_xtol,
                            polish=False, max_residual=self.boundary_max_residual, extra=extra,
                            values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats)
        out["newton"] = dict(steps=history, converged=converged, evaluations=evaluations, merit=cost,
                             subspace=bi.size, products="exact", seconds_hessian=seconds_hessian,
                             seconds=time.perf_counter() - t0, warm=False, projected=True)
        self.stats["newton_evaluations"] = self.stats.get("newton_evaluations", 0) + evaluations
        self.stats["seconds_newton"] = self.stats.get("seconds_newton", 0.0) + out["newton"]["seconds"]
        return out

    def _record(self, x):
        x = np.asarray(self._x(x), dtype=float)
        sol = self._solve(x)
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected(self._rejections[x.tobytes()])
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at the record ``sol`` (cached): its certified Jacobian, one reverse pass."""
        if "gradients" in sol:
            return sol["gradients"]
        t0 = time.perf_counter()
        try:
            return self._linearize_uncached(x, sol)
        finally:
            self.stats["seconds_gradient"] += time.perf_counter() - t0

    def _linearize_uncached(self, x, sol):
        if not self.exact_gradient:
            gradients = self._gauss_newton_gradients(x, sol)
        else:
            got = self._exact_gradients(x, sol)
            sol["exact_gradient"] = {k: got[k] for k in ("true_residual", "condition", "seconds_hessian")}
            self.stats["exact_gradient_true_residual"] = float(max(got["true_residual"]))
            gradients = got["gradients"]
        if not np.all(np.isfinite(gradients)):
            raise TrialRejected("nonfinite design gradient")
        sol["gradients"] = gradients
        return gradients

    def _gauss_newton_gradients(self, x, sol):
        """Every row's design gradient with the Gauss-Newton sensitivity of the fit (no curvature terms)."""
        xj, nb = jnp.asarray(x), self.model.x0.size
        J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
        state, params = sol["state"], sol["params"]
        try:
            params_bar = self._pullback(state, sol["mask"], params, xj)
        except (VmecError, RuntimeError) as error:  # an adjoint solve that does not certify
            raise TrialRejected(str(error)) from error
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        lam = _residual_multipliers(J_b, G_b)  # a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column))) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        return gradients

    def _cost_gradients(self, b, params, field, x):
        """At boundary coordinates ``b`` (fixed design ``x``): ``g = J_b^T r`` (the fit's stationarity), ``c = J_d^T r``
        (the design gradient of ``|r|^2 / 2`` at that fixed boundary: coil field and plasma parameters) and ``r``.

        One tight interior solve and one reverse pass through the equilibrium (:meth:`ThreeTermFreeBoundaryModel.pullback`
        with the rows as cotangent), instead of the Jacobian's every column.
        """
        model = self.model
        got = model.evaluate(np.asarray(b, dtype=float), params, field, tight=True)
        if got is None:
            raise TrialRejected("a stationarity difference point has no certified interior root")
        rows, aux = got
        state, mask, params_b, _ = aux
        r = jnp.asarray(rows)
        params_bar = self._cost_pullback(state, mask, params_b, r, jnp.asarray(x))
        b_bar, x_bar = self._coordinates(params_b, jnp.asarray(b), jnp.asarray(x), params_bar)
        g = np.asarray(b_bar)[0]
        c = np.asarray(x_bar)[0] + np.asarray(self._field_vjp(state, params_b, jnp.asarray(x), r))
        return g, c, np.asarray(rows)

    def stationarity_design_gradient(self, x, rows=None):
        """Every requested row's exact design gradient at ``x``'s record (:meth:`_exact_gradients`), with the
        Gauss-Newton one (``gauss_newton``) for comparison."""
        x = np.asarray(x, dtype=float)
        sol = self._record(x)
        got = self._exact_gradients(x, sol, rows)
        got["gauss_newton"] = np.asarray(self._gauss_newton_gradients(x, sol))[got["rows"]]
        return got

    def dense_hessian(self, x, sol, chunk=None):
        """The fit's exact Hessian ``H = J_b^T J_b + sum_i r_i d2r_i/db2`` at the record ``sol``: every unit column (in
        the model coordinates) from one call of :meth:`exact_products`, so its block factorization (shared with the
        state tangents) and multiplier are paid once per Hessian, with the state tangents solved ``chunk`` columns at a time (default
        :data:`EXACT_PRODUCTS_CHUNK`).  Returns ``(H, asymmetry)``: ``H`` symmetrized and the relative norm of its
        antisymmetric part (zero up to round-off for exact products)."""
        chunk = EXACT_PRODUCTS_CHUNK if chunk is None else chunk
        H = self.exact_products(x, sol, np.eye(self.model.x0.size), check=False, chunk=chunk)[0]
        asymmetry = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
        return 0.5 * (H + H.T), asymmetry

    def projected_hessian(self, x, sol, chunk=None):
        """With a bootstrap current: the fit's exact Hessian with the stiff current block eliminated, as a dict.

        The current rows carry ``bootstrap_weight`` (squared in ``H``), so in one matrix with the boundary's
        relabelling valley the condition number reaches ~1e12 and the dense solve and eigenvalues are round-off
        (indefinite at the minimum). Instead: the current columns ``H[:, c]`` (one :meth:`exact_products` call),
        ``K = H_cc^-1 H_cb``, then the products along the current-following directions ``[I; -K]`` give the reduced
        Hessian ``H_red = H_bb - H_bc K`` directly: along them the weighted current rows' first-order change cancels
        inside each product, not in a difference of large entries. ``H = T^T diag(H_red, H_cc) T`` with ``T = [[I,
        0], [K, I]]`` (:func:`_projected_solve`). ``leak`` is the relative size of the current rows of those
        products, zero up to round-off."""
        chunk = EXACT_PRODUCTS_CHUNK if chunk is None else chunk
        n, c = self.model.x0.size, np.asarray(self.model._projected)
        bi = np.setdiff1d(np.arange(n), c)
        E = np.eye(n)
        Hc = np.asarray(self.exact_products(x, sol, E[:, c], check=False, chunk=chunk)[0])
        Hcc = 0.5 * (Hc[c] + Hc[c].T)
        K = np.linalg.solve(Hcc, Hc[bi].T)
        V = E[:, bi].copy()
        V[c] = -K
        HP = np.asarray(self.exact_products(x, sol, V, check=False, chunk=chunk)[0])
        Hred = HP[bi]
        asymmetry = float(np.linalg.norm(Hred - Hred.T) / max(np.linalg.norm(Hred), 1e-300))
        leak = float(np.linalg.norm(HP[c]) / max(np.linalg.norm(Hc[c]), 1e-300))
        return dict(Hred=0.5 * (Hred + Hred.T), Hcc=Hcc, K=K, b=bi, c=c, leak=leak), asymmetry

    def _exact_dense_gradients(self, x, sol, rows=None, H=None):
        """Every row's exact design gradient with the dense exact Hessian: ``mu = H^-1 Q_b^T`` (one factorization for
        every row; with a bootstrap current the block solve of :meth:`projected_hessian`), then ``Q_d - F_d^T mu`` and
        each row's true residual ``|Q_b - H mu| / |Q_b|`` from one more exact call on the multipliers.  No difference
        step, no iteration; the Hessian is kept (``self._hessian``)."""
        import scipy.linalg

        xj = jnp.asarray(x)
        state, params_b = sol["state"], sol["params"]
        b0 = np.asarray(sol["x"], dtype=float)
        try:
            params_bar = self._pullback(state, sol["mask"], params_b, xj)
        except (VmecError, RuntimeError) as error:
            raise TrialRejected(str(error)) from error
        G_b, G_x = map(np.asarray, self._coordinates(params_b, jnp.asarray(b0), xj, params_bar))
        direct = np.asarray(self._direct(state, params_b, xj))
        rows = list(range(G_b.shape[0]) if rows is None else rows)
        t0 = time.perf_counter()
        kept = getattr(self, "_hessian", None)
        scale = np.asarray(self.model.x_scale, dtype=float)
        if H is not None:
            asymmetry = 0.0
        elif (kept is not None and np.array_equal(kept["x"], np.asarray(x, dtype=float))
                and float(np.linalg.norm((kept["b"] - b0) / scale)) <= 1e-6):
            H, asymmetry = kept["H"], 0.0  # the converged Newton polish's Hessian at this record
        elif self.model._projected is not None:
            H, asymmetry = self.projected_hessian(x, sol)
        else:
            H, asymmetry = self.dense_hessian(x, sol)
        seconds_hessian = time.perf_counter() - t0
        Q = G_b[rows].T  # a column per row
        if isinstance(H, dict):  # the current block eliminated (projected_hessian)
            mu = _projected_solve(H, Q)
        else:
            mu = scipy.linalg.lu_solve(scipy.linalg.lu_factor(H), Q)
        H_mu, Fd_mu, _ = self.exact_products(x, sol, mu, check=False)
        gradients = direct[rows] + G_x[rows] - Fd_mu.T
        norms = np.maximum(np.linalg.norm(G_b[rows], axis=1), 1e-300)
        true = (np.linalg.norm(Q - H_mu, axis=0) / norms).tolist()
        self._hessian = dict(b=b0, x=np.asarray(x, dtype=float).copy(), H=H)
        condition = float(np.linalg.cond(H["Hred"] if isinstance(H, dict) else H))
        if isinstance(sol.get("anchor"), dict):  # this record's exact Hessian: later trials' predicted start
            sol["anchor"]["hessian"] = H
        if not isinstance(H, dict):
            sol["hessian"] = H
        return dict(gradients=gradients, rows=rows, true_residual=true, asymmetry=asymmetry, condition=condition,
                    seconds_hessian=seconds_hessian)

    def _exact_gradients(self, x, sol, rows=None):
        """The exact derivative of the converged stationarity ``J_b^T r = 0`` for every requested row: ``dQ/dd = Q_d -
        F_d^T mu``, ``H mu = Q_b^T``, with both curvature blocks, the interior equilibrium's response and the coupled
        bootstrap-current coordinates (:meth:`_exact_dense_gradients`).  A kept Hessian whose multipliers miss
        (true residual above 1e-6) is rebuilt at this record; a fresh one that misses raises."""
        got = self._exact_dense_gradients(x, sol, rows)
        if max(got["true_residual"]) > 1e-6 and getattr(self, "_hessian", None) is not None:
            self._hessian = None
            got = self._exact_dense_gradients(x, sol, rows)
        if max(got["true_residual"]) > 1e-6:
            raise RuntimeError(f"dense exact multipliers missed: true residuals {got['true_residual']}")
        return got

    def exact_products(self, x, sol, V, check=True, chunk=None):
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
        consistency)``: ``consistency`` compares ``L_b``, ``L_x`` with :meth:`_cost_gradients` at the same point;
        every linear solve is certified (raises).
        """
        if getattr(self, "_exact_products_core", None) is None:
            self._exact_products_core = jax.jit(self._exact_products_traced, static_argnames=("chunk",))
        V = np.asarray(V, dtype=float).reshape(self.model.x0.size, -1)
        chunk = None if chunk is None or int(chunk) >= V.shape[1] else max(int(chunk), 1)
        got = self._exact_products_core(sol["state"], sol["mask"], jnp.asarray(x, dtype=float),
                                        jnp.asarray(sol["x"], dtype=float), jnp.asarray(V), chunk=chunk)
        HV, FdV, L_b, L_x, mu_ok, tangent_ok, adjoint_ok = got
        if not (bool(mu_ok) and np.asarray(tangent_ok).all() and np.asarray(adjoint_ok).all()):
            raise RuntimeError(f"an exact-product linear solve missed: multiplier {bool(mu_ok)}, tangents "
                               f"{np.asarray(tangent_ok).tolist()}, adjoints {np.asarray(adjoint_ok).tolist()}")
        if not check:
            return np.asarray(HV), np.asarray(FdV), None
        xj = jnp.asarray(x, dtype=float)
        params0 = self.model.with_boundary(self._params_at(xj), jnp.asarray(sol["x"], dtype=float))
        g_ref, c_ref, _ = self._cost_gradients(np.asarray(sol["x"], dtype=float), params0, self.field(xj),
                                               np.asarray(x, dtype=float))
        consistency = dict(g=float(np.linalg.norm(np.asarray(L_b) - g_ref) / max(np.linalg.norm(g_ref), 1e-300)),
                           c=float(np.linalg.norm(np.asarray(L_x) - c_ref) / max(np.linalg.norm(c_ref), 1e-300)))
        return np.asarray(HV), np.asarray(FdV), consistency

    def _exact_products_traced(self, state, mask, xj, b0, V, chunk=None):
        """The traced body of :meth:`exact_products` (the block system holds callables, so it is built here)."""
        model = self.model
        im, cfg = model._im, model.cfg
        frozen = jax.lax.stop_gradient(state)
        P = im._dof_projector(cfg, mask)
        edge = im._edge_mask(cfg)
        F_raw = im.residual_fn(cfg, frozen, mask, formulation="raw")
        z0 = P(frozen)
        params_at, field_at = self._params_at, self.field

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
            mu, residual, tolerance = im._block_adjoint(system, lambda w: vjp_z(w)[0], system.project(rhs), cfg)
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
            # (jax.linearize once for every column is slower and holds 2x the memory: 69 GB for 16 QA columns).
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

    def _promote(self, sol):
        if "anchor" in sol:
            self._anchor, self._continuation = sol["anchor"], sol["continuation"]

    def _value(self, x):
        return float(self._record(x)["values"][0])

    def _value_gradient(self, x):
        sol = self._record(x)
        gradients = self._linearize(x, sol)
        return float(sol["values"][0]), gradients[0].copy()

    def constraint_values(self, x):
        """The quantities' values, in their units."""
        return self._record(x)["values"][1:].copy()

    def constraint_jac(self, x):
        """The quantities' design gradients."""
        return self._linearize(x, self._record(x))[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the quantities supplied to ``from_loss``, in their units."""
        from .freeboundary_problem import _nonlinear_constraint

        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def accept_x(self, x):
        """Promote an evaluated point after the optimizer accepts it (its anchor and continuation become what
        later trials start from); drop the other trials."""
        key = np.asarray(self._x(x), dtype=float).tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(np.asarray(x, dtype=float).copy()), self.accepted_step + 1
        self._promote(self._cache[key])
        self._cache = {key: self._cache[key]}

    def state(self, x):
        """``(state, params)`` of ``x``'s free boundary (``params`` the ``ImplicitParams`` it was solved at)."""
        sol = self._record(x)
        return sol["state"], sol["params"]

    def boundary_residual(self, x) -> BoundaryResidual:
        """The three interface conditions at ``x``'s free boundary."""
        sol = self._record(x)
        return self.model.boundary_residual(sol["state"], sol["params"], self.field(jnp.asarray(x)))

    def equilibrium_from_x(self, x, *, rtol=1e-6):
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium``: the certified state the optimizer used.

        A fixed-boundary re-solve on its boundary, started from that state, refined onto the root of the model's
        own interior equations (its ``m1_complement``: :func:`~vmex.core.optimize.refine_equilibrium`) and
        recertified there.  The interior root is unique, so this reproduces the optimizer's state; if it moves any
        field by more than ``rtol`` (largest change relative to the field's size) this raises
        :class:`~vmex.core.errors.VmecConvergenceError` instead of returning a different equilibrium.
        """
        from .optimize import refine_equilibrium, solve_equilibrium

        sol = self._record(x)
        eq = refine_equilibrium(
            solve_equilibrium(self.model._im.input_with_params(self.model.fixed, sol["params"]),
                              initial_state=sol["state"]),
            m1_complement=self.model.cfg.m1_complement)
        moved = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(np.asarray(b)))), 1e-300))
                    for a, b in zip(jax.tree.leaves(eq.state), jax.tree.leaves(sol["state"])) if np.size(b))
        if not moved <= rtol:
            raise VmecConvergenceError(
                f"the exported equilibrium differs from the optimizer's certified state by {moved:.2e} (> {rtol:g})",
                hint="the certified state and the re-solve are on different roots: check the m1_complement policy")
        return eq

    def _definition(self):
        """What a checkpoint must match (schema 4): the deck the model was built on (its labelling reference,
        quadrature plan and axis come from it), the model settings (including the m = 1 complement policy), the fit
        settings, the design layout (size, coordinate names and scales, plasma coordinates), the number of value
        rows, and the caller's ``semantic_identity``."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            deck = self.model.fixed.to_indata(Path(folder) / "input.checkpoint").read_text()
        return json.loads(json.dumps(dict(
            deck=deck, model=self.model.options, boundary_ftol=self.boundary_ftol,
            boundary_max_nfev=self.boundary_max_nfev, boundary_max_residual=self.boundary_max_residual,
            boundary_xtol=self.boundary_xtol,
            design_size=int(np.size(self.accepted.parameters)), plasma_coordinates=self._plasma.tolist(),
            names=list(self.names), scales=np.asarray(self.scales, dtype=float).tolist(), rows=self._n_rows,
            semantic_identity=self.semantic_identity)))

    def _maps(self, x):
        """Hash of what ``x`` maps to: its coil field and plasma parameters (the design maps of the problem)."""
        xj = jnp.asarray(x, dtype=float)
        return _fingerprint(self.field(xj), self._params_at(xj))

    def save_checkpoint(self, path):
        """Save the accepted point (``.npz``): its design, certified boundary and interior state, values and
        certificate, and the problem's definition.  :meth:`restore_checkpoint` re-solves it; the optimizer's own
        history is not saved, so a restart is a warm restart of the accepted equilibrium."""
        x = self.accepted.parameters
        sol = self._record(x)
        leaves = jax.tree.leaves(sol["state"])
        meta = dict(schema=CHECKPOINT_SCHEMA, accepted_step=self.accepted_step, definition=self._definition(),
                    maps=self._maps(x), certificate=sol.get("certificate"), state_leaves=len(leaves))
        path = Path(path)
        partial = path.with_name(path.name + ".partial.npz")  # atomic: a killed run leaves the last whole file
        np.savez(partial, x=x, b=np.asarray(sol["x"]), values=sol["values"], rows=sol["rows"],
                 meta=json.dumps(meta), **{f"state{i}": np.asarray(a) for i, a in enumerate(leaves)})
        os.replace(partial, path)

    def restore_checkpoint(self, path, *, rtol=1e-6):
        """Make a saved accepted point (:meth:`save_checkpoint`) this problem's accepted point.

        The problem must have been built with the same definition (deck, settings, design layout) or this raises
        :class:`ValueError`.  The saved boundary itself is evaluated (its interior equilibrium re-solved from the
        saved state, no re-fit: a re-fit moves a soft boundary within the fit's tolerance) and certified; every
        value must agree with the saved one to ``rtol`` relative to its own size or this raises
        :class:`~vmex.core.errors.VmecError`.  Returns the largest relative value change.
        """
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {meta.get('schema')!r}")
        definition = self._definition()
        differ = sorted(k for k in set(definition) | set(meta["definition"])
                        if definition.get(k) != meta["definition"].get(k))
        if differ:
            raise ValueError(f"the checkpoint is of a different problem: {', '.join(differ)} differ")
        model, x = self.model, np.asarray(data["x"], dtype=float)
        if meta["maps"] != self._maps(x):
            raise ValueError("the checkpoint is of a different problem: its design's coil field or plasma "
                             "parameters differ")
        state = jax.tree.unflatten(jax.tree.structure(model.seed[0]),
                                   [jnp.asarray(data[f"state{i}"]) for i in range(meta["state_leaves"])])
        previous = self._continuation  # back in place unless the saved point restores and validates
        self._restore(dict(linearization=None, radius=None, hot={cfg: state for cfg in {model.cfg, model.cfg_trial}}))
        try:
            b, xj = np.asarray(data["b"], dtype=float), jnp.asarray(x)
            params, field = self._params_at(xj), self.field(xj)
            got = model.evaluate(b, params, field, seed=state, tight=True)
            if got is None:
                raise VmecError("the checkpoint's boundary has no certified equilibrium from its saved state")
            rows, aux = got
            fit = dict(x=b, rows=rows, aux=aux, nfev=1, njev=0, accepted=0, converged=True, radius=None,
                       reason="restored")
            try:
                # exactly the saved point (no Newton refinement: refine_accepted() is the explicit step)
                sol = self._finalize(x, params, field, fit, polish=False, newton=False)
            except TrialRejected as error:
                raise VmecError(f"the checkpoint's boundary is not certified: {error}") from error
            saved = np.asarray(data["values"])
            change = float(np.max(np.abs(sol["values"] - saved) / np.maximum(np.abs(saved), 1e-300)))
            if not change <= rtol:
                raise VmecError(f"the restored equilibrium changes the saved values by {change:.2e} (> {rtol:g})")
        except Exception:
            if previous is not None:
                self._restore(previous)
            raise
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), int(meta["accepted_step"])
        self._promote(sol)
        self._cache = {x.tobytes(): sol}
        return change

    def refine_accepted(self):
        """Newton-refine the accepted record (:meth:`_newton_refine`; ``newton_polish`` must be set), e.g. a restored
        checkpoint certified by its gain only.  Returns the largest relative change of its values."""
        if self.newton_polish is None:
            raise ValueError("refine_accepted needs newton_polish")
        x = self.accepted.parameters
        key = x.tobytes()
        sol = self._cache[key]
        xj = jnp.asarray(x)
        params, field = self._params_at(xj), self.field(xj)
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        new = self._newton_refine(x, params, field, sol, extra)
        nb = self.model.x0.size
        new.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=new["x"], J_b=new["J"][:, :nb], J_p=new["J"][:, nb:],
            state=new["state"], params=new["params"], cost=0.5 * new["rows"] @ new["rows"], hessian=new.get("hessian")))
        change = float(np.max(np.abs(new["values"] - sol["values"]) / np.maximum(np.abs(sol["values"]), 1e-300)))
        self._promote(new)
        self._cache[key] = new
        return change

    def close(self):
        """Drop the cached trials but the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}


def _fit_certificate(J, r, normal=None):
    """Stationarity and conditioning of a boundary fit with rows ``r`` and (scaled) Jacobian ``J`` at its point.

    ``stationarity`` is the relative cost reduction a Gauss-Newton step with this Jacobian predicts,
    ``|Q^T r|^2 / |r|^2``, and ``step`` that step's length, ``|J^+ r|`` in the (scaled) coordinates: in a flat
    direction a fit can be far from its stationary point while the cost barely changes, which only ``step`` sees.
    ``condition`` is ``J``'s condition number, and ``rank_deficient`` marks a smallest singular value at the
    rounding level, ``max(J.shape) * eps`` of the largest (NumPy's rank rule).  With ``normal`` (rows: the
    boundary's normal displacement per unit of each scaled coordinate, over the minor radius, on a grid),
    ``normal_step`` is that step's RMS normal displacement of the boundary over the minor radius: what the fit
    still has to move the surface itself, as opposed to its labelling (tangential motion).
    """
    sv = np.linalg.svd(J, compute_uv=False)
    Q, R = np.linalg.qr(J)
    cost = r @ r
    full = np.linalg.solve(R, Q.T @ r) if sv[-1] > 0 else None
    step = float(np.linalg.norm(full)) if full is not None else np.inf
    normal_step = None if normal is None or full is None else float(np.sqrt(np.mean((normal @ full) ** 2)))
    return dict(stationarity=float(np.sum((Q.T @ r) ** 2) / cost) if cost > 0 else 0.0, step=step,
                normal_step=normal_step, condition=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
                rank_deficient=bool(sv[-1] <= sv[0] * max(J.shape) * np.finfo(float).eps))


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
    residual vanishes or is affine; with them, the exact derivative of the re-solved point (the qualification
    reference).  Linear solves, no inverse.
    """
    J_b, J_d = np.asarray(J_b), np.asarray(J_d)
    F_b = J_b.T @ J_b + (0.0 if S_bb is None else np.asarray(S_bb))
    F_d = J_b.T @ J_d + (0.0 if S_bd is None else np.asarray(S_bd))
    return -np.linalg.solve(F_b, F_d)


def least_squares_sensitivity(J_y, J_d, S_yy=None, S_yd=None):
    """``dy/dd`` of a least-squares stationary point ``y(d)``, ``F(y, d) = J_y^T r = 0``: ``-F_y^-1 F_d``.

    ``F_y = J_y^T J_y + S_yy`` and ``F_d = J_y^T J_d + S_yd``, with the curvature terms ``S_yy = sum_i r_i
    d2r_i/dy2`` and the mixed ``S_yd = sum_i r_i d2r_i/dy dd``.  Without them it is the Gauss-Newton sensitivity
    ``-J_y^+ J_d`` that :class:`ThreeTermFreeBoundaryProblem` differentiates with; with them, the exact one (the
    qualification reference: both terms vanish only at a zero residual).
    """
    J_y, J_d = np.atleast_2d(J_y), np.asarray(J_d, dtype=float).reshape(np.shape(J_y)[0], -1)
    if S_yy is None and S_yd is None:
        return -np.linalg.lstsq(J_y, J_d, rcond=None)[0]
    F_y = J_y.T @ J_y + (0.0 if S_yy is None else np.atleast_2d(S_yy))
    F_d = J_y.T @ J_d + (0.0 if S_yd is None else np.asarray(S_yd, dtype=float).reshape(J_y.shape[1], -1))
    return -np.linalg.solve(F_y, F_d)


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray


#: Most Gauss-Newton corrections of the projected (bootstrap-current) coordinates after one boundary step.
PROJECTION_CORRECTIONS = 4

#: An absolute ceiling on the conditional-stationarity gate of the projected (current) coordinates: the gate is
#: ``min(0.1 ftol, PROJECTION_GATE)`` (None: ``0.1 ftol``).  Along the current-following path the rows are close to
#: their reduced linear model, but a current left at ``0.1 ftol`` adds evaluation noise of that order.
PROJECTION_GATE = None


def _projection_gate(ftol):
    """The conditional-stationarity gate of the projected coordinates (:data:`PROJECTION_GATE`)."""
    return 0.1 * ftol if PROJECTION_GATE is None else min(0.1 * ftol, float(PROJECTION_GATE))


def _projected_solve(P, rhs):
    """``H^-1 rhs`` for a Hessian with its current block eliminated (``projected_hessian``): with ``H = T^T diag(H_red,
    H_cc) T``, ``T = [[I, 0], [K, I]]``, each block solved on its own (no solve mixes their scales)."""
    rhs = np.asarray(rhs, dtype=float)
    b, c, K = P["b"], P["c"], P["K"]
    z_b = np.linalg.solve(P["Hred"], rhs[b] - K.T @ rhs[c])
    out = np.empty_like(rhs)
    out[b], out[c] = z_b, np.linalg.solve(P["Hcc"], rhs[c]) - K @ z_b
    return out


def _project(evaluate, x, got, Q_c, R_c, index, budget, tol):
    """Gauss-Newton corrections of the coordinates ``index`` of ``x`` (``J_c = Q_c R_c``, the columns of a nearby
    Jacobian), each kept only if it lowers the cost, until the conditional stationarity test holds: the next
    correction predicts a gain of at most ``tol`` of the cost, ``|Q_c^T r|^2 <= tol |r|^2``.  That is the
    least-squares minimum over those coordinates at the rest of ``x`` (variable projection), so the converged
    problem is unchanged.  Returns ``(x, (rows, aux), evaluations, stationary)``.
    """
    used = 0
    for _ in range(min(PROJECTION_CORRECTIONS, budget) + 1):
        g = Q_c.T @ got[0]
        if g @ g <= tol * (got[0] @ got[0]):
            return x, got, used, True
        if used == min(PROJECTION_CORRECTIONS, budget):
            break
        x_try = np.array(x, dtype=float)
        x_try[index] -= np.linalg.lstsq(R_c, g, rcond=None)[0]
        again = evaluate(x_try)
        used += 1
        if again is None or again[0] @ again[0] >= got[0] @ got[0]:
            break
        x, got = x_try, again
    return x, got, used, False


#: A current block whose smallest singular value is below this fraction of its largest is rank-deficient.
CURRENT_BLOCK_RCOND = 1e-12


def _levenberg_marquardt(evaluate, x, r, aux, J, *, ftol, max_nfev, verbose, radius=None, project=None,
                         current_columns=None):
    """Trust-region Gauss-Newton steps with a fixed Jacobian ``J`` from ``x`` (``evaluate(x) -> (r, aux)`` or ``None``).

    Each step costs one evaluation.  In Marquardt-scaled coordinates the step is the Levenberg-Marquardt step
    whose length is the trust radius (the Gauss-Newton step if shorter), from one SVD of ``J``.  The radius
    shrinks to a quarter of a step that gained less than a quarter of its predicted reduction, or that failed
    to solve, and doubles after a step at the radius that gained three quarters of it.  ``radius`` carries the
    last solve's radius over (at least 1/20 of the first Gauss-Newton step; default: that step's length).
    Stops at a relative cost change of ``ftol`` that the linear model predicted, after three steps in a row
    that each gained under 5% (the Jacobian is stale there), or when the radius falls below 1e-4 of the
    Gauss-Newton step; ``reason`` is ``"ftol"``, ``"stalled"``, ``"radius"`` or ``"max_nfev"``.

    ``project`` (indices of ``x``) are stiff closure coordinates, such as the bootstrap current's, whose rows a
    step of the others moves far off their linear model.  They are eliminated (variable projection): the trust
    region steps the other coordinates ``b`` only, on the reduced model ``(I - P_c) J_b`` (Kaufman's
    approximation; ``P_c`` the projector onto the range of ``J_c``), each step carries the projected
    coordinates' linear correction, and :func:`_project` then re-solves them at the new ``b`` to conditional
    stationarity.  The predicted reduction is that of this composite step, ``|r|^2 - |(I - P_c)(r + J_b db)|^2``
    (halved), so the gain ratio compares the model and the evaluations of one and the same proposal.
    ``projections`` counts the steps whose re-solve did not reach conditional stationarity.

    The current block must have full rank (smallest singular value above ``CURRENT_BLOCK_RCOND`` of the largest), or
    this raises ``RuntimeError`` (a rejected trial).  The corrections use the columns of ``J``, which rotate as the
    boundary moves; with ``current_columns`` (``(x, aux) -> (J_c, aux)``, a fresh current block) the finish is
    certified with the current operator: if the fresh columns show a conditional gain above ``0.1 ftol``, the
    corrections are rerun with them, and a finish that still misses it reports ``reason = "projection_unconverged"``
    and is not converged.  ``projection_certified`` is True, False, or None (no fresh check).
    """
    J = np.asarray(J)
    if not np.all(np.isfinite(J)):
        raise RuntimeError("nonfinite boundary Jacobian")
    n = x.size
    if project is not None:
        project = np.asarray(project)
    free = np.arange(n) if project is None else np.setdiff1d(np.arange(n), project)

    def decompose(J):
        """The current block's QR (rank-checked), the reduced boundary Jacobian and its scaled SVD."""
        if project is None:
            Q_c = R_c = None
            J_free = J
        else:
            Q_c, R_c = np.linalg.qr(J[:, project])
            block_sv = np.linalg.svd(R_c, compute_uv=False)
            if not (np.all(np.isfinite(block_sv)) and block_sv[-1] > CURRENT_BLOCK_RCOND * block_sv[0]):
                raise RuntimeError(f"rank-deficient current block (singular values {block_sv[0]:.2e} .. "
                                   f"{block_sv[-1]:.2e})")
            J_free = J[:, free] - Q_c @ (Q_c.T @ J[:, free])  # the reduced Jacobian of the boundary coordinates
        norms = np.linalg.norm(J_free, axis=0)
        norms[norms == 0.0] = 1.0
        try:
            U, sv, Vt = np.linalg.svd(J_free / norms, full_matrices=False)
        except np.linalg.LinAlgError as error:  # a rejected trial, not the end of the run
            raise RuntimeError(f"boundary Jacobian: {error}") from error
        return Q_c, R_c, norms, U, sv, Vt

    Q_c, R_c, norms, U, sv, Vt = decompose(J)
    updates = extrapolated = accelerated = 0
    nfev, accepted, converged, slow, reason, unprojected = 1, 0, False, 0, "max_nfev", 0
    started = time.perf_counter()

    def step(g, lam):
        return -(sv / (sv**2 + lam)) * g  # in the singular basis of the scaled (reduced) Jacobian

    while nfev < max_nfev:
        g = U.T @ r
        g_c = np.zeros(0) if Q_c is None else Q_c.T @ r  # the projected coordinates' own gain, if re-solved
        cost = 0.5 * r @ r
        if 0.5 * (g @ g + g_c @ g_c) <= ftol * cost:  # the Gauss-Newton step's predicted gain
            converged, reason = True, "ftol"
            break
        full = np.linalg.norm(step(g, 0.0))
        if nfev == 1:  # a carried radius, but not far below this Jacobian's own step
            radius = full if radius is None else max(radius, 0.05 * full)
        if radius < 1e-4 * full:
            reason = "radius"
            break
        lam = 0.0
        if full > radius:  # the damping whose step has the radius' length (bisection in log lam)
            lo, hi = np.log(sv[-1] ** 2 * 1e-12 + 1e-300), np.log(sv[0] ** 2 * (full / radius) + 1e-300)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if np.linalg.norm(step(g, np.exp(mid))) > radius else (lo, mid)
            lam = np.exp(hi)
        z = step(g, lam)
        length = np.linalg.norm(z)
        # (I - P_c) r = r - Q_c g_c: the reduced model's residual, plus the gain of the projected coordinates
        predicted = cost - 0.5 * (r @ r - g_c @ g_c - g @ g + np.sum((g + sv * z) ** 2))
        x_new = np.array(x, dtype=float)
        db = (Vt.T @ z) / norms
        x_new[free] += db
        if Q_c is not None:  # the projected coordinates' linear correction along the composite step
            x_new[project] -= np.linalg.lstsq(R_c, Q_c.T @ (r + J[:, free] @ db), rcond=None)[0]
        if GEODESIC and nfev + 2 <= max_nfev:
            dx = x_new - x
            probe = evaluate(x + GEODESIC_H * dx)
            nfev += 1
            if probe is not None:
                r2 = (2.0 / GEODESIC_H) * ((np.asarray(probe[0]) - r) / GEODESIC_H - J @ dx)
                g2 = U.T @ (r2 if Q_c is None else r2 - Q_c @ (Q_c.T @ r2))
                a_z = step(g2, lam)
                a = np.zeros(n)
                a[free] = (Vt.T @ a_z) / norms
                if Q_c is not None:
                    a[project] -= np.linalg.lstsq(R_c, Q_c.T @ (r2 + J[:, free] @ a[free]), rcond=None)[0]
                if np.all(np.isfinite(a)) and np.linalg.norm(a_z) <= GEODESIC_ALPHA * np.linalg.norm(z):
                    x_new = x_new + 0.5 * a
                    accelerated += 1
        got = evaluate(x_new)
        nfev += 1
        if got is not None and Q_c is not None:
            x_new, got, used, stationary = _project(evaluate, x_new, got, Q_c, R_c, project, max_nfev - nfev,
                                                    _projection_gate(ftol))
            nfev += used
            unprojected += not stationary
        cost_new = np.inf if got is None else 0.5 * got[0] @ got[0]
        rho = (cost - cost_new) / predicted if predicted > 0 else -np.inf
        if verbose:
            print(f"Levenberg-Marquardt step {nfev - 1}: |step| {length:.2e} (radius {radius:.2e}), cost "
                  f"{cost:.6e} -> {cost_new:.6e}, gain ratio {rho:.2f} ({time.perf_counter() - started:.1f} s)",
                  flush=True)
        if rho < 0.25:
            radius = 0.25 * length
        elif rho > 0.75 and length > 0.99 * radius:
            radius = 2.0 * radius
        if STALL_RULE == "model":
            slow = slow + 1 if rho < 0.25 else 0
        else:
            slow = slow + 1 if cost_new > (1.0 - STALL_GAIN) * cost else 0
        if cost_new < cost:
            if JACOBIAN_UPDATE == "broyden":  # rank-one secant update along the accepted step (Broyden's good)
                dx, dr = x_new - x, got[0] - r
                if dx @ dx > 0:
                    J = J + np.outer(dr - J @ dx, dx) / (dx @ dx)
                    Q_c, R_c, norms, U, sv, Vt = decompose(J)
                    updates += 1
            x_prev = x
            x, (r, aux), accepted = x_new, got, accepted + 1
            if GN_EXTRAPOLATE and lam == 0.0 and rho > 0.75:
                d, factor = x_new - x_prev, 1.0
                for _ in range(GN_EXTRAPOLATE_MAX):
                    if nfev >= max_nfev:
                        break
                    factor *= 2.0
                    x_try = x_prev + factor * d
                    trial = evaluate(x_try)
                    nfev += 1
                    if trial is not None and Q_c is not None:
                        x_try, trial, used, stationary = _project(evaluate, x_try, trial, Q_c, R_c, project,
                                                                  max_nfev - nfev, _projection_gate(ftol))
                        nfev += used
                        unprojected += not stationary
                    cost_try = np.inf if trial is None else 0.5 * trial[0] @ trial[0]
                    if verbose:
                        print(f"  extrapolated x{factor:.0f}: cost {cost_new:.6e} -> {cost_try:.6e}", flush=True)
                    if not cost_try < cost_new:
                        break
                    x, (r, aux), cost_new, extrapolated = x_try, trial, cost_try, extrapolated + 1
            if cost - cost_new <= ftol * cost and rho > 0.25:  # a small gain the linear model foresaw
                converged, reason = True, "ftol"
                break
        if slow >= STALL_STEPS:  # steps in a row gained under STALL_GAIN: the Jacobian no longer describes the rows
            reason = "stalled"
            break
    certified_projection = None
    if project is not None and current_columns is not None:
        gate = _projection_gate(ftol)

        def fresh(x, aux, r):
            J_c, aux = current_columns(x, aux)
            Q, R = np.linalg.qr(np.asarray(J_c))
            g = Q.T @ r
            return Q, R, aux, bool(g @ g <= gate * (r @ r))

        Q_f, R_f, aux, certified_projection = fresh(x, aux, r)
        if not certified_projection:  # the stale columns' stationarity is not the current operator's: refresh
            budget = max(min(PROJECTION_CORRECTIONS, max_nfev - nfev), 1)
            x, (r, aux), used, _ = _project(evaluate, x, (r, aux), Q_f, R_f, project, budget, gate)
            nfev += used
            Q_f, R_f, aux, certified_projection = fresh(x, aux, r)
            if not certified_projection:
                converged, reason = False, "projection_unconverged"
    return dict(x=x, rows=r, aux=aux, nfev=nfev, accepted=accepted, converged=converged, radius=radius,
                reason=reason, projections=unprojected, projection_certified=certified_projection,
                jacobian_updates=updates, extrapolated=extrapolated, accelerated=accelerated)
