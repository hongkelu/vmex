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


def _condensation_rows(gamma, mpol, ntor):
    """VMEC's spectral-condensation condition on the boundary (Hirshman & Breslau 1998, VMEC2000's ``alias.f``):
    the ``m = 1 .. mpol - 2``, ``|n| <= ntor`` Fourier components of
    ``G = R_theta sum m (m - 1) R_mn cos(m theta - n N phi) + Z_theta sum m (m - 1) Z_mn sin(m theta - n N phi)``,
    relative to the mean ``R_theta^2 + Z_theta^2``.  Free-boundary VMEC holds its boundary's poloidal angle by this
    condition once its anchor to the initial angle has decayed (``funct3d.f``).  Neither it nor equal arc holds
    exactly for a truncated surface without moving it: the rows trade against the interface rows by their weight.

    ``gamma`` is ``(3, nphi, ntheta)`` Cartesian points over one field period.  Returns the real and imaginary
    parts of those components, flattened.
    """
    nphi, ntheta = gamma.shape[-2:]
    k = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
    R, Z = jnp.fft.fft(jnp.hypot(gamma[0], gamma[1]), axis=-1), jnp.fft.fft(gamma[2], axis=-1)
    theta_derivative = lambda f: jnp.real(jnp.fft.ifft(1j * k * f, axis=-1))  # noqa: E731
    condensed = lambda f: jnp.real(jnp.fft.ifft(jnp.abs(k) * (jnp.abs(k) - 1) * f, axis=-1))  # noqa: E731
    R_theta, Z_theta = theta_derivative(R), theta_derivative(Z)
    G = (condensed(R) * R_theta + condensed(Z) * Z_theta) / jnp.mean(R_theta**2 + Z_theta**2)
    spectrum = jnp.fft.fft2(G) / G.size
    rows = spectrum[np.r_[0:ntor + 1, nphi - ntor:nphi]][:, 1:mpol - 1]
    return jnp.concatenate([jnp.real(rows).ravel(), jnp.imag(rows).ravel()])


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
    ``labelling="equal_arc"`` replaces them by equal arc length of the current surface, and
    ``labelling="condensed"`` by VMEC's spectral-condensation condition (:func:`_condensation_rows`).
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
        if labelling not in ("start", "equal_arc", "condensed"):
            raise ValueError(f"labelling must be 'start', 'equal_arc' or 'condensed', got {labelling!r}")
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
            elif labelling == "condensed":
                sizes.append(("phase", 2 * (2 * int(self.fixed.ntor) + 1) * (int(self.fixed.mpol) - 2) + int(nphi)))
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
            elif labelling == "condensed":  # VMEC's free-boundary labelling, with equal arc's m = 1 phase rows
                phase = _equal_arc_rows(gamma)[1] / np.sqrt(nphi)
                label = jnp.zeros_like(gamma[0])
                phase = label_weight * jnp.concatenate([jnp.sqrt(jnp.sum(area)) * _condensation_rows(
                    gamma, int(fixed.mpol), int(fixed.ntor)), phase])
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

        def tangents(params, state, mask, batch):
            """Projected state responses to the stacked parameter tangents (one block factorization, chunked), and
            whether each column's linear solve is certified."""
            solution, report = im._implicit_evolved_tangent_multi_rhs(
                params, cfg, jax.lax.stop_gradient(state), mask, batch, active_fields=active,
                probe_chunk_size=self.chunk, response_chunk_size=self.chunk,
                certify_rtol=float(cfg.jacobian_adjoint_tol), certify_maxiter=int(cfg.jacobian_adjoint_maxiter))
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

        self.pullback = pullback
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._radius = None  # the boundary steps' trust radius, carried from one solve to the next

    def solve(self, params, seed=None, tight=True):
        """Hot-restarted fixed-boundary equilibrium at ``params``: ``(state, mask)``, or ``None`` if not certified.

        ``tight=False`` solves to the trial tolerance, without the Newton refinement that certifies a state for
        differentiation (:meth:`linearize` re-solves tight first).
        """
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
            for round_ in range(POLISH_ROUNDS + 1):
                normal = None
                if self._normal_operator is not None:  # the remaining step's motion of the surface itself
                    N, a = map(np.asarray, self._normal_operator(jnp.asarray(sol["x"])))
                    normal = np.hstack([N * scale[:, :N.shape[1]], np.zeros((N.shape[0], n - N.shape[1]))]) / a
                sol["certificate"] = certificate = _fit_certificate(sol["J"][:, :n] * scale, sol["rows"], normal)
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
                stats["polish_rounds"] = stats.get("polish_rounds", 0) + 1
            if not settled:
                normal_note = "" if certificate["normal_step"] is None else (
                    f" and moves the boundary normally by {certificate['normal_step']:.2e} of the minor radius")
                raise TrialRejected(f"boundary fit not stationary: a Gauss-Newton step with its own Jacobian gains "
                                    f"{certificate['stationarity']:.2e} of the cost{normal_note}")
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
                  **model_options):
        """Solve the seed design's free boundary (or, with ``checkpoint``, restore a saved accepted point instead:
        :meth:`restore_checkpoint`; ``x0`` then only fixes the design's layout).  ``semantic_identity`` (any
        JSON-serializable value, e.g. a version and the names of the maps and objectives) is the caller's declaration
        of what the callbacks compute: a checkpoint restores only into a problem with the same declaration, since
        callbacks that agree at one design can still differ in their derivatives.

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
                   semantic_identity=semantic_identity)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_ftol, boundary_max_nfev, boundary_max_residual=None,
                 boundary_xtol=None, checkpoint=None, semantic_identity=None):
        im = model._im
        self.model, self.field = model, field_from_parameters
        self.boundary_ftol, self.boundary_max_nfev = float(boundary_ftol), int(boundary_max_nfev)
        self.boundary_max_residual = None if boundary_max_residual is None else float(boundary_max_residual)
        self.boundary_xtol = None if boundary_xtol is None else float(boundary_xtol)
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
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0, fit_reasons={},
                          max_condition=0.0)
        self._continuation = None
        # The base problem only stores the layout (names, scales), which the checkpoint definition includes.
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})
        if checkpoint is not None:
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0  # the layout restore_checkpoint checks
            self.restore_checkpoint(checkpoint)
        else:
            seed = self._solve(x0)
            if seed is None:
                raise VmecError(f"the seed design has no three-term free boundary: {self._rejections[x0.tobytes()]}")
            self._promote(seed)
            self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0

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
        if starts is not None:  # given starts (a restore: the saved boundary)
            anchor = {}
        elif anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        else:
            starts = [None]
        sol, why = None, []
        for i, start in enumerate(starts):
            if self._continuation is not None:  # every start from the accepted point's continuation
                self._restore(self._continuation)
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, ftol=self.boundary_ftol,
                                                max_nfev=self.boundary_max_nfev, refresh_on_stall=True,
                                                target_cost=4 * anchor["cost"] if anchor else None)
            except (VmecError, RuntimeError) as error:  # an uncertified equilibrium: try the next start
                why.append(f"start {i}: {error}")
                continue
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
            try:  # the certificate decides (a stall on a fresh Jacobian may be stationary)
                sol = self._finalize(x, params, field, fit, polish)
                break
            except TrialRejected as error:
                why.append(f"start {i}: {error}")
        if sol is None:
            return self._reject(key, "; ".join(why) or "no three-term free boundary at this design point")
        self._cache[key] = sol
        return sol

    def _reject(self, key, reason):
        self.stats["failed"] += 1
        self._cache[key], self._rejections[key] = None, reason
        return None

    def _finalize(self, x, params, field, fit, polish=True):
        """A trial's one record: its fit certified by :meth:`ThreeTermFreeBoundaryModel.certify` (re-solved to the
        deck's tolerance, polished with its own Jacobian when ``polish``, certified, and only then valued).  Every
        value, gradient, acceptance and checkpoint of ``x`` uses this record; differentiating it adds gradients and
        never moves its boundary.  Raises :class:`~vmex.core.errors.TrialRejected`."""
        model, xj, nb = self.model, jnp.asarray(x), self.model.x0.size
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        sol = model.certify(params, field, fit, gate=max(self.boundary_ftol, POLISH_FTOL), xtol=self.boundary_xtol,
                            polish=polish, max_residual=self.boundary_max_residual, extra=extra,
                            values=lambda state, params_b: self._rows(state, params_b, xj), stats=self.stats)
        J_b, J_p = sol["J"][:, :nb], sol["J"][:, nb:]
        # What later trials start from if the optimizer accepts this one (accept_x): its linearization now.
        sol.update(continuation=self._checkpoint(), anchor=dict(
            x=np.asarray(x, dtype=float).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=sol["state"],
            params=sol["params"], cost=0.5 * sol["rows"] @ sol["rows"]))
        return sol

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
        if not np.all(np.isfinite(gradients)):
            raise TrialRejected("nonfinite design gradient")
        sol["gradients"] = gradients
        return gradients

    def exact_design_gradient(self, x, row=0, *, h=1e-4, rtol=1e-8, maxiter=30):
        """The exact design gradient of value row ``row`` at ``x`` (qualification reference; costly).

        The optimizer's gradient (:meth:`_linearize`) differentiates the boundary fit with the Gauss-Newton
        sensitivity ``-(J_b^T J_b)^-1 J_b^T J_d``, exact only at a zero residual.  This is the derivative of the
        fit's stationarity condition ``g(b, d) = J_b^T r = 0`` itself:

            dQ/dd = Q_d - F_d^T mu,   H mu = Q_b^T,   H = dg/db = J_b^T J_b + sum_i r_i d2r_i/db2,
            F_d = dg/dd = J_b^T J_d + sum_i r_i d2r_i/db dd,

        with ``mu`` by conjugate gradients preconditioned with ``J_b^T J_b``.  Each product with ``H`` is a
        central difference (step ``h`` in the scaled coordinates) of ``g`` at boundaries ``b +- eps v`` (tight
        interior solves at fixed boundaries and design, no fits), and ``F_d^T mu`` the central difference along
        ``mu`` of the fixed-boundary design gradient of ``|r|^2 / 2`` (the coil field's VJP and the plasma
        columns).  With the curvature terms dropped this is exactly the Gauss-Newton gradient.  Cost: two
        linearizations per CG iteration and two for ``F_d^T mu``.
        """
        model, nb = self.model, self.model.x0.size
        x = np.asarray(x, dtype=float)
        xj = jnp.asarray(x)
        sol = self._record(x)
        gradients = self._linearize(x, sol)  # the Gauss-Newton gradients (and G_b, direct terms below)
        params, field = self._params_at(xj), self.field(xj)
        extra = self._plasma_directions(params, xj) if self._plasma.size else None
        b0, J_b0 = np.asarray(sol["x"], dtype=float), np.asarray(sol["J"])[:, :nb]
        scale = np.asarray(model.x_scale, dtype=float)
        state, params_b = sol["state"], sol["params"]
        try:
            params_bar = self._pullback(state, sol["mask"], params_b, xj)
        except (VmecError, RuntimeError) as error:
            raise TrialRejected(str(error)) from error
        G_b, G_x = map(np.asarray, self._coordinates(params_b, jnp.asarray(b0), xj, params_bar))
        Q_b = G_b[row]

        def at(bb):
            got = model.evaluate(bb, params, field, tight=True)
            if got is None:
                raise TrialRejected("an exact-gradient difference point has no certified interior root")
            rows, aux = got
            J, _, _, aux = model.linearize(bb, aux, field, extra=extra)
            J = np.asarray(J)
            return np.asarray(rows), J[:, :nb], (J[:, nb:] if extra is not None else None), aux

        def step(v):
            return h / max(float(np.linalg.norm(v / scale)), 1e-300)

        def g(bb):
            rows, J_b, _, _ = at(bb)
            return J_b.T @ rows

        def hessian(v):
            eps = step(v)
            return (g(b0 + eps * v) - g(b0 - eps * v)) / (2 * eps)

        Q, R = np.linalg.qr(J_b0)

        def precondition(v):  # (J_b^T J_b)^-1 v
            return np.linalg.solve(R, np.linalg.solve(R.T, v))

        mu = precondition(Q_b)  # the Gauss-Newton multiplier: the initial guess
        res = Q_b - hessian(mu)
        z = precondition(res)
        d = z.copy()
        rz = res @ z
        norm0 = float(np.linalg.norm(Q_b))
        iterations = 0
        for iterations in range(1, maxiter + 1):
            if float(np.linalg.norm(res)) <= rtol * norm0:
                break
            Hd = hessian(d)
            alpha = rz / (d @ Hd)
            mu = mu + alpha * d
            res = res - alpha * Hd
            z = precondition(res)
            rz, rz_old = res @ z, rz
            d = z + (rz / rz_old) * d
        converged = float(np.linalg.norm(res)) <= rtol * norm0

        def design_gradient_of_cost(bb):
            rows, _, J_p, aux = at(bb)
            st, _, pb, _ = aux
            out = np.asarray(self._field_vjp(st, pb, xj, jnp.asarray(rows)))
            if J_p is not None:
                out = out.copy()
                out[self._plasma] += J_p.T @ rows
            return out

        eps = step(mu)
        Fd_mu = (design_gradient_of_cost(b0 + eps * mu) - design_gradient_of_cost(b0 - eps * mu)) / (2 * eps)
        direct = np.asarray(self._direct(state, params_b, xj))[row]
        exact = direct + G_x[row] - Fd_mu
        return dict(gradient=exact, gauss_newton=np.asarray(gradients[row]), cg_iterations=iterations,
                    cg_converged=bool(converged), cg_residual=float(np.linalg.norm(res)) / max(norm0, 1e-300))

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
                sol = self._finalize(x, params, field, fit, polish=False)
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
    updates = extrapolated = 0
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
        got = evaluate(x_new)
        nfev += 1
        if got is not None and Q_c is not None:
            x_new, got, used, stationary = _project(evaluate, x_new, got, Q_c, R_c, project, max_nfev - nfev,
                                                    0.1 * ftol)
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
                                                                  max_nfev - nfev, 0.1 * ftol)
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
        gate = 0.1 * ftol

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
                jacobian_updates=updates, extrapolated=extrapolated)
