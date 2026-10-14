"""Free-boundary equilibria by virtual casing, with the boundary as the unknown.

At a free boundary ``S`` without a sheet current the plasma-vacuum interface conditions are
(Conlin et al. 2024, arXiv:2412.05680)

1. ``B_out . n = 0``,
2. ``|B_out|^2 = |B_in|^2 + 2 mu0 p``,
3. ``n x (B_out - B_in) = mu0 K = 0``,

with ``B_out = B_coil + B_plasma`` and ``B_plasma`` the plasma's own field from the virtual-casing principle
(:mod:`vmex.core.virtual_casing`).  Two curl- and divergence-free fields outside ``S`` with the same normal
component and the same net poloidal current ``G`` are equal, so (1) and ``G_plasma = G_coil`` imply (2) and (3)
when the edge pressure vanishes (a nonzero edge pressure needs a sheet current and is rejected).

The unknown is the displacement ``h(theta, zeta)`` of the deck's boundary along the deck equilibrium's radial
coordinate lines, measured normal to the surface (one function on the deck's boundary modes;
:func:`_normal_displacement_map`).  The deck fixes the boundary's poloidal labelling: at finite resolution the
labelling decides how well the truncated series represents the surface, and a labelling taken from the start (or
left free) makes the answer depend on the start or the fit crawl.  ``initial_boundary`` only sets the starting
``h``.  The equations are square in ``h``, with the net poloidal current row ``G_plasma = G_coil`` for the
``(0, 0)`` mode:

- vacuum (no pressure, no plasma current): ``B_out = B_coil`` is an explicit function of the boundary, and the
  rows are the stationarity of the area-weighted mean square of ``B_coil . n`` in ``h``
  (:func:`_mean_square_normal_field`): the least-squares free boundary, with its exact Hessian as the Newton
  Jacobian.  A projection of ``B . n`` onto the boundary modes instead leaves the residual outside them
  uncontrolled (LP QH 12 x 12: B.n 4.9e-4, above the deck's 1.8e-4; least squares 4.2e-5);
- finite beta: the same stationarity with the plasma's own response to ``h`` left out of the test functions:
  a normal displacement ``h`` changes ``B . N`` by ``-|N| B_t . grad h`` (``N`` the area normal, ``B_t`` the
  tangential field), so the rows are the area-weighted moments of ``B_out . n`` against ``B_out . grad
  cos(m theta - n nfp phi)``, one per boundary mode (the exact least-squares rows would need the equilibrium's
  Hessian).  The boundary keeps two modes fewer than the interior (``_BOUNDARY_MODE_GAP``: the interior's edge
  field is least accurate in its top modes).  LP QA at 2.5 % beta, 12 x 12, evaluated by a 16-mode re-solve:
  B.n / pressure jump / sheet current 2.8e-4 / 9.9e-5 / 2.8e-4, against 4.1e-4 / 1.1e-4 / 4.1e-4 with plain
  sine moments of ``B_out . n``.

Two curl- and divergence-free fields with the same normal component and the same ``G`` are equal, so the
pressure balance (2) and the sheet current (3) follow and are reported (:func:`boundary_residual`), not fitted.

Every trial boundary is a fixed-boundary equilibrium (Newton-refined; :mod:`vmex.core.implicit`), so the rows
and their exact implicit derivatives are smooth functions of the boundary, the coils and the plasma parameters.
VMEC2000 counterpart: none; this replaces NESTOR (``vacuum.f`` and its callees, :mod:`vmex.core.vacuum`) as the
free-boundary condition.

Entry points, from the most to the least packaged:

- a free-boundary deck with ``!@VMEX BOUNDARY_CONDITION = THREE_TERMS``
  (``vmex input.case``, :func:`~vmex.core.multigrid.solve_file`), the CLI flag
  ``--boundary-condition three-terms``, or
  ``solve_free_boundary_multigrid(..., boundary_condition="three_terms")``;
- :func:`solve_free_boundary_three_terms`: one solve with every control, and
  ``previous=`` to restart from an earlier result after the coils change;
- ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_terms")``
  (:class:`ThreeTermsFreeBoundaryProblem`): single-stage optimization, a loss
  and constraint rows of coils and plasma parameters at their free boundary,
  for :func:`vmex.core.optimize.minimize`;
- :class:`ThreeTermsFreeBoundaryModel`: the residual, its Jacobian, state
  tangents and pullbacks for one deck and any external field or plasma
  parameters, without recompiling -- the pieces an optimizer needs;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. to
  check a NESTOR result.

The names keep ``three_terms`` (the three interface conditions above) for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
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
    """Rows and weights of :func:`boundary_residual`, plus the boundary points ``(3, nphi, ntheta)``.

    ``precision=None`` with ``digits=None`` is the vacuum: no plasma current, so ``B_out`` is the external field
    alone (no virtual casing, whose quadrature would only add its error).
    """
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    if digits is None:
        B_in, normal = jnp.asarray(data.B_total), jnp.asarray(data.normal)
        B_out = vc.external_B_cartesian(external_field, jnp.asarray(data.gamma), jnp.asarray(data.phi))
        area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
        B2 = jnp.sum(B_in**2, axis=0)
        scale = jnp.sqrt(B2)
        rows = [jnp.sum(B_out * normal, axis=0) / scale, (jnp.sum(B_out**2, axis=0) - B2) / (2.0 * B2),
                jnp.cross(normal, B_out - B_in, axis=0) / scale]
        return jnp.concatenate([rows[0][None], rows[1][None], rows[2]]), area / jnp.sum(area), data.gamma
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
                      nphi: int, ntheta: int, digits: int = vc.VC_DIGITS, precision=None,
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
    rc, zs = np.asarray(wout.raxis_cc, dtype=float), np.asarray(wout.zaxis_cs, dtype=float)
    cos, sin = np.cos(np.outer(phi, harmonics)), np.sin(np.outer(phi, harmonics))
    R, Z = cos @ rc, -sin @ zs
    R_p, Z_p = -sin @ (rc * harmonics), -cos @ (zs * harmonics)  # exact tangent: the periodic sum is spectral
    points = np.stack([R * np.cos(phi), R * np.sin(phi), Z])
    tangent = np.stack([R_p * np.cos(phi) - R * np.sin(phi), R_p * np.sin(phi) + R * np.cos(phi), Z_p])
    return jnp.asarray(points), jnp.asarray(tangent), float(phi[1] - phi[0])


def _net_current(field, points, tangent, dphi):
    """Traceable ``G = (1/2 pi)`` circulation of ``field`` along a closed curve (:func:`_axis_loop`)."""
    B = vc.external_B_cartesian(field, points[:, :, None])[:, :, 0]
    return jnp.sum(B * tangent) * dphi / (2.0 * jnp.pi)


def _coil_net_current(external_field, wout) -> float:
    """``G = (1/2 pi)`` times the external field's circulation around the magnetic axis of ``wout``."""
    return float(_net_current(external_field, *_axis_loop(wout)))


def _surface_frame(rmnc, zmns, xm, xn, phi, theta):
    """Points, ``e_theta``, ``e_phi`` (each ``(3, nphi, ntheta)``) of the surface ``(rmnc, zmns)`` (traceable).

    ``phi, theta`` are the ``(nphi, ntheta)`` grid; the series is ``sum cos|sin(xm theta - xn phi)``.
    """
    phase = xm[:, None, None] * theta[None] - xn[:, None, None] * phi[None]
    cos, sin = jnp.cos(phase), jnp.sin(phase)
    R, Z = jnp.tensordot(rmnc, cos, 1), jnp.tensordot(zmns, sin, 1)
    R_t, Z_t = jnp.tensordot(-rmnc * xm, sin, 1), jnp.tensordot(zmns * xm, cos, 1)
    R_p, Z_p = jnp.tensordot(rmnc * xn, sin, 1), jnp.tensordot(-zmns * xn, cos, 1)
    c, s = jnp.cos(phi), jnp.sin(phi)
    return (jnp.stack([R * c, R * s, Z]), jnp.stack([R_t * c, R_t * s, Z_t]),
            jnp.stack([R_p * c - R * s, R_p * s + R * c, Z_p]))


def _mean_square_normal_field(rmnc, zmns, xm, xn, phi, theta, phi_grid, field):
    """Area-weighted mean of ``(B . n)^2`` of ``field`` on the surface ``(rmnc, zmns)`` (traceable).

    ``phi, theta`` are the ``(nphi, ntheta)`` grid (one field period), ``phi_grid`` its ``phi`` axis.
    """
    points, e_theta, e_phi = _surface_frame(rmnc, zmns, xm, xn, phi, theta)
    N = jnp.cross(e_theta, e_phi, axis=0)
    area = jnp.linalg.norm(N, axis=0)
    B = vc.external_B_cartesian(field, points, phi_grid)
    return jnp.sum(jnp.sum(B * N, axis=0) ** 2 / area) / jnp.sum(area)


#: Relative change of the displacement map under a doubled grid at which its grid counts as converged: the map's
#: error then moves a boundary displacement ``h`` by less than ``sqrt(eps) |h|``, below the Newton step tolerance
#: (:func:`newton_step_converged`).  LP QA 8 x 8 / 12 x 12 decks (``paper/runs/b23``): 64 x 32 -> 128 x 64 changes
#: the map by 1.8e-7 / 1.1e-4, 128 x 64 -> 256 x 128 by 9e-14 / 1.2e-12.
_MAP_GRID_RTOL = float(np.sqrt(np.finfo(float).eps))


def _normal_displacement_map(inp, max_mode, alpha=1.2, ntheta=None, nzeta=None, direction=None):
    """:func:`_normal_displacement_map_on` on a grid that resolves it: from four points per boundary harmonic in
    each direction (``4 mpol`` poloidal, ``4 ntor + 4`` per field period), doubled until a doubling changes the map
    by less than ``_MAP_GRID_RTOL`` of its largest entry.  The displacement times the direction field
    ``g = n / (n_R^2 + n_Z^2)`` is not band-limited, so the fit depends on the grid until it is resolved.
    ``ntheta, nzeta`` fix the grid instead (no convergence check)."""
    if ntheta is not None and nzeta is not None:
        return _normal_displacement_map_on(inp, max_mode, alpha, int(ntheta), int(nzeta), direction)
    ntor, mpol = (np.shape(inp.rbc)[0] - 1) // 2, np.shape(inp.rbc)[1]
    nt, nz = 4 * mpol, 4 * ntor + 4
    out = _normal_displacement_map_on(inp, max_mode, alpha, nt, nz, direction)
    while True:
        nt, nz = 2 * nt, 2 * nz
        finer = _normal_displacement_map_on(inp, max_mode, alpha, nt, nz, direction)
        change = np.max(np.abs(finer[0] - out[0])) / np.max(np.abs(finer[0]))
        out = finer
        if change <= _MAP_GRID_RTOL:
            return out
        if nt * nz > 2**22:  # 4M points: the map does not converge (a non-smooth direction field)
            raise ValueError(f"the boundary displacement map does not converge in its grid ({change:.1e} at "
                             f"{nt} x {nz})")


def _normal_displacement_map_on(inp, max_mode, alpha, ntheta, nzeta, direction):
    """Boundary coordinates as a normal displacement of ``inp``'s surface: ``(M, modes, x_scale)``.

    ``h(theta, zeta) = sum h_mn cos(m theta - n nfp zeta)`` over ``modes`` (those of
    :func:`~vmex.core.optimize.pack_boundary`, then ``(0, 0)``) moves each boundary point along the surface
    normal, within its ``phi = zeta`` plane: ``(dR, dZ) = h (N_R, N_Z) / (N_R^2 + N_Z^2)`` with ``N`` the unit
    normal, so its normal displacement is ``h``.  Its truncated Fourier fit is linear in ``h``: the packed
    boundary coordinates are ``x0 + M h``.  Tangential moves (relabellings ``theta -> theta + delta(theta,
    zeta)``, which no interface condition sees) are thereby absent from the unknowns.
    """
    from .optimize import _dof_modes, pack_boundary

    rbc, zbs, nfp = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float), int(inp.nfp)
    ntor, mpol = (rbc.shape[0] - 1) // 2, rbc.shape[1]
    th = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ze = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    T, Zg = np.meshgrid(th, ze, indexing="ij")
    ns, ms = np.arange(-ntor, ntor + 1), np.arange(mpol)
    ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * Zg[None, None]
    cos, sin = np.cos(ph), np.sin(ph)
    del ph
    R = np.einsum("nm,nmij->ij", rbc, cos)
    Rt = np.einsum("nm,nmij->ij", -rbc * ms[None, :], sin)
    Zt = np.einsum("nm,nmij->ij", zbs * ms[None, :], cos)
    NR, NZ = -R * Zt, R * Rt  # (e_theta x e_zeta) in its R and Z components; the phi one is not representable
    N = np.sqrt(NR**2 + NZ**2 + (np.einsum("nm,nmij->ij", zbs * ms[None, :], cos)
                                 * np.einsum("nm,nmij->ij", rbc * ns[:, None] * nfp, sin)
                                 - Rt * np.einsum("nm,nmij->ij", -zbs * ns[:, None] * nfp, cos)) ** 2)
    nR, nZ = NR / N, NZ / N
    if direction is None:
        gR, gZ = nR / (nR**2 + nZ**2), nZ / (nR**2 + nZ**2)
    else:  # move along a given (R, Z) direction field, scaled so the normal displacement is h
        dR, dZ = direction(T, Zg)
        gR, gZ = dR / (nR * dR + nZ * dZ), dZ / (nR * dR + nZ * dZ)
    keep = ~((ms[None, :] == 0) & (ns[:, None] <= 0))
    keep_r = ~((ms[None, :] == 0) & (ns[:, None] < 0))
    A_r = cos[keep_r].reshape(int(keep_r.sum()), -1).T
    A_z = sin[keep].reshape(int(keep.sum()), -1).T
    pinv_r, pinv_z = np.linalg.pinv(A_r), np.linalg.pinv(A_z)
    modes = list(_dof_modes(inp, max_mode)) + [(0, 0)]
    zero = replace(inp, rbc=np.zeros_like(rbc), zbs=np.zeros_like(zbs))
    columns = []
    for m, n in modes:
        h = np.cos(m * T - n * nfp * Zg)
        dr, dz = np.zeros_like(rbc), np.zeros_like(zbs)
        dr[keep_r], dz[keep] = pinv_r @ (h * gR).ravel(), pinv_z @ (h * gZ).ravel()
        columns.append(pack_boundary(replace(zero, rbc=dr, zbs=dz), max_mode, vary_major_radius=True))
    levels = np.asarray([max(abs(m), abs(n)) if (m, n) != (0, 0) else 1 for m, n in modes], dtype=float)
    return np.asarray(columns).T, modes, np.exp(-alpha * levels) / np.exp(-alpha)


def displacement_rms(h, modes) -> float:
    """RMS over the angles of ``sum h_mn cos(m theta - n nfp zeta)`` on ``modes`` (metres for metres).

    The normal displacement of the boundary for coefficients ``h`` on ``modes`` (``(m, n)`` pairs, as
    :attr:`ThreeTermsFreeBoundaryModel.boundary_modes`): every mode but ``(0, 0)`` has mean square ``h^2 / 2``.
    """
    h = np.asarray(h, dtype=float)
    weight = np.array([1.0 if (int(m), int(n)) == (0, 0) else 0.5 for m, n in modes])
    return float(np.sqrt(np.sum(weight * h**2)))


def newton_step_converged(step_rms: float, noise_rms: float, scale: float) -> bool:
    """Whether a Newton step of RMS size ``step_rms`` leaves nothing the rows can resolve: the free boundary's
    convergence test, shared by the Galerkin and least-squares solvers.

    Converged when ``step_rms <= max(noise_rms, sqrt(eps) * scale)``:

    - ``noise_rms`` is the step that the rows' measured noise alone produces (the difference of the rows of one
      boundary solved from two seeds, through the inverse Jacobian): a smaller step is not resolved by the rows;
    - ``sqrt(eps) * scale`` (``scale`` the coordinates' size, the major radius ``R0`` for a displacement) is
      round-off: Newton's error after a step ``p`` is ``O(|p|^2 / scale)``, so a step below ``sqrt(eps) scale``
      leaves an error at the coordinates' round-off ``eps * scale`` (the step tolerance of Dennis and Schnabel 1983, ch. 7).

    The step is in the coordinates' own units (metres of normal displacement: :func:`displacement_rms`), so the
    test does not depend on how the rows are scaled.
    """
    return bool(step_rms <= max(float(noise_rms), float(np.sqrt(np.finfo(float).eps)) * float(scale)))


def _newton_step(J, r):
    """The Newton step ``-J^-1 r`` of the square system, or ``None`` when ``J`` is numerically singular.

    Singular means no correct digit in the step: ``cond(J) * eps >= 1`` for ``J`` with its columns scaled to
    unit norm (the column scaling within ``sqrt(n)`` of the best one, van der Sluis 1969, so the coordinates'
    units do not decide it).
    """
    J, r = np.asarray(J, dtype=float), np.asarray(r, dtype=float)
    norms = np.linalg.norm(J, axis=0)
    if not np.all(np.isfinite(J)) or not np.all(norms > 0.0):
        return None
    if not np.linalg.cond(J / norms[None, :]) * np.finfo(float).eps < 1.0:
        return None
    return -np.linalg.solve(J, r)


#: Surface grid points per field period, in each direction, per harmonic of the deck's interior (``mpol``,
#: ``ntor``).  The rows integrate products of band-limited fields (edge field, geometry, test functions), which
#: needs more than 2 points per harmonic to be alias-free; 4 gives margin.  Measured on LP QA at 2.5 % beta
#: (``paper/runs/b23``): 4 instead of 6 points per harmonic move the free boundary by 0.41 um at 8 x 8 and 26 nm
#: at 12 x 12, with B.n unchanged: far below the boundary's dependence on the interior ``ns`` (~1 mm), and no
#: larger than the effect of the virtual-casing accuracy (``VC_DIGITS`` 4 -> 5: 0.4-1.0 um).  The reported pressure-balance RMS moves by
#: ~1 % at 8 x 8 because the report is integrated on the same grid; final numbers come from an independent,
#: converged evaluation, not from the solver's own report.
_GRID_POINTS_PER_HARMONIC = 4


def interface_grid(inp) -> tuple[int, int]:
    """``(nphi, ntheta)`` per field period of the interface rows of the deck ``inp``.

    ``_GRID_POINTS_PER_HARMONIC * max(mpol, ntor)`` in both directions (one rule for both: the edge field's
    harmonics follow the field lines, which mix the two).  The default of :class:`ThreeTermsFreeBoundaryModel`.
    """
    n = _GRID_POINTS_PER_HARMONIC * max(int(inp.mpol), int(inp.ntor), 1)
    return n, n


def interface_quadrature(nfp: int, nphi: int, ntheta: int) -> tuple[int, int]:
    """The virtual-casing singular quadrature ``(quad_nt, quad_np)`` for an ``nphi x ntheta`` field-period grid:
    ``(4 nfp nphi, 2 ntheta)``.  With ``quad_np = ntheta`` the tangential plasma field is off by ~3e-4 ``|B|`` on
    the LP QA 2.5 % beta boundary; planning the quadrature from its own error estimate instead can exhaust a GPU on
    a strongly shaped boundary.  The default of :class:`ThreeTermsFreeBoundaryModel`.
    """
    return 4 * int(nfp) * int(nphi), 2 * int(ntheta)


def solve_free_boundary_three_terms(
    inp: VmecInput,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_boundary: VmecInput | None = None,
    max_mode: int | None = None,
    nphi: int | None = None,
    ntheta: int | None = None,
    digits: int = vc.VC_DIGITS,
    max_nfev: int = 60,
    quadrature: tuple[int, int] | None = None,
    trial_ftol: float | None = None,
    chunk: int = 8,
    bootstrap=None,
    bootstrap_helicity: int = 0,
    bootstrap_weight: float = 100.0,
    previous=None,
    verbose: int = 0,
):
    """Free-boundary equilibrium of ``inp`` (the interface conditions of the module docstring).

    ``inp`` is a free-boundary deck (``PHIEDGE``, profiles, and the external
    field through ``external_field``, ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``).
    The free boundary is the deck's boundary displaced along the deck
    equilibrium's radial lines (:class:`ThreeTermsFreeBoundaryModel`): the deck is the fixed reference of
    the boundary's labelling, so the solution does not depend on where the
    solve starts; ``initial_boundary`` (another surface) only sets the starting
    displacement.  Every trial is a fixed-boundary equilibrium on the deck's
    ``NS_ARRAY`` ladder at fixed toroidal flux, pressure and current or iota.
    ``external_field`` is anything
    :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts (an
    :class:`~vmex.core.mgrid.MgridField`, a ``xyz -> B`` callable such as an
    ESSOS Biot-Savart field, or a coil field with ``b_cyl``).

    The equations (module docstring) are Newton-solved on an ``nphi x ntheta``
    grid over one field period (default :func:`interface_grid`), with the
    virtual-casing singular quadrature ``quadrature`` (default
    :func:`interface_quadrature`) and its singular patches planned once on
    the deck to ``digits``, to at most ``max_nfev`` evaluations.  ``previous``, an
    earlier result for the same deck (typically before a coil change), starts
    from its boundary, Jacobian and model: neither new coils nor a repeat
    solve compile anything.  ``trial_ftol`` and ``chunk`` are
    :class:`ThreeTermsFreeBoundaryModel`'s.  ``bootstrap`` (kinetic
    profiles), ``bootstrap_helicity`` and ``bootstrap_weight`` solve for a
    Redl-self-consistent current profile together with the boundary.

    Returns a :class:`scipy.optimize.OptimizeResult` with ``x`` (the boundary
    coordinates, then any bootstrap-current values), ``fun`` (the equations),
    ``jac`` (the last Jacobian), ``cost``, ``nfev``, ``njev``, ``success``
    (converged), ``input`` (the free boundary, and the current, as a
    fixed-boundary deck), ``equilibrium`` (solved to a final force residual of
    ``1e-17``, see ``_REPORT_FTOL``), ``model``, ``aux`` (``(state, mask,
    params, tight)``, for ``previous=``), ``boundary_residual`` and
    ``initial_boundary_residual`` (:class:`BoundaryResidual` of the reported
    equilibrium, and at the start) and ``normal_outside_modes``
    (:meth:`ThreeTermsFreeBoundaryModel.normal_outside_modes`).  A solve that
    does not lower ``B . n`` below the deck's warns (``RuntimeWarning``) with
    that out-of-band part.
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .implicit import input_with_params
    from .optimize import solve_equilibrium

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
        seed_state, seed_params = previous.aux[0], previous.aux[2]
    else:
        model = ThreeTermsFreeBoundaryModel(
            inp, max_mode=max_mode, nphi=nphi, ntheta=ntheta, digits=digits,
            quadrature=quadrature, trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap,
            bootstrap_helicity=bootstrap_helicity, bootstrap_weight=bootstrap_weight)
        x0 = None if initial_boundary is None else model.displacement_of(initial_boundary)
        jacobian = None
        seed_state, seed_params = model.seed[0], model.params0
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    # the warning's reference is the deck itself, also on a warm restart (whose start is ``previous``'s answer)
    deck = initial if previous is None else model.boundary_residual(model.seed[0], model.params0, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, jacobian=jacobian, max_nfev=max_nfev,
                               verbose=verbose)
    state, _, params, _ = out["aux"]
    boundary = input_with_params(model.fixed, params)
    # From the rows' own state on its own rung (a restart skips the deck's lower rungs; an ``initial_state`` would
    # seed the first rung and climb the ladder again: LP QA 2.5 %, 8 x 8, CURTOR x 0.99, that climb hit the
    # iteration cap at fsq 1e-4 after 545 s and was returned, QS 1.12e-4 against the rows' 1.71e-5), and an
    # equilibrium that does not reach the tolerance is an error, not a result.
    equilibrium = solve_equilibrium(
        boundary, restart_from=state, forward_ftol=min(_REPORT_FTOL, float(np.asarray(boundary.ftol_array)[-1])),
        forward_max_iterations=max(int(np.asarray(boundary.niter_array)[-1]), 200000), raise_on_max_iterations=True)
    final = model.boundary_residual(equilibrium.solution, params, external_field)
    outside = model.normal_outside_modes(equilibrium.solution, params, external_field)
    if final.normal >= deck.normal:
        import warnings

        warnings.warn(f"the free boundary did not lower B.n below the deck's ({final.normal:.2e} >= "
                      f"{deck.normal:.2e} RMS over |B|); {outside:.2e} of it lies outside the boundary's "
                      f"{len(model.boundary_modes)} modes, which the rows do not control", RuntimeWarning,
                      stacklevel=2)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=bool(out["converged"]), message=f"{out['accepted']} Newton steps",
        input=boundary, equilibrium=equilibrium, model=model, aux=out["aux"],
        boundary_residual=final, normal_outside_modes=outside, initial_boundary_residual=initial)


class ThreeTermsFreeBoundaryModel:
    """The free boundary of one deck for many external fields and plasma parameters.

    The fixed-boundary equilibria and their derivatives come from
    :mod:`vmex.core.implicit`, with the boundary, PHIEDGE and current profile
    as traced :class:`~vmex.core.implicit.ImplicitParams`, and the external
    field enters the compiled interface rows as an argument: neither new coils
    nor new plasma parameters recompile anything, which is what an optimizer
    that solves the free boundary at every step needs.

    Coordinates: ``x`` is the displacement ``h`` of the deck's boundary on
    its modes up to ``max_mode`` (``x0 = 0`` the deck; ``x_scale`` their ESS
    scales; :meth:`displacement_of` the coordinates of another surface),
    ``params`` an ``ImplicitParams`` (``params0`` the deck's) whose boundary
    ``with_boundary(params, x)`` replaces.  The rows are the square
    equations of the module docstring, with the quadrature plan and the field
    scale ``B0`` fixed on the deck (``seed``: its ``(state, mask)``).

    Methods: :meth:`solve_boundary` (the free boundary, cold or warm),
    :meth:`evaluate` (rows at a boundary), :meth:`linearize` (their Jacobian
    and the state responses), :meth:`push` (forward-mode columns of other
    functions of the equilibrium), :meth:`pullback` (their reverse-mode
    gradients), :meth:`boundary_residual`, :meth:`bootstrap_residual` and
    :meth:`solve`; ``rows(state, params, field)`` (and its interface part
    ``interface_rows``) are the traceable rows themselves.

    ``trial_ftol`` solves the trial equilibria of the boundary steps to a looser
    force residual (each is re-solved to the deck's before it is
    differentiated).  ``nphi x ntheta`` is the rows' grid over one field period (default
    :func:`interface_grid`); ``quadrature = (quad_nt, quad_np)`` is the virtual-casing singular quadrature
    (default :func:`interface_quadrature`; ``quad_nt`` a multiple of ``nfp * nphi``) and ``digits`` the accuracy
    its singular patches are planned to (default :data:`~vmex.core.virtual_casing.VC_DIGITS`).
    ``chunk`` bounds the Jacobian columns per batch.

    ``bootstrap`` (:class:`~vmex.core.bootstrap.KineticProfiles`, with
    ``bootstrap_helicity`` the quasisymmetry ``N`` of the Redl model) makes the
    current profile Redl's bootstrap current in the same solve: ``x`` gains its
    knot values (three inside the first surface, then one on every half-mesh
    surface) after the boundary coordinates (``n_boundary`` of them), and the
    rows gain one self-consistency row per knot, scaled by
    ``bootstrap_weight`` (see :class:`~vmex.core.bootstrap.HalfMeshCurrent`;
    ``ns <= 97``, a prescribed-current deck).  The current then follows the
    equilibrium instead of being an input, and :meth:`bootstrap_residual`
    reports how far it is from Redl's.  ``fixed_boundary=True`` (with
    ``bootstrap``) holds the boundary at that of ``params`` and solves only the
    current: ``x`` is the current values alone, the rows are the bootstrap ones
    and no external field or virtual casing enters -- the fixed-boundary
    single stage's equilibrium, with the same Jacobians and pullbacks.
    """

    def __init__(self, inp: VmecInput, *, max_mode=None, nphi=None, ntheta=None, digits=vc.VC_DIGITS, chunk=8,
                 device=None,
                 trial_ftol=None, quadrature=None, bootstrap=None, bootstrap_helicity=0, bootstrap_weight=100.0,
                 fixed_boundary=False):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent
        from .optimize import boundary_arrays_from_x, pack_boundary

        self._im = im
        self.fixed = fixed = replace(inp, lfreeb=False, mgrid_file="NONE")
        self.bootstrap = None if bootstrap is None else HalfMeshCurrent(fixed, bootstrap, bootstrap_helicity,
                                                                       bootstrap_weight)
        self.fixed_boundary = bool(fixed_boundary)

        # Vacuum: no pressure and no plasma current, so B_out is the coil field alone (no virtual casing).
        _, p_peak = _edge_pressure(fixed)
        self.vacuum = (bootstrap is None and p_peak == 0.0 and int(fixed.ncurr) == 1 and float(fixed.curtor) == 0.0
                       and (fixed.ac is None or not np.any(np.asarray(fixed.ac, dtype=float))))
        if self.fixed_boundary and self.bootstrap is None:
            raise ValueError("fixed_boundary=True solves only the bootstrap current: pass bootstrap=")
        if self.bootstrap is not None:
            self.fixed = fixed = self.bootstrap.deck
        # At finite beta the boundary keeps fewer modes than the interior (the fixed-boundary convention, mpol - 2
        # and ntor - 2): the interface conditions use the interior's field at the edge, least accurate in its top
        # modes, and a boundary that uses them fits that error (LP QA beta 2.5 %: interior/boundary 12/12 B.n
        # 4.1e-4, 16/12 1.6e-4) and gains a near-null direction there.  In vacuum the conditions need only the coils.
        top = max(fixed.mpol - 1, fixed.ntor)
        top = top if self.vacuum else max(top - _BOUNDARY_MODE_GAP, 1)
        self.max_mode = int(top if max_mode is None else max_mode)
        device = jax.devices()[0] if device is None else device
        # Hot-restarted solves are exact: the m=1 angle gauge is fixed (implicit.ImplicitConfig.freeze_m1), so
        # the state is a single-valued function of the boundary whatever the seed, and the implicit Jacobian is
        # the derivative of the re-solve map.
        # Certified means refined to |F| <= refine_tol: the rows and their implicit derivatives are taken at the
        # refined state, which must be the root (a refinement that stops short is a failed evaluation).
        cfg = im.make_config(fixed, multigrid=True, hot_restart=True, certify_refined=True)
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
        # Boundary coordinates: the normal displacement h of the deck's surface (see _normal_displacement_map):
        # one function for the one normal condition, no tangential (relabelling) directions among the unknowns.
        x_ref = pack_boundary(fixed, self.max_mode, vary_major_radius=True)
        self.R0 = float(np.asarray(fixed.rbc)[int(fixed.ntor), 0])  # the length scale of the convergence test
        if self.fixed_boundary:
            self.x0, self.x_scale, normal_map = x_ref[:0], x_ref[:0], None
        else:
            normal_map, self.boundary_modes, self.x_scale = _normal_displacement_map(fixed, self.max_mode)
            self.x0 = np.zeros(normal_map.shape[1])
        self.n_boundary = nb = self.x0.size
        if self.bootstrap is not None:
            self.x0 = np.r_[self.x0, self.bootstrap.x0]
            self.x_scale = np.r_[self.x_scale, self.bootstrap.x_scale]
        self.chunk = int(chunk)
        self._linearization = None
        self._last_linearize = None  # (key, extra key, result, _linearization) of the latest linearize
        self._x_ref = jnp.asarray(x_ref)
        self._normal_map = None if normal_map is None else jnp.asarray(normal_map)

        def with_boundary(params, x):
            if not self.fixed_boundary:
                rbc, zbs = boundary_arrays_from_x(fixed, self._x_ref + self._normal_map @ x[:nb], self.max_mode,
                                                  vary_major_radius=True)
                params = replace(params, rbc=rbc, zbs=zbs)
            return params if self.bootstrap is None else self.bootstrap.apply(params, x[nb:])

        self.with_boundary = with_boundary
        seed = self.solve(self.params0)
        if seed is None:
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium")
        state, mask = seed[:2]
        self.seed = (state, mask)
        if not self.fixed_boundary:
            # The boundary moves along the deck equilibrium's radial coordinate lines dX/ds at the edge (VMEC's
            # nested surfaces share their theta labelling along them), not along the surface normal: the labelling
            # convention of the deck is carried to the free boundary without sliding (known-target benchmark,
            # deck 11 mm away: 8 x 8 field lines 1.23 -> 0.36 mm, K 1.9e-3 -> 3.8e-4).
            sp = vc._state_field_spectra(fixed, state, im.runtime_from_params(self.params0, self.cfg))
            xm, xn = np.asarray(sp["xm"]), np.asarray(sp["xn"])
            ds = 1.0 / (int(sp["ns"]) - 1)
            dr = (np.asarray(sp["rmnc"])[-1] - np.asarray(sp["rmnc"])[-2]) / ds
            dz = (np.asarray(sp["zmns"])[-1] - np.asarray(sp["zmns"])[-2]) / ds

            def radial(T, Zg):
                a = xm[:, None, None] * T[None] - xn[:, None, None] * Zg[None]
                return np.tensordot(dr, np.cos(a), axes=1), np.tensordot(dz, np.sin(a), axes=1)

            normal_map, _, _ = _normal_displacement_map(fixed, self.max_mode, direction=radial)
            self._normal_map = jnp.asarray(normal_map)
        if not self.fixed_boundary:
            grid = interface_grid(fixed)
            self.nphi, self.ntheta = int(nphi or grid[0]), int(ntheta or grid[1])
            self.quadrature = tuple(map(int, quadrature or interface_quadrature(fixed.nfp, self.nphi, self.ntheta)))
            self._interface_setup(state, self.nphi, self.ntheta, digits, self.quadrature)
        self._implicit_setup()

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature):
        """The interface rows, with the quadrature plan and the field scale ``B0`` of the seed ``state``."""
        from .freeboundary import _vacuum_scalars
        from .optimize import solve_equilibrium

        im, cfg, fixed = self._im, self.cfg, self.fixed
        nfp = int(fixed.nfp)
        runtime = im.runtime_from_params(self.params0, cfg)
        surface = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        quad_nt, quad_np = map(int, quadrature)
        if self.vacuum:  # the coils' field alone: no virtual-casing quadrature to plan
            precision, digits = None, None
        else:
            precision = vc.plan_vc_precision(surface, digits=digits, quad_nt=quad_nt, quad_np=quad_np)
        self.precision = precision
        # A closed curve inside the coils for G_coil: the seed's magnetic axis.
        axis = _axis_loop(solve_equilibrium(fixed, initial_state=state).wout)
        vacuum = self.vacuum
        area = np.linalg.norm(np.asarray(surface.area_vector), axis=0)
        B0 = float(np.sqrt(np.sum(area * np.sum(np.asarray(surface.B_total) ** 2, axis=0)) / np.sum(area)))
        self.B0 = B0
        # Surface data are (nphi, ntheta) over one field period.
        phi, theta = np.meshgrid(np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False),
                                 np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False), indexing="ij")
        # The boundary arrays' entries as series in (m, n nfp).
        ntor_b, mpol_b = np.shape(self.params0.rbc)[0] // 2, np.shape(self.params0.rbc)[1]
        modes = np.meshgrid(np.arange(-ntor_b, ntor_b + 1) * nfp, np.arange(mpol_b), indexing="ij")
        xm, xn = jnp.asarray(modes[1].ravel(), dtype=float), jnp.asarray(modes[0].ravel(), dtype=float)
        # R0 d/dh: B.grad = (B/R)(d_phi + iota d_theta), so R0 times the h-gradient of the mean square of
        # B.n/B0 is a dimensionless moment of B.n/B0 (as are the finite-beta rows, with the same factor).
        length = float(np.asarray(fixed.rbc)[int(fixed.ntor), 0])
        phi_grid = jnp.asarray(phi[:, 0])
        m_b = jnp.asarray([m for m, _ in self.boundary_modes[:-1]], dtype=float)[:, None, None]
        n_b = jnp.asarray([n * nfp for _, n in self.boundary_modes[:-1]], dtype=float)[:, None, None]
        test_sine = jnp.sin(m_b * theta[None] - n_b * phi[None])  # boundary modes but (0, 0): B.n is odd
        if vacuum:
            # The boundary arrays are linear in x: their derivatives along x.
            boundary = jax.jacfwd(lambda x: self.with_boundary(self.params0, x))(jnp.asarray(self.x0))
            to_x = (boundary.rbc.reshape(-1, self.x0.size), boundary.zbs.reshape(-1, self.x0.size))

        def interface_rows(state, params, field, moments=False):
            """The interface rows of the module docstring at ``state`` (solved on ``params``' boundary).

            ``moments=True`` (finite beta): plain sine moments of ``B_out . n`` instead of the transported ones,
            the robust start of :meth:`solve_boundary` (no field in the test functions).
            """
            runtime = im.runtime_from_params(params, cfg)
            G_coil = jnp.abs(_net_current(field, *axis))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            current = ((G - G_coil) / G_coil)[None]
            if vacuum:  # no plasma current: the field outside is the coils' own, an explicit function of h
                gradient = jax.grad(lambda r, z: _mean_square_normal_field(
                    r.ravel(), z.ravel(), xm, xn, phi, theta, phi_grid, field), argnums=(0, 1))(params.rbc, params.zbs)
                rows = to_x[0].T @ gradient[0].ravel() + to_x[1].T @ gradient[1].ravel()
                return jnp.concatenate([length * rows[:-1] / (2.0 * B0**2), current])
            data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
            area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
            weight = (area / jnp.sum(area)).ravel()
            B_out = vc.PlasmaVacuumInterface.from_surface_data(
                data, p_edge=0.0, digits=digits, precision=precision).total_B_out(field)
            Bn = (jnp.sum(B_out * jnp.asarray(data.normal), axis=0) / B0).ravel()
            if moments:
                return jnp.concatenate([2.0 * test_sine.reshape(test_sine.shape[0], -1) @ (weight * Bn), current])
            # test functions B.grad(cos(m theta - n nfp phi)) R0 / B0: a normal displacement h changes B . N by
            # -|N| B_t . grad h (the surface divergence of h B_t), the plasma's own response left out
            _, e_theta, e_phi = _surface_frame(jnp.ravel(params.rbc), jnp.ravel(params.zbs), xm, xn, phi, theta)
            N = jnp.cross(e_theta, e_phi, axis=0)
            N2 = jnp.sum(N * N, axis=0)
            b_theta = jnp.sum(jnp.cross(B_out, e_phi, axis=0) * N, axis=0) / N2
            b_phi = jnp.sum(jnp.cross(e_theta, B_out, axis=0) * N, axis=0) / N2
            tests = (n_b * b_phi[None] - m_b * b_theta[None]) * test_sine * (length / B0)
            return jnp.concatenate([tests.reshape(tests.shape[0], -1) @ (weight * Bn), current])

        self.interface_rows = interface_rows
        self._terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field, runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])

    def _implicit_setup(self):
        """The rows (interface, then bootstrap) and their implicit tangents, pushes and pullbacks."""
        im, cfg = self._im, self.cfg

        def rows(state, params, field, moments=False):
            parts = [] if self.fixed_boundary else [self.interface_rows(state, params, field, moments)]
            if self.bootstrap is not None:
                parts.append(self.bootstrap.rows(state, im.runtime_from_params(params, cfg), params))
            return jnp.concatenate(parts)

        self.rows = rows
        self._rows = jax.jit(rows, static_argnames="moments")
        self._moments = False  # solve_boundary's robust start (finite beta) uses the plain moments
        active = im._active_state_fields(cfg)
        edge = im._edge_mask(cfg)

        def tangents(params, state, mask, batch):
            """Projected state responses to the stacked parameter tangents: one block factorization, chunked.

            Staged programs (:func:`~vmex.core.implicit.staged_tangent_multi_rhs`): the factors from the
            forward refinement's compiled program, then the block solve and the GMRES corrector."""
            return im.staged_tangent_multi_rhs(
                params, cfg, state, mask, batch, chunk=self.chunk,
                certify_rtol=cfg.jacobian_adjoint_tol, certify_maxiter=cfg.jacobian_adjoint_maxiter)[0]

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

        self._tangents = tangents
        self.push = push
        push_rows = jax.jit(lambda state, mask, params, dz, pb, field, moments=False: push(
            lambda s, p: rows(s, p, field, moments), state, mask, params, dz, pb), static_argnames="moments")

        def push_rows_padded(state, mask, params, dz, pb, field, moments=False):
            n = jax.tree.leaves(dz)[0].shape[0]  # padded to whole chunks: no tail-width copy of the program
            chunk = min(self.chunk, n)
            return push_rows(state, mask, params, im._padded(dz, chunk), im._padded(pb, chunk), field,
                             moments=moments)[:n]

        self._push_rows = push_rows_padded
        # First-order trial state from the last linearization (upstream's perturbation warm start): it keeps
        # the trial on the path the Jacobian was taken along, instead of a hot restart that may drift.
        self._predict = jax.jit(predict)

        def pullback(fun, state, mask, params, cotangents, *args, factors=None):
            """``ImplicitParams`` gradients of ``cotangents[i] . fun(state, params, *args)`` through the equilibrium.

            Reverse mode with one block factorization shared by every row (upstream's implicit pullback): for a few
            scalar rows of an expensive function this costs a few reverse passes, not one forward pass per
            parameter direction.  ``fun`` may be a tuple of functions with ``cotangents`` a tuple of their rows'
            cotangents: each row then runs only its own function's reverse pass, and the rows come out in order.
            """
            frozen = jax.lax.stop_gradient(state)
            back = im._block_state_pullback(params, cfg, frozen, mask, active_fields=active,
                                            probe_chunk_size=self.chunk, factors=factors)
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

    def displacement_of(self, boundary, ntheta=96, nzeta=48):
        """Coordinates ``x`` (normal displacement of the reference surface) closest to ``boundary``'s surface.

        Each reference point is moved along its normal to the nearest point of the other surface (nearest-point
        iteration on a dense sampling), and the displacements are fitted on the normal modes.
        """
        from .optimize import boundary_arrays_from_x

        nfp = int(self.fixed.nfp)

        def points(rbc, zbs, nt, nz, full=False):
            rbc, zbs = np.asarray(rbc, dtype=float), np.asarray(zbs, dtype=float)
            ntor = (rbc.shape[0] - 1) // 2
            ms, ns = np.arange(rbc.shape[1]), np.arange(-ntor, ntor + 1)
            th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
            ze = np.linspace(0.0, 2.0 * np.pi / (1 if full else nfp), nz, endpoint=False)
            T, P = np.meshgrid(th, ze, indexing="ij")
            ph = ms[None, :, None, None] * T[None, None] - ns[:, None, None, None] * nfp * P[None, None]
            R = np.einsum("nm,nmij->ij", rbc, np.cos(ph)); Z = np.einsum("nm,nmij->ij", zbs, np.sin(ph))
            Rt = np.einsum("nm,nmij->ij", -rbc * ms, np.sin(ph)); Zt = np.einsum("nm,nmij->ij", zbs * ms, np.cos(ph))
            Rp = np.einsum("nm,nmij->ij", rbc * ns[:, None] * nfp, np.sin(ph))
            Zp = np.einsum("nm,nmij->ij", -zbs * ns[:, None] * nfp, np.cos(ph))
            X = np.stack([R * np.cos(P), R * np.sin(P), Z], -1)
            et = np.stack([Rt * np.cos(P), Rt * np.sin(P), Zt], -1)
            ep = np.stack([Rp * np.cos(P) - R * np.sin(P), Rp * np.sin(P) + R * np.cos(P), Zp], -1)
            N = np.cross(et, ep)
            return T, P, X, N / np.linalg.norm(N, axis=-1, keepdims=True)

        from scipy.spatial import cKDTree
        r0, z0 = boundary_arrays_from_x(self.fixed, self._x_ref, self.max_mode, vary_major_radius=True)
        T, P, X, n = points(r0, z0, ntheta, nzeta)
        _, _, Y, _ = points(boundary.rbc, boundary.zbs, 4 * ntheta, 4 * nzeta * nfp, full=True)
        tree = cKDTree(Y.reshape(-1, 3))
        t = np.zeros(T.shape)
        for _ in range(4):  # the point of the other surface on the normal line, by nearest-point iteration
            _, k = tree.query((X + t[..., None] * n).reshape(-1, 3))
            t = np.sum((Y.reshape(-1, 3)[k].reshape(X.shape) - X) * n, axis=-1)
        basis = np.stack([np.cos(m * T - k * nfp * P).ravel() for m, k in self.boundary_modes], axis=1)
        h = np.linalg.lstsq(basis, t.ravel(), rcond=None)[0]
        return np.r_[h, np.zeros(self.n_boundary - h.size), self.x0[self.n_boundary:]]

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
            with im._device_context(cfg):
                try:
                    state, mask = im._host_solve_and_mask_impl(cfg, params_np, refine=False)
                except (VmecError, RuntimeError):  # RuntimeError: the relayed sentinel of a VmecError
                    return None
            result = im._LAST_SOLVE[cfg][1]
            fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
            if not (bool(result.converged) or fsq / cfg.ftol <= cfg.max_fsq_ratio):
                return None
        return jax.device_put(jax.tree.map(jnp.asarray, (state, mask)), cfg.device)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`BoundaryResidual` of a solved state."""
        return summarize_boundary_residual(*self._terms(state, params, field))

    def normal_outside_modes(self, state, params, field) -> float:
        """RMS (over ``|B|``, area-weighted) of the part of ``B_out . n`` outside the boundary's modes.

        The rows control ``B . n`` through the boundary's modes; what lies outside them (higher poloidal or
        toroidal harmonics) only resolution lowers.  Harmonics of the grid's FFT, in field periods.
        """
        rows, weights = map(np.asarray, self._terms(state, params, field))
        f = rows[0] * np.sqrt(weights * weights.size)  # mean of f^2 = the weighted mean square of B.n
        power = np.abs(np.fft.fft2(f) / f.size) ** 2  # axes (phi, theta)
        k_phi = np.fft.fftfreq(f.shape[0], 1.0 / f.shape[0]).astype(int)
        k_theta = np.fft.fftfreq(f.shape[1], 1.0 / f.shape[1]).astype(int)
        inside = np.zeros(power.shape, dtype=bool)
        for m, n in self.boundary_modes:  # sin(m theta - n nfp phi) holds (-n, m) and (n, -m)
            inside |= (k_phi[:, None] == -n) & (k_theta[None, :] == m)
            inside |= (k_phi[:, None] == n) & (k_theta[None, :] == -m)
        return float(np.sqrt(max(power[~inside].sum(), 0.0)))

    def bootstrap_residual(self, state, params) -> float:
        """``max_j |I'(s_j) - I'_Redl(s_j)| / max_j |I'_Redl(s_j)|`` of a solved state (with ``bootstrap``)."""
        difference, target = map(np.asarray, self.bootstrap.mismatch(
            state, self._im.runtime_from_params(params, self.cfg), params))
        return float(np.max(np.abs(difference)) / np.max(np.abs(target)))

    def evaluate(self, x, params, field, seed=None, tight=False):
        """Rows at boundary ``x``, or ``None`` if the solve failed.

        Returns ``(rows, (state, mask, params_x, tight))``; ``tight`` marks a state solved to the deck's
        tolerance rather than the trial one.
        """
        params_x = self.with_boundary(params, jnp.asarray(x))
        if seed is None and self._linearization is not None:
            x_ref, state_ref, mask_ref, dz = self._linearization
            seed = self._predict(state_ref, mask_ref, dz, jnp.asarray(np.asarray(x) - x_ref), params_x)
        tight = tight or self.cfg_trial is self.cfg
        solved = self.solve(params_x, seed, tight)
        if solved is None:
            return None
        state, mask = solved
        return np.asarray(self._rows(state, params_x, field, moments=self._moments)), (state, mask, params_x, tight)

    def boundary_directions(self, params, x):
        """``ImplicitParams`` tangents of the boundary coordinates, stacked."""
        eye = jnp.eye(np.size(x))
        return jax.vmap(lambda e: jax.jvp(lambda xx: self.with_boundary(params, xx), (jnp.asarray(x),), (e,))[1])(eye)

    def linearize(self, x, aux, field, extra=None):
        """Columns of the rows along the model coordinates and the ``extra`` parameter tangents.

        The model coordinates are the boundary, then the bootstrap current.  A trial state is first solved to
        the deck's tolerance (from itself).  Returns ``(J, dz, params_batch, aux)``: the Jacobian, the projected
        state responses and parameter tangents of its columns, and the (tight) state they belong to.  The
        responses also seed the next trials (first-order predicted states).  A call at the boundary, plasma
        parameters and field of the latest one returns its result (without ``extra``'s columns when not asked
        for): a polish that stalls asks again for the Jacobian it was given.
        """
        state, mask, params_x, tight = aux
        key = _fingerprint(x, params_x, field) + ("moments" if self._moments else "")
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
        dz_all = self._tangents(params_x, state, mask, jax.tree.map(join, *groups))
        ends = np.cumsum([jax.tree.leaves(batch)[0].shape[0] for batch in groups])
        parts = [(jax.tree.map(lambda a: a[lo:hi], dz_all), batch)
                 for lo, hi, batch in zip(np.r_[0, ends[:-1]], ends, groups)]
        columns = np.hstack([np.asarray(self._push_rows(state, mask, params_x, dz, batch, field,
                                                        moments=self._moments)).T
                             for dz, batch in parts])
        dz = jax.tree.map(join, *[dz for dz, _ in parts[:n_coordinates]])
        self._linearization = (np.asarray(x, dtype=float).copy(), state, mask, dz)
        if extra is not None:
            dz = jax.tree.map(join, dz, parts[-1][0])
        batch = jax.tree.map(join, *[batch for _, batch in parts])
        self._last_linearize = (key, extra_key, (columns, dz, batch, aux), self._linearization)
        return columns, dz, batch, aux

    def rows_at(self, aux, field):
        """The rows of a solved ``aux = (state, mask, params, tight)`` (no solve)."""
        state, _, params_x, _ = aux
        return np.asarray(self._rows(state, params_x, field, moments=self._moments))

    def repeat(self, x, params, field, aux):
        """The rows at ``x`` solved again, to the deck's tolerance, from a second seed; ``None`` if that fails.

        The difference from the rows of ``aux`` (the same boundary, solved from another start) is the rows'
        repeatability, the noise below which Newton cannot converge (:func:`newton_step_converged`).  The second
        seed is the deck's equilibrium, or a cold start when ``x`` is the deck's boundary itself; the solver's
        one-entry memos are cleared so that the boundary is really solved again.
        """
        im, cfg = self._im, self.cfg
        params_x = self.with_boundary(params, jnp.asarray(x))
        at_deck = not np.any(np.asarray(x)[:self.n_boundary] - self.x0[:self.n_boundary])
        for memo in (im._LAST_SOLVE, im._LAST_REFINED, im._LAST_REFINEMENT_CORRECTION) + (
                (im._HOT_CACHE,) if at_deck else ()):
            memo.pop(cfg, None)
        solved = self.solve(params_x, None if at_deck else self.seed[0], tight=True)
        return None if solved is None else self.rows_at((*solved, params_x, True), field)

    def _step_converged(self, step, noise_step, x):
        """:func:`newton_step_converged` for the boundary part of ``step`` (metres, against ``R0``) and for the
        bootstrap-current part (against the current values ``x``); ``noise_step`` is the step of the measured
        row noise, or ``None`` before it is measured."""
        nb = self.n_boundary
        noise = np.zeros_like(step) if noise_step is None else noise_step
        ok = nb == 0 or newton_step_converged(displacement_rms(step[:nb], self.boundary_modes),
                                              displacement_rms(noise[:nb], self.boundary_modes), self.R0)
        if step.size > nb:  # the bootstrap current: plain RMS of its knot values
            rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
            ok = ok and newton_step_converged(rms(step[nb:]), rms(noise[nb:]), rms(x[nb:]))
        return ok

    def _newton_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0):
        """The boundary where the square rows vanish: Newton steps with backtracking.

        A given ``jacobian`` (of a nearby point) is reused while it reduces the residual ten-fold a step;
        otherwise, and whenever a step fails, the Jacobian is refreshed at the current point.  Converged when the
        Newton step at a tight evaluation is within :func:`newton_step_converged`'s tolerance: round-off, or the
        step that the rows' measured noise (:meth:`repeat`, measured once a fresh Newton step finds no descent)
        produces.  A singular Jacobian, or a fresh step with no descent that is larger than that, is a stall
        (``converged=False``): a minimum of ``|rows|^2`` that is no root is never reported as converged.  Trial
        points solved to ``trial_ftol`` are re-solved to the deck's tolerance before convergence is tested.
        """
        x = np.asarray(self.x0 if x0 is None else x0, dtype=float)
        got = self.evaluate(x, params, field)
        for fraction in (0.75, 0.5, 0.25, 0.0):  # a far start may have no equilibrium: back off toward the deck
            if got is not None or not np.any(x):  # (this only moves the start, never the answer)
                break
            x = fraction * x
            got = self.evaluate(x, params, field)
        if got is None:
            raise RuntimeError("the starting boundary has no certified equilibrium")
        r, aux = got
        J, fresh, slow, nfev, njev, steps, started = jacobian, False, False, 1, 0, 0, time.perf_counter()
        # the rows' measured noise (a vector of row differences), kept by the model for later solves of its rows
        noises = self.__dict__.setdefault("_row_noise", {})
        converged, tight, noise = False, self.cfg_trial is self.cfg, noises.get(self._moments)
        while nfev < max_nfev:
            if J is None:
                J, _, _, aux = self.linearize(x, aux, field)
                J, fresh, njev = np.asarray(J), True, njev + 1
                if not got[1][3] and aux[3]:  # the rows of the tight state the Jacobian belongs to
                    r, got = self.rows_at(aux, field), (None, aux)
            p = _newton_step(J, r)
            if p is None:  # singular Jacobian: no Newton step
                if not fresh:
                    J = None
                    continue
                if verbose:
                    print("Newton stalled: singular Jacobian", flush=True)
                break
            # the step of a Jacobian at this point or a nearby one (the previous point, ``previous=``, the
            # single-stage anchor) is the distance to the root to first order
            if self._step_converged(p, None if noise is None else _newton_step(J, noise), x):
                if got[1][3]:  # converged: the last step leaves an error O(|p|^2), kept if it does not raise the rows
                    converged, final = True, self.evaluate(x + p, params, field, tight=True)
                    nfev += 1
                    if final is not None and final[0] @ final[0] <= r @ r:
                        x, (r, aux), steps = x + p, final, steps + 1
                    break
                tight, got = True, self.evaluate(x, params, field, tight=True)  # converge on tight rows only
                nfev += 1
                if got is None:
                    break
                r, aux = got
                continue
            if slow:  # the last step was slow: test convergence with its Jacobian, then refresh it
                J, slow = None, False
                continue
            cost = 0.5 * float(r @ r)
            t, accepted = 1.0, None
            while t >= (1.0 / 64 if fresh else 1.0) and nfev < max_nfev:  # a reused J: its full step, or refresh
                trial = self.evaluate(x + t * p, params, field, tight=tight)
                nfev += 1
                if trial is not None and 0.5 * float(trial[0] @ trial[0]) < (1.0 - 1e-4 * t) * cost:
                    accepted = trial
                    break
                t *= 0.25
            if verbose:
                new = np.inf if accepted is None else 0.5 * float(accepted[0] @ accepted[0])
                print(f"Newton step {steps + 1}{' (fresh J)' if fresh else ''}: t {t:.3g}, cost {cost:.6e} -> "
                      f"{new:.6e} ({time.perf_counter() - started:.1f} s)", flush=True)
            if accepted is None:
                if fresh and not tight:  # the trials' looser force tolerance is the rows' noise: solve them tight
                    tight, J, got = True, None, self.evaluate(x, params, field, tight=True)
                    nfev += 1
                    if got is None:  # no certified equilibrium at the deck's tolerance
                        break
                    r, aux = got
                    continue
                if fresh:  # no descent along a fresh Newton step: the rows' noise, or a stall
                    if got[1][3] and nfev < max_nfev:
                        again = self.repeat(x, params, field, aux)
                        nfev += 1
                        if again is not None:
                            noise = again - r if noise is None or (np.linalg.norm(again - r)
                                                                   > np.linalg.norm(noise)) else noise
                            noises[self._moments] = noise
                            converged = self._step_converged(p, _newton_step(J, noise), x)
                    if not converged and verbose:
                        print(f"Newton stalled: cost {cost:.6e}, step not within the rows' noise", flush=True)
                    break
                J = None
                continue
            new_cost = 0.5 * float(accepted[0] @ accepted[0])
            x, got, steps = x + t * p, accepted, steps + 1
            r, aux = got
            slow = new_cost > 1e-2 * cost  # slower than a 10x residual reduction a step (a chord method)
            fresh = False  # the Jacobian (if kept) is now of the previous point
        return dict(x=x, rows=np.asarray(r), jacobian=J, aux=aux, nfev=nfev, njev=njev, accepted=steps,
                    converged=converged, noise=None if noise is None else float(np.linalg.norm(noise)))

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0):
        """The free boundary for ``params`` and ``field``, from ``x0`` (default: the reference surface).

        See :meth:`_newton_boundary`; returns a dict with ``x``, ``rows``, ``jacobian``, ``aux`` (state, mask,
        params, tight), ``nfev``, ``njev``, ``accepted`` (steps) and ``converged``.
        """
        out = self._newton_boundary(params, field, x0=x0, jacobian=jacobian, max_nfev=max_nfev, verbose=verbose)
        if out["converged"] or self.vacuum or self.fixed_boundary:
            return out
        # The transported rows carry the field in their test functions, and far from the answer Newton on them
        # can stall; the plain moments (the same boundary modes) converge from farther: solve them first, then
        # the transported rows from there.
        if verbose:
            print("Newton on the transported rows stalled: plain moments first", flush=True)
        self._moments = True
        try:
            start = self._newton_boundary(params, field, x0=x0, max_nfev=max_nfev, verbose=verbose)
        finally:
            self._moments = False
        final = self._newton_boundary(params, field, x0=start["x"], max_nfev=max_nfev, verbose=verbose)
        for key in ("nfev", "njev"):
            final[key] += out[key] + start[key]
        return final


#: Boundary modes fewer than the interior's at finite beta (see ThreeTermsFreeBoundaryModel).
_BOUNDARY_MODE_GAP = 2

#: Force residual of the reported equilibrium's final rung: plain descent stopped at the usual 1e-13 leaves ~1 % in
#: QS ~ 1e-6 (the boundary solve's own states are Newton-refined and need no such tolerance).
_REPORT_FTOL = 1e-17



@dataclass(frozen=True)
class _AcceptedPoint:
    """The design the optimizer last accepted (``ThreeTermsFreeBoundaryProblem.accepted``)."""

    parameters: np.ndarray


class ThreeTermsFreeBoundaryProblem(FunctionProblem):
    """A scalar loss and constraint rows of a design ``x`` at its three-terms free boundary, for SLSQP.

    Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_terms")``
    (:meth:`~vmex.core.freeboundary_problem.FreeBoundaryProblem.from_loss`); pass it to
    :func:`vmex.core.optimize.minimize`.  ``loss``, ``quantities``, ``parameter_quantities``,
    ``field_from_parameters`` and ``plasma_from_parameters`` are as there.  Every trial is the K = 0 boundary
    of the field and plasma parameters of ``x`` (:class:`ThreeTermsFreeBoundaryModel`), warm-started from the
    first-order prediction of the latest linearized solution; a trial without a certified solution, or whose
    boundary fit did not converge from any start (a stalled Newton solve is no free boundary, whatever its
    residual), raises :class:`~vmex.core.errors.TrialRejected`.  A row ``h`` (loss or quantity) has the design
    gradient of the implicit function theorem of the (square) boundary equations,

        dh/dx = dh/dx|_b - lambda^T dr/dx,   lambda = J_b^-T dh/db,

    with ``r`` the interface rows, ``J_b`` their Jacobian in the model's coordinates ``b`` (boundary, and the
    bootstrap current with ``bootstrap``) and ``dr/dx`` through the plasma parameters and the field; every
    row's ``dh/db`` and plasma term come from one reverse pass through the equilibrium.  Nothing recompiles
    when ``x`` changes.  ``model`` is the :class:`ThreeTermsFreeBoundaryModel`; :meth:`boundary_residual` and
    ``stats`` (counts of trials, failures, boundary evaluations and linearizations) report progress.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_max_nfev=100, **model_options):
        """Solve the seed design's free boundary.

        ``model_options`` are :class:`ThreeTermsFreeBoundaryModel` keywords (``nphi``, ``ntheta``,
        ``quadrature``, ``chunk``, ``trial_ftol``, ``bootstrap``, ...).  A trial's boundary is Newton-solved from
        the accepted point's first-order prediction with its Jacobian (refreshed when it no longer converges
        fast), to at most ``boundary_max_nfev`` evaluations; a fit that does not converge within them is no free
        boundary: the trial tries its next start, or fails.
        """
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if plasma_from_parameters is not None:
            functions += (plasma_from_parameters,)
        if not all(map(callable, functions)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = ThreeTermsFreeBoundaryModel(inp, **model_options)
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_max_nfev=boundary_max_nfev)

    def __init__(self, model, loss, x0, *, field_from_parameters, plasma_from_parameters, scales, names, quantities,
                 parameter_quantities, boundary_max_nfev):
        from .implicit import runtime_from_params

        self.model, self.field = model, field_from_parameters
        self.boundary_max_nfev = int(boundary_max_nfev)
        x0 = np.asarray(x0, dtype=float)
        params_at = (lambda params, x: params) if plasma_from_parameters is None else plasma_from_parameters

        def seed_params_at(x):
            return params_at(model.params0, x)

        self._params_at = jax.jit(seed_params_at)

        def part(f, with_x):
            def value(state, params, x):
                runtime = runtime_from_params(params, model.cfg)
                return jnp.ravel(jnp.asarray(f(state, runtime, x) if with_x else f(state, runtime)))
            return value

        parts = (part(loss, True), *(part(f, False) for f in quantities),
                 *(part(f, True) for f in parameter_quantities))

        def rows(state, params, x):
            return jnp.concatenate([f(state, params, x) for f in parts])

        self._rows = jax.jit(rows)
        # The coordinates of x that move plasma parameters: their columns of the interface rows enter the gradient.
        tangent = jax.jacfwd(seed_params_at)(jnp.asarray(x0))
        moved = sum(np.any(np.asarray(a).reshape(-1, x0.size) != 0, axis=0) for a in jax.tree.leaves(tangent))
        self._plasma = np.flatnonzero(moved)
        eye = jnp.eye(x0.size)[self._plasma]
        self._plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
            lambda z: params_at(params, z), (x,), (e,))[1])(eye))

        def pullback(state, mask, params, x, factors):
            """Every row's ``ImplicitParams`` gradient, each through its own function's reverse pass only.

            ``factors``: the block factors at ``state`` (:func:`~vmex.core.implicit.block_factors_at`, an
            executable compiled once per config), not a second block assembly inside this program."""
            sizes = [jax.eval_shape(f, state, params, x).size for f in parts]
            return model.pullback(parts, state, mask, params, tuple(jnp.eye(k) for k in sizes), x, factors=factors)

        self._pullback = jax.jit(pullback)

        @jax.jit
        def coordinates(params, b, x, params_bar):
            """``ImplicitParams`` cotangents onto the model coordinates ``b`` and onto ``x`` (plasma parameters)."""
            _, b_vjp = jax.vjp(lambda z: model.with_boundary(params, z), b)
            _, x_vjp = jax.vjp(lambda z: params_at(params, z), x)
            return jax.vmap(lambda bar: (b_vjp(bar)[0], x_vjp(bar)[0]))(params_bar)

        self._coordinates = coordinates
        self._direct = jax.jit(jax.jacrev(rows, argnums=2))
        self._field_vjp = jax.jit(lambda state, params, x, lam: jax.vjp(
            lambda z: model.rows(state, params, field_from_parameters(z)), x)[1](lam)[0])
        self._field_jvp = jax.jit(lambda state, params, x, dx: jax.jvp(
            lambda z: model.rows(state, params, field_from_parameters(z)), (x,), (dx,))[1])
        self._cache, self._anchor = {}, {}
        self.stats = dict(trials=0, failed=0, boundary_evaluations=0, linearizations=0)
        if self._solve(x0) is None:
            raise VmecError("the seed design has no three-terms free boundary")
        self.accepted, self.accepted_step = _AcceptedPoint(x0.copy()), 0
        super().__init__(x0, names=names, scales=scales, fun=self._value, value_and_grad=self._value_gradient,
                         metadata={"holder": {"failed_trials": 0}})

    def _solve(self, x):
        """The trial at ``x`` (cached): boundary fit, state and rows, or ``None``."""
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key in self._cache:
            return self._cache[key]
        self.stats["trials"] += 1
        params, field = self._params_at(jnp.asarray(x)), self.field(jnp.asarray(x))
        anchor, starts, J = self._anchor, [None], None
        if anchor:  # the anchor's first-order boundary prediction, then the anchor's own boundary
            dx = x - anchor["x"]
            dr = anchor["J_p"] @ dx[self._plasma] + np.asarray(self._field_jvp(
                anchor["state"], anchor["params"], jnp.asarray(anchor["x"]), jnp.asarray(dx)))
            starts, J = [anchor["b"] - np.linalg.lstsq(anchor["J_b"], dr, rcond=None)[0], anchor["b"]], anchor["J_b"]
        fit = None
        for start in starts:
            try:
                fit = self.model.solve_boundary(params, field, x0=start, jacobian=J, max_nfev=self.boundary_max_nfev)
            except (VmecError, RuntimeError):  # an uncertified equilibrium: try the next start
                continue
            if not fit["converged"]:  # a stalled fit is no free boundary (its rows are not zero): the next start
                self.stats["boundary_evaluations"] += fit["nfev"]
                fit = None
                continue
            break
        if fit is None:
            self.stats["failed"] += 1
            self._cache[key] = None
            return None
        self.stats["boundary_evaluations"] += fit["nfev"]
        state, mask, params_b, _ = fit["aux"]
        self._cache[key] = dict(fit, state=state, mask=mask, params=params_b,
                                values=np.asarray(self._rows(state, params_b, jnp.asarray(x))))
        return self._cache[key]

    def _record(self, x):
        sol = self._solve(self._x(x))
        if sol is None:
            self.metadata["holder"]["failed_trials"] += 1
            raise TrialRejected("no three-terms free boundary at this design point")
        return sol

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached), re-solved tight and polished with its own Jacobian."""
        if "gradients" in sol:
            return sol["gradients"]
        model, xj = self.model, jnp.asarray(x)
        field, nb = self.field(xj), model.x0.size

        def relinearize():
            extra = self._plasma_directions(sol["params"], xj) if self._plasma.size else None
            try:
                J, _, _, aux = model.linearize(sol["x"], sol["aux"], field, extra=extra)
            except RuntimeError as error:  # the trial's state does not tighten to the deck's tolerance
                raise TrialRejected(str(error)) from error
            state, mask, params, _ = aux
            sol.update(aux=aux, state=state, mask=mask, params=params,
                       rows=np.asarray(model._rows(state, params, field)),
                       values=np.asarray(self._rows(state, params, xj)))
            self.stats["linearizations"] += 1
            return J

        J = relinearize()
        # to this Jacobian's fixed point, solved tight; J stays the anchor of the next trials
        params_x = self._params_at(xj)
        polished = model.solve_boundary(params_x, field, x0=sol["x"], jacobian=J[:, :nb],
                                        max_nfev=self.boundary_max_nfev)
        self.stats["boundary_evaluations"] += polished["nfev"]
        if not polished["converged"]:
            raise TrialRejected("the polish of the boundary did not converge")
        if polished["njev"]:  # J's own steps stalled and fresh ones finished: the anchor is relinearized there
            sol.update(x=polished["x"], aux=polished["aux"])
            J = relinearize()
        else:
            tight = model.evaluate(polished["x"], params_x, field, tight=True)
            if tight is None:
                raise TrialRejected("the polished boundary has no certified equilibrium")
            rows, aux = tight
            state, mask, params, _ = aux
            sol.update(x=polished["x"], aux=aux, state=state, mask=mask, params=params, rows=rows,
                       values=np.asarray(self._rows(state, params, xj)))
        J_b, J_p = J[:, :nb], J[:, nb:]
        state, params = sol["state"], sol["params"]
        factors = model._im.block_factors_at(model.cfg, params, state, sol["mask"])
        params_bar = self._pullback(state, sol["mask"], params, xj, factors)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        Q, R = np.linalg.qr(J_b)
        lam = Q @ np.linalg.solve(R.T, G_b.T)  # residual-space multipliers, a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        gradients -= np.stack([np.asarray(self._field_vjp(state, params, xj, jnp.asarray(column))) for column in lam.T])
        gradients[:, self._plasma] -= lam.T @ J_p
        self._anchor = dict(x=self._x(x).copy(), b=sol["x"], J_b=J_b, J_p=J_p, state=state,
                            params=params, cost=0.5 * sol["rows"] @ sol["rows"])
        sol["gradients"] = gradients
        return gradients

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
        """Promote an evaluated point after the optimizer accepts it; drop the other trials."""
        x = self._x(x)
        key = x.tobytes()
        if self._cache.get(key) is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accepted, self.accepted_step = _AcceptedPoint(x.copy()), self.accepted_step + 1
        self._cache = {key: self._cache[key]}

    def state(self, x):
        """``(state, params)`` of ``x``'s free boundary (``params`` the ``ImplicitParams`` it was solved at)."""
        sol = self._record(x)
        return sol["state"], sol["params"]

    def boundary_residual(self, x) -> BoundaryResidual:
        """The three interface conditions at ``x``'s free boundary."""
        sol = self._record(x)
        return self.model.boundary_residual(sol["state"], sol["params"], self.field(jnp.asarray(x)))

    def equilibrium_from_x(self, x):
        """The free-boundary equilibrium of ``x`` as an ``Equilibrium`` (fixed-boundary solve on its boundary)."""
        from .implicit import input_with_params
        from .optimize import solve_equilibrium

        sol = self._record(x)
        return solve_equilibrium(input_with_params(self.model.fixed, sol["params"]), restart_from=sol["state"])

    def close(self):
        """Drop every cached trial except the accepted one."""
        key = self.accepted.parameters.tobytes()
        self._cache = {k: v for k, v in self._cache.items() if k == key}
