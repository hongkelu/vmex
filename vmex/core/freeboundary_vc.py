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
(:func:`vmex.core.optimize.least_squares`, ``jac="implicit"``).  At an exact
solution all three vanish together; at finite resolution they stop at a floor
set mainly by ``mpol`` (about 1e-3 relative at ``mpol = 5``, 5e-4 at 7), where
the relative ``weights`` decide the balance.

Condition (1) alone leaves one direction open: the flux of ``B_out`` through
the closed surface vanishes identically, and a mismatch of the net poloidal
current adds a field tangent to ``S``.  Conditions (2) and (3) close it.  With
``p = 0`` on the boundary a solution satisfies all three exactly; a nonzero edge
pressure needs a sheet current and is rejected.
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
    label_weight: float = 1e-2,
    max_nfev: int = 30,
    verbose: int = 0,
    **least_squares_kwargs,
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
    and 5e-5 at 64 x 64 (whose quadrature needs several times the memory).  Rows of the boundary points' displacement
    along the surface in ``theta``, relative to the initial boundary and scaled
    by ``label_weight``, fix the boundary's poloidal labelling, which the
    interface conditions leave free.  The residual and Jacobian graphs run on
    JAX's default device (the GPU when there is one; pass ``device`` to
    override).  Remaining keywords go to
    :func:`vmex.core.optimize.least_squares` (and on to SciPy).

    Returns the :func:`~vmex.core.optimize.least_squares` result with two more
    attributes: ``boundary_residual`` (:class:`BoundaryResidual` of the final
    equilibrium) and ``initial_boundary_residual``.  ``result.equilibrium`` is
    the free-boundary equilibrium and ``result.input`` its deck (as a fixed
    boundary).
    """
    from .freeboundary import _external_field_from_input
    from .optimize import least_squares, solve_equilibrium

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    start = inp if initial_boundary is None else replace(
        inp, rbc=initial_boundary.rbc, zbs=initial_boundary.zbs,
        raxis_c=initial_boundary.raxis_c, zaxis_s=initial_boundary.zaxis_s)
    fixed = replace(start, lfreeb=False, mgrid_file="NONE")
    nphi, ntheta = int(nphi or 48), int(ntheta or 48)
    scale = jnp.repeat(jnp.asarray(weights, dtype=float), jnp.asarray([1, 1, 3]))[:, None, None]

    seed = solve_equilibrium(fixed)
    surface = vc.surface_field_data_from_state(fixed, seed.solution, runtime=seed.solver_context,
                                               nphi=nphi, ntheta=ntheta)
    precision = vc.plan_vc_precision(surface, digits=digits)
    # The interface conditions do not see how the boundary is labelled in
    # theta, so moving points along the surface is a null direction of the
    # residual.  Rows of the in-plane tangential displacement from the initial
    # boundary, relative to its mean |d x / d theta|, remove it; a normal
    # displacement leaves them zero to first order.
    gamma0 = jnp.asarray(surface.gamma)
    wavenumber = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)
    e_theta = jnp.real(jnp.fft.ifft(1j * wavenumber * jnp.fft.fft(gamma0, axis=-1), axis=-1))
    length = jnp.linalg.norm(e_theta, axis=0)
    tangent = e_theta / (length * jnp.mean(length))[None]

    def evaluate(state, runtime):
        return _interface_terms(fixed, state, external_field, runtime=runtime, nphi=nphi, ntheta=ntheta,
                                digits=digits, precision=precision, p_edge=0.0)

    def rows(state, runtime):
        values, area, gamma = evaluate(state, runtime)
        label = label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0)
        return (jnp.sqrt(area)[None] * jnp.concatenate([scale * values, label[None]])).ravel()

    def summary(eq):
        return summarize_boundary_residual(*evaluate(eq.solution, eq.solver_context)[:2])

    # Virtual casing dominates a Gauss-Newton step and runs ~200x faster on an
    # accelerator than on the CPU that implicit graphs default to.  Its
    # forward-mode columns are memory hungry: at ns = 101 and 8x8 modes the
    # automatic chunk width asked a 24 GB GPU for 62 GB, so 8 columns at a time.
    options = dict(jac="implicit", use_ess=True, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                   device=jax.devices()[0], jac_chunk_size=8)
    options.update(least_squares_kwargs)
    result = least_squares(
        [(rows, 0.0, 1.0)], fixed,
        max_mode=int(max(fixed.mpol - 1, fixed.ntor) if max_mode is None else max_mode),
        vary_major_radius=True, max_nfev=max_nfev, verbose=verbose, **options)
    result.initial_boundary_residual = summary(seed)
    result.boundary_residual = summary(result.equilibrium)
    return result
