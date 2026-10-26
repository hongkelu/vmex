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
(:class:`ThreeTermsFreeBoundaryModel`).  At an exact
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

The interface conditions do not fix the boundary's poloidal angle (a
relabelling ``theta -> theta + f`` moves no point of the surface), so the least
squares is flat along the relabellings.  ``ThreeTermsConfig.labelling`` chooses
how that is handled:

- ``"gauge"`` (default): VMEC's spectral condensation held exactly on the
  boundary (Hirshman & Meier 1985) and Newton on the fit's cost on that gauge
  (:mod:`vmex.core.freeboundary_three_terms_implicit`), so the stationary point
  and its derivatives are determined;
- ``"none"``: the plain Gauss-Newton least squares ("NL"), whose labelling is
  whatever the fit ends at.

Entry points, from the most to the least packaged:

- an ``LFREEB = T`` deck with ``LTHREE_TERMS = T`` (and optionally
  ``THREE_TERMS_LABELLING = 'none'``) through ``vmex input.case`` or
  :func:`~vmex.core.multigrid.solve_file`, or
  ``solve_free_boundary_multigrid(..., three_terms=ThreeTermsConfig())``;
- :func:`solve_free_boundary_three_terms`: the same solve with every control,
  and ``previous=`` to continue from an earlier result after the coils change;
- :class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`:
  single-stage optimization, a loss and constraint rows of coils and plasma
  parameters at their free boundary, for :func:`vmex.core.optimize.minimize`;
- :class:`ThreeTermsFreeBoundaryModel`: the residual, its Jacobian, state
  tangents and pullbacks for one deck and any external field or plasma
  parameters, without recompiling -- the pieces an optimizer needs;
- :func:`boundary_residual`: the three conditions on any equilibrium, e.g. to
  check a NESTOR result.

The solve needs ``jax_enable_x64`` and ``virtual-casing-jax``, a
stellarator-symmetric deck (``LASYM = F``) and zero edge pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .bootstrap import CLOSURE_ATOL, CLOSURE_RTOL
from .device import AUTO
from .errors import TrialRejected, VmecConvergenceError, VmecError
from .input import VmecInput
from .printing import emit_flushed

__all__ = [
    "BoundaryResidual",
    "ThreeTermsConfig",
    "ThreeTermsFreeBoundaryModel",
    "ThreeTermsResult",
    "boundary_residual",
    "resolve_three_terms",
    "solve_free_boundary_three_terms",
    "summarize_boundary_residual",
]


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


def _gauge_rows(R, Z, mpol, ntor):
    """The exact gauge of the boundary's poloidal angle: VMEC's spectral condensation (Hirshman & Meier 1985,
    Hirshman & Breslau 1998) as equations, not penalty rows.

    ``R``, ``Z`` are ``(ntheta, nphi)`` on a uniform grid over one field period.  ``G = R_theta sum m (m - 1) R_mn
    cos(m theta - n N phi) + Z_theta sum m (m - 1) Z_mn sin(...)``, relative to the mean ``R_theta^2 + Z_theta^2``, is
    odd for a stellarator-symmetric surface, so its sine components carry every condition: those with
    ``m = 1 .. mpol - 2`` and ``|n| <= ntor`` fix the ``m >= 1`` relabellings.  The ``m = 0`` relabellings (a
    shift of theta on each toroidal cut) leave ``G`` unchanged; the phase of the ``m = 1`` part of ``R + i Z`` per cut
    (odd in phi) fixes them, by its sine components ``n = 1 .. ntor``.  Returns the conditions, flattened.
    """
    ntheta, nphi = R.shape
    k = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)[:, None]
    FR, FZ = jnp.fft.fft(R, axis=0), jnp.fft.fft(Z, axis=0)
    R_theta, Z_theta = (jnp.real(jnp.fft.ifft(1j * k * F, axis=0)) for F in (FR, FZ))
    weight = jnp.abs(k) * (jnp.abs(k) - 1)  # VMEC's m (m - 1)
    R_c, Z_c = (jnp.real(jnp.fft.ifft(weight * F, axis=0)) for F in (FR, FZ))
    G = (R_c * R_theta + Z_c * Z_theta) / jnp.mean(R_theta**2 + Z_theta**2)
    spectrum = jnp.fft.fft2(G) / G.size
    toroidal = np.r_[0:ntor + 1, nphi - ntor:nphi]
    condensation = jnp.imag(spectrum[1:mpol - 1][:, toroidal]).ravel()
    c1 = jnp.fft.fft(R + 1j * Z, axis=0)[1]
    phase = jnp.arctan2(jnp.imag(c1), jnp.real(c1))
    phase_sine = -jnp.imag(jnp.fft.fft(phase))[1:ntor + 1] / nphi
    return jnp.concatenate([condensation, phase_sine])


#: Stationarity (the gain a Gauss-Newton step predicts) a trial's polish rounds aim at.
POLISH_FTOL = 1e-4
#: Polish rounds (each with a fresh Jacobian) a trial gets to reach its certificate; past them it keeps polishing,
#: up to POLISH_ROUNDS_MAX, while each round still cuts the stationarity by POLISH_PROGRESS (a converging polish that
#: started far: the gauged m = 1 complement's bootstrap fit, 2.8e-2 -> 4.9e-4 in five rounds at ~0.4 per round).
POLISH_ROUNDS = 5
POLISH_ROUNDS_MAX = 12
POLISH_PROGRESS = 0.5
#: Most evaluations of one polish round (Levenberg-Marquardt steps with that round's Jacobian).
POLISH_MAX_NFEV = 20
#: Levenberg-Marquardt stops as stalled after this many steps in a row that each gained less than STALL_GAIN of the
#: cost (the Jacobian no longer describes the rows there).
STALL_STEPS = 3
STALL_GAIN = 0.05
#: Most fresh Jacobians of the cold ``fit="secant"`` boundary fit (:meth:`ThreeTermsFreeBoundaryModel.solve_boundary`)
#: before the trust region finishes from its end point.
FIT_ROUNDS = 12
TANGENT_WIDTH = 16  # tangent batches are padded to a multiple of this: one compiled program for nearby column counts

#: Largest RMS interface residual (relative to ``|B|``) of a three-term free boundary that counts as solved: a guard
#: against wrong stationary points, not an accuracy standard (about upstream #614's DEL-BSQ gate of 2e-2 on B^2).
ACCEPTED_RESIDUAL = 1e-2

#: The boundary labellings (:attr:`ThreeTermsConfig.labelling`) and the cold fits (:attr:`ThreeTermsConfig.fit`).
LABELLINGS = ("gauge", "none")
FITS = ("secant", "trf")


def _newton_defaults() -> dict:
    return dict(step_tol=1e-8, max_steps=60, quasi=True)


@dataclass(frozen=True)
class ThreeTermsConfig:
    """Settings of the three-term free boundary (``three_terms=`` of the free-boundary solvers).

    ``labelling`` is how the boundary's poloidal angle is fixed: ``"gauge"`` (VMEC's spectral condensation held
    exactly, Newton on the gauge: a determined stationary point; the default) or ``"none"`` (the plain least
    squares, "NL").  ``m1_complement`` is the interior's m = 1 complement (``ImplicitConfig.m1_complement``:
    ``"gauged"`` fixes it to ``sqrt(s)`` times its edge value, a gauge that certifies at any resolution;
    ``"frozen"`` is VMEC2000's released complement).

    The discrete problem: ``max_mode`` (the boundary modes solved for, default all), the ``nphi x ntheta``
    interface grid over one field period, the virtual-casing ``quadrature = (quad_nt, quad_np)`` (fixed; use
    ``(4 nfp nphi, 2 ntheta)``; ``None`` plans it to ``digits``), ``weights`` of the ``(normal, pressure, sheet
    current)`` rows and ``net_current_weight`` of the net poloidal current row (see
    :class:`ThreeTermsFreeBoundaryModel`).

    The solve: ``fit`` is the cold boundary fit, ``"secant"`` (rounds of Levenberg-Marquardt steps on one fresh,
    Broyden-updated Jacobian; LBD 8 x 8 cold gauge fit 899 s -> 299 s, 36 -> 6 Jacobians, the same certified
    boundary) or ``"trf"`` (SciPy's trust region, a fresh Jacobian after every accepted step).  It runs to a
    relative cost change of ``jacobian_ftol`` with fresh Jacobians and to ``ftol`` with its last one, at most
    ``max_nfev`` evaluations; ``trial_ftol`` solves its trial equilibria to a looser force residual.  With the
    gauge, ``newton`` (``step_tol``, ``max_steps``, ``quasi``, ``verbose``) then converges the fit's point on the
    gauge.  ``max_residual`` is the largest RMS interface residual of a converged free boundary (a guard against
    wrong stationary points).  ``chunk`` bounds the Jacobian columns per batch.
    """

    labelling: str = "gauge"
    m1_complement: str = "gauged"
    max_mode: int | None = None
    nphi: int = 48
    ntheta: int = 48
    quadrature: tuple[int, int] | None = None
    digits: int = 4
    trial_ftol: float | None = None
    fit: str = "secant"
    ftol: float = 1e-4
    jacobian_ftol: float | None = 1e-2
    max_nfev: int = 60
    max_residual: float = ACCEPTED_RESIDUAL
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    net_current_weight: float = 1.0
    chunk: int = 8
    newton: dict = field(default_factory=_newton_defaults)

    def __post_init__(self) -> None:
        if self.labelling not in LABELLINGS:
            raise ValueError(f"labelling must be one of {LABELLINGS}, got {self.labelling!r}")
        if self.fit not in FITS:
            raise ValueError(f"fit must be one of {FITS}, got {self.fit!r}")
        if self.m1_complement not in ("gauged", "frozen"):
            raise ValueError(f"m1_complement must be 'gauged' or 'frozen', got {self.m1_complement!r}")
        object.__setattr__(self, "newton", {**_newton_defaults(), **dict(self.newton or {})})

    @classmethod
    def from_input(cls, inp: Any) -> "ThreeTermsConfig | None":
        """The three-term solve a deck requests (``LTHREE_TERMS``, ``THREE_TERMS_LABELLING``), else ``None``."""
        if not bool(getattr(inp, "lthree_terms", False)):
            return None
        return cls(labelling=str(getattr(inp, "three_terms_labelling", "gauge")))

    def model_options(self) -> dict:
        """The :class:`ThreeTermsFreeBoundaryModel` keywords of these settings."""
        return dict(labelling=self.labelling, m1_complement=self.m1_complement, max_mode=self.max_mode,
                    nphi=int(self.nphi), ntheta=int(self.ntheta), quadrature=self.quadrature, digits=self.digits,
                    trial_ftol=self.trial_ftol, fit=self.fit, weights=tuple(self.weights),
                    net_current_weight=self.net_current_weight, chunk=self.chunk)


@dataclass(frozen=True)
class ThreeTermsResult:
    """Outcome of a three-term free-boundary solve, returned as ``result.three_terms``.

    ``certified`` means the fit's final point passed the acceptance rule (re-solved tight; with ``"none"``
    labelling polished until a Gauss-Newton step gains at most ``max(ftol, POLISH_FTOL)`` of the cost, with
    ``"gauge"`` Newton-converged on the gauge); ``certificate`` is that record's (``None`` when not certified) and
    ``message`` says how the solve ended.  ``boundary_residual`` and ``initial_boundary_residual`` are the
    :class:`BoundaryResidual` at the end and at the start; ``input`` is the free boundary as a fixed-boundary deck;
    ``x`` its model coordinates (:meth:`ThreeTermsFreeBoundaryModel.solve_boundary`), ``jacobian`` the interface
    rows' last Jacobian along them, ``hessian`` the cost's Hessian of the gauge Newton (``None`` for ``"none"``),
    ``nfev``/``njev`` the boundary evaluations and Jacobians, ``newton`` the gauge Newton's record.  ``model`` (the
    :class:`ThreeTermsFreeBoundaryModel`) and ``aux`` (the final interior state) let ``previous=`` continue from
    here without compiling anything.
    """

    labelling: str
    certified: bool
    certificate: dict | None
    message: str
    boundary_residual: BoundaryResidual
    initial_boundary_residual: BoundaryResidual
    input: VmecInput = field(repr=False)
    x: np.ndarray = field(repr=False)
    jacobian: np.ndarray = field(repr=False)
    hessian: np.ndarray | None = field(default=None, repr=False)
    nfev: int = 0
    njev: int = 0
    newton: dict | None = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    aux: Any = field(default=None, repr=False)


def resolve_three_terms(requested: Any, inp: Any) -> "ThreeTermsConfig | None":
    """An explicit config wins; ``None`` follows the deck (``LTHREE_TERMS``); ``False`` disables it; ``True`` is the
    deck's labelling with the default settings."""
    if requested is None:
        return ThreeTermsConfig.from_input(inp)
    if requested is False:
        return None
    if requested is True:
        return ThreeTermsConfig(labelling=str(getattr(inp, "three_terms_labelling", "gauge")))
    if not isinstance(requested, ThreeTermsConfig):
        raise TypeError(f"three_terms must be a ThreeTermsConfig, a bool or None, got {type(requested).__name__}")
    return requested


def _start_boundary(inp: VmecInput, initial_state, restart_from) -> VmecInput:
    """``inp`` with the boundary (and axis) of ``restart_from`` (a wout path, :class:`~vmex.core.wout.WoutData` or
    :class:`~vmex.core.solver.SolveResult`) or of ``initial_state`` (a state on the deck's radial grid): a start."""
    from .solver import SolveResult
    from .wout import WoutData, read_wout, wout_from_state

    if initial_state is not None and restart_from is not None:
        raise ValueError("initial_state and restart_from are mutually exclusive")
    source = restart_from if restart_from is not None else initial_state
    if source is None:
        return inp
    if isinstance(source, SolveResult):
        source = source.state
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        source = read_wout(source)
    if not isinstance(source, WoutData):  # a SpectralState
        source = wout_from_state(inp=inp, state=source, fsqr=0.0, fsqz=0.0, fsql=0.0, converged=True)
    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(source.xm, int), np.asarray(source.xn, int) // int(source.nfp),
                          np.asarray(source.rmnc)[-1], np.asarray(source.zmns)[-1]):
        if m < rbc.shape[1] and abs(n) <= int(inp.ntor):
            rbc[n + int(inp.ntor), m], zbs[n + int(inp.ntor), m] = r, z
    axis = int(inp.ntor) + 1
    return replace(inp, rbc=rbc, zbs=zbs, raxis_c=np.asarray(source.raxis_cc)[:axis],
                   zaxis_s=np.asarray(source.zaxis_cs)[:axis])


def solve_free_boundary_three_terms(
    inp: VmecInput,
    *,
    external_field: Any = None,
    mgrid_path=None,
    initial_state: Any = None,
    restart_from: Any = None,
    previous: Any = None,
    config: ThreeTermsConfig | None = None,
    verbose: bool | int = False,
    emit=emit_flushed,
    device: Any = AUTO,
):
    """Free-boundary equilibrium of ``inp`` from the three interface conditions, as a
    :class:`~vmex.core.solver.SolveResult`.

    ``inp`` is a free-boundary deck (``PHIEDGE``, profiles, and the external
    field through ``external_field``, ``mgrid_path`` or ``MGRID_FILE``/``EXTCUR``);
    its boundary is the initial guess, so a fixed-boundary design and the coils
    fitted to it make a natural start.  ``restart_from`` (a wout path,
    :class:`~vmex.core.wout.WoutData` or ``SolveResult``, e.g. a NESTOR solve) or
    ``initial_state`` (a state on the deck's radial grid) starts from that
    equilibrium's boundary instead.  The boundary Fourier coefficients up to
    ``config.max_mode`` (default: all) and ``RBC(0,0)`` are solved for at fixed
    toroidal flux, pressure and current or iota, every trial being a
    fixed-boundary equilibrium on the deck's ``NS_ARRAY`` ladder.
    ``external_field`` is anything
    :func:`~vmex.core.virtual_casing.external_B_cartesian` accepts (an
    :class:`~vmex.core.mgrid.MgridField`, a ``xyz -> B`` callable such as an
    ESSOS Biot-Savart field, or a coil field with ``b_cyl``).

    The residual is ``sqrt(area weight)`` times ``config.weights`` times the rows
    of :func:`boundary_residual` on the ``nphi x ntheta`` grid (see
    :class:`ThreeTermsConfig`).  The grid, not ``digits``, limits accuracy: for
    a current-free state virtual casing returns ``|B_plasma| / |B|`` of about
    2e-4 at 48 x 48 and 5e-5 at 64 x 64.  A boundary fit with exact implicit
    Jacobians runs to ``jacobian_ftol`` and ``ftol``; its end point is then
    accepted by the same rule as every trial of the optimization
    (:class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`):
    with ``labelling="none"`` it is re-solved tight and polished until a
    Gauss-Newton step with its own Jacobian gains at most ``max(ftol,
    POLISH_FTOL)`` of the cost; with ``labelling="gauge"`` it is relabelled onto
    the exact gauge and Newton-converged there (``config.newton``).  A stationary
    point need not be the free boundary (a wrong local minimum is one too), and
    the interface rows cannot reach zero at a finite resolution, so
    ``result.converged`` also requires every RMS interface residual
    (:class:`BoundaryResidual`) at or below ``config.max_residual``.

    ``previous`` (an earlier result for the same deck and settings, typically
    before a coil change) starts from its boundary, Jacobian (and, on the gauge,
    its Hessian) and model: neither new coils nor a repeat solve compile the
    model again.  The returned state is the fixed-boundary solve of the final
    stage on the free boundary, started from the certified interior state;
    ``result.three_terms`` is the :class:`ThreeTermsResult` and
    ``result.vacuum`` is ``None``.
    """
    from .freeboundary import _resolve_mgrid
    from .freeboundary_three_terms_implicit import _accept, _Design, _solve_point
    from .mgrid import MgridField
    from .multigrid import solve_multigrid

    config = ThreeTermsConfig() if config is None else config
    if not jax.config.jax_enable_x64:
        raise ValueError("the three-term free boundary needs jax_enable_x64")
    vc._require_vcj()  # ImportError naming vmex[freeb] when virtual-casing-jax is missing or too old
    if bool(inp.lasym):
        raise NotImplementedError("the three-term free boundary supports lasym = False only")
    if _edge_pressure_error(inp):
        raise ValueError(_edge_pressure_error(inp))
    if external_field is None:  # tricubic: virtual casing samples between the mgrid planes, where the trilinear
        # interpolant has kinks that NESTOR (on the planes) never sees; on the CTH-like deck they cost 0.4 mm
        external_field = MgridField.from_input(inp, _resolve_mgrid(inp, mgrid_path), order=3)
    previous = getattr(previous, "three_terms", previous)
    if previous is not None:
        if previous.labelling != config.labelling:
            raise ValueError(f"previous was solved with labelling {previous.labelling!r}, not {config.labelling!r}")
        model, x0, jacobian = previous.model, previous.x, previous.jacobian
        seed_state, seed_params = previous.aux[0], previous.aux[2]
    else:
        model = ThreeTermsFreeBoundaryModel(_start_boundary(inp, initial_state, restart_from),
                                           **config.model_options())
        x0 = jacobian = None
        seed_state, seed_params = model.seed[0], model.params0
    initial = model.boundary_residual(seed_state, seed_params, external_field)
    stats: dict[str, Any] = {}
    kept = {}
    gauge = config.labelling == "gauge"
    design = _Design.fixed(model, external_field)

    def accept(fit):  # the optimization's acceptance rule: one definition of a solved free boundary
        return _accept(design, np.zeros(0), model.params0, external_field, fit,
                       gate=max(config.ftol, POLISH_FTOL), newton=config.newton if gauge else None, stats=stats,
                       anchor_H=None if previous is None else previous.hessian, keep_hessian=lambda record: kept.update(record or {}))

    ftol = config.ftol
    starts, direct = [x0], None
    if gauge and previous is not None:  # the previous boundary straight into the gauge Newton, then fitted
        starts, direct = [np.asarray(x0, dtype=float), x0], 0
    sol, why, fit = _solve_point(
        model, model.params0, external_field, starts, accept=accept, jacobian=jacobian, direct=direct, stats=stats,
        fit_options=dict(ftol=ftol, jacobian_ftol=ftol if config.jacobian_ftol is None else config.jacobian_ftol,
                         max_nfev=config.max_nfev, refresh_on_stall=True, verbose=int(verbose)))
    if fit is None:
        raise VmecConvergenceError(f"three-term free boundary: {'; '.join(why)}")
    certified = sol is not None
    if certified:
        out = sol
        how = (f"Newton-converged on the gauge ({len(out['newton']['steps'])} steps)" if gauge else
               f"certified (Gauss-Newton gain {out['certificate']['stationarity']:.1e})")
        jacobian_out = out["J"][:, :model.x0.size]
    else:
        out, how, jacobian_out = fit, f"not certified: {'; '.join(why)}", fit["jacobian"]
    state, _, params, _ = out["aux"]
    boundary = model._im.input_with_params(model.fixed, params)
    residual = model.boundary_residual(state, params, external_field)
    largest = max(residual.normal, residual.pressure, residual.sheet_current)
    message = (f"three-term free boundary ({config.labelling} labelling): boundary fit {fit['reason']}, "
               f"{int(stats.get('boundary_evaluations', 0))} evaluations; {how}; interface residuals B.n "
               f"{residual.normal:.2e}, pressure {residual.pressure:.2e}, sheet current {residual.sheet_current:.2e}"
               + ("" if largest <= config.max_residual else f" (above the accepted {config.max_residual:.0e})"))
    if verbose:
        print(message, flush=True)
    stage = dict(ns_array=[int(np.asarray(state.R_cos).shape[0])],
                 ftol_array=[float(np.atleast_1d(boundary.ftol_array)[-1])],
                 niter_array=[int(np.atleast_1d(boundary.niter_array)[-1])])
    result = solve_multigrid(boundary, **stage, initial_state=state, verbose=bool(verbose), emit=emit, device=device)
    hessian = kept.get("H_cost") if certified and gauge else None
    return replace(result, converged=bool(result.converged) and certified and largest <= config.max_residual,
                   three_terms=ThreeTermsResult(
                       labelling=config.labelling, certified=certified,
                       certificate=out.get("certificate") if certified else None, message=message,
                       boundary_residual=residual, initial_boundary_residual=initial, input=boundary,
                       x=np.asarray(out["x"]), jacobian=np.asarray(jacobian_out), hessian=hessian,
                       nfev=int(stats.get("boundary_evaluations", 0)),
                       njev=int(fit.get("njev", 0) + stats.get("linearizations", 0)),
                       newton=out.get("newton"), model=model, aux=out["aux"]))


class ThreeTermsFreeBoundaryModel:
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
    replaces.  The rows are those of :func:`solve_free_boundary_three_terms`,
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

    ``labelling`` fixes the boundary's poloidal angle, which the interface
    rows leave free: ``"gauge"`` (default) builds VMEC's spectral condensation
    as exact geometric conditions (``gauge_rows``, :func:`_gauge_rows`) and the
    relabelling onto them (``relabel``) for the gauge Newton of
    :mod:`vmex.core.freeboundary_three_terms_implicit`; ``"none"`` leaves the
    labelling to the fit.  ``label_weight`` (default 0: off) adds rows of the
    boundary points' displacement along the surface in ``theta``, relative to
    the deck's boundary.  ``fit`` is the cold boundary fit of
    :meth:`solve_boundary` (:attr:`ThreeTermsConfig.fit`) and ``m1_complement``
    the interior's m = 1 complement policy (:attr:`ThreeTermsConfig.m1_complement`).

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

    def __init__(self, inp: VmecInput, *, labelling="gauge", max_mode=None, nphi=48, ntheta=48, digits=4,
                 weights=(1.0, 1.0, 1.0), net_current_weight=1.0, label_weight=0.0, chunk=8, device=None,
                 trial_ftol=None, quadrature=None, fit="secant", bootstrap=None, bootstrap_helicity=0,
                 bootstrap_weight=100.0, fixed_boundary=False, bootstrap_g_convention="simsopt",
                 m1_complement="gauged"):
        from . import implicit as im
        from .bootstrap import HalfMeshCurrent
        from .optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

        self._im = im
        if labelling not in LABELLINGS:
            raise ValueError(f"labelling must be one of {LABELLINGS}, got {labelling!r}")
        if fit not in FITS:
            raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
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
        self.fixed_boundary, self.labelling, self.fit = bool(fixed_boundary), labelling, fit
        #: The exact gauge of the boundary's poloidal angle (:func:`_gauge_rows`): None (``labelling="none"``: the
        #: labelling is whatever minimizes the interface residual) or "condensation" (VMEC's spectral condensation held
        #: exactly: every iterate of the gauge Newton is projected onto it, which needs geometry only).
        self.gauge = "condensation" if labelling == "gauge" else None
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

        # The boundary's sample grid (gauge conditions, relabelling, normal operator): the gauge conditions are
        # products of two series of the boundary's modes (up to K = its highest m or n), so their coefficients up to
        # K are alias-free only on more than 3 K points; 48 up to 12 boundary modes, more above (at 16 the 48 grid
        # aliased the toroidal conditions and left a null direction on the gauge: Newton condition 5e14)
        mmax, nmax = int(fixed.mpol) - 1, min(int(fixed.ntor), self.max_mode)
        ng = max(48, -(-(3 * max(mmax, nmax) + 2) // 16) * 16)
        self.surface_grid = ng

        def surface(x):
            """``(R, Z)`` of the boundary of coordinates ``x`` on the ``surface_grid`` grid over one field period."""
            rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
            th = jnp.linspace(0.0, 2 * jnp.pi, ng, endpoint=False)
            ph = jnp.linspace(0.0, 2 * jnp.pi / int(fixed.nfp), ng, endpoint=False)
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
            k = jnp.fft.fftfreq(ng, 1.0 / ng)
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
        self.surface = jax.jit(surface) if nb else None
        if self.gauge is not None and nb:
            mpol_g, ntor_g = min(int(fixed.mpol), self.max_mode + 1), min(int(fixed.ntor), self.max_mode)  # the boundary's

            def gauge_rows(x):
                R, Z = surface(x[:nb])
                return _gauge_rows(R, Z, mpol_g, ntor_g)

            self.gauge_rows = jax.jit(gauge_rows)
            self.gauge_jacobian = jax.jit(jax.jacfwd(gauge_rows))
            self.gauge_curvature = jax.jit(lambda x, lam: jax.hessian(lambda y: jnp.vdot(lam, gauge_rows(y)))(x))

            # A relabelling theta -> theta + f of the same surface, f = sum f_k sin(m_k theta - n_k N phi), with as many
            # modes as gauge conditions (m = 1 .. mpol - 2, |n| <= ntor; and m = 0, n = 1 .. ntor). The surface is
            # resampled at the new angle and refitted to the coordinates (the coordinates-to-samples map is affine),
            # so the shape changes only at the truncation level.
            nfp_g = int(fixed.nfp)
            modes = [(m, n) for m in range(1, mpol_g - 1) for n in range(-ntor_g, ntor_g + 1)]
            modes += [(0, n) for n in range(1, ntor_g + 1)]
            m_f = jnp.asarray([m for m, _ in modes], dtype=float)
            n_f = jnp.asarray([n for _, n in modes], dtype=float)

            def samples(x):
                R, Z = surface(x[:nb])
                return jnp.concatenate([R.ravel(), Z.ravel()])

            A = np.asarray(jax.jacfwd(samples)(jnp.asarray(self.x0)))[:, :nb]
            A_pinv = jnp.asarray(np.linalg.pinv(A))

            def relabel(x, fc):
                rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], self.max_mode, vary_major_radius=True)
                th = jnp.linspace(0.0, 2 * jnp.pi, ng, endpoint=False)[:, None]
                ph = jnp.linspace(0.0, 2 * jnp.pi / nfp_g, ng, endpoint=False)[None, :]
                f = jnp.einsum("k,ktp->tp", fc, jnp.sin(m_f[:, None, None] * th[None] - n_f[:, None, None] * nfp_g
                                                        * ph[None]))
                n = (jnp.arange(rbc.shape[0]) - (rbc.shape[0] - 1) // 2) * nfp_g
                m = jnp.arange(rbc.shape[1])
                angle = m[None, None, None, :] * (th + f)[:, :, None, None] - n[None, None, :, None] * ph[:, :, None, None]
                R = jnp.einsum("tpnm,nm->tp", jnp.cos(angle), rbc)
                Z = jnp.einsum("tpnm,nm->tp", jnp.sin(angle), zbs)
                target = jnp.concatenate([R.ravel(), Z.ravel()])
                return x.at[:nb].add(A_pinv @ (target - samples(x)))

            self.relabel_modes = len(modes)
            self.relabel = jax.jit(relabel)
            self.relabel_gauge = jax.jit(lambda x, fc: gauge_rows(relabel(x, fc)))
            self.relabel_gauge_jacobian = jax.jit(jax.jacfwd(lambda x, fc: gauge_rows(relabel(x, fc)), argnums=1))
        seed = self.solve(self.params0)
        if seed is None:
            raise VmecConvergenceError("the initial boundary has no converged fixed-boundary equilibrium: "
                                       f"{getattr(self, '_last_failure', None)}")
        state, mask = seed[:2]
        self.seed = (state, mask)
        if not self.fixed_boundary:
            self._interface_setup(state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight)
        self._implicit_setup()
        self.row_blocks, self.column_blocks = self._layout(nphi, ntheta, net_current_weight)
        #: Called with a complete record at every certificate round and at the outcome of :meth:`certify` (``None``:
        #: nothing recorded).  See :meth:`certify`.
        self.recorder = None

    def _layout(self, nphi, ntheta, net_current_weight):
        """``(row_blocks, column_blocks)``: ``(name, start, stop)`` of each block of the rows and of the model
        coordinates.  Interface rows are component-major: ``B.n``, ``pressure``, ``K_x``, ``K_y``, ``K_z``, ``label``,
        each ``nphi * ntheta`` (``(nphi, ntheta)`` row-major), then the net-current row and the bootstrap rows."""
        sizes = []
        if not self.fixed_boundary:
            grid = int(nphi) * int(ntheta)
            sizes += [(name, grid) for name in ("B.n", "pressure", "K_x", "K_y", "K_z", "label")]
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

    def _interface_setup(self, state, nphi, ntheta, digits, quadrature, weights, label_weight, net_current_weight):
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
            label = label_weight * jnp.sum((gamma - gamma0) * tangent, axis=0)
            interface = (jnp.sqrt(area)[None] * jnp.concatenate([scale * values, label[None]])).ravel()
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
        jit_tangents = jax.jit(tangents)

        def padded_tangents(params, state, mask, batch, system=None):
            """``tangents`` on a batch padded to a multiple of TANGENT_WIDTH columns (copies of the last; their
            results dropped): the boundary block and the boundary block plus a plasma column share one program
            (255 and 256 columns each compiled ``tangents``, ~100 s of a solve)."""
            n = int(jax.tree.leaves(batch)[0].shape[0])
            width = -(-n // TANGENT_WIDTH) * TANGENT_WIDTH
            if width != n:
                batch = jax.tree.map(lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], width - n, axis=0)]), batch)
            solution, certified = jit_tangents(params, state, mask, batch, system)
            return jax.tree.map(lambda a: a[:n], solution), certified[:n]

        self._tangents = padded_tangents
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
            state, mask, status, fsq, ratio = im._host_solve_and_mask_status(cfg, params_np)
            if int(status) != 0:
                self._last_failure = dict(status=int(status), fsq=float(fsq), ratio=float(ratio),
                                          error=repr(im._LAST_STATUS_ERROR.get(cfg))[:300])
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
                       target_cost=None, refresh_on_stall=False, tight=False, fresh_current=False, verbose=0,
                       fit=None):
        """K = 0 boundary for ``params`` and ``field``.

        With ``jacobian`` (from a nearby solve) Levenberg-Marquardt steps reuse it from ``x0``; otherwise, or if no
        step reduces the cost, or they end above ``target_cost``, a trust region with fresh Jacobians runs to
        ``jacobian_ftol`` and the reuse steps finish to ``ftol`` (see :func:`solve_free_boundary_three_terms`);
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
        if (self.fit if fit is None else fit) == "secant":  # cold, or the warm reuse fell short: secant rounds
            got = self.evaluate(x, params, field, tight=tight)
            if got is None:
                raise RuntimeError("the start boundary has no certified equilibrium")
            r, aux = got
            nfev += 1

            def lm(x, r, aux, J, tol):
                return _levenberg_marquardt(lambda z: self.evaluate(z, params, field, tight=tight), x, r, aux, J,
                                            ftol=tol, max_nfev=max_nfev, verbose=verbose, radius=self._radius,
                                            project=self._projected, current_columns=columns, update="broyden")

            # As the trust region's two stages: rounds on fresh Jacobians to ``jacobian_ftol`` (each round its own
            # evaluation budget, the rounds bounded by their Jacobians), then the finish to ``ftol`` on the last one
            # (updated), with one fresh Jacobian if it stalls.  The certificate's polish takes it from there.
            for _ in range(FIT_ROUNDS):
                J, _, _, aux = self.linearize(x, aux, field)
                njev += 1
                out = lm(x, r, aux, J, max(ftol, jacobian_ftol))
                self._radius = out["radius"]
                nfev += out["nfev"]
                x, r, aux = out["x"], out["rows"], out["aux"]
                if out["converged"]:
                    break
            else:  # FIT_ROUNDS fresh Jacobians and still not converged: the trust region below finishes from here
                self.fit_fallbacks = getattr(self, "fit_fallbacks", 0) + 1
                out = None
            if out is not None:
                out = lm(x, r, aux, J, ftol)
                nfev += out["nfev"]
                if refresh_on_stall and not out["converged"]:
                    J, _, _, aux = self.linearize(out["x"], out["aux"], field)
                    njev += 1
                    out = lm(out["x"], out["rows"], aux, J, ftol)
                    nfev += out["nfev"]
                self._radius = out["radius"]
                return dict(out, jacobian=J, nfev=nfev, njev=njev)
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
        params)`` are computed only then.  The standalone solve (:func:`solve_free_boundary_three_terms`) and the
        optimization (:class:`~vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`) accept a state
        through this one method.  ``stats``
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
                                                                  "projection_certified", "jacobian_updates")}
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
            polish_error = ""
            previous = np.inf
            cap = max(POLISH_ROUNDS, POLISH_ROUNDS_MAX)
            for round_ in range(cap + 1):
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
                progressing = np.isfinite(previous) and certificate["stationarity"] <= POLISH_PROGRESS * previous
                previous = certificate["stationarity"]
                budget = round_ >= POLISH_ROUNDS and not (progressing and round_ < cap)
                if done or not polish or budget:
                    out_of_rounds = polish and not done and budget
                    break
                # Short of the polish target: polish with this (fresh, tight) Jacobian and certify again.
                try:
                    # Tight evaluations: the polish makes the final record, judged on tight states.  On trial ones
                    # (QH 2.5%: trial FTOL 1e-11 against the deck's 1e-15) Redl's target differs enough that the
                    # current solved there leaves 98% of the tight cost in the bootstrap rows, and the polish cycles.
                    # fresh Jacobians (fit="trf"): the polish finishes along the weak relabelling directions,
                    # where a secant-updated Jacobian stalls short of the gate (the cheap secant fit only gets close)
                    polished = self.solve_boundary(params, field, x0=sol["x"], jacobian=sol["J"][:, :n],
                                                   ftol=0.1 * min(gate, POLISH_FTOL), max_nfev=POLISH_MAX_NFEV,
                                                   tight=True, fresh_current=True, fit="trf")
                except (VmecError, RuntimeError) as error:
                    stats["failed_polishes"] = stats.get("failed_polishes", 0) + 1
                    polish_error = f"; its polish failed: {error}"
                    break
                stats["boundary_evaluations"] = stats.get("boundary_evaluations", 0) + polished["nfev"]
                if not polished["rows"] @ polished["rows"] < sol["rows"] @ sol["rows"]:
                    break  # the polish gains nothing: the certificate below decides
                sol.update(tight(polished["x"], polished["aux"]))  # the same fit, nearer its stationary point
                sol.update({k: polished[k] for k in ("nfev", "accepted", "reason", "projections", "projection_certified",
                                                      "jacobian_updates") if k in polished})
                stats["polish_rounds"] = stats.get("polish_rounds", 0) + 1
            if not settled:
                normal_note = "" if certificate["normal_step"] is None else (
                    f" and moves the boundary normally by {certificate['normal_step']:.2e} of the minor radius")
                budget_note = f" when the polish budget of {rounds[0] - 1} rounds ran out" if out_of_rounds else ""
                raise TrialRejected(f"boundary fit not stationary: a Gauss-Newton step with its own Jacobian gains "
                                    f"{certificate['stationarity']:.2e} of the cost{normal_note}{budget_note}"
                                    f"{polish_error}")
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


#: Most Gauss-Newton corrections of the projected (bootstrap-current) coordinates after one boundary step.
PROJECTION_CORRECTIONS = 4


def _projection_gate(ftol):
    """The conditional-stationarity gate of the projected (current) coordinates: ``0.1 ftol``."""
    return 0.1 * ftol


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
                         current_columns=None, update=None):
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

    ``update="broyden"`` updates ``J`` by a rank-one secant (Broyden) step after every accepted step (the secant fit
    rounds, ``fit="secant"``); ``jacobian_updates`` counts them.  Default: ``J`` stays fixed.
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
    updates = 0
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
        slow = slow + 1 if cost_new > (1.0 - STALL_GAIN) * cost else 0
        if cost_new < cost:
            if update == "broyden":  # rank-one secant update along the accepted step
                dx, dr = x_new - x, got[0] - r
                if dx @ dx > 0:
                    J = J + np.outer(dr - J @ dx, dx) / (dx @ dx)
                    Q_c, R_c, norms, U, sv, Vt = decompose(J)
                    updates += 1
            x, (r, aux), accepted = x_new, got, accepted + 1
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
                jacobian_updates=updates)

