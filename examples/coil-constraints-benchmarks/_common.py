"""Helpers shared by the coil-constraint benchmarks and their post-processing.

The case seed (``seed_input``) and target (``target_residual``); the
finite-beta and bootstrap seeds (``finite_beta_input``, ``redl_profiles``,
``bootstrap_input``, ``bootstrap_mismatch``); a finished run's restart deck
(``restart_input``); coil utilities (``resize_coils``, ``scale_coil_currents``);
and the coil-field B.n diagnostics (``coil_field``, ``normal_field_rms``,
``boundary_diagnostics``). ``free_boundary_single_stage_optimization.py``,
``single_stage_optimization.py``, ``postprocess.py`` and ``fit_coils.py`` all
import from here, never from each other.

JAX, ESSOS and VMEX are imported inside the functions, so a script can set
``JAX_ENABLE_X64`` and ``JAX_PLATFORMS`` before the first JAX import.
"""
from dataclasses import replace

import parameters as P

NORMAL_FIELD_CONSTRAINT = 0.008   # fixed arm: limit on the area-weighted RMS B.n/|B|


def seed_input():
    """The case's fixed-boundary seed at the optimization resolution.

    With ``P.SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T;
    otherwise it is ``P.INPUT_FILE``'s. ``finite_beta_input`` then calibrates PHIEDGE.
    """
    import numpy as np
    import vmex as vj

    mpol, ntor, ns = P.RESOLUTION
    inp = vj.VmecInput.from_file(P.INPUT_FILE)
    if P.SEED is not None:
        inp = replace(inp, nfp=P.SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=P.GRID[0], nzeta=P.GRID[1])
    if P.SEED is not None:
        _, aspect, ratio = P.SEED
        minor = P.RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = P.RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([P.EQUILIBRIUM_FTOL]), lfreeb=False)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``P.HELICITY``, or constructed QI."""
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if P.HELICITY is None:
        return ConstructedQIResidual(np.asarray(P.QI_SURFACES), **P.QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(P.QA_SURFACES), *P.HELICITY)


def finite_beta_input(inp, beta, device, am=(1.0, -1.0)):
    """Give ``inp`` a pressure p ~ ``am`` (power series in s, default 1 - s) whose fixed-boundary seed has ``beta``.

    ``P.BETA_DEFINITION`` "volume": ``beta`` is <beta>; "axis": WOUT ``betaxis``.
    The pressure is ramped in with hot restarts, then three corrections rescale
    PHIEDGE until the edge R B_phi (the coils' mu0 I / 2 pi) is ``P.B0`` R0, as in
    upstream #426, and the pressure to ``beta``. Returns the input and the last
    fixed-boundary solve.
    """
    import numpy as np
    from vmex import optimize as opt

    axis, r0 = P.BETA_DEFINITION == "axis", float(inp.rbc[inp.ntor, 0])
    b0 = P.B0
    pressure = beta * b0**2 / (8e-7 * np.pi) if axis else beta / (4e-7 * np.pi)
    shape = np.zeros_like(np.asarray(inp.am, dtype=float))
    shape[: len(am)] = am
    inp = replace(inp, am=shape)
    fixed = None
    ramp = (0.25, 0.5, 0.75, 1.0) if beta > 0 else (1.0,)
    for index in range(len(ramp) + 3):
        if index < len(ramp):
            inp = replace(inp, pres_scale=ramp[index] * pressure)
        else:
            # beta ~ p / PHIEDGE^2 at fixed shape, so a flux rescale carries its pressure along
            measured = float(fixed.wout.betaxis if axis else fixed.wout.betatotal)
            flux = b0 * r0 / abs(float(fixed.wout.rbtor))
            inp = replace(inp, phiedge=float(inp.phiedge) * flux,
                          pres_scale=inp.pres_scale * flux**2 * (beta / measured if beta > 0 else 1.0))
        fixed = opt.solve_equilibrium(inp, initial_state=None if fixed is None else fixed.state, device=device,
                                      raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"PRES_SCALE = {inp.pres_scale:.6e} Pa, PHIEDGE = {float(inp.phiedge):.6f} Wb: betaxis = "
          f"{float(w.betaxis):.4%}, <beta> = {float(w.betatotal):.4%}, edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    import numpy as np
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0, b0 = float(inp.rbc[inp.ntor, 0]), P.B0
    t0 = P.REACTOR_T0 * (b0 / P.REACTOR_B0) ** (2 / 3) * (r0 / P.REACTOR_R0) ** (1 / 3)
    n0 = P.REACTOR_N0 * (b0 / P.REACTOR_B0) ** (4 / 3) * (P.REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, 0, np.asarray(P.REDL_SURFACES), n_lambda=P.REDL_N_LAMBDA)


def bootstrap_input(inp, beta, device):
    """``finite_beta_input`` with reactor-like kinetic profiles and their self-consistent Redl current.

    ne = n0 (1 - s^5) and Te = Ti = T0 (1 - s), so p = 2 e ne Te ~ (1 - s)(1 - s^5). n0 and T0
    start at the Helios-like reactor's collisionality nu* ~ n R / T^2 and beta ~ n T / B^2 moved
    to this R0 and B0, and follow the beta calibration as n ~ p^(2/3), T ~ p^(1/3), which keeps
    nu*. A Picard loop then makes the current Redl's, and it is resampled onto
    ``P.CURRENT_KNOTS`` spline knots. Above ``P.BOOTSTRAP_BETA_STEP`` beta is ramped in
    steps of at most that size, each step's pressure ramp carrying the previous step's
    bootstrap current, so its transform holds the equilibrium together as beta rises;
    with ``P.BOOTSTRAP_BETA_START`` the ramp first doubles beta from that value.
    Returns the input, the equilibrium and the Redl mismatch.
    """
    import math

    import numpy as np
    from vmex import optimize as opt
    from vmex.core.bootstrap import self_consistent_bootstrap

    ac = np.zeros_like(np.asarray(inp.ac, dtype=float))
    ac[0] = 1.0
    inp = replace(inp, ncurr=1, pcurr_type="power_series", ac=ac, curtor=0.0)
    steps = max(1, math.ceil(round(beta / P.BOOTSTRAP_BETA_STEP, 9)))
    betas = [beta * k / steps for k in range(1, steps + 1)]
    start = P.BOOTSTRAP_BETA_START
    while start is not None and start < betas[-steps]:  # double up to the first linear stage
        betas.insert(len(betas) - steps, start)
        start *= 2
    stages = len(betas)
    for stage, stage_beta in enumerate(betas, 1):
        inp, _ = finite_beta_input(inp, stage_beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
        profiles, redl = redl_profiles(inp)
        picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=P.PICARD_ITERATIONS, tol=P.PICARD_TOLERANCE,
                                           relax=P.PICARD_RELAX,
                                           degree=P.CURRENT_KNOTS - 1, s_eval=np.asarray(P.REDL_SURFACES),
                                           solve_kwargs=dict(device=device))
        inp = picard.input
        if stages > 1:
            print(f"bootstrap ramp {stage}/{stages}: beta {stage_beta:.4f}, CURTOR = "
                  f"{float(inp.curtor):.1f} A, Picard {picard.iterations} iterations (converged {picard.converged})",
                  flush=True)
    n0, t0 = float(profiles.ne_coeffs[0]), float(profiles.Te_coeffs[0])
    inp = opt.resample_current_profile(inp, P.CURRENT_KNOTS)
    fixed = opt.solve_equilibrium(inp, initial_state=picard.equilibrium.state, device=device,
                                  raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"Redl seed: n0 = {n0:.4e} 1/m^3, T0 = {t0:.1f} eV, Picard {picard.iterations} iterations "
          f"(converged {picard.converged}), CURTOR = {float(inp.curtor):.1f} A, mismatch = "
          f"{float(redl.total(w)):.3e}; <beta> = {float(w.betatotal):.4%}, iota = {float(np.min(np.abs(w.iotaf))):.4f}"
          f"..{float(np.max(np.abs(w.iotaf))):.4f}, edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed, redl


def max_abs_iota(state, runtime):
    """Largest |iota| over the half-mesh surfaces (axis excluded), the counterpart of ``opt.min_abs_iota``."""
    import jax.numpy as jnp
    from vmex.core.statephysics import _iotas_half

    return jnp.max(jnp.abs(_iotas_half(state, runtime)[1:]))


def bootstrap_mismatch(inp, redl):
    """(state, runtime) -> the bootstrap self-consistency mismatch held under ``P.REDL_TOLERANCE``.

    Redl's, or with ``P.BOOTSTRAP_MODEL = "dkx"`` the equilibrium's <j.B> against DKX's
    drift-kinetic one on ``P.DKX_SURFACES``, in Redl's normalized form (Redl assumes quasisymmetry).
    """
    if P.BOOTSTRAP_MODEL == "redl":
        return redl.total_state
    import numpy as np
    from dkx.bootstrap import KineticBootstrapMismatch

    kinetic = KineticBootstrapMismatch(redl_profiles(inp)[0], surfaces=P.DKX_SURFACES,
                                       collision_operator=P.DKX_COLLISION_OPERATOR,
                                       mboz=P.QI_OPTIONS["mboz"], nboz=P.QI_OPTIONS["nboz"])

    def mismatch(state, runtime):
        # DKX reads the radial grid on the host. Under jit it is a tracer, but it is
        # always linspace(0, 1, ns), so hand DKX that concrete grid.
        grid = np.linspace(0.0, 1.0, runtime.setup.s_full.shape[0])
        return kinetic.total(state, replace(runtime, setup=replace(runtime.setup, s_full=grid)))

    return mismatch


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""
    import numpy as np

    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import numpy as np
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    inp = replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False)
    if int(inp.ncurr) == 1:
        spline = "spline" in str(inp.pcurr_type)
        field_name, values = ("ac_aux_f", w.ac_aux_f) if spline else ("ac", w.ac)
        inp = replace(inp, curtor=float(w.ctor),
                      **{field_name: np.asarray(values, dtype=float)[: np.size(getattr(inp, field_name))]})
    return inp


def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


def scale_coil_currents(coils, rbtor):
    """Coils with every current scaled so the linked mu0 I / 2 pi (R B_phi at R = 1 m, Z = 0) is ``rbtor``."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils
    from essos.fields import BiotSavart

    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = jnp.asarray(np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    field = np.asarray(jax.vmap(BiotSavart(coils).B)(points))
    linked = abs(float(np.mean(-np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1])))
    return Coils(coils.curves, coils.dofs_currents_raw * (rbtor / linked), currents_scale=coils.currents_scale)


def coil_field(coils):
    """points (..., 3) -> the coils' Biot-Savart field B (..., 3)."""
    import jax
    from essos.fields import BiotSavart

    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


def weighted_rms(weights, values):
    """sqrt(sum weights values^2): the RMS for area weights that sum to one."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(weights * values**2))


def total_normal_field(interface, field):
    """(B_coils + B_plasma).n/|B| on a ``vmex.PlasmaVacuumInterface``, B_coils = ``field``."""
    import jax.numpy as jnp

    return interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of the coil-only B.n/|B| on an ESSOS surface."""
    import jax.numpy as jnp

    field = coil_field(coils)(surface.gamma)
    normal = jnp.sum(field * surface.unitnormal, axis=2) / jnp.linalg.norm(field, axis=2)
    return weighted_rms(surface.area_element / jnp.sum(surface.area_element), normal)


def boundary_diagnostics(wout, coils, nphi=61, ntheta=64):
    """Interface diagnostics of a WOUT in its coils' field; virtual casing is planned on this grid.

    Returns beta, the RMS/max of (B_coils + B_plasma).n/|B|, the RMS of the
    coil-only B.n/|B|, and the RMS of the pressure-balance residual
    (|B_out|^2 - |B_in|^2 - 2 mu0 p) / |B_in|^2.
    """
    import jax.numpy as jnp
    import vmex as vj

    field = coil_field(coils)
    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=nphi, ntheta=ntheta)
    weights = interface.weights
    total = total_normal_field(interface, field)
    coil_only = interface.external_Bn(field) / jnp.linalg.norm(interface.external_B(field), axis=0)
    balance = interface.pressure_balance_residual(field) / interface.Bin_mag2
    return dict(beta=float(wout.betatotal), normal_field_rms=float(weighted_rms(weights, total)),
                normal_field_max=float(jnp.max(jnp.abs(total))),
                coil_normal_field_rms=float(weighted_rms(weights, coil_only)),
                pressure_balance_rms=float(weighted_rms(weights, balance)))
