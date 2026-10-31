"""Effective-ripple diagnostics through the optional NEO_JAX backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "EffectiveRippleResidual",
    "diagnostic_neo_config",
    "objective_neo_config",
    "epsilon_effective_from_boozer",
    "epsilon_effective_from_wout",
]


def _neo_imports():
    from .._compat import require_optional

    require_optional("neo-jax", "effective ripple")
    try:
        from neo_jax import NeoConfig, run_neo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "effective ripple requires NEO_JAX; install VMEX with "
            "`pip install vmex[neoclassical]`") from exc
    return NeoConfig, run_neo


def diagnostic_neo_config():
    """Return the bounded NEO resolution used by VMEX summary figures.

    On the bundled NFP=2 QA and NFP=4 QI finite-beta decks this stays within
    4% of NEO_JAX's default ``NeoConfig``.  A 16 x 16 spline grid aliases the
    summary Boozer spectrum (``mboz=16``) and fewer than 20 steps per field
    period or 200 field periods leave 7-30% errors.  Pass a
    ``neo_jax.NeoConfig`` to :func:`epsilon_effective_from_wout` for
    publication calculations.
    """
    NeoConfig, _ = _neo_imports()
    return NeoConfig(
        theta_n=32, phi_n=32, npart=24, multra=1, no_bins=50,
        nstep_per=20, nstep_min=200, nstep_max=500, acc_req=0.02,
        max_rational_field_periods=100000,
    )


def objective_neo_config():
    """Return the NEO resolution of :class:`EffectiveRippleResidual`.

    The diagnostic resolution on a 64 x 64 spline grid (the objective's
    ``mboz = nboz = 16`` spectrum), without the rational-surface work guard:
    the guard reads iota as a Python float, which a traced state does not have.
    """
    from dataclasses import replace

    return replace(diagnostic_neo_config(), theta_n=64, phi_n=64, max_rational_field_periods=0)


def _neo_boozer_data(bx, indices):
    """Share one mode-first Boozer transform with NEO, including sine partners."""
    booz = {
        "nfp_b": int(bx.nfp), "ns_b": len(indices), "mode_first": True,
        "asym": bool(getattr(bx, "asym", False)),
        "ixm_b": np.asarray(bx.xm_b), "ixn_b": np.asarray(bx.xn_b),
        "iota_b": np.asarray(bx.iota)[indices],
        "buco_b": np.asarray(bx.Boozer_I), "bvco_b": np.asarray(bx.Boozer_G),
        "rmnc_b": np.asarray(bx.rmnc_b), "zmns_b": np.asarray(bx.zmns_b),
        "pmns_b": -np.asarray(bx.numns_b), "bmnc_b": np.asarray(bx.bmnc_b),
        "s_b": np.asarray(bx.s_b),
    }
    if booz["asym"]:
        booz.update(rmns_b=np.asarray(bx.rmns_b), zmnc_b=np.asarray(bx.zmnc_b),
                    pmnc_b=-np.asarray(bx.numnc_b), bmns_b=np.asarray(bx.bmns_b))
    return booz


def epsilon_effective_from_boozer(booz: Any, *, config=None):
    r"""Return ``(s, epsilon_eff**(3/2))`` from Boozer-coordinate data.

    ``booz`` may be a NEO_JAX ``BoozerData`` or a booz_xform-style mapping.
    The returned NEO quantity is the effective-ripple transport measure
    :math:`\epsilon_\mathrm{eff}^{3/2}`, conventionally named ``epstot``.
    JAX arrays remain differentiable when the mapping and selected NEO path
    are JAX-native.
    """
    NeoConfig, run_neo = _neo_imports()
    cfg = NeoConfig() if config is None else config
    result = run_neo(
        booz, config=cfg, use_jax=True, progress=False, jax_surface_scan=True)
    values = getattr(result, "eps_eff", None)
    if values is None:
        values = result.epsilon_effective
    if hasattr(booz, "es"):
        surfaces = booz.es
    elif isinstance(booz, dict) and "s_b" in booz:
        surfaces = booz["s_b"]
    else:
        surfaces = np.arange(np.asarray(values).size, dtype=float)
    return surfaces, values


def epsilon_effective_from_wout(
    wout,
    *,
    surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95),
    mboz: int = 16,
    nboz: int = 12,
    config=None,
    clear_jax_caches: bool = False,
):
    """Compute ``epsilon_eff**(3/2)`` directly from an in-memory VMEX wout.

    The same in-memory Boozer transform feeds NEO, including all LASYM
    partners. Caches remain available unless ``clear_jax_caches=True``.
    """
    if clear_jax_caches:
        import gc

        import jax

        jax.clear_caches(); gc.collect()
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    indices = sorted({int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces})
    bx.compute_surfs = indices
    bx.run()
    booz = _neo_boozer_data(bx, indices)
    return epsilon_effective_from_boozer(booz, config=config)


class EffectiveRippleResidual:
    r"""Effective ripple of a live equilibrium as least-squares rows ``sqrt(w_s) eps_eff,s^(3/2) / reference``.

    One traceable chain: :func:`~vmex.core.omnigenity.boozer_spectrum_state`
    with its geometry, then NEO_JAX's surface scan (``run_neo_from_boozer_jax``),
    whose field-line integration is a fixed-length ``lax.scan`` and so
    reverse-mode differentiable. ``total`` is ``sum_s w_s (eps_eff,s^(3/2) / reference)^2``
    with the weights normalized to sum 1 (equal by default). The requested surfaces snap
    to the nearest half-mesh ones (``s_b``). The derivative holds the bounce-well
    topology and NEO's rational-surface step count fixed.
    """

    name = "epsilon_effective"

    def __init__(self, surfaces, *, weights=None, reference=1.0, mboz=16, nboz=16, oversample=2, config=None):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        weights = np.ones(self.surfaces.size) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != self.surfaces.shape or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be nonnegative, one per surface, and not all zero")
        self.weights = weights / weights.sum()
        if not reference > 0:
            raise ValueError("reference must be positive")
        self.reference = float(reference)
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        self.config = objective_neo_config() if config is None else config

    def compute_state(self, state, rt):
        """Return the rows, their total, ``epsilon_effective`` (eps_eff^(3/2)) and ``s_b``."""
        import jax.numpy as jnp
        from neo_jax.driver import run_neo_from_boozer_jax
        from neo_jax.io import booz_xform_to_boozerdata_jax

        from .omnigenity import boozer_spectrum_state

        _neo_imports()
        booz = boozer_spectrum_state(state, rt, surfaces=self.surfaces, mboz=self.mboz, nboz=self.nboz,
                                     oversample=self.oversample, geometry=True)
        lasym = bool(rt.setup.lasym)
        names = ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b") + (("rmns_b", "zmnc_b", "pmnc_b", "bmns_b") if lasym else ())
        data = booz_xform_to_boozerdata_jax(
            {"nfp_b": booz["nfp"], "ixm_b": booz["xm_b"], "ixn_b": booz["xn_b"], "s_b": booz["s_b"],
             "iota_b": booz["iota_b"], "buco_b": booz["I_b"], "bvco_b": booz["G_b"],
             **{name: booz[name] for name in names}},
            mode_first=False, asym_override=lasym, mode_indices=np.arange(booz["bmnc_b"].shape[1]))
        out = run_neo_from_boozer_jax(data, self.config.to_control(), skip_fourier_mask=True,
                                      max_rational_field_periods=0,
                                      rational_surface_policy=self.config.rational_surface_policy)
        rows = jnp.sqrt(jnp.asarray(self.weights)) * out.eps_eff / self.reference
        return {"residuals1d": rows, "total": jnp.vdot(rows, rows), "epsilon_effective": out.eps_eff,
                "s_b": booz["s_b"]}

    def residuals_state(self, state, rt):
        """Return the flat traceable least-squares rows."""
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        """Return the sum of squared rows on a live state."""
        return self.compute_state(state, rt)["total"]

    def residuals(self, eq):
        """Evaluate on a converged equilibrium."""
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = residuals

    def total(self, eq):
        """Return the scalar objective on a converged equilibrium."""
        return self.total_state(eq.state, eq.runtime)
