"""Maximum-J objective term for the three-term single stage (``--extension``).

TERMS adds ``MAXJ_WEIGHT / 2 |r|^2`` with ``r`` the residual rows of :class:`vmex.core.maxj.MaximumJResidual`:
the bounce action J of the same trapped particles must not grow outward between neighbouring SURFACES. As in
``QI_maxJ_continuation.py`` (its first stage: target 0, weight 500), the pitches are held fixed: common trapped
pitches of the starting equilibrium (``MAXJ_WOUT``, its wout), at TRAPPING_DEPTHS of the shared trapping interval.

Environment: ``MAXJ_WOUT`` (required), ``MAXJ_WEIGHT`` (500), ``MAXJ_TARGET`` (0); a cheaper term with
``MAXJ_SURFACES`` (colon-separated s values) and ``MAXJ_NALPHA``, ``MAXJ_POINTS``, ``MAXJ_PERIODS``, ``MAXJ_WELLS``,
``MAXJ_QUADRATURE`` (field lines, points per period, periods, wells, quadrature order of the bounce actions).
"""

import os

import jax.numpy as jnp
import numpy as np

from vmex.core.maxj import MaximumJResidual, common_trapped_pitches
from vmex.core.optimize import boozer_modes_from_wout
from vmex.core.wout import read_wout

SURFACES = np.array([float(v) for v in os.environ.get("MAXJ_SURFACES", "0.20:0.35:0.50:0.65:0.80:0.90").split(":")])
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
ACTION = dict(nalpha=int(os.environ.get("MAXJ_NALPHA", 5)), points_per_period=int(os.environ.get("MAXJ_POINTS", 24)),
              num_periods=int(os.environ.get("MAXJ_PERIODS", 6)), max_wells=int(os.environ.get("MAXJ_WELLS", 16)),
              quadrature_order=int(os.environ.get("MAXJ_QUADRATURE", 16)))  # defaults: the example's COARSE_ACTION
MBOZ = 8
WEIGHT = float(os.environ.get("MAXJ_WEIGHT", 500.0))
TARGET = float(os.environ.get("MAXJ_TARGET", 0.0))


def starting_pitches(path, nalpha=9, points_per_period=64, num_periods=4):
    """``common_trapped_pitches_state`` on a wout: |B| along field lines theta = alpha + iota phi (Boozer)."""
    booz = boozer_modes_from_wout(read_wout(path), surfaces=SURFACES, mboz=12, nboz=12)
    alpha = 2.0 * np.pi * np.arange(nalpha) / nalpha
    phi = 2.0 * np.pi * np.arange(points_per_period * num_periods) / (int(booz["nfp"]) * points_per_period)
    iota = np.asarray(booz["iota_b"])
    theta = alpha[None, :, None] + iota[:, None, None] * phi[None, None, :]
    angle = theta[..., None] * np.asarray(booz["xm_b"]) - phi[None, None, :, None] * np.asarray(booz["xn_b"])
    bmag = (np.einsum("sapm,sm->sap", np.cos(angle), np.asarray(booz["bmnc_b"]))
            + np.einsum("sapm,sm->sap", np.sin(angle), np.asarray(booz["bmns_b"])))
    return np.asarray(common_trapped_pitches(jnp.asarray(np.swapaxes(bmag, 1, 2)), TRAPPING_DEPTHS))


PITCH = starting_pitches(os.environ["MAXJ_WOUT"])
print(f"[maxj] pitches 1/lambda = {np.round(1 / PITCH, 5).tolist()} T, weight {WEIGHT}, target {TARGET}, "
      f"surfaces {SURFACES.tolist()}, {ACTION}", flush=True)
_maxj = MaximumJResidual(SURFACES, PITCH, mboz=MBOZ, nboz=MBOZ, target=TARGET, **ACTION)

def maximum_j_rows(state, rt):
    """The residual rows with a pitch that is untrapped on a surface pair (NaN in MaximumJResidual) set to 0:
    the pitch drops out there instead of making the loss NaN. The NaN is a constant branch: the gradient stays
    finite."""
    rows = _maxj.residuals_state(state, rt)
    return jnp.where(jnp.isnan(rows), 0.0, rows)


ROWS = {}
TERMS = {"maximum_j": (maximum_j_rows, WEIGHT)}
