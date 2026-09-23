"""Increase Fourier order without changing the seed's physical curves or currents."""
import jax.numpy as jnp
from essos.coils import Coils, Curves


def resize_coils(coils, order, n_segments):
    """Zero-pad higher modes, preserve scaling, and choose field quadrature.

    Refuse truncation: lowering order requires a separate, explicitly fitted seed.
    """
    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} seed to order {order}")
    if order == coils.order and n_segments == coils.n_segments:
        return coils
    old = coils.curves
    raw = old.dofs / old.scaling
    raw = jnp.pad(raw, ((0, 0), (0, 0), (0, 2*(order-coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor,
                    scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)
