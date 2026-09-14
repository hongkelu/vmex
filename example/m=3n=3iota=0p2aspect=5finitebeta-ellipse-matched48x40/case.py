"""Authenticated M3/N3 inputs, original coil chart and physical rows."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from vmex.core.transforms import register_pytree_dataclass
from vmex.core import statephysics
from vmex.core.input import VmecInput
from vmex.core.optimize import QuasisymmetryRatioResidual
from vmex.core.wout import read_wout

N_BASE_COILS, NFP, FOURIER_ORDER, SOLVE_COIL_SEGMENTS = 4, 2, 4, 75
STELLSYM = True
CURRENT_GROUPS = (1, 2, 3)
PARAMETER_COUNT = 111
PARAMETER_SCALES = np.r_[np.full(3, .06), np.tile([.002,.002,.002,.0005,.0005,.002/9,.002/9,.002/16,.002/16],12)]
CONSTRAINT_SCALES = np.array([.005,.05,.01])
ROW_ORDER = ('qs_norm','mean_iota','aspect_ratio','b0')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Exact JAX-traceable filament field used by the qualified anchor."""

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement)
        differential = differential * inv_radius3[..., None]
        point_ndim = xyz.ndim - 1
        current_shape = (1,) * point_ndim + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]

register_pytree_dataclass(DirectCoilField)

@dataclass(frozen=True, eq=False)
class Full111FieldBuilder:
    """Three relative currents plus every order-4 Cartesian coefficient."""

    nominal_dofs_curves: Any
    nominal_base_currents: Any

    def _parameters(self, parameters: Any):
        values = jnp.asarray(parameters)
        if values.shape != (PARAMETER_COUNT,):
            raise ValueError(f"full111 parameters must have shape ({PARAMETER_COUNT},)")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("full111 parameters must use a floating dtype")
        return values.astype(jnp.asarray(self.nominal_dofs_curves).dtype)

    def base_currents_at(self, parameters: Any):
        values = self._parameters(parameters)
        nominal = jnp.asarray(self.nominal_base_currents)
        varied = nominal
        for local_index, group in enumerate(CURRENT_GROUPS):
            varied = varied.at[group].add(values[local_index] * nominal[group])
        return varied

    def curve_dofs_at(self, parameters: Any):
        values = self._parameters(parameters)
        displacement = jnp.reshape(values[len(CURRENT_GROUPS) :], jnp.shape(self.nominal_dofs_curves))
        return jnp.asarray(self.nominal_dofs_curves) + displacement

    def __call__(self, parameters: Any) -> DirectCoilField:
        from essos.coils import Coils, Curves

        curves = Curves(
            self.curve_dofs_at(parameters),
            SOLVE_COIL_SEGMENTS,
            NFP,
            STELLSYM,
        )
        coils = Coils(curves, self.base_currents_at(parameters))
        return DirectCoilField(
            gamma=jnp.asarray(coils.gamma),
            gamma_dash=jnp.asarray(coils.gamma_dash),
            currents=jnp.asarray(coils.currents),
        )

