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
CONSTRAINT_SCALES = np.array([.005,.01,.01,.01])
ROW_ORDER = ('qs_norm','mean_iota','major_radius','minor_radius','b0')

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

def load_case(payload_path, expected_sha):
    if sha256(payload_path) != expected_sha:
        raise ValueError("payload authentication failed")
    payload = json.loads(Path(payload_path).read_text())
    if payload['case_id'] != 'rotating_ellipse_qa1e-1_mpol3_ntor3_ns31_maxmode1_full111':
        raise ValueError("case identity mismatch")
    paths = {}
    for name, binding in payload['artifacts'].items():
        path = Path(binding['path'])
        if sha256(path) != binding['sha256']:
            raise ValueError(f"artifact authentication failed: {name}")
        paths[name] = path
    inp = VmecInput.from_json_text(paths['input'].read_text())
    if (inp.mpol,inp.ntor,inp.nfp,inp.lasym,list(inp.ns_array)) != (3,3,2,False,[31]):
        raise ValueError("resolution or symmetry mismatch")
    raw = json.loads(paths['coils'].read_text())
    curves, currents = np.asarray(raw['dofs_curves']), np.asarray(raw['dofs_currents'])
    if curves.shape != (4,3,9) or currents.shape != (4,) or not np.all(np.isfinite(curves)) or not np.all(np.isfinite(currents)):
        raise ValueError("coil chart mismatch")
    if (raw['order'],raw['nfp'],raw['stellsym'],raw['n_segments']) != (4,2,True,160):
        raise ValueError("coil topology mismatch")
    builder = Full111FieldBuilder(jnp.asarray(curves),jnp.asarray(currents))
    w = read_wout(paths['wout'])
    expected = payload['expected']['metrics']
    targets = np.array([expected['mean_iota'],expected['major_radius'],float(w.Aminor_p),float(w.b0)])
    return inp, builder, targets, float(expected['loss_scale']), payload


def make_rows(runtime, targets, loss_scale):
    qs = QuasisymmetryRatioResidual((.25,.5,.75,1.),1,0)
    def rows(state):
        norm = jnp.linalg.norm(qs.residuals_state(state,runtime))/loss_scale
        physical = jnp.array([statephysics.mean_iota(state,runtime),
            statephysics.major_radius(state,runtime),statephysics.minor_radius(state,runtime),
            statephysics.on_axis_magnetic_field(state,runtime)])
        return jnp.r_[norm,(physical-targets)/CONSTRAINT_SCALES]
    return jax.jit(rows)


def metrics(values, targets, loss_scale):
    values = np.asarray(values)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite objective rows")
    return dict(qs_error_raw=float((values[0]*loss_scale)**2),
        qs_objective_normalized=float(.5*values[0]**2),
        physical=dict(zip(ROW_ORDER[1:],(targets+values[1:]*CONSTRAINT_SCALES).tolist())),
        constraint_inf=float(np.max(np.abs(values[1:]))))
