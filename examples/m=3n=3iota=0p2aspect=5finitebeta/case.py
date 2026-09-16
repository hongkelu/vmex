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
from vmex.core.wout import read_wout  # noqa: F401

N_BASE_COILS, NFP, FOURIER_ORDER, SOLVE_COIL_SEGMENTS = 4, 2, 4, 75
STELLSYM = True
CURRENT_GROUPS = (1, 2, 3)
PARAMETER_COUNT = 111
PARAMETER_SCALES = np.r_[
    np.full(3, 0.06), np.tile([0.002, 0.002, 0.002, 0.0005, 0.0005, 0.002 / 9, 0.002 / 9, 0.002 / 16, 0.002 / 16], 12)
]
CONSTRAINT_SCALES = np.array([0.005, 0.05, 0.01])
ROW_ORDER = ("qs_norm", "mean_iota", "aspect_ratio", "b0")


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


CASE = Path(__file__).resolve().parent
STATE_NAMES = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def load_case():
    from vmex.core.solver import SpectralState

    contract = json.loads((CASE / "case_contract.json").read_text())
    expected = json.loads((CASE / "inputs/manifest.json").read_text())
    for name, digest in expected.items():
        if sha256(CASE / "inputs" / name) != digest:
            raise ValueError("input hash mismatch: " + name)
    inp = VmecInput.from_json_text((CASE / "inputs/input.json").read_text())
    if (inp.mpol, inp.ntor, inp.nfp, inp.lasym, list(inp.ns_array)) != (3, 3, 2, False, [31]):
        raise ValueError("resolution/symmetry mismatch")
    raw = json.loads((CASE / "inputs/coils.json").read_text())
    curves, currents = np.asarray(raw["dofs_curves"]), np.asarray(raw["dofs_currents"])
    if (
        curves.shape != (4, 3, 9)
        or currents.shape != (4,)
        or not np.all(np.isfinite(curves))
        or not np.all(np.isfinite(currents))
    ):
        raise ValueError("invalid coil chart")
    if (raw["order"], raw["nfp"], raw["stellsym"]) != (4, 2, True):
        raise ValueError("coil topology mismatch")
    with np.load(CASE / "inputs/vacuum_warm_start.npz", allow_pickle=False) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    if str(data["schema_version"]) != "vmex.finitebeta-vacuum-guess/v1" or bool(data["eligible_for_resume"]):
        raise ValueError("finite-beta initialization requires an explicitly uncertified vacuum guess")
    if float(data["phiedge"]) != inp.phiedge:
        raise ValueError("warm-start signed flux mismatch")
    if data["parameters"].shape != (111,) or np.any(data["parameters"] != 0):
        raise ValueError("fresh case requires original zero parameters at step zero")
    if not np.array_equal(data["parameter_scales"], PARAMETER_SCALES):
        raise ValueError("parameter scale mismatch")
    for key in STATE_NAMES:
        if data[key].dtype != np.float64 or not np.all(np.isfinite(data[key])):
            raise ValueError("invalid seed state")
    return (
        inp,
        Full111FieldBuilder(jnp.asarray(curves), jnp.asarray(currents)),
        SpectralState(*(data[k] for k in STATE_NAMES)),
        contract,
        expected,
    )


def physical_rows(state, rt):
    return jnp.array(
        [
            statephysics.mean_iota(state, rt),
            statephysics.aspect_ratio(state, rt),
            statephysics.on_axis_magnetic_field(state, rt),
        ]
    )


def resolve_targets(initial_physical):
    values = np.asarray(initial_physical, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or values[2] == 0:
        raise ValueError("finite, nonzero initial signed B0 required")
    return np.array([0.2, 5.0, 5.1])


def make_rows(runtime, targets, loss_scale):
    qs = QuasisymmetryRatioResidual((0.25, 0.5, 0.75, 1.0), 1, 0)

    def rows(state):
        norm = jnp.linalg.norm(qs.residuals_state(state, runtime)) / loss_scale
        return jnp.r_[norm, (physical_rows(state, runtime) - targets) / CONSTRAINT_SCALES]

    return jax.jit(rows)


def metrics(values, targets, loss_scale):
    values = np.asarray(values)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid objective/constraint rows")
    physical = np.asarray(targets) + values[1:] * CONSTRAINT_SCALES
    return dict(
        qs_error_raw=float((values[0] * loss_scale) ** 2),
        qs_objective_normalized=float(0.5 * values[0] ** 2),
        physical=dict(zip(ROW_ORDER[1:], physical.tolist())),
        constraint_inf=float(np.max(np.abs(values[1:]))),
        target_errors=dict(zip(ROW_ORDER[1:], (physical - targets).tolist())),
    )
