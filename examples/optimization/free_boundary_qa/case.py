"""Authenticated M3/N3 inputs, original coil chart and physical rows."""

import hashlib
import json
from pathlib import Path
import numpy as np
from vmex.core.input import VmecInput

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


from vmex.core.coil_parameters import CoilParameters


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
    if (inp.ntheta, inp.nzeta) != (48, 40):
        raise ValueError("explicit angular grid must be 48x40")
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
    with np.load(CASE / "inputs/seed_checkpoint.npz", allow_pickle=False) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    if int(data["accepted_step"]) != 0 or data["parameters"].shape != (111,) or np.any(data["parameters"] != 0):
        raise ValueError("fresh case requires original zero parameters at step zero")
    if not np.array_equal(data["parameter_scales"], PARAMETER_SCALES):
        raise ValueError("parameter scale mismatch")
    for key in STATE_NAMES:
        if data[key].dtype != np.float64 or not np.all(np.isfinite(data[key])):
            raise ValueError("invalid seed state")
    return (
        inp,
        CoilParameters(curves, currents, current_dofs=CURRENT_GROUPS, max_coil_mode=FOURIER_ORDER,
                       nfp=NFP, stellsym=STELLSYM, n_segments=SOLVE_COIL_SEGMENTS, scales=PARAMETER_SCALES),
        SpectralState(*(data[k] for k in STATE_NAMES)),
        contract,
        expected,
    )


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


def sampled_displacement(delta):
    """Evaluate the proposed coil displacement at the 75 diagnostic angles."""
    delta = np.asarray(delta)
    if delta.shape != (111,) or not np.all(np.isfinite(delta)):
        raise ValueError('expected finite 111-vector')
    t = np.arange(75)/75
    basis = np.ones((75, 9))
    for k in range(1, 5):
        basis[:, 2*k-1] = np.sin(2*np.pi*k*t)
        basis[:, 2*k] = np.cos(2*np.pi*k*t)
    return np.einsum('cdk,sk->csd', delta[3:].reshape(4, 3, 9), basis)
