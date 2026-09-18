import hashlib  # noqa: F401
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location("angular_" + name, CASE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


case = module("case")
checkpoint = module("checkpoint")


def test_refined_initial_field_does_not_redefine_signed_b0_target():
    entry = module("single_stage_free_boundary_optimization")
    np.testing.assert_array_equal([entry.IOTA_TARGET, entry.ASPECT_TARGET, entry.B0_TARGET],
                                  [0.2, 5.0, -0.17506474574437714])


def test_explicit_grid_and_unchanged_spectral_seed():
    inp, builder, state, contract, hashes = case.load_case()
    assert (inp.ntheta, inp.nzeta, inp.mpol, inp.ntor, list(inp.ns_array)) == (48, 40, 3, 3, [31])
    assert contract["initial_equilibrium_solves"] == 1 and not contract["root_polishing"]
    assert hashes["seed_checkpoint.npz"] == "bbc25674652973ccdb2f0e1305216af6016a84c6a4ce9775afe6ca479290d9f1"


@pytest.mark.parametrize("shape,valid", [((31, 25, 40), True), ((31, 7, 10), False)])
def test_checkpoint_cannot_mix_coarse_and_refined_baselines(tmp_path, shape, valid):
    inp, builder, state, contract, hashes = case.load_case()
    digest = case.sha256(CASE / "case_contract.json")
    data = dict(
        schema_version=np.asarray("vmex.iota02-aspect5-accepted/v1"),
        accepted_step=np.asarray(0),
        parameters=np.zeros(111),
        parameter_scales=case.PARAMETER_SCALES,
        targets=np.array([0.2, 5.0, -0.17506474574437714]),
        constraint_scales=case.CONSTRAINT_SCALES,
        loss_scale=np.asarray(0.3),
        phiedge=np.asarray(inp.phiedge),
        rcon0=np.zeros(shape),
        zcon0=np.zeros(shape),
        contract_sha256=np.asarray(digest),
        provenance_json=np.asarray(json.dumps({"input_hashes": hashes, "contract": contract})),
    )
    for n in case.STATE_NAMES:
        data[n] = np.asarray(getattr(state, n))
        data["mask_" + n] = np.ones((31, 18))
    path = tmp_path / "checkpoint.npz"
    np.savez_compressed(path, **data)
    kw = dict(
        contract_sha256=digest,
        input_hashes=hashes,
        parameter_scales=case.PARAMETER_SCALES,
        constraint_scales=case.CONSTRAINT_SCALES,
        phiedge=inp.phiedge,
        target_step=3,
    )
    if valid:
        assert checkpoint.load_checkpoint(path, case.sha256(path), **kw)["rcon0"].shape == shape
    else:
        with pytest.raises(ValueError, match="rcon0"):
            checkpoint.load_checkpoint(path, case.sha256(path), **kw)
