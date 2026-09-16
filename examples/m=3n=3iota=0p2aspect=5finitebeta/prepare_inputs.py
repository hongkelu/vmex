"""Reproduce the field-scaled rotating-ellipse inputs without solving VMEX."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

CASE = Path(__file__).resolve().parent
PARENT = CASE.parent / 'm=3n=3iota=0p2aspect=5fresh'
REFERENCE = CASE.parents[1] / 'examples/data/input.nfp2_QA_finite_beta'
ORIGINAL_B0_T = -0.17506474574437714
TARGET_B0_T = 5.1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_profiles(text):
    """Read the explicit single-line assignments in this reference deck.

    This is deliberately not a general Fortran namelist parser. Missing or
    duplicate assignments fail; VMEX's parser independently checks the result.
    """
    def value(name):
        found = re.findall(r'^\s*' + name + r'\s*=\s*([^!\n]+)', text, re.M | re.I)
        if len(found) != 1:
            raise ValueError(f'expected one reference assignment for {name}')
        return found[0].strip()

    result = {name: value(name).strip('"\'').lower()
              for name in ('pmass_type', 'pcurr_type')}
    for name in ('am', 'ac', 'ac_aux_s', 'ac_aux_f'):
        result[name] = [float(x.replace('D', 'e').replace('d', 'e'))
                        for x in value(name).replace(',', ' ').split()]
    result.update(pres_scale=float(value('pres_scale')),
                  curtor=float(value('curtor')), ncurr=int(value('ncurr')),
                  gamma=0.0, am_aux_s=[], am_aux_f=[])
    return result


def build_inputs():
    expected = json.loads((PARENT / 'inputs/manifest.json').read_text())
    for name, expected_hash in expected.items():
        if digest(PARENT / 'inputs' / name) != expected_hash:
            raise ValueError('parent input hash mismatch: ' + name)
    original = json.loads((PARENT / 'inputs/input.json').read_text())
    old_coils = json.loads((PARENT / 'inputs/coils.json').read_text())
    factor = TARGET_B0_T / ORIGINAL_B0_T
    inp, coils = deepcopy(original), deepcopy(old_coils)
    # Keep all length coordinates fixed. Signed field reversal is intentional.
    coils['dofs_currents'] = [factor * i for i in old_coils['dofs_currents']]
    inp['phiedge'] = factor * original['phiedge']
    inp['extcur'] = [factor * i for i in original['extcur']]
    # These are the reference plasma amplitudes, not the vacuum field scaling.
    inp.update(reference_profiles(REFERENCE.read_text()))
    metadata = dict(
        status='prepared inputs; no finite-beta equilibrium certified',
        reference_input=str(REFERENCE.relative_to(CASE.parents[1])),
        reference_input_sha256=digest(REFERENCE), parent_input_sha256=expected,
        original_b0_T=ORIGINAL_B0_T, target_b0_T=TARGET_B0_T,
        signed_coil_current_and_flux_factor=factor, length_factor=1.0,
        original_phiedge_Wb=original['phiedge'], phiedge_Wb=inp['phiedge'],
        original_base_coil_currents_A=old_coils['dofs_currents'],
        base_coil_currents_A=coils['dofs_currents'],
        central_pressure_Pa=inp['pres_scale'] * inp['am'][0],
        total_plasma_current_A=inp['curtor'],
        field_scaling_claim='5.1 T applies to the uniformly scaled vacuum starting state; finite-beta B0 must be solved and measured',
        plasma_profile_scaling='Reference pressure and plasma current copied exactly; no extra B-scale or length-scale multiplier',
        lambda_sign_factor=-1.0,
        warm_start='vacuum_warm_start.npz contains unchanged R/Z and sign-reversed internal lambda for the scaled vacuum guess; it is not an accepted finite-beta checkpoint',
    )
    return inp, coils, metadata


def main():
    import numpy as np
    inp, coils, metadata = build_inputs()
    output = CASE / 'inputs'
    if not output.resolve().is_relative_to(CASE):
        raise ValueError('input output must stay inside this case')
    output.mkdir(exist_ok=False)
    for name, obj in [('input.json', inp), ('coils.json', coils),
                      ('scaling.json', metadata)]:
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    (output / 'original_vacuum_seed.npz').write_bytes(
        (PARENT / 'inputs/seed_checkpoint.npz').read_bytes())
    with np.load(output / 'original_vacuum_seed.npz', allow_pickle=False) as data:
        guess = {name: data[name].copy() for name in
                 ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin',
                  'parameters', 'parameter_scales')}
    for name in ('L_cos', 'L_sin'):
        # Internal lambda enters B multiplied by nonnegative lamscale ~ |flux|.
        # Its sign must reverse with the signed flux to preserve physical lambda.
        guess[name] *= metadata['lambda_sign_factor']
    guess.update(schema_version=np.asarray('vmex.finitebeta-vacuum-guess/v1'),
                 eligible_for_resume=np.asarray(False),
                 phiedge=np.asarray(inp['phiedge']),
                 source_seed_sha256=np.asarray(metadata['parent_input_sha256']['seed_checkpoint.npz']))
    with (output / 'vacuum_warm_start.npz').open('xb') as stream:
        np.savez_compressed(stream, **guess)
    (output / 'manifest.json').write_text(json.dumps(
        {p.name: digest(p) for p in sorted(output.iterdir())}, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
