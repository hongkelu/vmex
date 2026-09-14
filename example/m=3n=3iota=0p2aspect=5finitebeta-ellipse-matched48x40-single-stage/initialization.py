"""Ordinary finite-beta initialization; a vacuum guess is never an accepted root."""
import hashlib
import json
from pathlib import Path
import time


def verify_runtime():
    import vmex
    case = Path(__file__).resolve().parent
    manifest = json.loads((case / 'runtime_manifest.json').read_text())
    runtime = Path(vmex.__file__).resolve().parent
    for name, expected in manifest['files'].items():
        path = runtime / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('VMEX runtime differs from pinned commit ' + manifest['commit'] + ': ' + name)
    return runtime



def initial_baselines():
    # Fixed-boundary seed has no free-boundary continuation baselines.
    return None, None


def ordinary_initial_state(inp, builder, parameters, seed, solver, contract,
                           event, deadline, output):
    """One ordinary accuracy refinement; no root polish or same-coil retries."""
    import numpy as np
    import jax.numpy as jnp
    from vmex.core.freeboundary import _solve_free_boundary_stage
    from vmex.core.errors import VmecError
    if time.monotonic() >= deadline:
        raise TimeoutError('walltime before ordinary initialization')
    event('ordinary_initialization_start',forward_solve_tolerance=contract['forward_solve_tolerance'],
          force_acceptance_tolerance=contract['force_tolerance'],root_polishing=False)
    stage=_solve_free_boundary_stage(inp,external_field=builder(jnp.asarray(parameters)),
        resolution=solver.resolution,ftol=contract['forward_solve_tolerance'],
        max_iterations=contract['max_iterations'],initial_state=seed,verbose=True,
        constraint_continuation=None,
        include_edge_in_convergence=False,error_on_no_convergence=False,
        jacobian_retries=0,allow_initial_axis_reguess=False,use_fft=False)
    forces={name:float(getattr(stage.result,name)) for name in ('fsqr','fsqz','fsql','fedge')}
    event('ordinary_initialization_result',converged=bool(stage.result.converged),
          iterations=int(stage.result.iterations),forces=forces)
    with (Path(output)/'ordinary_initial_state_uncertified.npz').open('xb') as f:
        np.savez_compressed(f,eligible_for_resume=np.asarray(False),parameters=np.asarray(parameters),
            rcon0=np.asarray(stage.rcon0),zcon0=np.asarray(stage.zcon0),
            **{n:np.asarray(getattr(stage.result.state,n)) for n in
               ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')})
    if time.monotonic() >= deadline:
        raise TimeoutError('walltime during ordinary initialization')
    if not stage.result.converged or any(not np.isfinite(forces[n]) or not 0<=forces[n]<=contract['force_tolerance'] for n in ('fsqr','fsqz','fsql')):
        raise VmecError('ordinary initialization failed; no root polish or same-coil retry')
    return stage
