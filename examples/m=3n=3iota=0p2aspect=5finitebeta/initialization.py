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


def ordinary_initial_state(inp, builder, parameters, seed, runtime, solver,
                           contract, event, deadline, output):
    import jax.numpy as jnp
    import numpy as np
    from vmex.core.freeboundary import _solve_free_boundary_stage

    if time.monotonic() >= deadline:
        raise TimeoutError('walltime before finite-beta initialization')
    event('ordinary_initialization_start',
          phiedge_Wb=float(inp.phiedge), central_pressure_Pa=float(inp.pres_scale * inp.am[0]),
          total_plasma_current_A=float(inp.curtor), root_polishing=False,
          warm_start_is_accepted_equilibrium=False)
    stage = _solve_free_boundary_stage(
        inp, external_field=builder(jnp.asarray(parameters)),
        resolution=solver.resolution, ftol=contract['force_tolerance'],
        max_iterations=contract['max_iterations'], initial_state=seed,
        constraint_continuation=(jnp.zeros_like(runtime.rcon0), jnp.zeros_like(runtime.zcon0)),
        include_edge_in_convergence=True,
        edge_force_tolerance=contract['edge_force_tolerance'],
        error_on_no_convergence=False, jacobian_retries=0,
        allow_initial_axis_reguess=False, use_fft=False,
    )
    forces = {name: float(getattr(stage.result, name))
              for name in ('fsqr', 'fsqz', 'fsql', 'fedge')}
    event('ordinary_initialization_result', converged=bool(stage.result.converged),
          iterations=int(stage.result.iterations), forces=forces)
    # Preserve the result, including failures, without claiming certification.
    with (Path(output) / 'ordinary_initial_state_uncertified.npz').open('xb') as stream:
        np.savez_compressed(
            stream, eligible_for_resume=np.asarray(False),
            parameters=np.asarray(parameters), rcon0=np.asarray(stage.rcon0),
            zcon0=np.asarray(stage.zcon0),
            **{name: np.asarray(getattr(stage.result.state, name))
               for name in ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')})
    if time.monotonic() >= deadline:
        raise TimeoutError('walltime during finite-beta initialization')
    limits = {name: contract['edge_force_tolerance'] if name == 'fedge'
              else contract['force_tolerance'] for name in forces}
    if not bool(stage.result.converged) or any(
            not np.isfinite(value) or value < 0.0 or value > limits[name]
            for name, value in forces.items()):
        raise RuntimeError('ordinary finite-beta initialization failed its force gate; no root polish or same-coil retry')
    return stage  # The caller must still perform fresh force/root certification.
