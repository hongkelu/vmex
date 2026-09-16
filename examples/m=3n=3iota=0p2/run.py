#!/usr/bin/env python3
"""Strict main-native 1-mm optimization from an existing accepted state."""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import signal
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload',type=Path,required=True)
    parser.add_argument('--payload-sha256',required=True)
    parser.add_argument('--resume-checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--target-step',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-wall-hours',type=float,default=4.)
    parser.add_argument('--checkpoint-every',type=int,default=1,
        help='Save at absolute step multiples, plus the initial and final accepted state.')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--validate-tangent-only',action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_hours) or args.max_wall_hours <= 0:
        parser.error('walltime must be finite and positive')
    if args.checkpoint_every < 1:
        parser.error('checkpoint interval must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for variable, name in [('JAX_COMPILATION_CACHE_DIR','jax'),('MPLCONFIGDIR','mpl'),
                           ('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda')]:
        path = output/'cache'/name
        path.mkdir(parents=True)
        os.environ[variable] = str(path)
    os.environ['JAX_ENABLE_X64'] = 'True'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import solvax
    from vmex.core import implicit as im, freeboundary_implicit as fbi
    from vmex.core import freeboundary_continuation as fc
    from vmex.core.freeboundary import _solve_free_boundary_stage
    from vmex.core.wout import wout_from_state, write_wout
    from case import load_case, make_rows, metrics, sha256, PARAMETER_SCALES
    from optimization import proposal, displacement
    from checkpoint import load, save, write_json, STATE_NAMES
    def stop(signum, frame):
        raise TimeoutError(f"received signal {signum}")
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    started = time.monotonic()
    deadline = started+args.max_wall_hours*3600
    accepted = cfg = rows = None
    events = []
    step = initial_step = None
    status = 'initializing'
    saved_step = None
    def event(phase, **values):
        record = dict(phase=phase,elapsed_s=time.monotonic()-started,**values)
        events.append(record)
        with (output/'progress.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    provenance = dict(jax=jax.__version__,solvax=solvax.__version__,
        payload_sha256=args.payload_sha256,checkpoint_sha256=args.checkpoint_sha256,
        target_step=args.target_step,checkpoint_every=args.checkpoint_every,
        max_wall_hours=args.max_wall_hours,root_polishing=False,equilibrium_recompute=False,
        force_tolerance=1e-14,projected_root_atol=2e-6,maximum_coil_step_m=.001)
    try:
        inp,builder,targets,loss_scale,payload = load_case(args.payload,args.payload_sha256)
        checkpoint = load(args.resume_checkpoint,args.checkpoint_sha256,targets)
        step = initial_step = checkpoint['step']
        if args.target_step < step or (args.target_step == step and not args.validate_only):
            raise ValueError('target must exceed the loaded absolute step')
        device = jax.devices('gpu' if args.device.startswith('cuda') else 'cpu')[0]
        import inspect, sys
        from essos.coils import Coils
        provenance['python'] = sys.version
        provenance['essos_coils_sha256'] = sha256(inspect.getsourcefile(Coils))
        provenance['device'] = str(device)
        provenance['initial_step'] = step
        provenance['requested_promotions'] = args.target_step-step
        source = Path(__file__).resolve().parents[2]
        provenance['source_sha256'] = {str(p.relative_to(source)):sha256(p)
            for folder in (source/'vmex',Path(__file__).resolve().parent)
            for p in folder.rglob('*.py') if 'runs' not in p.parts}
        write_json(output/'manifest.json',provenance)
        solver = fbi.make_free_boundary_config(inp,builder(jnp.asarray(checkpoint['parameters'])),
            field_from_parameters=builder,device=device,ftol=1e-14,max_iterations=12000,
            adjoint_solver='reverse_gcrot',adjoint_tol=2e-5,adjoint_residual_rtol=2e-5,
            adjoint_gcrot_m=30,adjoint_gcrot_k=10,adjoint_maxiter=100,adjoint_fail='error',
            include_edge_in_convergence=True,edge_force_tolerance=1e-14)
        params = im.params_from_input(inp)
        rt = im.runtime_from_params(params,solver.implicit)
        if checkpoint['legacy']:
            # The hash-pinned historical accepted-state evaluator defines zero
            # baselines explicitly; this is not a guess or a relaxation.
            rcon0,zcon0 = jnp.zeros_like(rt.rcon0),jnp.zeros_like(rt.zcon0)
        else:
            rcon0,zcon0 = checkpoint['data']['rcon0'],checkpoint['data']['zcon0']
        cfg = fc.make_free_boundary_continuation_config_from_state(solver,params,
            checkpoint['parameters'],state=checkpoint['state'],rcon0=rcon0,zcon0=zcon0,
            parameter_scales=PARAMETER_SCALES,continuation_step=.1,
            max_continuation_steps=64,root_residual_atol=2e-6)
        imported = cfg._anchor
        for n in STATE_NAMES:
            if not np.array_equal(np.asarray(getattr(imported.state,n)),getattr(checkpoint['state'],n)):
                raise RuntimeError('checkpoint restoration changed the state')
            if not checkpoint['legacy'] and not np.array_equal(
                    np.asarray(getattr(imported.dof_mask,n)),checkpoint['data']['mask_'+n]):
                raise RuntimeError('checkpoint active mask changed')
        rows = make_rows(rt,jnp.asarray(targets),loss_scale)
        accepted = imported
        values = np.asarray(rows(accepted.state))
        initial_metrics = metrics(values,targets,loss_scale)
        event('restored',absolute_step=step,segment_step=0,state_preserved_exactly=True,
            root_residual=accepted.root_residual_norm,forces={n:float(getattr(accepted.result,n))
            for n in ('fsqr','fsqz','fsql','fedge')},metrics=initial_metrics)
        save(output,accepted,step,targets,float(checkpoint['data']['trust_radius']),provenance)
        saved_step = step
        if args.validate_tangent_only:
            solver=dataclasses.replace(solver,implicit=dataclasses.replace(
                solver.implicit,adjoint_tol=2e-6),adjoint_residual_rtol=2e-6)
            direction=jnp.zeros(111,dtype=jnp.float64).at[3].set(1e-6)
            diagnostics=[]
            tangent=fbi.free_boundary_state_tangent(cfg.params,jnp.asarray(accepted.parameters),
                solver,accepted.state,accepted.dof_mask,direction,
                rcon0=accepted.rcon0,zcon0=accepted.zcon0,diagnostics=diagnostics)
            event('tangent_validated',rows=diagnostics,
                tangent_norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(tangent)))),
                maximum_direction_m=float(np.max(np.linalg.norm(displacement(np.asarray(direction)),axis=-1))))
            status='tangent_validated'
        elif args.validate_only:
            status = 'validated'
        else:
            # Only linear solves differ during this study; the state is fixed.
            jacobians = []
            rhs = jax.jacrev(rows)(accepted.state)
            for tolerance in (2e-5,2e-6,2e-7):
                if time.monotonic() >= deadline:
                    raise TimeoutError('walltime reached')
                trial_solver = dataclasses.replace(solver,implicit=dataclasses.replace(
                    solver.implicit,adjoint_tol=tolerance),adjoint_residual_rtol=tolerance)
                trial_cfg = dataclasses.replace(cfg,solver=trial_solver)
                diagnostics = []
                t = time.monotonic()
                jac = np.asarray(fc.free_boundary_continuation_state_pullback(
                    accepted,trial_cfg,rhs,diagnostics=diagnostics))
                jacobians.append(jac)
                event('tolerance_study',rtol=tolerance,seconds=time.monotonic()-t,rows=diagnostics)
            def row_errors(a,b):
                return np.linalg.norm((a-b)*PARAMETER_SCALES,axis=1)/np.maximum(
                    np.linalg.norm(b*PARAMETER_SCALES,axis=1),1e-30)
            errors = row_errors(jacobians[0],jacobians[-1])
            tight_errors = row_errors(jacobians[1],jacobians[-1])
            steps = [proposal(values,j,PARAMETER_SCALES) for j in jacobians]
            motion_errors = [float(np.max(np.linalg.norm(displacement(p-steps[-1]),axis=-1))) for p in steps[:2]]
            # Predeclared derivative/proposal stability gates: 0.1%, 1 micron.
            if max(tight_errors) > .001 or motion_errors[1] > 1e-6:
                raise RuntimeError('tighter derivatives/proposal are not stable')
            chosen = 0 if max(errors) <= .001 and motion_errors[0] <= 1e-6 else 1
            tolerance = (2e-5,2e-6)[chosen]
            solver = dataclasses.replace(solver,implicit=dataclasses.replace(
                solver.implicit,adjoint_tol=tolerance),adjoint_residual_rtol=tolerance)
            cfg = dataclasses.replace(cfg,solver=solver)
            event('tolerance_selected',rtol=tolerance,row_errors=errors.tolist(),
                tight_row_errors=tight_errors.tolist(),proposal_difference_m=motion_errors)
            provenance['selected_adjoint_rtol'] = tolerance
            write_json(output/'manifest.json',provenance)
            jac = jacobians[chosen]
            while step < args.target_step:
                if time.monotonic() >= deadline:
                    status='walltime';break
                if step != initial_step:
                    diagnostics=[];t=time.monotonic()
                    rhs=jax.jacrev(rows)(accepted.state)
                    jac=np.asarray(fc.free_boundary_continuation_state_pullback(accepted,cfg,rhs,diagnostics=diagnostics))
                    event('adjoint',seconds=time.monotonic()-t,rows=diagnostics)
                delta=proposal(values,jac,PARAMETER_SCALES)
                count=max(1,int(np.ceil(np.max(np.abs(delta/PARAMETER_SCALES))/.1)))
                if count>64:raise RuntimeError('proposal exceeds continuation budget')
                tangent=fbi.free_boundary_state_tangent(cfg.params,jnp.asarray(accepted.parameters),
                    solver,accepted.state,accepted.dof_mask,jnp.asarray(delta),
                    rcon0=accepted.rcon0,zcon0=accepted.zcon0)
                previous=accepted
                for index in range(1,count+1):
                    if time.monotonic() >= deadline:raise TimeoutError('walltime reached')
                    point=accepted.parameters+delta*(index/count)
                    predicted=jax.tree.map(lambda x,dx:x+dx/count,previous.state,tangent)
                    t=time.monotonic()
                    stage=_solve_free_boundary_stage(inp,external_field=builder(jnp.asarray(point)),
                        resolution=solver.resolution,ftol=1e-14,max_iterations=12000,
                        initial_state=predicted,constraint_continuation=(previous.rcon0,previous.zcon0),
                        include_edge_in_convergence=True,edge_force_tolerance=1e-14,
                        error_on_no_convergence=False,jacobian_retries=0,allow_initial_axis_reguess=False,use_fft=False)
                    event('ordinary_correction',seconds=time.monotonic()-t,point=index,points=count,
                        absolute_step=step+1,converged=bool(stage.result.converged),
                        iterations=int(stage.result.iterations),
                        forces={n:float(getattr(stage.result,n)) for n in ('fsqr','fsqz','fsql','fedge')})
                    t=time.monotonic()
                    previous=fc.certify_free_boundary_continuation_state(cfg,point,stage.result.state,
                        rcon0=stage.rcon0,zcon0=stage.zcon0,result=stage.result)
                    event('certification',seconds=time.monotonic()-t,root_residual=previous.root_residual_norm,
                        forces={n:float(getattr(previous.result,n)) for n in ('fsqr','fsqz','fsql','fedge')})
                cfg=fc.reanchor_free_boundary_continuation_config(cfg,previous)
                accepted=cfg._anchor;step+=1
                values=np.asarray(rows(accepted.state))
                record = None
                if step % args.checkpoint_every == 0:
                    record=save(output,accepted,step,targets,float(checkpoint['data']['trust_radius']),provenance)
                    saved_step = step
                event('promoted',absolute_step=step,segment_step=step-initial_step,
                    maximum_step_m=float(np.max(np.linalg.norm(displacement(delta),axis=-1))),
                    metrics=metrics(values,targets,loss_scale),checkpoint=record,
                    forces={n:float(getattr(accepted.result,n)) for n in ('fsqr','fsqz','fsql','fedge')})
            else:status='target_reached'
    except Exception as exc:
        status='failed'
        event('failure',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        summary=dict(status=status,initial_step=initial_step,final_step=step,
            promoted_steps=0 if step is None else step-initial_step,elapsed_s=time.monotonic()-started)
        if accepted is not None:
            if saved_step != step:
                save(output,accepted,step,targets,float(checkpoint['data']['trust_radius']),provenance)
            summary['final_metrics']=metrics(np.asarray(rows(accepted.state)),targets,loss_scale)
            summary['root_residual']=accepted.root_residual_norm
            if (output/'latest_checkpoint.json').exists():
                summary['final_checkpoint']=json.loads((output/'latest_checkpoint.json').read_text())
            w=wout_from_state(inp=inp,state=accepted.state,niter=accepted.result.iterations,
                fsqr=accepted.result.fsqr,fsqz=accepted.result.fsqz,fsql=accepted.result.fsql,
                vacuum_output=accepted.result.vacuum)
            path=output/'wout_final.nc';write_wout(path,w)
            summary['final_wout']=dict(path=str(path),sha256=sha256(path))
        write_json(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
