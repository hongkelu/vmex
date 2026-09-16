"""Read-only audit of saved states; no equilibrium solve or optimization."""
from pathlib import Path
import os,sys,json,dataclasses,time,hashlib,subprocess
ROOT=Path('/root/autodl-tmp/vmex-finitebeta-matched48x40-single-stage-20260914T203732Z')
CASE=ROOT/'source/example/m=3n=3iota=0p2aspect=5finitebeta-ellipse-matched48x40-single-stage'
OUT=ROOT/'analysis/initialization_failure'
OUT.mkdir(parents=True,exist_ok=False)
GPU='GPU-0b009025-c499-c2ad-030f-91e852095542'
assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
for key,name in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('JAX_COMPILATION_CACHE_DIR','jax'),('CUDA_CACHE_PATH','cuda')]:
 p=OUT/'cache'/name;p.mkdir(parents=True);os.environ[key]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,JAX_ENABLE_X64='1',JAX_PLATFORMS='cuda,cpu',PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
sys.path[:0]=[str(ROOT/'source'),str(CASE),'/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/external/ESSOS']
import jax,jax.numpy as jnp,numpy as np
from case import load_case,physical_rows,STATE_NAMES,sha256
from initialization import verify_runtime
from vmex.core import implicit as im,freeboundary_implicit as fbi
from vmex.core.freeboundary import _vacuum_scalars
from vmex.core.solver import SpectralState,evaluate_forces
verify_runtime();inp,builder,seed,contract,hashes=load_case();p=jnp.zeros(111);started=time.monotonic()
cfg=fbi.make_free_boundary_config(inp,builder(p),field_from_parameters=builder,device=jax.devices('gpu')[0],ftol=1e-10,include_edge_in_convergence=False)
params=im.params_from_input(inp);base=im.runtime_from_params(params,cfg.implicit)
base=dataclasses.replace(base,lfreeb=True,jmax=50,presf_ns_scale=fbi._presf_ns_scale_traceable(params,inp,50))
result=dict(ordinary_solves=0,coils_changed=False,ntheta=48,nzeta=40,ns=50,states={})
for name,file in [('fixed',CASE/'inputs/initial_fixed_state.npz'),('relaxed',CASE/'runs/steps010/ordinary_initial_state_uncertified.npz')]:
 print(json.dumps(dict(phase='audit_state',name=name)),flush=True)
 with np.load(file,allow_pickle=False) as d:
  state=SpectralState(*(jnp.asarray(d[n]) for n in STATE_NAMES))
  rt=dataclasses.replace(base,rcon0=jnp.asarray(d['rcon0']),zcon0=jnp.asarray(d['zcon0'])) if 'rcon0' in d else base
 original=[np.asarray(x).copy() for x in jax.tree.leaves(state)]
 vac=cfg.vacuum_program.full(state,rt,builder(p));scalars=_vacuum_scalars(state,rt)
 inside=np.asarray(scalars[4]);outside=np.asarray(vac['bsqvac'])+float(scalars[5])*float(rt.presf_ns_scale);jump=outside-inside
 assert jump.shape==(25,40)
 full=np.empty((48,40));full[:25]=jump
 for i in range(25,48):full[i]=jump[48-i,(-np.arange(40))%40]
 fourier=np.fft.fft2(full)/full.size;ms=np.fft.fftfreq(48)*48;ns=np.fft.fftfreq(40)*40
 low=(abs(ms[:,None])<3)&(abs(ns[None,:])<=3);energy=abs(fourier)**2
 rtforce=dataclasses.replace(rt,bsqvac_edge=vac['bsqvac'])
 gc,forces,diag=evaluate_forces(state,rtforce)
 vals=np.asarray(physical_rows(state,rt))
 indices=np.argsort(energy.ravel())[-12:][::-1]
 record=dict(sha256=sha256(file),physical=dict(zip(['mean_iota','aspect','B0_T','Rmajor_m'],vals.tolist())),fresh_delbsq=float(vac['delbsq_num']/vac['delbsq_den']),fresh_forces={n:float(getattr(forces,n)) for n in ['fsqr','fsqz','fsql','fedge']},preconditioned_force_norm=float(np.sqrt(sum(np.sum(np.asarray(a)**2) for a in jax.tree.leaves(gc)))),pressure_jump_rms=float(np.sqrt(np.mean(full**2))),pressure_jump_outside_m2_n3_energy_fraction=float(energy[~low].sum()/energy.sum()),pressure_jump_mean=float(full.mean()),pressure_jump_top_modes=[dict(m=int(round(ms[np.unravel_index(i,energy.shape)[0]])),n=int(round(ns[np.unravel_index(i,energy.shape)[1]])),complex_amplitude_abs=float(abs(fourier.ravel()[i]))) for i in indices],axis_R_phi0_m=float(scalars[2][0]),pressure_edge_scaled=float(scalars[5])*float(rt.presf_ns_scale),constraint_baseline_norm=float(np.linalg.norm(np.asarray(rt.rcon0)))+float(np.linalg.norm(np.asarray(rt.zcon0))))
 assert all(np.array_equal(a,np.asarray(b)) for a,b in zip(original,jax.tree.leaves(state)))
 np.savez_compressed(OUT/(name+'_pressure.npz'),jump=jump,inside=inside,outside=outside)
 result['states'][name]=record
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(record),flush=True)
result.update(completed=True,seconds=time.monotonic()-started)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
