"""Read-only NESTOR pressure diagnostic on the exact fixed target and coils."""
from pathlib import Path
import os,sys,hashlib,json,time,subprocess,dataclasses
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/frozen_initial'
OUT.mkdir(parents=True,exist_ok=False)
GPU=os.environ['CUDA_VISIBLE_DEVICES']
used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert GPU not in used,'GPU occupied; diagnostic not started'
for k,n in [('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('CUDA_CACHE_PATH','cuda')]:
 p=OUT/'cache'/n;p.mkdir(parents=True);os.environ[k]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,JAX_ENABLE_X64='1',JAX_PLATFORMS='cuda,cpu',PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
sys.path[:0]=[str(ROOT.parents[1]),str(ROOT)]
import jax,jax.numpy as jnp,numpy as np
from case import load_case,STATE_NAMES,sha256
from initialization import verify_runtime
from gates import pressure_diagnostic
from vmex.core import implicit as im,freeboundary_implicit as fbi
verify_runtime();inp,builder,state,contract,hashes=load_case();p=jnp.zeros(111);started=time.monotonic()
original={n:np.asarray(getattr(state,n)).copy() for n in STATE_NAMES}
state=jax.tree.map(jnp.asarray,state)
print(json.dumps(dict(phase='frozen_nestor_start',state_sha256=hashes['initial_fixed_state.npz'],coil_sha256=hashes['coils.json'])),flush=True)
solver=fbi.make_free_boundary_config(inp,builder(p),field_from_parameters=builder,device=jax.devices('gpu')[0],ftol=1e-10,include_edge_in_convergence=False)
params=im.params_from_input(inp);rt=im.runtime_from_params(params,solver.implicit)
rt=dataclasses.replace(rt,lfreeb=True,jmax=50,presf_ns_scale=fbi._presf_ns_scale_traceable(params,inp,50))
out=solver.vacuum_program.full(state,rt,builder(p))
metric=pressure_diagnostic(out['delbsq_num'],out['delbsq_den'],.01)
assert all(np.array_equal(np.asarray(getattr(state,n)),original[n]) for n in STATE_NAMES)
result=dict(completed=True,seconds=time.monotonic()-started,exact_fixed_state_unchanged=True,ordinary_equilibrium_solves=0,coil_parameters_zero=True,state_sha256=hashes['initial_fixed_state.npz'],coil_sha256=hashes['coils.json'],phiedge_Wb=float(inp.phiedge),resolution=dict(ns=int(solver.resolution.ns),ntheta=int(solver.resolution.ntheta),nzeta=int(solver.resolution.nzeta)),diagnostic=metric,script_sha256=sha256(Path(__file__)))
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
