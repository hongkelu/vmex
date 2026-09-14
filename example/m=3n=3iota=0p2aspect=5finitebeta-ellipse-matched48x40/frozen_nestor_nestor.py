"""Actual frozen-state NESTOR gate for the newly calibrated target and refitted coils."""
from pathlib import Path
import os,sys,json,hashlib,time,subprocess,dataclasses
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'frozen_nestor_nestor'
assert not OUT.exists()
GPU='GPU-0b009025-c499-c2ad-030f-91e852095542'
assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
OUT.mkdir()
for k,n in [('TMPDIR','tmp'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('JAX_COMPILATION_CACHE_DIR','jax'),('CUDA_CACHE_PATH','cuda')]:
 p=OUT/'cache'/n;p.mkdir(parents=True);os.environ[k]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,JAX_ENABLE_X64='1',JAX_PLATFORMS='cuda,cpu',PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
sys.path[:0]=[str(ROOT.parents[1]),str(ROOT),'/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/external/ESSOS']
import jax,jax.numpy as jnp,numpy as np
from case import Full111FieldBuilder
from initialization import verify_runtime
from vmex.core.input import VmecInput
from vmex.core.solver import SpectralState
from vmex.core import implicit as im,freeboundary_implicit as fbi
from vmex.core.wout import read_wout
verify_runtime();started=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fixed=json.loads((ROOT/'fixed/summary.json').read_text());fit=json.loads((ROOT/'coils_nestor/summary.json').read_text())
assert fixed['qualified_fixed_target'] and fit['eligible_for_frozen_nestor_check']
assert fixed['wout_sha256']==sha(ROOT/'fixed/wout.nc')
w=read_wout(ROOT/'fixed/wout.nc')
assert abs(float(np.mean(w.iotas[1:]))-.2)<=1e-5 and abs(float(w.b0)-5.1)<=1e-4
assert abs(float(w.aspect)-5)<=1e-9 and abs(float(w.Rmajor_p)-11.067033897730942)<=1e-8
inp=VmecInput.from_file(ROOT/'fixed/input.fixed.json')
assert (inp.ntheta,inp.nzeta,inp.mpol,inp.ntor)==(48,40,3,3)
inp=dataclasses.replace(inp,lfreeb=True,mgrid_file='DIRECT_ESSOS_BIOT_SAVART',ns_array=(50,),ftol_array=(1e-13,),niter_array=(12000,))
assert inp.lfreeb
raw=json.loads((ROOT/'coils_nestor/coils_candidate.json').read_text());builder=Full111FieldBuilder(jnp.asarray(raw['dofs_curves']),jnp.asarray(raw['dofs_currents']));p=jnp.zeros(111)
names=('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')
with np.load(ROOT/'fixed/state.npz',allow_pickle=False) as d:
 assert bool(d['qualified_fixed_target']);original={n:d[n].copy() for n in names}
state=SpectralState(*(jnp.asarray(original[n]) for n in names))
solver=fbi.make_free_boundary_config(inp,builder(p),field_from_parameters=builder,device=jax.devices('gpu')[0],ftol=1e-10,include_edge_in_convergence=False)
assert (solver.resolution.ns,solver.resolution.ntheta,solver.resolution.ntheta3,solver.resolution.nzeta)==(50,48,25,40)
params=im.params_from_input(inp);rt=im.runtime_from_params(params,solver.implicit)
rt=dataclasses.replace(rt,lfreeb=True,jmax=50,presf_ns_scale=fbi._presf_ns_scale_traceable(params,inp,50))
out=solver.vacuum_program.full(state,rt,builder(p))
num,den=map(float,(out['delbsq_num'],out['delbsq_den']))
assert np.isfinite(num) and np.isfinite(den) and num>=0 and den>0
ratio=num/den
assert all(np.array_equal(np.asarray(getattr(state,n)),original[n]) for n in names)
summary=dict(completed=True,passed=ratio<=.01,delbsq=ratio,delbsq_percent=100*ratio,limit=.01,numerator=num,denominator=den,ntheta=48,nzeta=40,ns=50,vacuum_mf=4,vacuum_nf=3,coil_segments=75,ordinary_equilibrium_solves=0,state_unchanged=True,wout_sha256=fixed['wout_sha256'],state_sha256=sha(ROOT/'fixed/state.npz'),coil_sha256=sha(ROOT/'coils_nestor/coils_candidate.json'),seconds=time.monotonic()-started)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if summary['passed']:
 (ROOT/'coils_nestor/coils_selected.json').write_bytes((ROOT/'coils_nestor/coils_candidate.json').read_bytes())
 inp.to_json(OUT/'input.free.json')
