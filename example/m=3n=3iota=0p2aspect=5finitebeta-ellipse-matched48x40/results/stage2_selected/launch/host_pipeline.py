"""Finite, guarded matched-resolution rebuild; no automatic QA campaign."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
CASE=SOURCE/'example/m=3n=3iota=0p2aspect=5finitebeta-ellipse-matched48x40'
PYTHON='/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/venv/bin/python'
GPU='GPU-0b009025-c499-c2ad-030f-91e852095542'
def event(phase,**kw):
 d=dict(phase=phase,time_unix=time.time(),**kw)
 (ROOT/'host_status.json').write_text(json.dumps(d,indent=2)+'\n')
 with (ROOT/'host_progress.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps(d),flush=True)
def main():
 for manifest in ('source_manifest.json','host_manifest.json'):
  for n,h in json.loads((ROOT/manifest).read_text()).items():
   p=(SOURCE/n).resolve();assert p.is_relative_to(SOURCE) and hashlib.sha256(p.read_bytes()).hexdigest()==h,n
 assert json.loads((ROOT/'quadrature_preflight.json').read_text())['passed']
 assert '1 passed' in (ROOT/'tests.log').read_text()
 assert json.loads((ROOT/'preflight.json').read_text())['passed']
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 if sys.argv[1]=='start':
  (ROOT/'host.lock').mkdir()
  with (ROOT/'host.log').open('xb') as log:
   child=subprocess.Popen([PYTHON,'-B',str(__file__),'run'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (ROOT/'host_launch.json').write_text(json.dumps(dict(pid=child.pid,started_unix=time.time(),gpu_uuid=GPU,max_gpu_wait_seconds=1800,phase_timeouts_seconds=[3300,240],angular_grid=[48,40],no_qa_launch=True),indent=2));print(child.pid);return
 assert sys.argv[1]=='run'
 event('waiting_for_gpu',max_wait_seconds=1800,required_idle_seconds=60)
 deadline=time.monotonic()+1800;idle_since=None
 while time.monotonic()<deadline:
  used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  if GPU in used:idle_since=None
  elif idle_since is None:idle_since=time.monotonic()
  elif time.monotonic()-idle_since>=60:break
  time.sleep(10)
 else:raise TimeoutError('No sustained GPU availability within 30-minute resource wait; nothing launched')
 hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True)
 (ROOT/'host_hardware.txt').write_text(hardware)
 for phase,script,budget in [('coil_fit','fit_coils_host.py',3300),('frozen_nestor','frozen_nestor_host.py',240)]:
  if phase=='coil_fit':assert json.loads((CASE/'fixed/summary.json').read_text())['qualified_fixed_target']
  if phase=='frozen_nestor':assert json.loads((CASE/'coils_host/summary.json').read_text())['eligible_for_frozen_nestor_check']
  event(phase,script=script,max_seconds=budget)
  cmd=['timeout','--kill-after=30',str(budget),PYTHON,'-B','-u',str(CASE/script)]
  with (ROOT/('host_'+phase+'.log')).open('xb') as log:
   ret=subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=budget+45).returncode
  if ret:raise RuntimeError(f'{phase} exited {ret}; subsequent phases not launched')
 frozen=json.loads((CASE/'frozen_nestor_host/summary.json').read_text())
 event('completed' if frozen['passed'] else 'frozen_pressure_gate_failed',frozen_nestor=frozen,free_boundary_solve_launched=False,qa_steps=0)
if __name__=='__main__':
 try:main()
 except Exception as e:
  event('failed',error=type(e).__name__+': '+str(e));raise
