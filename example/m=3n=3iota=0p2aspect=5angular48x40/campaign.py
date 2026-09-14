"""Bounded, locked restart controller for the independent angular-48x40 campaign."""
from pathlib import Path
import argparse,collections,csv,datetime,fcntl,hashlib,io,json,math,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
CASE_REL='example/m=3n=3iota=0p2aspect=5angular48x40'
PYTHON=str(ROOT/'venv/bin/python')
ESSOS=str(ROOT/'external/ESSOS')
GPU='GPU-da1ee515-5328-d768-95bc-4fbe4817911e'
PHYSICAL=('mean_iota','aspect_ratio','b0_T','qs_error_raw','qs_objective_normalized')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def lines(p):
    if not p.exists():return []
    data=p.read_bytes().splitlines(keepends=True)
    return [json.loads(x) for x in data if x.endswith(b'\n')]
def safe(p):
    q=Path(p).resolve();assert q.is_relative_to(ROOT),str(q);return q
def processes():return subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
def owned(ps):
    return [p for p in ps if (str(SOURCE/CASE_REL/'run.py') in p and str(ROOT/'segments') in p) or (str(ROOT/'campaign.py')+' supervise ' in p)]
def authenticate_checkpoint(row):
    p=Path(row['checkpoint_path']).resolve();assert sha(p)==row['checkpoint_sha256'],'checkpoint hash mismatch'
    import numpy as np
    with np.load(p,allow_pickle=False) as z:
        assert int(z['accepted_step'])==row['step']
        assert str(z['schema_version'])=='vmex.iota02-aspect5-accepted/v1'
        assert z['parameters'].shape==(111,) and np.isfinite(z['parameters']).all()
        np.testing.assert_array_equal(z['targets'],[.2,5.,-.17506474574437714])
        assert float(z['loss_scale'])==read(ROOT/'policy.json')['loss_scale']
        assert z['rcon0'].shape==z['zcon0'].shape==(31,25,40)
        prov=json.loads(str(z['provenance_json']))
        assert (prov['contract']['ntheta'],prov['contract']['nzeta'])==(48,40)
        assert str(z['contract_sha256'])==sha(SOURCE/CASE_REL/'case_contract.json')
        assert str(z['optimizer_policy_sha256'])==sha(SOURCE/CASE_REL/'optimizer_policy.json')
        for n in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin'):
            assert np.isfinite(z[n]).all() and z[n].dtype==np.float64
    assert row['root_residual']<=2e-6 and max(row[n] for n in ('fsqr','fsqz','fsql','fedge'))<=1e-11
    return p

def append_rows(history,new,output):
    if not new:return history
    assert new[0]['step']==history[-1]['step'],'resume anchor does not match history'
    for k in PHYSICAL:assert math.isclose(new[0][k],history[-1][k],rel_tol=1e-12,abs_tol=1e-14),k
    offset=history[-1]['elapsed_s']
    for row in new[1:]:
        assert row['step']==history[-1]['step']+1,'noncontiguous history'
        history.append(dict(row,elapsed_s=offset+row['elapsed_s'],checkpoint_path=str(output/row['checkpoint_path']),segment_root=str(output)))
    return history

def verify_restoration(segment):
    out=segment/'output';launch=read(segment/'launch.json');events=lines(out/'progress.jsonl')
    initialized=next((e for e in events if e['phase']=='initialized'),None)
    if initialized is None:return False
    proof=segment/'resume_verification.json'
    if proof.exists():return True
    import numpy as np
    restored=out/f"checkpoint_step_{launch['initial_step']:04d}.npz"
    with np.load(segment/'resume_checkpoint.npz',allow_pickle=False) as before,np.load(restored,allow_pickle=False) as after:
        assert set(before.files)==set(after.files)
        for k in before.files:
            if k!='provenance_json':np.testing.assert_array_equal(before[k],after[k])
        prov=json.loads(str(after['provenance_json']));assert prov['target_step']==launch['target_step'] and prov['initial_step']==launch['initial_step']
        assert prov['contract']['force_tolerance']==1e-11 and prov['contract']['adjoint_residual_rtol']==2e-5 and not prov['contract']['root_polishing']
    assert initialized['root_residual']<=2e-6 and max(initialized['forces'].values())<=1e-11
    write(proof,dict(verified_at_utc=now(),exact_checkpoint_restoration=True,initial=initialized,input_sha256=sha(segment/'resume_checkpoint.npz'),restored_sha256=sha(restored)))
    return True

def collect(registry):
    history=lines(ROOT/'prefix/metrics.jsonl');assert [r['step'] for r in history]==list(range(len(history)))
    snapshots=lines(ROOT/'prefix/snapshots.jsonl');exits=[];active_seconds=0.;streak=0
    for seg in registry['segments']:
        folder=safe(ROOT/seg['directory']);out=folder/'output';new=lines(out/'metrics.jsonl')
        if new:assert new[0]['step']==seg['initial_step']
        history=append_rows(history,new,out)
        if verify_restoration(folder):seg['restoration_verified']=True
        for item in lines(out/'diagnostics/snapshots.jsonl'):
            candidate=dict(item,segment_root=str(out),checkpoint_path=str(out/item['checkpoint_path']),files={k:dict(v,path=str(out/v['path'])) for k,v in item['files'].items()})
            previous=next((v for v in snapshots if v['step']==item['step']),None)
            if previous is None:snapshots.append(candidate)
            else:
                # A resumed initial/final snapshot is retained in its segment,
                # but the combined history keeps one verified pair per step.
                assert item['step']==seg['initial_step'] and seg.get('restoration_verified'),'unverified duplicate snapshot'
                proof=read(folder/'resume_verification.json')
                assert item['checkpoint_sha256']==proof['restored_sha256'],'snapshot is not the restored anchor'
                for artifact in candidate['files'].values():assert sha(safe(Path(artifact['path'])))==artifact['sha256']
                for k in ('mean_iota','aspect_ratio','b0'):
                    assert math.isclose(previous['metrics']['physical'][k],item['metrics']['physical'][k],rel_tol=1e-12,abs_tol=1e-14),'snapshot anchor metrics differ'
                assert math.isclose(previous['metrics']['qs_error_raw'],item['metrics']['qs_error_raw'],rel_tol=1e-12,abs_tol=1e-14)

        ended=read(folder/'exit.json');exits.append(ended)
        active_seconds+=max(0.,(ended['finished_unix'] if ended else time.time())-seg['started_unix'])
        # Only finished segments count toward the no-progress circuit breaker.
        if ended:streak=streak+1 if history[-1]['step']<=seg['initial_step'] else 0
    allowed=[Path(p).resolve() for p in read(ROOT/'policy.json')['allowed_artifact_roots']]+[ROOT]
    for item in snapshots:
        for artifact in item['files'].values():
            p=Path(artifact['path']).resolve();assert any(p.is_relative_to(a) for a in allowed) and sha(p)==artifact['sha256'],'snapshot integrity failure'
    assert len({s['step'] for s in snapshots})==len(snapshots),'duplicate snapshot step'
    latest=history[-1];authenticate_checkpoint(latest)
    ps=owned(processes());current=registry['segments'][-1] if registry['segments'] else None
    folder=ROOT/current['directory'] if current else None;out=folder/'output' if folder else None
    ev=lines(out/'progress.jsonl') if out else [];directions=[e for e in ev if e['phase']=='direction'];summary=read(out/'summary.json') if out else None
    cp=read(out/'latest_checkpoint.json') if out else None
    if cp and not ps:
        assert cp['accepted_step']==latest['step'],'checkpoint/history handoff incomplete'
        if current and latest['step']==current['initial_step']:
            assert current.get('restoration_verified'),'unverified resume anchor'
        else:assert cp['sha256']==latest['checkpoint_sha256'],'checkpoint/history hash mismatch'
    if summary and not ps and summary.get('final_checkpoint'):
        assert summary['final_checkpoint']==cp and summary['final_step']==latest['step'],'terminal summary/checkpoint mismatch'
    policy=read(ROOT/'policy.json');missing=sorted(set(range(0,latest['step']+1,20))-{s['step'] for s in snapshots})
    if not ps:assert not missing,'missing scheduled snapshots in stopped segment'
    disk=shutil.disk_usage(ROOT)
    status=dict(checked_at_utc=now(),target_step=policy['target_step'],last_recorded_step=latest['step'],metrics=latest,history_rows=len(history),snapshot_count=len(snapshots),last_snapshot_step=max(s['step'] for s in snapshots),missing_periodic_snapshots=missing,owned_run_processes=ps,segment_count=len(registry['segments']),current_segment=None if folder is None else str(folder),active_seconds=active_seconds,nonprogress_interruptions=streak,wall_seconds=time.time()-registry['created_unix'],summary=summary,exit=exits[-1] if exits else None,last_event=ev[-1] if ev else None,last_direction=directions[-1] if directions else None,numerical_failure_events=[e for e in ev if e['phase']=='failure' and e.get('error_type')!='TimeoutError'],rejection_reasons=dict(collections.Counter(e.get('reason') for e in ev if e['phase']=='trial_decision' and not e['accepted'])),max_dense_relative_residual=max((x['relative_residual'] for e in ev if e['phase']=='adjoint' for x in e['rows']),default=None),disk_free_bytes=disk.free,restoration_verified=bool(current and current.get('restoration_verified')),last_segment_started_unix=current['started_unix'] if current else None)
    output=ROOT/'campaign';output.mkdir(exist_ok=True)
    for name,rows in [('metrics.jsonl',history),('snapshots.jsonl',snapshots)]:
        p=output/name;tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(p)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history);p=output/'metrics.csv';p.with_suffix('.tmp').write_text(stream.getvalue());p.with_suffix('.tmp').replace(p)
    return status

def decision(s,p):
    if s['owned_run_processes']:return 'running'
    if s['last_recorded_step']>=p['target_step']:
        return 'complete' if s['summary'] and s['summary'].get('status') in ('step_budget_reached','converged') and s['summary']['final_step']==s['last_recorded_step'] and s['exit'] and s['exit']['returncode']==0 else 'target_reached_needs_terminal_review'
    if s['numerical_failure_events']:return 'numerical_failure_needs_review'
    if s['summary'] and s['summary']['status'] in ('converged','stagnated'):return s['summary']['status']
    if s['segment_count'] and not s['exit']:
        return 'missing_exit_needs_review'
    handoff=(p.get('validation_checkpoint_sha256') is not None and s['last_recorded_step']==10 and s['exit'] and s['exit']['returncode']==0 and s['summary'] and s['summary'].get('status')=='step_budget_reached' and s['summary'].get('target_step')==10 and s.get('metrics',{}).get('checkpoint_sha256')==p['validation_checkpoint_sha256'])
    if s['segment_count'] and s['exit']['returncode'] not in (-9,137,-15,143,124) and not handoff:
        # A normal explicit walltime failure is restartable if budget remains.
        last=s.get('last_event') or {}
        if not (s['summary'] and s['summary']['status']=='failed' and last.get('error_type')=='TimeoutError'):
            return 'unexpected_exit_needs_review'
    if s['active_seconds']>=p['max_active_seconds'] or s['wall_seconds']>=p['max_wall_seconds']:return 'campaign_time_budget_reached'
    if s['segment_count']>=p['max_segments']:return 'restart_budget_reached'
    if s['nonprogress_interruptions']>=p['max_nonprogress_interruptions']:return 'no_progress_needs_review'
    if s['disk_free_bytes']<p['min_free_bytes']:return 'insufficient_disk_space'
    return 'restart_ready' if s['segment_count'] else 'start_ready'

def verify_source():
    for n,h in read(ROOT/'source_manifest.json').items():
        p=(SOURCE/n).resolve();assert p.is_relative_to(SOURCE) and sha(p)==h,'source drift: '+n
    for n,h in read(ROOT/'external_manifest.json').items():
        p=safe(ROOT/n);assert sha(p)==h,'external source drift: '+n
    assert read(ROOT/'tests_passed.json')['source_manifest_sha256']==sha(ROOT/'source_manifest.json')

def launch(registry,s,policy):
    verify_source();assert not owned(processes()),'owned process already exists'
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    if GPU in used:return 'waiting_for_gpu'
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True);assert GPU in hardware
    remaining=min(86400,int(policy['max_active_seconds']-s['active_seconds']),int(policy['max_wall_seconds']-s['wall_seconds']))
    if remaining<120:return 'campaign_time_budget_reached'
    cp=authenticate_checkpoint(s['metrics']);folder=safe(ROOT/'segments'/f"segment_{len(registry['segments'])+1:04d}");folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(cp,folder/'resume_checkpoint.npz');assert sha(folder/'resume_checkpoint.npz')==s['metrics']['checkpoint_sha256']
    command=['timeout','--signal=TERM','--kill-after=60',str(remaining),PYTHON,'-B',str(SOURCE/CASE_REL/'run.py'),'--resume-checkpoint',str(folder/'resume_checkpoint.npz'),'--checkpoint-sha256',sha(folder/'resume_checkpoint.npz'),'--target-step',str(policy['target_step']),'--device','cuda:0','--output-dir',str(folder/'output'),'--max-wall-hours',str(remaining/3600)]
    rec=dict(directory=str(folder.relative_to(ROOT)),initial_step=s['last_recorded_step'],started_unix=time.time(),checkpoint_sha256=sha(folder/'resume_checkpoint.npz'),remaining_budget_seconds=remaining)
    write(folder/'launch.json',dict(rec,command=command,gpu_uuid=GPU,hardware=hardware,target_step=policy['target_step'],main_commit=policy['main_commit']))
    registry['segments'].append(rec);write(ROOT/'registry.json',registry)
    with (folder/'supervisor.log').open('xb') as log:
        proc=subprocess.Popen([PYTHON,'-B',str(ROOT/'campaign.py'),'supervise','--segment',rec['directory']],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rec['supervisor_pid']=proc.pid;write(ROOT/'registry.json',registry)
    return 'restarted' if len(registry['segments'])>1 else 'started'

def supervise(directory):
    folder=safe(ROOT/directory);rec=read(folder/'launch.json');verify_source()
    env=dict(os.environ,PYTHONPATH=f'{SOURCE}:{ESSOS}',PYTHONDONTWRITEBYTECODE='1',JAX_ENABLE_X64='1',PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_VISIBLE_DEVICES=GPU,JAX_PLATFORMS='cuda,cpu')
    for key,name in [('JAX_COMPILATION_CACHE_DIR','jax'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda')]:
        p=folder/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    try:
        with (folder/'run.log').open('xb') as f:
            proc=subprocess.Popen(rec['command'],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT)
            write(folder/'process.json',dict(timeout_pid=proc.pid,supervisor_pid=os.getpid(),started_unix=time.time()))
            code=proc.wait()
        write(folder/'exit.json',dict(returncode=code,finished_unix=time.time()))
    except BaseException as exc:
        write(folder/'supervisor_error.json',dict(error_type=type(exc).__name__,error=str(exc),time=now()));raise

def tick(restart=False):
    with (ROOT/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'action':'controller_busy','checked_at_utc':now()}
        reg=read(ROOT/'registry.json');policy=read(ROOT/'policy.json')
        try:
            s=collect(reg);s['decision']=decision(s,policy);s['action']='none'
            if restart and s['decision'] in ('start_ready','restart_ready'):
                s['action']=launch(reg,s,policy);s['current_segment']=str(ROOT/reg['segments'][-1]['directory']) if reg['segments'] else None
                s['segment_count']=len(reg['segments'])
                if s['action'] in ('started','restarted'):
                    action=s['action'];previous_exit=s['exit']
                    s=collect(reg);s.update(action=action,decision='initializing',previous_segment_exit=previous_exit)
            write(ROOT/'registry.json',reg);write(ROOT/'campaign/status.json',s)
        except Exception as exc:
            s=dict(checked_at_utc=now(),decision='integrity_or_controller_error',action='none',error_type=type(exc).__name__,error=str(exc));write(ROOT/'campaign/controller_error.json',s)
        with (ROOT/'controller_history.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        return s

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['tick','supervise']);p.add_argument('--restart',action='store_true');p.add_argument('--segment');a=p.parse_args()
    if a.mode=='supervise':supervise(a.segment)
    else:print(json.dumps(tick(a.restart)))
if __name__=='__main__':main()
