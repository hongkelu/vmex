"""Monitor only the 48x40 campaign; retain the established entrypoint filename."""
import argparse,datetime,fcntl,json,subprocess
from pathlib import Path
CASES={
    'refined_lhk3_6_gpu0':('lhk3-6','/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/venv/bin/python','/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/workflow.py'),
}
def check(name,restart):
    host,python,controller=CASES[name]
    remote=f'{python} -B {controller} tick'+(' --restart' if restart else '')
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,remote],capture_output=True,text=True,timeout=60)
        if result.returncode:
            return dict(decision='live_status_unavailable',action='none',ssh_returncode=result.returncode,error=result.stderr[-1000:])
        data=json.loads(result.stdout)
        assert isinstance(data,dict)
        return data
    except Exception as exc:
        return dict(decision='live_status_unavailable',action='none',error_type=type(exc).__name__,error=str(exc))
def main():
    p=argparse.ArgumentParser();p.add_argument('--restart',action='store_true');a=p.parse_args()
    out=Path(__file__).resolve().parent/'runs/high-resolution-monitor';out.mkdir(parents=True,exist_ok=True)
    with (out/'monitor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'action':'monitor_busy'}));return
        previous=json.loads((out/'latest.json').read_text()) if (out/'latest.json').exists() else {}
        cases={name:check(name,a.restart) for name in CASES}
        for name,data in cases.items():
            before=previous.get('cases',{}).get(name,{}).get('last_recorded_step')
            after=data.get('last_recorded_step')
            if before is not None and after is not None:data['accepted_gain_since_previous_check']=after-before
        now=datetime.datetime.now(datetime.timezone.utc)
        result=dict(checked_at_utc=now.isoformat(),restart_enabled=a.restart,cases=cases)
        payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
        (out/('status_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')).write_text(payload)
        temp=out/'latest.tmp';temp.write_text(payload);temp.replace(out/'latest.json')
        print(json.dumps(result))
if __name__=='__main__':main()
