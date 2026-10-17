"""Continue optimization runs until SLSQP converges: when a followed process ends at its step budget, relaunch it
with ``--restart`` from its final coils and boundary on an empty GPU (its own first), same command and environment.

    python extra/continue_until_converged.py <pid> [<pid> ...]     (cwd: the vmex tree; log to stdout)

A run that converged, stopped for another optimizer reason or crashed is not relaunched; at most MAX_CONTINUATIONS each.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONTINUATIONS = 5
BUDGET_STOPS = ("accepted_step_budget_reached", "Iteration limit reached", "iteration limit")


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def process(pid):
    proc = Path(f"/proc/{pid}")
    argv = (proc / "cmdline").read_bytes().split(b"\0")[:-1]
    env = dict(kv.split(b"=", 1) for kv in (proc / "environ").read_bytes().split(b"\0") if b"=" in kv)
    return dict(pid=pid, argv=[a.decode() for a in argv], env={k.decode(): v.decode() for k, v in env.items()},
                cwd=os.readlink(proc / "cwd"), count=0)


def follow(pid):
    r = process(pid)
    r["base"] = option(r["argv"], "--output")
    return r


def option(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def with_option(argv, name, value):
    argv = list(argv)
    if name in argv:
        argv[argv.index(name) + 1] = value
    else:
        argv += [name, value]
    return argv


def empty_gpus():
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()
    rows = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return [u for u, m in (r.split(", ") for r in rows) if int(m) == 0 and u not in busy]


def alive(pid):
    """Running, not a zombie: a run this watcher relaunched stays in /proc until it is reaped here."""
    try:
        if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            return True
    except FileNotFoundError:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return False


runs = [follow(int(p)) for p in sys.argv[1:]]
for r in runs:
    log("following", r["pid"], option(r["argv"], "--output"), "on", r["env"].get("CUDA_VISIBLE_DEVICES"))
while runs:
    time.sleep(120)
    for r in list(runs):
        if alive(r["pid"]):
            continue
        out = Path(r["cwd"]) / option(r["argv"], "--output")
        summary = out / "summary.json"
        if not summary.exists():
            log("ended without summary (crash?):", out, "- not relaunched")
            runs.remove(r)
            continue
        info = json.loads(summary.read_text())
        reason = str(info.get("stop_reason") or info.get("message"))
        if not any(b in reason for b in BUDGET_STOPS) or r["count"] >= MAX_CONTINUATIONS:
            log("done:", out, "-", reason, "(success)" if info.get("success") else "")
            runs.remove(r)
            continue
        if not (out / "coils.json").exists() or not (out / "wout.nc").exists():
            log("budget reached but no final coils/wout in", out, "- not relaunched")
            runs.remove(r)
            continue
        free = empty_gpus()
        own = r["env"].get("CUDA_VISIBLE_DEVICES")
        gpu = own if own in free else (free[0] if free else None)
        if gpu is None:
            continue  # its GPU was taken: try again next round
        new_out = f"{r['base']}-c{r['count'] + 1}"
        argv = with_option(with_option(r["argv"], "--restart", str(out.relative_to(r["cwd"]))), "--output", new_out)
        env = dict(r["env"], CUDA_VISIBLE_DEVICES=gpu)
        with open(Path(r["cwd"]) / f"{new_out}.log", "w") as stream:
            child = subprocess.Popen(argv, cwd=r["cwd"], env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"{out.name}: {reason} at step {info.get('accepted_steps')}; continued as {new_out} (pid {child.pid}) on {gpu}")
        r.update(pid=child.pid, argv=argv, env=env, count=r["count"] + 1)
log("all followed runs are done")
