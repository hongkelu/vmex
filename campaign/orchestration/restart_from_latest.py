"""Stop runs and restart each from its latest saved step with the current code, same command, environment and GPU.

    python <repo>/campaign/orchestration/restart_from_latest.py <suffix> <pid> [<pid> ...]     (cwd: the vmex tree)

The restart directory ``<output>.<suffix>-from-<step>`` gets input.run, coils.json and wout.nc of that step; the new
run writes ``<output>-<suffix>``. Prints the new pids (for campaign/orchestration/continue_until_converged.py).
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def option(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def with_option(argv, name, value):
    argv = list(argv)
    if name in argv:
        argv[argv.index(name) + 1] = value
    else:
        argv += [name, value]
    return argv


suffix, pids = sys.argv[1], [int(p) for p in sys.argv[2:]]
new = []
for pid in pids:
    proc = Path(f"/proc/{pid}")
    argv = [a.decode() for a in (proc / "cmdline").read_bytes().split(b"\0")[:-1]]
    env = {k.decode(): v.decode() for k, v in (kv.split(b"=", 1) for kv in (proc / "environ").read_bytes().split(b"\0")
                                               if b"=" in kv)}
    cwd = Path(os.readlink(proc / "cwd"))
    out = option(argv, "--output")
    os.kill(pid, signal.SIGTERM)
    # a zombie (child of a continue_until_converged watcher, reaped by it later) has ended too
    while proc.exists() and (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
        time.sleep(2)
    run = cwd / out
    step = max(int(re.search(r"step(\d+)", p.name).group(1)) for p in run.glob("wout.step*.nc"))
    restart = cwd / f"{out}.{suffix}-from-{step}"
    restart.mkdir(parents=True, exist_ok=False)
    shutil.copy(run / "input.run", restart / "input.run")
    shutil.copy(run / f"coils.step{step}.json", restart / "coils.json")
    shutil.copy(run / f"wout.step{step}.nc", restart / "wout.nc")
    argv = with_option(with_option(argv, "--restart", str(restart.relative_to(cwd))), "--output", f"{out}-{suffix}")
    with open(cwd / f"{out}-{suffix}.log", "w") as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"{out}: stopped at step {step}, restarted as {out}-{suffix} (pid {child.pid})", flush=True)
    new.append(child.pid)
print("NEW_PIDS", *new)
