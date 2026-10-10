"""Build a --restart directory from the latest saved step of a three-term run.

    python make_restart.py <run_dir> <restart_dir>

The driver's --restart wants RUN/coils.json, RUN/wout.nc and RUN/input.run; a
run killed by the allocation limit has only coils.stepN.json / wout.stepN.nc,
so the latest pair is copied under those names.  Prints the step used.
"""
import re
import shutil
import sys
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
steps = sorted(int(m.group(1)) for p in run.glob("wout.step*.nc") if (m := re.search(r"step(\d+)\.nc$", p.name))
               and (run / f"coils.step{m.group(1)}.json").exists())
if not steps:
    sys.exit(f"no saved steps in {run}")
n = steps[-1]
out.mkdir(parents=True, exist_ok=True)
shutil.copy(run / f"wout.step{n}.nc", out / "wout.nc")
shutil.copy(run / f"coils.step{n}.json", out / "coils.json")
shutil.copy(run / "input.run", out / "input.run")
(out / "RESTARTED_FROM").write_text(f"{run} step {n}\n")
print(f"restart from {run} step {n} -> {out}")
