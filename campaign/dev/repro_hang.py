"""Reproduce the tokamak-seed vacuum-iota hang, dumping all thread stacks every 120 s.

    COIL_CASE=qa4-beta-tok python extra/repro_hang.py <tok run dir with input.run>     (cwd: vmex tree, GPU)
"""
import dataclasses
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402

faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
t = time.perf_counter()
eq = opt.solve_equilibrium(inp)
print(f"finite-beta seed solved {time.perf_counter() - t:.0f} s", flush=True)
vac = tt.vacuum_iota(inp)
for label, fn in (("eager", vac), ("jit", jax.jit(vac))):
    t = time.perf_counter()
    try:
        v = np.asarray(fn(eq.state, eq.runtime))
        print(f"{label}: {time.perf_counter() - t:.0f} s, vacuum iota min {v.min():.3e}", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"{label}: {time.perf_counter() - t:.0f} s, raised {type(error).__name__}: {str(error)[:300]}", flush=True)
