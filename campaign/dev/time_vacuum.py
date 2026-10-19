"""Time the vacuum-iota row: eager and jitted, repeated and at a moved boundary.

    COIL_CASE=qa4-beta python extra/time_vacuum.py <run dir with input.run>     (cwd: the vmex tree, GPU)
"""
import dataclasses
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
from vmex.core import implicit as im  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
eqs = [opt.solve_equilibrium(inp)]
rbc = np.array(inp.rbc); rbc[inp.ntor - 1, 1] += 1e-3
eqs.append(opt.solve_equilibrium(dataclasses.replace(inp, rbc=rbc), initial_state=eqs[0].state))
vac = tt.vacuum_iota(inp)
f = jax.jit(vac)


def timed(label, fun, eq):
    t = time.perf_counter()
    v = np.asarray(fun(eq.state, eq.runtime))
    print(f"{label:34s} {time.perf_counter() - t:8.2f} s  min {v.min():.6f}", flush=True)


timed("eager, first", vac, eqs[0])
timed("eager, same boundary", vac, eqs[0])
timed("jit, first (compile)", f, eqs[0])
timed("jit, same boundary", f, eqs[0])
timed("jit, moved boundary", f, eqs[1])
timed("jit, moved back", f, eqs[0])
timed("jit, moved again", f, eqs[1])
g = jax.jit(jax.grad(lambda s, r: jax.numpy.min(vac(s, r))))
for label, eq in (("grad, first (compile)", eqs[0]), ("grad, same", eqs[0]), ("grad, moved", eqs[1])):
    t = time.perf_counter()
    jax.block_until_ready(g(eq.state, eq.runtime))
    print(f"{label:34s} {time.perf_counter() - t:8.2f} s", flush=True)
