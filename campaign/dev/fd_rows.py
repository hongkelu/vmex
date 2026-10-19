"""FD check of the new rows (vacuum iota min, axis |B|) through a finite-beta fixed-boundary implicit solve.

    COIL_CASE=qa4-beta python extra/fd_rows.py     (cwd: the vmex tree, GPU)
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

t = time.time()
inp, fixed = tt.finite_beta_input(tt.seed_input(), 0.01, "gpu")
print(f"seed {time.time() - t:.0f} s, axis |B| {float(opt.axis_field_strength(fixed.state, fixed.runtime)):.6f}",
      flush=True)
vac = tt.vacuum_iota(inp)
cfg = im.make_config(inp, hot_restart=True)
p0 = im.params_from_input(inp)
ntor = int(inp.ntor)
# directions: the elongation-rotation modes that carry the vacuum transform, and the major radius
dirs = [(ntor - 1, 1, "rbc"), (ntor - 1, 1, "zbs"), (ntor, 2, "rbc"), (ntor + 1, 1, "zbs")]


def rows(z):
    p = p0
    for k, (i, j, name) in enumerate(dirs):
        p = dataclasses.replace(p, **{name: getattr(p, name).at[i, j].add(z[k])})
    state = im.solve_implicit(p, cfg)
    rt = im.runtime_from_params(p, cfg)
    v = vac(state, rt)
    return jnp.stack([jnp.min(v), v[0], v[-1], opt.axis_field_strength(state, rt)])


z0 = jnp.zeros(len(dirs))
t = time.time()
val = rows(z0)
print("values (vac min, vac axis, vac edge, B_axis):", np.asarray(val), f"{time.time() - t:.0f} s", flush=True)
t = time.time()
J = np.asarray(jax.jacrev(rows)(z0))
print(f"jacrev {time.time() - t:.0f} s", flush=True)
h = 1e-5
for k in range(len(dirs)):
    e = np.zeros(len(dirs)); e[k] = h
    fd = (np.asarray(rows(jnp.asarray(e))) - np.asarray(rows(jnp.asarray(-e)))) / (2 * h)
    rel = np.abs(J[:, k] - fd) / np.maximum(np.abs(fd), 1e-8)
    print(f"dir {dirs[k]}: AD {J[:, k]}\n                FD {fd}\n                rel {rel}", flush=True)
