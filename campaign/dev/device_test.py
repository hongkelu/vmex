"""Where the certification factorization runs, and its time inside vs outside the solve callback.

    python extra/device_test.py <run dir with input.run>     (cwd: vmex tree, GPU)
"""
import dataclasses
import sys
import threading
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core.device import resolve_implicit_device  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
inp = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                          ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None)
seen = {}
original = im._refine_block_factors


def factors(cfg, params, frozen, mask, z):
    t = time.perf_counter()
    out = jax.block_until_ready(original(cfg, params, frozen, mask, z))
    dt = time.perf_counter() - t
    leaf = jax.tree.leaves(out)[0]
    print(f"    factorization {dt:6.2f} s on {leaf.devices()} (z on {jax.tree.leaves(z)[0].devices()}), "
          f"thread {threading.current_thread().name}", flush=True)
    seen["args"] = (cfg, params, frozen, mask, z)
    return out


im._refine_block_factors = factors
for name, device in (("cfg.device None", None), ("cfg.device gpu", resolve_implicit_device("gpu"))):
    cfg = im.make_config(inp, multigrid=True, hot_restart=True, device=device)
    p0 = im.params_from_input(inp, device=device)
    print(f"== {name} -> {cfg.device}", flush=True)
    for k in range(3):
        rbc = np.array(inp.rbc); rbc[inp.ntor - 1, 1] += k * 1e-3
        p = dataclasses.replace(p0, rbc=jax.device_put(rbc, device) if device else jax.numpy.asarray(rbc))
        t = time.perf_counter()
        jax.block_until_ready(im.solve_implicit(p, cfg))
        print(f"  boundary {k}: {time.perf_counter() - t:.1f} s", flush=True)
    for _ in range(2):  # the same factorization, called directly on the main thread
        factors(*seen["args"])
