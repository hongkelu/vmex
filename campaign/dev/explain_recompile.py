"""Why the certified solve recompiles: log compiles and cache misses over three moved boundaries.

    python extra/explain_recompile.py <run dir with input.run> [vacuum]     (cwd: vmex tree, GPU)
"""
import dataclasses
import logging
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
if len(sys.argv) > 2:
    inp = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                              ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None)
cfg = im.make_config(inp, multigrid=True, hot_restart=True)
p0 = im.params_from_input(inp)
for k in range(3):
    if k == 2:
        jax.config.update("jax_log_compiles", True)
        jax.config.update("jax_explain_cache_misses", True)
        logging.basicConfig(level=logging.WARNING)
    rbc = np.array(inp.rbc); rbc[inp.ntor - 1, 1] += k * 1e-3
    p = dataclasses.replace(p0, rbc=jax.numpy.asarray(rbc))
    t = time.perf_counter()
    jax.block_until_ready(im.solve_implicit(p, cfg))
    print(f"=== boundary {k}: {time.perf_counter() - t:.1f} s", flush=True)
