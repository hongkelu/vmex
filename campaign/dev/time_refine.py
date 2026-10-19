"""Phase timings of one certified implicit solve at a moved boundary (vacuum and finite beta).

    COIL_CASE=qa4-beta python extra/time_refine.py <run dir with input.run>     (cwd: the vmex tree, GPU)
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import vmex as vj  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

inp = dataclasses.replace(vj.VmecInput.from_file(Path(sys.argv[1]) / "input.run"), lfreeb=False)
vacuum = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                             ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None)


def wrap(name, fn, show):
    def inner(*a, **k):
        jax.effects_barrier()
        t = time.perf_counter()
        out = fn(*a, **k)
        jax.block_until_ready(out)
        print(f"    {name:22s} {time.perf_counter() - t:7.2f} s {show(out)}", flush=True)
        return out
    return inner


im._host_solve = wrap("host VMEC solve", im._host_solve, lambda o: "")
im._refine_block_factors = wrap("block factorization", im._refine_block_factors, lambda o: "")
im._refine_block_step = wrap("block Newton step", im._refine_block_step,
                             lambda o: f"|F| {float(o[2]):.2e} linear {float(o[3]):.1e}")
im._refine_step = wrap("GCROT Newton step", im._refine_step, lambda o: f"|F| {float(o[2]):.2e}")

for label, deck in (("vacuum", vacuum), ("finite beta", inp)):
    cfg = im.make_config(deck, multigrid=True, hot_restart=True)
    p0 = im.params_from_input(deck)
    print(f"== {label}: ftol {cfg.ftol:g}, refine_tol {cfg.refine_tol:g}", flush=True)
    for k in range(3):
        rbc = np.array(deck.rbc); rbc[deck.ntor - 1, 1] += k * 1e-3
        p = dataclasses.replace(p0, rbc=jax.numpy.asarray(rbc))
        before = dict(im._SOLVE_STATS.get(cfg, {}))
        t = time.perf_counter()
        jax.block_until_ready(im.solve_implicit(p, cfg))
        stats = im._SOLVE_STATS.get(cfg, {})
        diff = {key: stats[key] - before.get(key, 0) for key in stats if isinstance(stats[key], (int, float))
                and stats[key] != before.get(key, 0)}
        print(f"  boundary {k}: total {time.perf_counter() - t:.2f} s  {diff}", flush=True)
