"""Vacuum iota (no pressure, no current) of saved run boundaries, beside their finite-beta iota.

    python extra/vacuum_iota_runs.py <run dir> <wout name>...     (cwd: the vmex tree, GPU)
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core.statephysics import _iotas_half  # noqa: E402

run = Path(sys.argv[1])
deck = vj.VmecInput.from_file(run / "input.run")
for name in sys.argv[2:]:
    w = vj.read_wout(run / name)
    inp = tt.boundary_from_wout(deck, w)
    vac = dataclasses.replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                              ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None,
                              lfreeb=False)
    eq = opt.solve_equilibrium(vac)
    half = np.abs(np.asarray(_iotas_half(eq.state, eq.runtime))[1:])
    axis, edge = abs(float(opt.axis_iota(eq.state, eq.runtime))), 1.5 * half[-1] - 0.5 * half[-2]
    ns = half.size
    row = dict(run=run.name, wout=name, vacuum_axis=round(axis, 4), vacuum_mid=round(float(half[ns // 2]), 4),
               vacuum_edge=round(float(edge), 4), vacuum_min=round(float(min(axis, half.min(), edge)), 4),
               beta_iota_axis=round(abs(float(w.iotaf[0])), 4), beta_iota_mid=round(abs(float(w.iotas[1 + ns // 2])), 4),
               beta_iota_edge=round(abs(float(w.iotaf[-1])), 4), betatotal=round(float(w.betatotal), 5),
               aspect=round(float(w.aspect), 3), curtor_kA=round(float(w.ctor) / 1e3, 1), converged=bool(eq.result.converged))
    print(json.dumps(row), flush=True)
