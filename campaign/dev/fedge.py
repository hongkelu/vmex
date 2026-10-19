"""Force residuals (fsqr, fsqz, fsql and the spectral edge force fedge) of a WOUT's equilibrium, re-solved from it.

    python extra/fedge.py <input.run> <wout> [<wout> ...]     (cwd: vmex tree, GPU)

The deck's profiles with the WOUT's boundary, PHIEDGE and current, restarted from the WOUT's state at its resolution.
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402

deck = vj.VmecInput.from_file(sys.argv[1])
for name in sys.argv[2:]:
    w = vj.read_wout(name)
    inp = tt.current_from_wout(dataclasses.replace(tt.boundary_from_wout(deck, w), phiedge=float(w.phi[-1]),
                                                   lfreeb=False), w)
    inp = dataclasses.replace(inp, ns_array=[int(w.ns)], ftol_array=[float(deck.ftol_array[-1])])
    eq = opt.solve_equilibrium(inp, initial_state=vj.state_from_wout(w, inp=inp, ns=int(w.ns)))
    r = eq.result
    print(json.dumps(dict(wout=name, ns=int(w.ns), converged=bool(r.converged), fsqr=float(r.fsqr),
                          fsqz=float(r.fsqz), fsql=float(r.fsql), fedge=float(r.fedge),
                          wout_fsqr=float(w.fsqr), wout_fsqz=float(w.fsqz), wout_fsql=float(w.fsql))), flush=True)
