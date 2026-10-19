import dataclasses, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt
import vmex as vj
from vmex import optimize as opt
from vmex.core.fields import surface_currents
from vmex.core.statephysics import _field_chain
run = Path(sys.argv[1])
w = vj.read_wout(run / "wout.nc")
deck = vj.VmecInput.from_file(run / "input.run")
inp = tt.current_from_wout(dataclasses.replace(tt.boundary_from_wout(deck, w), phiedge=float(w.phi[-1]), lfreeb=False), w)
eq = opt.solve_equilibrium(inp, initial_state=vj.state_from_wout(w, inp=inp, ns=int(w.ns)))
f = _field_chain(eq.state, eq.runtime)[3]
c = surface_currents(bsubu=f.bsubu, bsubv=f.bsubv, trig=eq.runtime.trig, s=jnp.asarray(eq.runtime.setup.s_full), signgs=eq.runtime.setup.signgs)
print("wout ctor", float(w.ctor), "surface_currents ctor", float(c.ctor), "rbtor", float(c.rbtor), "buco[-3:]", c.buco[-3:])
print("iota_I", float(tt.current_iota(eq.state, eq.runtime)))
