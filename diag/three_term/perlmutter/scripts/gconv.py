"""G vs G + N I in Redl's QH prefactor on a solved deck: N I / G per surface, the <J.B> change, and the change in
the Redl-implied I'(s) and its integral (CURTOR).  usage: gconv.py <input.run> <helicity_n> <profiles: n0 T0>"""
import json
import sys

import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core import bootstrap as bs

inp = vj.VmecInput.from_file(sys.argv[1])
inp = vj.VmecInput.from_file(sys.argv[1])
from dataclasses import replace
inp = replace(inp, lfreeb=False)
helicity, n0, T0 = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
profiles = bs.KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), T0 * np.array([1.0, -1.0]),
                              T0 * np.array([1.0, -1.0]))
eq = opt.solve_equilibrium(inp)
state, rt = eq.solution, eq.solver_context
hm = bs._half_mesh_fields(state, rt)
geom = bs._geometry_from_half(hm, hm.s_half, n_lambda=bs.N_LAMBDA, refine_extrema=True)
shift = np.asarray(bs.isomorphism_g_shift(geom, helicity))
out = {}
for conv in bs.G_CONVENTIONS:
    jr = bs.j_dot_B_redl(profiles, geom, helicity, g_convention=conv)[0]
    dI = np.asarray(bs._current_derivative(hm, jr))
    s = np.asarray(hm.s_half)
    out[conv] = dict(jdotb=np.asarray(jr).tolist(), dIds=dI.tolist(), curtor=float(np.sum(dI) * (s[1] - s[0])))
rel = (np.array(out["isomorphism"]["jdotb"]) - np.array(out["simsopt"]["jdotb"])) / np.max(
    np.abs(out["simsopt"]["jdotb"]))
summary = dict(deck=sys.argv[1], helicity_n=helicity, nfp=int(inp.nfp), ctor_deck=float(eq.wout.ctor),
               NI_over_G=dict(min=float(shift.min()), max=float(shift.max()), edge=float(shift[-1])),
               jdotb_rel_change_max=float(np.max(np.abs(rel))),
               curtor=dict(simsopt=out["simsopt"]["curtor"], isomorphism=out["isomorphism"]["curtor"]),
               curtor_rel_change=out["isomorphism"]["curtor"] / out["simsopt"]["curtor"] - 1)
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, s=np.asarray(hm.s_half).tolist(), shift=shift.tolist(), **out),
          open(sys.argv[1] + ".gconv.json", "w"))
