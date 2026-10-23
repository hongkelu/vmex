"""B1: the fixed-boundary bootstrap model (Gauss-Newton current) against the Picard loop, with the m = 1 policy fixed
for the whole process (env M1: solved | frozen), cross-evaluating both currents.

usage: M1=solved python model_vs_picard.py OUT.json   (cwd: a vmex tree; a GPU)

test_fixed_boundary_bootstrap_model_matches_picard's deck (LP QA beta 0.5% bootstrap, mpol 3, ns 25).  Reported:
both current profiles (ac_aux_f), their largest relative difference, the integrated current (CURTOR), and each
current's bootstrap closure as the model measures it (model.bootstrap_residual at the model's evaluator).
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

M1 = os.environ.get("M1", "solved")

sys.path.insert(0, "tests")
import test_freeboundary_vc as t  # noqa: E402
from vmex.core import bootstrap as bs  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402


def main():
    """Model and Picard currents, cross-evaluated; write OUT.json."""
    inp = dataclasses.replace(t._deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False,
                              ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = t._lp_beta0p5_profiles(inp)
    model = fvc.ThreeTermFreeBoundaryModel(inp, bootstrap=profiles, fixed_boundary=True, m1_complement=M1)
    out = model.solve_boundary(model.params0, None, ftol=1e-10)
    state, _, params, _ = out["aux"]
    model_current = np.asarray(params.ac_aux_f)
    picard = bs.self_consistent_bootstrap(inp, profiles, 0, n_iter=60, tol=1e-9, relax=0.5, profile="half_mesh")
    picard_current = np.asarray(picard.input.ac_aux_f)
    scale = np.max(np.abs(picard_current))
    res = dict(m1=M1, model_current=model_current.tolist(), picard_current=picard_current.tolist(),
               max_rel=float(np.max(np.abs(model_current - picard_current)) / scale),
               model_curtor=float(np.asarray(params.curtor)), picard_curtor=float(picard.input.curtor),
               picard_status=str(picard.status), picard_iterations=int(picard.iterations),
               picard_closure_error=float(picard.history[-1].get("closure_error", np.nan)),
               model_closure=float(model.bootstrap_residual(state, params)))
    # the Picard current evaluated by the model's own evaluator (same knots): its closure there
    try:
        p_params = dataclasses.replace(params, ac_aux_f=picard.input.ac_aux_f, curtor=picard.input.curtor)
        st_p = model.solve(p_params, None, True)[0]
        res["picard_current_model_closure"] = float(model.bootstrap_residual(st_p, p_params))
    except Exception as error:  # report, do not hide
        res["picard_current_model_closure_error"] = repr(error)[:300]
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
