"""Which labelling's 8 x 8 answer is closer to the resolution-converged one? Re-solve saved boundaries at higher modes.

usage: python labelling_resolution.py CHECKPOINT [CHECKPOINT ...] --modes 8 12 16   (one JSON line per checkpoint)

Each checkpoint's boundary ``b`` (8 x 8 model coordinates) is written into its run's deck and solved fixed-boundary,
with the deck's own mpol = ntor raised to each requested M (the boundary keeps its coefficients; higher ones are
zero), at the deck's ns. Reported per M:
- ``vacuum_iota_min``: min over the half mesh of the vacuum |iota| (no pressure or current), the optimizer's
  vacuum-iota row;
- ``iota_min``: min |iota| of the deck's finite-beta equilibrium (prescribed pressure and current);
- ``dmerc_min``: min of Mercier's D_Merc on s in [0.2, 1) (the optimizer's range);
- ``qs_total`` (with ``--helicity M N``): the quasisymmetry ratio residual's sum of squares on s = 0.1..1 (the
  optimizer's QA/QH target, unscaled), of the finite-beta equilibrium.

The same physical surface carried in two labellings should give the same values as M grows. Their 8 x 8 values
differ by each labelling's discretization error.
"""
import argparse
import json
import tempfile
from dataclasses import replace

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--modes", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--max-mode", type=int, default=8)
    p.add_argument("--helicity", type=int, nargs=2, metavar=("M", "N"))
    args = p.parse_args()

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.optimize import boundary_arrays_from_x

    qs = None if args.helicity is None else opt.QuasisymmetryRatioResidual(
        np.array([i / 10 for i in range(1, 11)]), *args.helicity)
    for path in args.checkpoints:
        data = np.load(path, allow_pickle=False)
        deck = json.loads(str(data["meta"]))["definition"]["deck"]
        with tempfile.NamedTemporaryFile("w", suffix=".run", delete=False) as f:
            f.write(deck)
        base = vj.VmecInput.from_file(f.name)
        rbc, zbs = map(np.asarray, boundary_arrays_from_x(base, np.asarray(data["b"], float), args.max_mode,
                                                         vary_major_radius=True))
        base = replace(base, rbc=rbc, zbs=zbs, lfreeb=False)
        out = dict(checkpoint=path, modes={})
        for M in args.modes:
            inp = base.change_resolution(mpol=M, ntor=M)
            row = {}
            for name, deck_m in (("finite_beta", inp),
                                 ("vacuum", replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1,
                                                    pcurr_type="power_series",
                                                    ac=np.zeros_like(np.asarray(inp.ac, dtype=float)),
                                                    ac_aux_s=None, ac_aux_f=None))):
                try:
                    eq = opt.solve_equilibrium(deck_m, raise_on_max_iterations=True)
                    iota = np.abs(np.asarray(eq.wout.iotas)[1:])
                    row[name] = dict(iota_min=float(iota.min()), iota_edge=float(np.abs(np.asarray(eq.wout.iotaf)[-1])),
                                     iota_axis=float(np.abs(np.asarray(eq.wout.iotaf)[0])))
                    dmerc = np.asarray(getattr(eq.wout, "DMerc", np.zeros(0)), dtype=float)
                    if dmerc.size:  # Mercier on s in [0.2, 1) (the optimizer's range), the full-mesh interior
                        s_full = np.linspace(0.0, 1.0, dmerc.size)
                        keep = (s_full >= 0.2) & (s_full < 1.0)
                        row[name]["dmerc_min"] = float(dmerc[keep].min())
                    if qs is not None and name == "finite_beta":
                        row[name]["qs_total"] = float(qs.compute(eq.wout)["total"])
                except Exception as error:  # a resolution that does not converge is reported, not fatal
                    row[name] = dict(error=f"{type(error).__name__}: {str(error)[:160]}")
            out["modes"][M] = row
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
