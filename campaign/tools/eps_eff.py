"""Effective ripple eps_eff^{3/2}(s) of WOUTs (NEO_JAX on a booz_xform_jax transform, as the plot_wout summary).

    python extra/eps_eff.py <wout.nc | booz_small.npz> [...]     (cwd: vmex tree, GPU)

A ``.npz`` holds only the WOUT fields booz_xform_jax reads (``<field>_cols``: the nonzero columns of a Nyquist array).
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

import vmex as vj
from vmex.core import plotting as pl
from vmex.core.neoclassical import diagnostic_neo_config, epsilon_effective_from_boozer


def load(name):
    if not name.endswith(".npz"):
        return vj.read_wout(name)
    d = dict(np.load(name))
    for k in [k for k in d if k.endswith("_cols")]:
        f = k.removesuffix("_cols")
        full = np.zeros((d[f].shape[0], d["xm_nyq"].size))
        full[:, d.pop(k)] = d[f]
        d[f] = full
    return SimpleNamespace(**{k: (v.item() if v.ndim == 0 else v) for k, v in d.items()})


for name in sys.argv[1:]:
    booz = pl._boozer_summary_data(load(name), n_surfaces=10)
    s, e = epsilon_effective_from_boozer(booz["neo_booz"], config=diagnostic_neo_config())
    s, e = np.asarray(s, float), np.asarray(e, float)
    print(json.dumps(dict(wout=name, s=s.round(3).tolist(), eps_eff_32=e.tolist(),
                          eps_eff=np.cbrt(e**2).tolist())), flush=True)
