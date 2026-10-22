"""Fast-ion confinement term for the three-term single stage (``--extension``): smooth Gamma_c.

TERMS adds ``GC_WEIGHT / 2 sum_s Gamma_c,smooth(s)^2`` with :class:`vmex.core.gammac.GammaCSmooth`, the
derivative-safe Gamma_c surrogate of ``omnigenity_epsilon_gammac_maxj.py`` (its surfaces 0.35, 0.6, 0.85,
temperature 0.15 and sampling budget). As there, the weight is set so the term equals the symmetry term at the
start: ``GC_WEIGHT = (QS or QI error) / sum Gamma_c,smooth^2`` of the starting equilibrium. Report hard Gamma_c and
alpha tracing for the result; the surrogate does not promise zero losses.

Environment: ``GC_WEIGHT`` (required), ``GC_SURFACES`` (colon-separated, default 0.35:0.6:0.85), ``GC_TEMPERATURE``
(0.15).
"""

import os

import numpy as np

from vmex.core.gammac import GammaCSmooth

SURFACES = tuple(float(v) for v in os.environ.get("GC_SURFACES", "0.35:0.6:0.85").split(":"))
TEMPERATURE = float(os.environ.get("GC_TEMPERATURE", 0.15))
BUDGET = dict(nalpha=7, num_transit=3, points_per_transit=64, num_pitch=24, quadrature_order=32)
WEIGHT = float(os.environ["GC_WEIGHT"])
print(f"[gamma_c] surfaces {list(SURFACES)}, temperature {TEMPERATURE}, weight {WEIGHT}, {BUDGET}", flush=True)
_gamma_c = GammaCSmooth(SURFACES, temperature=TEMPERATURE, **BUDGET)

ROWS = {}
TERMS = {"gamma_c_smooth": (_gamma_c.residuals_state, WEIGHT)}
