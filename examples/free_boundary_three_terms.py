#!/usr/bin/env python
"""Free boundary from the three plasma-vacuum interface conditions, next to NESTOR.

A free boundary without an edge sheet current satisfies three conditions on
the plasma boundary: ``B . n = 0``, pressure balance and continuity of the
tangential field (Conlin et al. 2024).  NESTOR enforces the first two and
accepts any tangential jump, so a VMEC/NESTOR equilibrium may carry a sheet
current.  ``three_terms=`` of the free-boundary solver solves for the boundary
on which all three vanish instead, with the plasma's own field from virtual
casing (``pip install vmex[freeb]``).

This solves the Landreman & Paul (2021) QA plasma at 0.5% beta in its fitted
ESSOS coils with NESTOR and with the three-term solver in its labellings
(``"start"``, ``"equal_arc"`` and ``"none"``: how the boundary's poloidal
angle, which the interface conditions leave free, is fixed), and reports each
result's interface residuals, edge rotational transform and distance from
NESTOR's last closed surface.  The same solve runs from a deck with
``LTHREE_TERMS = T`` (``vmex input.case``).
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# The fitted coils and the plasma deck that seeds the initial guess:
DATA_DIR = Path(__file__).resolve().parent / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_beta0p5_bootstrap"

# Plasma resolution (the boundary's Fourier modes are solved up to MPOL - 1 = NTOR):
MPOL, NTOR, NS = 4, 4, 25
NITER, FTOL = 5000, 1e-10

# Coil-field table; NZETA must divide its toroidal plane count for NESTOR:
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
NZETA = 16

# The interface grid over one field period and its fixed singular quadrature, (4 nfp N, 2 N):
N_INTERFACE = 48
LABELLINGS = ["start", "equal_arc", "none"]

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs: a coarse
# plasma, one labelling and a short fit:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
MAX_NFEV = 60
if ci_smoke:
    MPOL, NTOR, NS, N_INTERFACE, MAX_NFEV = 3, 3, 13, 24, 20
    LABELLINGS = ["equal_arc"]

###############################################################################
# End of input parameters.
###############################################################################

from essos.coils import Coils  # noqa: E402 (optional heavy import)

coils = Coils.from_json(str(COILS_FILE))
# Tricubic: virtual casing samples the field between the table's planes, where a trilinear table has kinks.
field = vj.MgridField.from_coils(coils, order=3, **GRID)

inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6,
                                                           nzeta=NZETA)
inp = replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([NS]),
              niter_array=np.array([NITER]), ftol_array=np.array([FTOL]))


def lcfs_distance_mm(a, b, nphi=5, ntheta=181):
    """Largest distance from ``a``'s last closed surface to ``b``'s over a few toroidal planes [mm]."""
    worst = 0.0
    theta, fine = np.linspace(0, 2 * np.pi, ntheta), np.linspace(0, 2 * np.pi, 4 * ntheta)
    for phi in np.linspace(0, np.pi / int(a.nfp), nphi):
        def rz(w, t):
            angle = np.outer(t, np.asarray(w.xm)) - np.asarray(w.xn) * phi
            return np.cos(angle) @ np.asarray(w.rmnc)[-1], np.sin(angle) @ np.asarray(w.zmns)[-1]
        (R, Z), (R0, Z0) = rz(a, theta), rz(b, fine)
        worst = max(worst, float(np.max(np.min(np.hypot(R[:, None] - R0[None], Z[:, None] - Z0[None]), axis=1))))
    return 1e3 * worst


nestor = vj.wout_from_result(inp, vj.solve_free_boundary_multigrid(inp, external_field=field))
print(f"NESTOR: iota edge {float(np.asarray(nestor.iotaf)[-1]):+.5f}")
for labelling in LABELLINGS:
    config = vj.ThreeTermsConfig(labelling=labelling, start="deck", nphi=N_INTERFACE, ntheta=N_INTERFACE,
                                 quadrature=(4 * int(inp.nfp) * N_INTERFACE, 2 * N_INTERFACE), max_nfev=MAX_NFEV)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, three_terms=config)
    three, res = result.three_terms, result.three_terms.boundary_residual
    wout = vj.wout_from_result(inp, result)
    vj.write_wout(f"wout_three_terms_{labelling}.nc", wout)
    print(f"three terms, {labelling} labelling: B.n {res.normal:.2e}, pressure balance {res.pressure:.2e}, "
          f"sheet current {res.sheet_current:.2e} (start: B.n {three.initial_boundary_residual.normal:.2e}, "
          f"sheet current {three.initial_boundary_residual.sheet_current:.2e}); iota edge "
          f"{float(np.asarray(wout.iotaf)[-1]):+.5f}; {lcfs_distance_mm(wout, nestor):.2f} mm from NESTOR; "
          f"{'solved' if three.success else 'not solved: ' + three.message}")
    if not ci_smoke and not three.success:
        raise SystemExit(1)
