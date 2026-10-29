#!/usr/bin/env python
"""Solve a free boundary from the three plasma-vacuum interface conditions.

NESTOR (``vj.solve_free_boundary_multigrid``) makes the boundary a flux surface
of the vacuum field and balances only ``|B|`` across it, so its equilibrium may
carry an edge sheet current.  The three-term solve instead finds the boundary
on which all three interface conditions hold -- ``B . n = 0``, pressure balance
and no sheet current, ``n x (B_out - B_in) = 0`` -- with the plasma's own field
from virtual casing (see docs/howto/free-boundary.md).

The case is the Landreman & Paul (2021) precise QA at 0.5% beta with the
ESSOS coils fitted to it.  Its fixed-boundary design is the start, and the free
boundary is solved twice: with the default ``labelling="gauge"`` (VMEC's
spectral condensation held exactly on the boundary and Newton on that gauge:
the boundary and its derivatives are determined) and with ``labelling="none"``
(the plain least squares).  Both reduce the design's interface residuals to
the floor of the resolution and land on nearly the same surface.  Then the
coil currents change by 1% and ``previous=`` continues from the gauge result
without compiling the model again.  The same solve runs from a deck with
``LTHREE_TERMS = T`` (``vmex input.case``) or through
``vj.solve_free_boundary_multigrid(..., three_terms=vj.ThreeTermsConfig())``.

Needs ``vmex[coils,freeb]`` (ESSOS and virtual-casing-jax) and a GPU: about
5 minutes on one A100, most of it compilation.  ``VMEX_EXAMPLES_CI=1`` runs a
coarse smoke pass.
"""

import os
from dataclasses import replace
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import vmex as vj  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_beta0p5_bootstrap"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"

# Plasma resolution: poloidal and toroidal modes, radial surfaces, force tolerance:
MPOL = NTOR = 4
NS, FTOL = 25, 1e-12

# The interface grid per field period and the fixed virtual-casing quadrature
# (4 nfp nphi, 2 ntheta); the boundary modes solved for (default: all):
GRID = 32
MAX_MODE = None

# The coil-current change of the warm continuation:
CURRENT_CHANGE = 0.01

OUTPUT_DIR = Path("output_free_boundary_three_terms")

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MPOL = NTOR = 3
    NS, FTOL, GRID = 11, 1e-11, 16

###############################################################################
# End of input parameters.
###############################################################################

### Coils and deck ############################################################

from essos.coils import Coils  # noqa: E402 (optional heavy import)
from essos.fields import BiotSavart  # noqa: E402

coils = Coils.from_json(str(COILS_FILE))
field = BiotSavart(coils)

inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([NS]),
              niter_array=np.array([20000]), ftol_array=np.array([FTOL]), delt=0.5)
nfp = int(inp.nfp)
settings = dict(nphi=GRID, ntheta=GRID, quadrature=(4 * nfp * GRID, 2 * GRID), max_mode=MAX_MODE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def report(name, result, seconds):
    tt = result.three_terms
    print(f"{name:>14s}: {'converged' if result.converged else 'NOT converged'} in {seconds:.0f} s "
          f"({tt.nfev} boundary evaluations, {tt.njev} Jacobians); interface residuals B.n "
          f"{tt.boundary_residual.normal:.2e}, pressure {tt.boundary_residual.pressure:.2e}, "
          f"sheet current {tt.boundary_residual.sheet_current:.2e}")


def lcfs_distance(a, b, ntheta=361):
    """Largest distance [m] from the LCFS of wout ``a`` to that of wout ``b`` on two cross-sections."""
    theta, worst = np.linspace(0.0, 2.0 * np.pi, ntheta), 0.0
    for phi in (0.0, np.pi / nfp):
        def rz(w):
            angle = np.outer(theta, np.asarray(w.xm)) - np.asarray(w.xn) * phi
            return np.cos(angle) @ np.asarray(w.rmnc)[-1], np.sin(angle) @ np.asarray(w.zmns)[-1]
        (R, Z), (R0, Z0) = rz(a), rz(b)
        worst = max(worst, float(np.max(np.min(np.hypot(R[:, None] - R0[None], Z[:, None] - Z0[None]), axis=1))))
    return worst


### Solve with both labellings ################################################

results, wouts = {}, {}
for labelling in ("gauge", "none"):
    started = time.perf_counter()
    results[labelling] = vj.solve_free_boundary_three_terms(
        inp, external_field=field, config=vj.ThreeTermsConfig(labelling=labelling, **settings))
    report(labelling, results[labelling], time.perf_counter() - started)
    wouts[labelling] = vj.wout_from_result(results[labelling].three_terms.input, results[labelling])
    print(f"Wrote {vj.write_wout(OUTPUT_DIR / f'wout_three_terms_{labelling}.nc', wouts[labelling])}")

start = results["gauge"].three_terms.initial_boundary_residual
print(f"{'design':>14s}: interface residuals B.n {start.normal:.2e}, pressure {start.pressure:.2e}, "
      f"sheet current {start.sheet_current:.2e} (the fixed-boundary start)")
print(f"gauge and none: LCFS {1e3 * lcfs_distance(wouts['gauge'], wouts['none']):.2f} mm apart, "
      f"gauge Newton {len(results['gauge'].three_terms.newton['steps'])} steps")

### Continue after a coil change ##############################################

n_curves = coils.dofs_curves.size
moved = BiotSavart(coils.with_dofs(jnp.concatenate([coils.dofs[:n_curves],
                                                    coils.dofs_currents * (1.0 + CURRENT_CHANGE)])))
started = time.perf_counter()
warm = vj.solve_free_boundary_three_terms(inp, external_field=moved, previous=results["gauge"],
                                         config=vj.ThreeTermsConfig(**settings))
report(f"currents +{100 * CURRENT_CHANGE:.0f}%", warm, time.perf_counter() - started)
print(f"the coil change moves the LCFS by "
      f"{1e3 * lcfs_distance(vj.wout_from_result(warm.three_terms.input, warm), wouts['gauge']):.2f} mm")

ok = all(result.converged for result in (*results.values(), warm))
if not ok and not ci_smoke:
    raise SystemExit("a three-term solve did not converge")
