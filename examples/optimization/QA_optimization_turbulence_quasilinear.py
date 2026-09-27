#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary against a quasilinear heat flux.

``QA_optimization.py`` with one more objective tuple: GKX's mixing-length
quasilinear heat flux ``gamma * W_Q / <k_perp^2>`` of the most unstable
gyrokinetic eigenmode, on a set of flux tubes chosen by physical radius ``s``
(normalized toroidal flux) and field-line label ``alpha``, reduced by a mean
or, with SOFTMAX_TEMPERATURE, by a smooth maximum. ``W_Q`` is the eigenmode's
heat flux per unit field energy and ``<k_perp^2>`` its field-weighted
perpendicular wavenumber, so the objective sees the mode structure, not only
its growth rate (``QA_optimization_turbulence_linear.py``).

Both weights depend on the eigenvector of a non-symmetric operator; GKX
differentiates it (``enable_eigvec_derivs``, JAX >= 0.10.1) in forward and
reverse mode, so SciPy receives VMEX's exact implicit Jacobian of every
residual row. Needs ``pip install 'vmex[turbulence]'``.

The mixing-length rule is a model of saturation with an uncalibrated
amplitude; ``QA_optimization_turbulence_nonlinear.py`` optimizes the heat flux
of the saturated nonlinear state instead.

Measured cost (office host, CPU, default settings): see the numbers printed
per stage and the table in examples/README.md.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.turbulence import quasilinear_flux_proxy

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]

# Targets:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Flux tubes: every (s, alpha) pair is one tube. s is the normalized toroidal
# flux, the same radius at every radial resolution. Add radii or lines here:
TUBE_S = [0.5]
TUBE_ALPHAS = [0.0]
SOFTMAX_TEMPERATURE = None        # None: mean over tubes; e.g. 0.01: smooth max

# Gyrokinetic model on each tube: drive in a/L, one binormal wavenumber
# ky = 2 pi / LY (in rho_ref units), and the Hermite-Laguerre resolution:
A_OVER_LT, A_OVER_LN = 3.0, 1.0
LY = 12.0
N_LAGUERRE, N_HERMITE = 4, 8
NTHETA = 32                       # parallel grid points over one poloidal turn

# Weight of the quasilinear heat-flux term:
FLUX_WEIGHT = 1.0

# Step control, as in QA_optimization.py:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2

# Equilibrium resolution: mode numbers max_mode + 2, never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 51
FINAL_FTOL = 1.0e-13
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QA_turbulence_quasilinear_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [3]
    N_LAGUERRE, N_HERMITE, NTHETA = 2, 3, 16
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

### Set up the objective ######################################################

def iota_floor(state, runtime):
    """Hinge on the profile minimum of |iota|."""
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


def heat_flux(state, runtime):
    """Mixing-length quasilinear heat flux, reduced over the flux tubes."""
    fluxes = jnp.stack([
        quasilinear_flux_proxy(
            state, runtime, s=s, alpha=alpha, ntheta=NTHETA,
            n_laguerre=N_LAGUERRE, n_hermite=N_HERMITE, ly=LY,
            a_over_lt=A_OVER_LT, a_over_ln=A_OVER_LN)
        for s in TUBE_S for alpha in TUBE_ALPHAS])
    if SOFTMAX_TEMPERATURE is None:
        return jnp.mean(fluxes)
    return SOFTMAX_TEMPERATURE * logsumexp(fluxes / SOFTMAX_TEMPERATURE)


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (heat_flux, 0.0, FLUX_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("QL heat flux", heat_flux, ".5f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
seed_flux = report("seed", equilibrium)["QL heat flux"]
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA + quasilinear flux stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode, use_ess=True,
        ess_alpha=ESS_ALPHA, restart_from=equilibrium)
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# Re-solve on a finer radial grid; the tubes sit at the same s there.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

final_flux = report("final", final_equilibrium)["QL heat flux"]
print(f"\nQL heat flux {seed_flux:.5f} -> {final_flux:.5f} at NS = {FINAL_NS}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
