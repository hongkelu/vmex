#!/usr/bin/env python
"""Single-stage fixed-boundary plasma and ESSOS coil optimization.

Use the commented ``Coils.from_simsopt`` line to replace the generated coils
with a SIMSOPT coil JSON while keeping the objective and derivative code.
Preview: this script needs ESSOS branch ``rj/vmex-optimization-interfaces``.
"""

from dataclasses import replace
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

# Keep the original defaults; comparison runs explicitly reuse the saved fit.
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--coils", type=Path, help="reuse fitted coils and skip the preliminary fit")
parser.add_argument("--maxiter", type=int, help="override MAXITER below")
parser.add_argument("--ftol", type=float, help="override equilibrium force tolerance")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
parser.add_argument("--no-plots", action="store_true")
parser.add_argument("--constrained", action="store_true", help="SLSQP with shared iota and physical-radius inequalities")
args = parser.parse_args()
if (args.maxiter is not None and args.maxiter < 1) or (args.ftol is not None and not 0 < args.ftol < 1):
    parser.error("invalid iteration budget or force tolerance")
if args.coils:
    args.coils = args.coils.resolve()
if args.output:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
    path = Path.cwd() / "cache" / key
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
                  VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im

import jax
import jax.numpy as jnp
from _scalar_diagnostics import StepHistory
import _scalar_constraints as limits

if jax.default_backend() != args.device or not jax.config.x64_enabled:
    raise RuntimeError("requested device and float64 precision are required")

try:
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary
except ImportError as error:
    raise ImportError(
        "This example needs ESSOS branch rj/vmex-optimization-interfaces "
        "(uwplasma/ESSOS#58)."
    ) from error

nfp = 2  # number of field periods
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", "B.n/B", or a callable ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODE = 3  # final mode limit of qa_optimization.py; one scalar solve
MAXITER = 100
COIL_FIT_MAXITER = 200  # preliminary coil-only fit on the frozen seed boundary
ASPECT_TARGET = 5.0
ASPECT_WEIGHT = 1.0
IOTA_FLOOR = 0.19  # matches the current qa_optimization.py
IOTA_WEIGHT = 10.0
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
PARAMETER_STEP = 0.1
COIL_STEP = 0.05
ESS_ALPHA = 1.2

N_COILS = 3
COIL_ORDER = 5
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

NORMAL_FIELD_WEIGHT = 1.0e3
NORMAL_FIELD_LIMIT = 0.01
NORMAL_FIELD_OBJECTIVE_LIMIT = 0.008  # margin for the independent final grid
NORMAL_FIELD_LIMIT_WEIGHT = 2.0e5
LENGTH_TARGET = 5
LENGTH_WEIGHT = 1.0
CURVATURE_LIMIT = 7.0
CURVATURE_OBJECTIVE_LIMIT = 6.9  # margin for the independent final grid
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_WEIGHT = 1.0e3
COIL_SURFACE_DISTANCE_LIMIT = 0.20
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e3

# A toroidal grid commensurate with the coil count can alias narrow B.n/B structure.
NPHI, NTHETA = 37, 32
METHOD = "BFGS"  # also accepts "L-BFGS-B"
PARAMETER_BOUND = 5.0  # used only when METHOD is L-BFGS-B
OPTIONS = {"maxiter": MAXITER, "gtol": 1.0e-8}
if args.maxiter is not None:
    OPTIONS["maxiter"] = args.maxiter
if METHOD == "L-BFGS-B":
    OPTIONS.update(maxls=20, ftol=1e-12, maxcor=20)
if args.constrained:
    METHOD = "SLSQP"
    OPTIONS = {"maxiter": args.maxiter or MAXITER, "ftol": 1e-10}

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, N_SEGMENTS, NPHI, NTHETA = 1, 24, 8, 8
    COIL_FIT_MAXITER = 2
    COIL_ORDER = 2
    OPTIONS = {"maxiter": MAXITER, "gtol": 1.0e-5}

DATA = Path(__file__).resolve().parent / "input.rotating_ellipse"
inp = vj.VmecInput.from_file(DATA)
# Use the shared input boundary and resolution unchanged.
if args.constrained and args.ftol is None:
    args.ftol = limits.FORCE_TOLERANCE
if args.ftol is not None:
    inp = replace(inp, ftol_array=np.array([args.ftol]))

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_terms = [
    (qs.residuals_state, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (iota_floor, 0.0, IOTA_WEIGHT),
]
def plasma_loss(state, runtime):
    rows = opt.residuals_from_tuples(state, runtime, plasma_terms)
    return 0.5 * jnp.vdot(rows, rows)


plasma_problem = opt.VmecProblem.from_loss(
    inp, plasma_loss, max_mode=MAX_MODE, vary_major_radius=VARY_MAJOR_RADIUS or args.constrained,
    use_ess=True, ess_alpha=ESS_ALPHA, progress=not ci_smoke, device=args.device)

# Shared starting geometry/currents; refine coils on the seed boundary below.
coils0 = Coils.from_json(str(DATA.with_name("coils.initial.scalar.json")))
curves0 = coils0.curves
# To start from a SIMSOPT coil file instead, use:
# coils0 = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=STELLSYM)
# curves0 = coils0.curves

def normalized_normal_field(coils, surface):
    field = BiotSavart(coils)
    magnetic_field = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(magnetic_field * surface.unitnormal, axis=2) / jnp.linalg.norm(magnetic_field, axis=2)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

def normal_field_residual(coils, surface):
    """Area-weighted rows of B.n/|B| over the target boundary.

    A flux surface requires B.n = 0 on it, so these rows are the quadrature
    of the surface-averaged square error the coils leave behind, normalized
    by |B| to make it dimensionless and by the area element so refining the
    grid does not change the objective. Driving them to zero is what makes
    the coil set reproduce the plasma boundary.
    """
    weights = surface.area_element / jnp.sum(surface.area_element)
    values = normalized_normal_field(coils, surface)
    return (jnp.sqrt(weights) * values).ravel()

def normal_field_excess(coils, surface):
    """Hinge on the largest local B.n/|B|, zero until the limit is exceeded.

    The residual above is an average, and an average tolerates one bad patch
    by trading it against a well-matched remainder; an island forms where the
    error is locally worst, not where it is worst on average. This term reads
    the maximum instead (through a logsumexp so it stays differentiable) and
    contributes nothing until that maximum passes
    NORMAL_FIELD_OBJECTIVE_LIMIT, which keeps it from competing with the
    average term over shapes that already satisfy the bound.
    """
    values = jnp.sqrt(normalized_normal_field(coils, surface)**2 + 1.0e-12)
    smooth_maximum = jax.scipy.special.logsumexp(2000.0 * values) / 2000.0
    return jnp.maximum(smooth_maximum - NORMAL_FIELD_OBJECTIVE_LIMIT, 0.0)

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)

coil_terms = [
    (normal_field_residual, 0.0, NORMAL_FIELD_WEIGHT),
    (normal_field_excess, 0.0, NORMAL_FIELD_LIMIT_WEIGHT),
    (coil_lengths, LENGTH_TARGET, LENGTH_WEIGHT),
    (coil_curvature_excess, 0.0, CURVATURE_WEIGHT),
]
coil_term_names = tuple(function.__name__ for function, _target, _weight in coil_terms) + (
    "coil separation", "coil-surface separation")

# The public VMEX problem owns the boundary-mode convention and RBC(0,0) choice.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)

n_curve_dofs = curves0.dofs.size
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
dof_names = plasma_problem.dof_names + curves0.dof_names

# SciPy works in dimensionless increments u, with x = x0 + scales*u.
scales = np.concatenate([
    PARAMETER_STEP * plasma_problem.scales, np.full(n_curve_dofs, COIL_STEP)])


def objects_from_x(x):
    x_boundary, x_coils = x[:x_boundary0.size], x[x_boundary0.size:]
    rbc, zbs = plasma_problem.boundary_from_x(x_boundary)
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

    coils = coils0.with_dofs(jnp.concatenate((x_coils, coils0.dofs_currents)))
    return surface, coils


def coil_costs(x):
    surface, coils = objects_from_x(x)
    rows = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
            for function, target, weight in coil_terms]
    return jnp.concatenate([jnp.stack([0.5 * jnp.vdot(row, row) for row in rows]), jnp.asarray([
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])])

solver_config = plasma_problem.metadata["config"]
accepted_state = im._LAST_SOLVE[solver_config][1].state

if args.constrained:
    from vmex.core.statephysics import major_radius
    constraint_problem = opt.VmecProblem.from_tuples(inp,
        [(opt.min_abs_iota, 0.0, 1.0), (major_radius, 0.0, 1.0)],
        max_mode=MAX_MODE, vary_major_radius=True, use_ess=True, ess_alpha=ESS_ALPHA,
        implicit_jacobian_method="reverse_adjoint", device=args.device)

    def constraint_values(u):
        im._HOT_CACHE[constraint_problem.metadata["config"]] = accepted_state
        x = x0 + scales*np.asarray(u)
        values = constraint_problem.residual(x[:x_boundary0.size])
        return np.asarray(limits.inequalities(values))

    def constraint_jacobian(u):
        constraint_values(u)
        x = x0 + scales*np.asarray(u)
        values = constraint_problem.residual(x[:x_boundary0.size])
        jac = np.asarray(jax.jacfwd(limits.inequalities)(jnp.asarray(values))) @ constraint_problem.residual_jac(x[:x_boundary0.size])
        return np.pad(jac*scales[:x_boundary0.size], ((0,0),(0,n_curve_dofs)))


def plasma_component(u):
    # Every trial starts from the last accepted equilibrium.
    im._HOT_CACHE[solver_config] = accepted_state
    im._PERTURB_SEED.pop(solver_config, None)
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    x_boundary = x[:x_boundary0.size]
    value, gradient = plasma_problem.value_and_grad(np.asarray(x_boundary))
    # Keep monitoring separate from differentiation of the scalar loss.
    plasma_costs = {"plasma total": float(value)}
    scaled_gradient = gradient * jnp.asarray(scales[:x_boundary0.size])
    return (value, plasma_costs), jnp.pad(scaled_gradient, (0, x0.size - x_boundary0.size))

def coil_objective(u):
    costs = coil_costs(jnp.asarray(x0) + jnp.asarray(scales) * u)
    return jnp.sum(costs), costs


monitor = opt.OptimizationMonitor()
# The host API selects certified Jacobian/fallback work without tracing both branches.
plasma_value_and_grad = plasma_component
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

# VMEX supplies the exact equilibrium derivative; JAX differentiates the coil
# objective. Their values and gradients add directly for any SciPy optimizer.
def value_and_grad(u):
    (plasma_value, plasma_costs), plasma_gradient = plasma_value_and_grad(u)
    (coil_value, coil_cost_values), coil_gradient = coil_value_and_grad(u)
    terms = dict(plasma_costs)
    terms.update(zip(coil_term_names, map(float, np.asarray(coil_cost_values))))
    return monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)


print("Running single_stage_optimization_scalar.py (same weighted objective; scalar adjoint)")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, one scalar plasma adjoint and JAX coil gradients")
print(f"dof_names = {dof_names}")

# Stage two: optimize only the existing coil objective, with the initial
# plasma boundary pinned by equal bounds. Currents stay fixed in objects_from_x.
# This fit calls no plasma objective, equilibrium solver, or plasma adjoint.
print(f"\n[stage-two coil fit] Up to {COIL_FIT_MAXITER} L-BFGS-B iterations "
      "on the frozen seed boundary...", flush=True)
coil_fit_monitor = opt.OptimizationMonitor(print_every=10, trace=False)

def coil_fit_value_and_grad(u):
    (value, _costs), gradient = coil_value_and_grad(jnp.asarray(u))
    return coil_fit_monitor.cache_evaluation(u, value, gradient)

if args.coils:
    from scipy.optimize import OptimizeResult
    fitted = Coils.from_json(str(args.coils))
    if fitted.dofs_curves.shape != coils0.dofs_curves.shape or fitted.n_segments != N_SEGMENTS:
        raise ValueError("fitted coil resolution differs from the control")
    np.testing.assert_allclose(fitted.dofs_currents_raw, coils0.dofs_currents_raw, rtol=1e-14, atol=0)
    coils0.dofs_currents_raw = fitted.dofs_currents_raw
    fitted_u = np.zeros_like(x0)
    fitted_u[x_boundary0.size:] = (np.asarray(fitted.curves.dofs).ravel()-x_coils0) / COIL_STEP
    coil_fit = OptimizeResult(x=fitted_u, nit=0, message="reused saved fit")
else:
    coil_fit = minimize(
        coil_fit_value_and_grad, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(0.0, 0.0)] * x_boundary0.size + [(-PARAMETER_BOUND, PARAMETER_BOUND)] * n_curve_dofs,
        callback=coil_fit_monitor,
        options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1.0e-15, "gtol": 1.0e-10})
    if not np.isfinite(coil_fit.fun) or not np.all(np.isfinite(coil_fit.x)):
        raise RuntimeError("Stage-two coil fit returned nonfinite values")
surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
fit_rms = float(jnp.linalg.norm(normal_field_residual(coils_fit, surface_seed)))
coil_fit_path = Path(f"coils_single_stage_scalar_fitted_{time.time_ns()}.json")
coils_fit.to_json(str(coil_fit_path))
print(f"[stage-two coil fit] {coil_fit.nit} iterations; B.n/B RMS = {100 * fit_rms:.3f}%; "
      f"status: {coil_fit.message}\nWrote {coil_fit_path}", flush=True)
print("\n[single stage] Starting joint optimization from the fitted coils...", flush=True)
joint_problem = vj.FunctionProblem.from_functions(
    coil_fit.x, value_and_grad=value_and_grad)
joint_problem.compile_value_and_gradient(report_interval=10.0)
history = StepHistory(Path.cwd())


def record_step(u):
    global accepted_state
    monitor(u)
    x = x0 + scales * u
    eq = plasma_problem.equilibrium_from_x(x[:x_boundary0.size])
    accepted_state = eq.state
    surface, coils = objects_from_x(jnp.asarray(x))
    result_state = im._LAST_SOLVE[solver_config][1]
    history.record(u, coils, surface, objective=monitor.records[-1].cost,
        qa=float(qs.total_state(eq.state, eq.runtime)),
        aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
        minimum_iota=float(opt.min_abs_iota(eq.state, eq.runtime)),
        aspect_target=ASPECT_TARGET, iota_floor=IOTA_FLOOR, coil_step=COIL_STEP,
        rbc00_m=float(plasma_problem.boundary_from_x(x[:x_boundary0.size])[0][inp.ntor, 0]),
        **limits.diagnostics(eq.state, eq.runtime),
        **{key: float(getattr(result_state, key)) for key in ("fsqr", "fsqz", "fsql")})
    np.savez_compressed(f"equilibrium_{len(history.rows)-1:04d}.npz",
        **{key: np.asarray(getattr(eq.state, key)) for key in
           ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    monitor.save("single_stage_scalar_objectives.csv")
    if args.constrained:
        row = history.rows[-1]
        print(f"  constraints: min |iota|={row['min_abs_iota']:.7f}, R={row['major_radius_m']:.7f} m, "
              f"physical limits met={bool(row['constraints_feasible'])}", flush=True)


sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
Path("provenance.json").write_text(json.dumps(dict(
    command=sys.argv, device=str(jax.devices()[0]), python=sys.version, jax=jax.__version__,
    input_sha256=sha(DATA), coils_sha256=sha(args.coils) if args.coils else sha(coil_fit_path),
    script_sha256=sha(__file__), ftol=float(inp.ftol_array[-1]), method=METHOD,
    dependencies={name: version(name) for name in ("numpy", "scipy", "essos", "solvax", "equinox", "diffrax", "lineax", "optimistix", "pyevtk")},
    core_sha256={str(p.relative_to(HERE.parents[1])): sha(p) for p in sorted((HERE.parents[1]/"vmex/core").glob("*.py"))},
    optimizer_options=OPTIONS, coil_step=COIL_STEP, vary_major_radius=VARY_MAJOR_RADIUS or args.constrained,
    constrained=args.constrained, constraint_settings={k:v for k,v in vars(limits).items() if k.isupper()},
    initial_u=coil_fit.x.tolist(), scales=scales.tolist()), indent=2)+"\n")
Path("control_source.py").write_text(Path(__file__).read_text())
Path("_scalar_diagnostics.py").write_text((HERE/"_scalar_diagnostics.py").read_text())
Path("_scalar_constraints.py").write_text((HERE/"_scalar_constraints.py").read_text())
record_step(joint_problem.x0)
if args.constrained:
    direction = np.random.default_rng(0).normal(size=x0.size)
    direction /= np.linalg.norm(direction)
    analytic = constraint_jacobian(joint_problem.x0) @ direction
    checks = []
    for h in (3e-3, 1e-3):
        fd = (constraint_values(joint_problem.x0+h*direction)-constraint_values(joint_problem.x0-h*direction))/(2*h)
        errors = np.abs(fd-analytic)/np.maximum(np.maximum(np.abs(fd),np.abs(analytic)),1e-8)
        checks.append(dict(h=h, analytic=analytic.tolist(), finite_difference=fd.tolist(), relative_error=errors.tolist()))
    Path("constraint_gradient_check.json").write_text(json.dumps(checks,indent=2)+"\n")
    if any(not all(np.isfinite(e) and e < 1e-3 for e in check["relative_error"]) for check in checks):
        raise RuntimeError("constraint directional derivative check failed")
def slsqp_gradient(u):
    # SLSQP asks for gradients after accepting its line search. Its callback
    # instead fires at the first trial of a major iteration, before backtracking.
    _, gradient = joint_problem.value_and_grad(u)
    if not np.array_equal(u, history.previous["u"]):
        record_step(u)
    return gradient


result = minimize(joint_problem.fun if args.constrained else joint_problem.value_and_grad, joint_problem.x0,
                  jac=slsqp_gradient if args.constrained else True, method=METHOD,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
                  constraints=[dict(type="ineq", fun=constraint_values, jac=constraint_jacobian)] if args.constrained else (),
                  callback=None if args.constrained else record_step, options=OPTIONS)
if args.constrained and not np.array_equal(result.x, history.previous["u"]):
    record_step(result.x)
initial_value = monitor.records[0].cost

x_final = x0 + scales * result.x
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_boundary0.size])
final_input = plasma_problem.input_from_x(x_final[:x_boundary0.size])
final_input = replace(final_input,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)

surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), inp.nfp,
    nphi=61, ntheta=64)
normal_field = np.asarray(normalized_normal_field(coils_final, surface_final))
area_weights = np.asarray(surface_final.area_element); area_weights = area_weights / area_weights.sum()
normal_field_rms = float(np.sqrt(np.sum(area_weights * normal_field**2)))
normal_field_max = float(np.max(np.abs(normal_field)))
coil_points, surface_points = np.asarray(coils_final.gamma), np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(float(np.linalg.norm(points[:, None] - surface_points[None], axis=2).min())
                            for points in coil_points)
coil_pairs = [(i, j) for i in range(len(coil_points)) for j in range(i + 1, len(coil_points))]
coil_distance = min(float(np.linalg.norm(coil_points[i][:, None] - coil_points[j][None], axis=2).min())
                    for i, j in coil_pairs)
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))

# Print results
report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
final_value = float(result.fun)
report("final", final_equilibrium)
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} in {result.nit} {METHOD} iterations")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"B.n/B: area-weighted RMS = {100 * normal_field_rms:.3f}%, max = {100 * normal_field_max:.3f}% "
      f"(target < {100 * NORMAL_FIELD_LIMIT:.1f}%)")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")

# The coil metrics above are each printed against their limit; the plasma
# targets were not, so a run that drove the coil terms down while leaving the
# rotational transform near zero read as a success. A boundary with no
# transform also makes the quasisymmetry residual trivially small, so report
# both plasma targets explicitly and say plainly whether they were met.
minimum_iota = float(opt.min_abs_iota(final_equilibrium.state, final_equilibrium.runtime))
final_aspect = float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime))
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target {ASPECT_TARGET:.4f})")
unmet = []
if minimum_iota < IOTA_FLOOR:
    unmet.append(f"minimum |iota| {minimum_iota:.4f} below the {IOTA_FLOOR:.4f} floor")
if normal_field_rms > NORMAL_FIELD_LIMIT:
    unmet.append(f"B.n/B RMS {100 * normal_field_rms:.3f}% above {100 * NORMAL_FIELD_LIMIT:.1f}%")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    print("A lower weighted objective with an unmet target is not a design. "
          "Raise the weight, reseed, or extend the budget before using it.")

# Save results
input_path = final_input.to_indata("input.single_stage_scalar_optimized")
wout_path = vj.write_wout("wout_single_stage_scalar_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_scalar_optimized.json")
# ESSOS writes |B| and B.n/B on the surface and the coil filaments for ParaView.
surface_initial = surfacerzfourier_from_boundary(
    rbc0, zbs0, inp.nfp, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_scalar_initial", field=BiotSavart(coils_fit))
coils_fit.to_vtk("coils_single_stage_scalar_initial")
field_final = BiotSavart(coils_final)
surface_final.to_vtk("surface_single_stage_scalar_optimized", field=field_final)
coils_final.to_vtk("coils_single_stage_scalar_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_scalar_optimized.json")
print("Wrote initial and optimized surface/coils VTK files")

Path("optimization_summary.json").write_text(json.dumps(dict(
    optimizer_success=bool(result.success), message=str(result.message),
    accepted_steps=int(result.nit), evaluations=int(result.nfev),
    best_feasible_step=min((row for row in history.rows if row["constraints_feasible"]), key=lambda row:row["objective"], default={}).get("step"),
    final_qa=float(qs.total(final_equilibrium)), final_aspect=final_aspect,
    **limits.diagnostics(final_equilibrium.state, final_equilibrium.runtime),
    final_min_abs_iota=minimum_iota), indent=2)+"\n")
if args.no_plots:
    raise SystemExit(0)

# Plot results
print("Plotting results...")
vj.plot_optimization_objects("single_stage_scalar_optimization.png",
    ("Initial (stage-two fit)", surface_initial, coils_fit), ("Optimized", surface_final, coils_final))
monitor.save("single_stage_scalar_objectives.csv")
monitor.plot("single_stage_scalar_objectives.png", title="Single-stage objective terms")
print("Wrote single_stage_scalar_optimization.png")
print("Wrote single_stage_scalar_objectives.csv and single_stage_scalar_objectives.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils("single_stage_scalar_optimization.gif", objects_from_x,
        x0=x0, scales=scales, surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA, cmap="jet")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
