#!/usr/bin/env python
"""Scalar free-boundary single-stage optimization through the public VMEX API.

Edit the parameters below, then follow build_problem -> run_optimizer ->
verify_endpoint -> postprocess. Like the fixed-boundary examples, the loss is
visible here; opt.minimize owns scaling and optimizer acceptance. VMEX owns
solves, total derivatives and prediction from the last accepted equilibrium.
Derivative tests live in verify_free_boundary_single_stage.py and run separately.
Use --qualification to reuse a checked seed; --dry-run only prints settings.
This example is vacuum-only; finite-beta API requirements are in the README.
"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
# Also support importlib-based tooling without initializing VMEX.
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
from single_stage_support import common, free

INPUT = HERE / "input.rotating_ellipse"
COILS = None  # supply --coils to reuse a fitted set; otherwise perform stage two
COIL_FIT_MAXITER = 200
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.5, 2.7e5
NORMAL_FIELD_WEIGHT, NORMAL_FIELD_OBJECTIVE_LIMIT, NORMAL_FIELD_LIMIT_WEIGHT = 1e3, 0.008, 2e5

# Optimization and equilibrium accuracy (different stopping criteria).
ACCEPTED_STEPS = 100
OPTIMIZER_FTOL = 1e-10
OPTIMIZER_GTOL = 1e-10
PARAMETER_BOUND = 5.0  # L-BFGS-B bounds in scaled coil coordinates
LBFGSB_MAXCOR, LBFGSB_MAXLS = 20, 20
COIL_STEP = 0.05
MAX_TRIALS = 500
RESOLUTION = (8, 8, 51)  # MPOL, NTOR, NS
GRID = (64, 64)  # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
ROOT_TOLERANCE = 2e-6  # ordinary-root admission before coupled Newton refinement
ROOT_POLISH_TOLERANCE = 1e-12
INITIALIZATION_SECONDS = 3600
OPTIMIZATION_SECONDS = 43200
VERIFICATION_SECONDS = 1800

# Same QA, aspect, iota and coil penalties as the scalar reference.
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
ASPECT_TARGET, ASPECT_WEIGHT = 5.0, 1.0
IOTA_FLOOR, IOTA_WEIGHT = 0.19, 10.0
RADIUS_TARGET, RADIUS_TOLERANCE = 1.0, 0.01
IOTA_MARGIN, RADIUS_MARGIN = 0.0005, 0.001
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 5, 64
NPHI, NTHETA = 37, 32
LENGTH_TARGET, LENGTH_WEIGHT = 5.0, 1.0
CURVATURE_OBJECTIVE_LIMIT, CURVATURE_LIMIT, CURVATURE_WEIGHT = 6.9, 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.15, 1e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1e3
# Free-boundary equilibrium determines B.n; it is a diagnostic, not a penalty.
NORMAL_FIELD_LIMIT = 0.01

# Checked matrix-free adjoints; dense recovery refreshes only on acceptance.
ADJOINT_RESIDUAL_RTOL = 1e-9
ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 32, 20000
MATRIXFREE_RTOL = 1e-11
MATRIXFREE_RESTART, MATRIXFREE_MAX_CYCLES, MATRIXFREE_RHS_BATCH_SIZE = 100, 3, 3
LU_REFRESH_HORIZON = 10  # rebuild when estimated savings repay its measured cost
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = 201, 1e-15, 12000

# Match the scalar reference: figures, accepted-iterate movie and WOUT plots.
MAKE_MOVIE = True
MOVIE_SURFACE_COLOR = "absB"  # total equilibrium field; alternatively "B.n/B"
POSTPROCESSING_SECONDS = 1800


def parse_args(argv=None):
    return common.parse_options(argv, parameters=globals(), description=__doc__, formulation="free")

def sha(path):
    """Identify an input, source or saved result by its contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    """Write a finite, readable report inside the selected run directory."""
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")

def setup_run(args):
    """Set project-local output/cache paths before importing JAX."""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        directory = out / "cache" / name
        directory.mkdir(parents=True)
        os.environ[name] = str(directory)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
                      VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
    sys.path.insert(0, str(HERE.parents[1]))
    import jax
    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested backend and float64 precision are required")
    return out

def qualification_contract(args):
    """Identify the numerical setup shared by verification and production."""
    from vmex import optimize as opt

    excluded = {"HERE", "INPUT", "COILS", "ACCEPTED_STEPS", "MAX_TRIALS", "MAKE_MOVIE", "MOVIE_SURFACE_COLOR"}
    parameters = {name: value for name, value in globals().items()
                  if name.isupper() and name not in excluded and not name.endswith("_SECONDS")}
    return opt.OptimizationQualification.signature(parameters=parameters,
        input_path=args.input, wout_path=args.wout, resolution=args.resolution,
        grid=args.grid, ftol=args.ftol, device=args.device,
        functions=(build_problem, configure_solver, run_optimizer,
                   common.load_input, common.initial_coils, common.physical_constraint))

def read_qualification(args):
    """Reuse the same authenticated fitted coils and accepted seed."""
    from vmex import optimize as opt

    if args.qualification is None:
        return None
    qualified = opt.OptimizationQualification.read(args.qualification, contract=qualification_contract(args))
    if args.coils is not None and sha(args.coils) != qualified.report["artifacts"]["coils"]["sha256"]:
        raise ValueError("supplied coils differ from the qualified fitted coils")
    return qualified

def build_problem(args, *, event=None, qualified=None):
    """Prepare the input, stage-two coils and common scalar VMEX problem.

    Without a qualification bundle, load or fit coils and solve the seed from
    input/WOUT. A supplied bundle restores its exact checked seed, skipping
    stage-two fitting and the initial equilibrium solve. No derivative tests run.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from vmex import optimize as opt
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary

    out = args.output.resolve()
    inp, seed = common.load_input(args, restore_state=qualified is None)
    qs = opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), 1, 0)

    def surface(state, runtime):
        rbc, zbs, _, _ = opt.boundary_from_state(state, runtime)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

    def iota_floor(state, runtime):
        return jnp.maximum(IOTA_FLOOR-opt.min_abs_iota(state, runtime), 0.0)

    plasma_terms = [(qs.residuals_state, 0.0, 1.0),
                    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
                    (iota_floor, 0.0, IOTA_WEIGHT)]

    def coil_costs(coils, surf):
        return jnp.array([
            0.5*LENGTH_WEIGHT*jnp.sum((coils.length[:N_COILS]-LENGTH_TARGET)**2),
            0.5*CURVATURE_WEIGHT*jnp.sum(jnp.maximum(coils.curvature[:N_COILS]-CURVATURE_OBJECTIVE_LIMIT, 0)**2),
            0.5*COIL_DISTANCE_WEIGHT*loss_coil_separation(coils, COIL_DISTANCE_LIMIT, block_size=32),
            0.5*COIL_SURFACE_DISTANCE_WEIGHT*loss_coil_surface_distance(
                coils, surf, COIL_SURFACE_DISTANCE_LIMIT, block_size=32)])

    def loss(state, runtime, coils):
        residuals = opt.residuals_from_tuples(state, runtime, plasma_terms)
        return 0.5*jnp.vdot(residuals, residuals) + jnp.sum(coil_costs(coils, surface(state, runtime)))

    def inequalities(values):
        iota, radius = values
        width = RADIUS_TOLERANCE-RADIUS_MARGIN
        return np.array([(iota-IOTA_FLOOR-IOTA_MARGIN)/IOTA_FLOOR,
                         (radius-RADIUS_TARGET+width)/RADIUS_TOLERANCE,
                         (RADIUS_TARGET+width-radius)/RADIUS_TOLERANCE])

    constraint_transform = np.array([[1/IOTA_FLOOR, 0], [0, 1/RADIUS_TOLERANCE], [0, -1/RADIUS_TOLERANCE]])

    coil_path = qualified[1]["coils"] if qualified is not None else args.coils or args.initial_coils
    coils0 = common.initial_coils(inp, coil_path, parameters=globals())
    fit_report = dict(reused=True, iterations=0)
    if qualified is None and args.coils is None:
        # Stage two changes only geometry on the frozen input/WOUT surface.
        # These are the scalar reference's two B.n penalties plus coil terms.
        seed_surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs),
                                                      inp.nfp, nphi=NPHI, ntheta=NTHETA)
        x0 = np.asarray(coils0.curves.dofs).ravel()

        def fit_loss(u):
            coils = coils0.with_dofs(jnp.concatenate((jnp.asarray(x0)+COIL_STEP*u, coils0.dofs_currents)))
            field = jax.vmap(BiotSavart(coils).B)(seed_surface.gamma.reshape(-1, 3)).reshape(seed_surface.gamma.shape)
            normal = jnp.sum(field*seed_surface.unitnormal, axis=-1)/jnp.linalg.norm(field, axis=-1)
            weights = seed_surface.area_element/jnp.sum(seed_surface.area_element)
            maximum = jax.scipy.special.logsumexp(2000*jnp.sqrt(normal**2+1e-12))/2000
            return (0.5*NORMAL_FIELD_WEIGHT*jnp.sum(weights*normal**2)
                    + 0.5*NORMAL_FIELD_LIMIT_WEIGHT*jnp.maximum(maximum-NORMAL_FIELD_OBJECTIVE_LIMIT, 0)**2
                    + jnp.sum(coil_costs(coils, seed_surface)))

        fit_gradient = jax.jit(jax.value_and_grad(fit_loss))
        initial_cost = float(fit_loss(jnp.zeros_like(jnp.asarray(x0))))
        fit = minimize(fit_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                       bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)]*x0.size,
                       options=dict(maxiter=args.coil_fit_maxiter, maxcor=20, ftol=1e-15, gtol=1e-10))
        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)) or fit.fun > initial_cost + 1e-10:
            raise RuntimeError("stage-two fitting returned invalid or worse coils")
        currents = np.asarray(coils0.dofs_currents_raw).copy()
        coils0 = coils0.with_dofs(jnp.concatenate((jnp.asarray(x0)+COIL_STEP*fit.x, coils0.dofs_currents)))
        if not np.array_equal(coils0.dofs_currents_raw, currents):
            raise RuntimeError("stage-two fitting changed fixed coil currents")
        fit_report = dict(reused=False, iterations=int(fit.nit), optimizer_success=bool(fit.success),
                          message=str(fit.message), initial_objective=initial_cost, objective=float(fit.fun))
    # Reload the saved representation once so qualification and production
    # construct identical charts, including floating-point serialization.
    fitted_path = out / "coils.stage2.json"
    coils0.to_json(str(fitted_path))
    coils0 = Coils.from_json(str(fitted_path))
    write_json(out / "stage_two.json", fit_report)
    scales = COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=(), scales=scales)
    if qualified is None and seed is None:
        seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                     polish_force_balance=False).state
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ftol_array=np.array([args.ftol]))
    contract = qualification_contract(args)
    restart = dict(restart_from=seed) if qualified is None else dict(
        checkpoint=qualified[1]["checkpoint"], checkpoint_sha256=qualified[0]["artifacts"]["checkpoint"]["sha256"])
    write_json(out / "provenance.json", dict(contract=contract, arguments=vars(args), command=sys.argv, script_sha256=sha(__file__),
        currents_A=chart.currents.tolist(), scales=scales.tolist(), fitted_coils_sha256=sha(fitted_path)))
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    lbfgsb_source = HERE / "free_boundary_single_stage_optimization.py"
    (out / lbfgsb_source.name).write_text(lbfgsb_source.read_text())
    (out / "single_stage_common.py").write_text(Path(common.__file__).read_text())
    problem = opt.FreeBoundaryProblem.from_loss(inp, loss, quantities=(opt.min_abs_iota, opt.major_radius),
        parameterization=chart, root_residual_atol=ROOT_TOLERANCE, event=event, checkpoint_identity=contract,
        solver_options=dict(device=args.device, ftol=args.ftol, edge_force_tolerance=args.ftol,
            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL), **restart)
    if problem.accepted_step != 0:
        problem.close()
        raise ValueError("qualification must describe an initial equilibrium at accepted step zero")
    try:
        problem.enable_root_polishing(tolerance=ROOT_POLISH_TOLERANCE)
    except BaseException:
        problem.close()
        raise
    return SimpleNamespace(problem=problem, inp=inp, chart=chart, coils=coils0, qs=qs,
                           surface=surface, coil_costs=coil_costs, inequalities=inequalities,
                           constraint_transform=constraint_transform, contract=contract)

def configure_solver(problem, args):
    """Prepare the production solver without qualification experiments."""
    problem.enable_matrix_free(rtol=MATRIXFREE_RTOL, restart=MATRIXFREE_RESTART,
        max_restarts=MATRIXFREE_MAX_CYCLES, rhs_batch_size=MATRIXFREE_RHS_BATCH_SIZE,
        refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.accepted_steps)

def run_optimizer(stage, args, record_step, *, method):
    """Choose the optimizer and physical bounds; VMEX owns scaling and acceptance."""
    from scipy.optimize import Bounds
    from vmex import optimize as opt

    problem = stage.problem
    options = dict(maxiter=args.accepted_steps, ftol=OPTIMIZER_FTOL)
    constraints, bounds = (), None
    if method == "SLSQP":
        constraints = common.physical_constraint(problem, parameters=globals())
    else:
        bounds = Bounds(-PARAMETER_BOUND*problem.scales, PARAMETER_BOUND*problem.scales)
        options.update(maxfun=MAX_TRIALS, gtol=OPTIMIZER_GTOL,
                       maxcor=LBFGSB_MAXCOR, maxls=LBFGSB_MAXLS)
    return opt.minimize(problem, method=method, bounds=bounds, constraints=constraints,
                        callback=lambda x: record_step(), options=options)

def verify_endpoint(stage, args, summary, history, initial_equilibrium):

    return free.verify_endpoint(SimpleNamespace(**globals()), None, stage, args, summary, history, initial_equilibrium)


def postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified):
    return free.postprocess(SimpleNamespace(**globals()), stage, args, summary, monitor, history, initial_equilibrium, verified)


def main(argv=None, *, method="SLSQP"):
    """Prepare the case, optimize, and independently solve the endpoint."""

    return free.run(parse_args(argv), case=SimpleNamespace(**globals()), method=method)


if __name__ == "__main__":
    raise SystemExit(main())
