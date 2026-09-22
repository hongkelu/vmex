#!/usr/bin/env python
"""Scalar free-boundary single-stage optimization through the public VMEX API.

Like single_stage_optimization_scalar.py, this example defines its objective
and calls SciPy directly. VMEX owns equilibrium solves, total derivatives and
accepted-state prediction. Edit the parameters below; --dry-run only prints them.
"""

import argparse
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
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
ROOT_TOLERANCE = 2e-6
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
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = 201, 1e-15, 12000

# Match the scalar reference: figures, accepted-iterate movie and WOUT plots.
MAKE_MOVIE = True
MOVIE_SURFACE_COLOR = "absB"  # total equilibrium field; alternatively "B.n/B"
POSTPROCESSING_SECONDS = 1800


def parse_args(argv=None):
    """Read ordinary example options without initializing JAX or writing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--wout", type=Path, help="WOUT restart; requires its corresponding input deck")
    parser.add_argument("--qualification", type=Path, help="passing report from verify_free_boundary_single_stage.py")
    coils = parser.add_mutually_exclusive_group()
    coils.add_argument("--coils", type=Path, default=COILS, help="reuse fitted coils without stage two")
    coils.add_argument("--initial-coils", type=Path, help="fit these coils instead of generating circles")
    parser.add_argument("--coil-fit-maxiter", type=int, default=COIL_FIT_MAXITER)
    parser.add_argument("--output", type=Path, default=HERE / "runs" / f"free-boundary-{time.time_ns()}")
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--accepted-steps", type=int, default=ACCEPTED_STEPS)
    parser.add_argument("--ftol", type=float)
    parser.add_argument("--resolution", type=int, nargs=3)
    parser.add_argument("--grid", type=int, nargs=2)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", dest="no_plots", action="store_false")
    parser.add_argument("--movie", action=argparse.BooleanOptionalAction, default=MAKE_MOVIE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    custom_input = args.input is not None
    if args.qualification is not None:
        report = json.loads(args.qualification.read_text())
        for name in ("input", "wout", "resolution", "grid", "ftol", "device"):
            if getattr(args, name) is None:
                value = report["configuration"][name]
                setattr(args, name, Path(value) if name in ("input", "wout") and value else value)
        if args.initial_coils is not None:
            parser.error("a qualified run reuses fitted coils; --initial-coils requires new qualification")
    elif args.wout is not None and args.input is None:
        parser.error("--wout requires --input to define the pressure/current profiles and solver settings")
    if args.input is None:
        args.input = INPUT
    if not custom_input and args.wout is None and args.qualification is None:
        args.resolution = RESOLUTION if args.resolution is None else args.resolution
        args.grid = GRID if args.grid is None else args.grid
    args.ftol = EQUILIBRIUM_FTOL if args.ftol is None else args.ftol
    args.device = args.device or "gpu"
    if args.coil_fit_maxiter < 1:
        parser.error("positive coil-fit iteration budget required")
    if args.accepted_steps < 1 or not math.isfinite(args.ftol) or args.ftol <= 0:
        parser.error("positive step budget and finite positive force tolerance required")
    if ((args.resolution is not None and (args.resolution[0] < 1 or args.resolution[1] < 0 or args.resolution[2] < 3))
            or (args.grid is not None and min(args.grid) < 4)):
        parser.error("invalid equilibrium resolution or grid")
    if not 0 <= RADIUS_MARGIN < RADIUS_TOLERANCE < RADIUS_TARGET or IOTA_MARGIN < 0:
        parser.error("invalid physical constraint margins")
    if MOVIE_SURFACE_COLOR not in ("absB", "B.n/B", None):
        parser.error("movie surface color must be absB, B.n/B or None")
    return args


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
    """Bind qualification to numerical settings, inputs, code and runtime."""
    import jax
    excluded = {"HERE", "INPUT", "COILS", "ACCEPTED_STEPS", "MAX_TRIALS", "MAKE_MOVIE", "MOVIE_SURFACE_COLOR"}
    parameters = {name: value for name, value in globals().items()
                  if name.isupper() and name not in excluded and not name.endswith("_SECONDS")}
    contract = dict(parameters=parameters, input_sha256=sha(args.input),
        wout_sha256=None if args.wout is None else sha(args.wout),
        resolution=args.resolution, grid=args.grid, ftol=args.ftol,
        device=args.device, hardware=jax.devices()[0].device_kind,
        dependencies={name: version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "essos", "solvax")},
        script_sha256=sha(__file__),
        lbfgsb_script_sha256=sha(HERE / "free_boundary_single_stage_optimization.py"), verification_sha256=sha(HERE / "verify_free_boundary_single_stage.py"),
        core_sha256={p.name: sha(p) for p in sorted((HERE.parents[1]/"vmex/core").glob("*.py"))})
    return json.loads(json.dumps(contract))


def read_qualification(args):
    """Validate a passing report and its fitted coils/checkpoint before reuse."""
    if args.qualification is None:
        raise ValueError("production requires --qualification from verify_free_boundary_single_stage.py")
    report = json.loads(args.qualification.read_text())
    if report.get("schema") != "vmex.single-stage-qualification/v1" or report.get("passed") is not True:
        raise ValueError("qualification report has not passed")
    if any(report.get("checks", {}).get(name, {}).get("passed") is not True
           for name in ("finite_difference", "matrix_free")):
        raise ValueError("qualification checks have not passed")
    if report["contract"] != qualification_contract(args):
        raise ValueError("qualification differs from current code, input, physics, resolution or runtime; qualify this setup first")
    assets = {}
    for name in ("coils", "checkpoint"):
        item = report["artifacts"][name]
        if Path(item["file"]).name != item["file"]:
            raise ValueError("qualification artifacts must be files beside the report")
        path = args.qualification.resolve().parent / item["file"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"qualified {name} hash mismatch")
        assets[name] = path
    if args.coils is not None and sha(args.coils) != report["artifacts"]["coils"]["sha256"]:
        raise ValueError("supplied coils differ from the qualified fitted coils")
    return report, assets


def build_problem(args, *, event=None, qualified=None):
    """Prepare the input, stage-two coils and common scalar VMEX problem.

    Qualification calls this without qualified to solve and certify a seed.
    Production supplies the checked report/assets to restore that exact seed,
    skipping both stage-two fitting and the initial equilibrium solve.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary

    out = args.output.resolve()
    inp = vj.VmecInput.from_file(args.input.resolve())
    mpol, ntor, ns = args.resolution or (inp.mpol, inp.ntor, int(inp.ns_array[-1]))
    ntheta, nphi = args.grid or (inp.ntheta, inp.nzeta)
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nphi)
    inp = replace(inp, ns_array=np.array([ns]), lfreeb=False)
    if inp.lasym:
        raise ValueError("this example requires stellarator symmetry")
    seed = None
    if args.wout is not None:
        wout = vj.read_wout(args.wout)
        # VMEX checks NFP/symmetry and remaps the radial/Fourier coordinates.
        if qualified is None:
            seed = vj.state_from_wout(wout, inp=inp, ns=ns)
        elif int(wout.nfp) != inp.nfp or bool(wout.lasym):
            raise ValueError("WOUT symmetry differs from the input")
        rbc, zbs, rbs, zbc = opt.boundary_from_wout(wout, mpol=mpol, ntor=ntor)
        inp = replace(inp, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc)
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
    if coil_path is not None:
        coils0 = Coils.from_json(str(coil_path.resolve()))
    else:
        curves = CreateEquallySpacedCurves(N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
                                          n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=True)
        coils0 = Coils(curves, jnp.full(N_COILS, COIL_CURRENT))
    if (coils0.nfp != inp.nfp or not coils0.stellsym or coils0.n_segments != N_SEGMENTS
            or coils0.dofs_curves.shape != (N_COILS, 3, 2*COIL_ORDER+1)):
        raise ValueError("coils must match symmetry, count, Fourier order and quadrature")
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
                       bounds=[(-5., 5.)]*x0.size,
                       options=dict(maxiter=args.coil_fit_maxiter, maxcor=20, ftol=1e-15, gtol=1e-10))
        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)) or fit.fun > initial_cost + 1e-10:
            raise RuntimeError("stage-two fitting returned invalid or worse coils")
        currents = np.asarray(coils0.dofs_currents_raw).copy()
        coils0 = coils0.with_dofs(jnp.concatenate((jnp.asarray(x0)+COIL_STEP*fit.x, coils0.dofs_currents)))
        np.testing.assert_array_equal(coils0.dofs_currents_raw, currents)
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
    write_json(out / "provenance.json", dict(contract=contract, arguments=vars(args), command=sys.argv,
        currents_A=chart.currents.tolist(), scales=scales.tolist(), fitted_coils_sha256=sha(fitted_path)))
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    lbfgsb_source = HERE / "free_boundary_single_stage_optimization.py"
    (out / lbfgsb_source.name).write_text(lbfgsb_source.read_text())
    problem = opt.FreeBoundaryProblem.from_loss(inp, loss, quantities=(opt.min_abs_iota, opt.major_radius),
        parameterization=chart, root_residual_atol=ROOT_TOLERANCE, event=event, checkpoint_identity=contract,
        solver_options=dict(device=args.device, ftol=args.ftol, edge_force_tolerance=args.ftol,
            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL), **restart)
    if problem.accepted_step != 0:
        problem.close()
        raise ValueError("qualification must describe an initial equilibrium at accepted step zero")
    return SimpleNamespace(problem=problem, inp=inp, chart=chart, coils=coils0, qs=qs,
                           surface=surface, coil_costs=coil_costs, inequalities=inequalities,
                           constraint_transform=constraint_transform, contract=contract)


def run_optimizer(problem, scales, inequalities, constraint_transform, args, record_step, *, method):
    """Run SciPy while promoting only iterates accepted by the chosen optimizer."""
    import numpy as np
    from scipy.optimize import minimize
    from vmex import optimize as opt

    def accept_step(u):
        x = scales*u
        if not np.array_equal(x, problem.accepted.parameters):
            problem.accept_x(x)
            record_step()
            if problem.accepted_step >= args.accepted_steps:
                raise StopIteration("accepted step budget reached")

    if method == "L-BFGS-B":
        def value_and_grad(u):
            # A line search requests gradients at unaccepted points. Never
            # promote here or fabricate a gradient for a failed equilibrium.
            value, gradient = problem.value_and_grad(scales*u)
            return value, gradient*scales

        return minimize(value_and_grad, problem.x0/scales, jac=True, method=method,
            bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)]*problem.x0.size, callback=accept_step,
            options=dict(maxiter=args.accepted_steps, maxfun=MAX_TRIALS, ftol=OPTIMIZER_FTOL,
                         gtol=OPTIMIZER_GTOL, maxcor=LBFGSB_MAXCOR, maxls=LBFGSB_MAXLS))

    if method != "SLSQP":
        raise ValueError(f"unsupported optimizer: {method}")

    def objective(u):
        try:
            return problem.fun(scales*u)
        except opt.TrialRejected:
            return np.inf

    def constraint_values(u):
        try:
            return inequalities(problem.constraint_values(scales*u))
        except opt.TrialRejected:
            return np.full(3, -1e6)

    def constraint_jacobian(u):
        return (constraint_transform @ problem.constraint_jac(scales*u))*scales

    def slsqp_gradient(u):
        # SLSQP requests gradients after accepting its line search; its major
        # iteration callback can precede backtracking and must not promote.
        _, gradient = problem.value_and_grad(scales*u)
        accept_step(u)
        return gradient*scales

    result = minimize(objective, problem.x0/scales, jac=slsqp_gradient, method=method,
        constraints=[dict(type="ineq", fun=constraint_values, jac=constraint_jacobian)],
        options=dict(maxiter=args.accepted_steps+1, ftol=OPTIMIZER_FTOL))
    slsqp_gradient(result.x)
    return result


def main(argv=None, *, method="SLSQP"):
    """Restore a qualified start, optimize and independently verify the endpoint."""
    if method not in ("SLSQP", "L-BFGS-B"):
        raise ValueError(f"unsupported optimizer: {method}")
    args = parse_args(argv)
    settings = {name: value for name, value in globals().items() if name.isupper() and name != "HERE"}
    if args.dry_run:
        print(json.dumps(dict(optimizer=method, parameters=settings, arguments=vars(args)), indent=2, default=str))
        return 0
    if args.qualification is None:
        raise ValueError("run verify_free_boundary_single_stage.py first, then pass --qualification REPORT")
    out = setup_run(args)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier
    qualified = read_qualification(args)

    def write_json(name, data):
        (out / name).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")

    def timeout(*_):
        raise TimeoutError("phase wall-time budget reached")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(INITIALIZATION_SECONDS)
    problem = verification = None
    phase = "initialization"
    started = time.perf_counter()
    try:
        timings = dict(adjoint=0.0, tangent=0.0, correction=0.0)
        trials = 0

        def event(name, **data):
            nonlocal trials
            if name in timings:
                timings[name] += data["seconds"]
            if name == "proposal":
                trials += 1
                if trials > MAX_TRIALS:
                    raise StopIteration("trial budget reached")
            if name in ("adjoint", "tangent", "matrixfree_check", "dense_recovery", "preconditioner_refresh"):
                # Keep diagnostic rows and failure reasons without serializing root arrays.
                record = {key: value for key, value in data.items() if key != "candidate"}
                with (out / "solver_events.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(event=name, trial=trials, **record), default=str) + "\n")

        stage = build_problem(args, event=event, qualified=qualified)
        problem, inp, chart, coils0 = stage.problem, stage.inp, stage.chart, stage.coils
        qs, surface, coil_costs = stage.qs, stage.surface, stage.coil_costs
        inequalities, constraint_transform = stage.inequalities, stage.constraint_transform
        scales = chart.scales
        initial_equilibrium = problem.equilibrium_from_x(problem.x0)
        monitor = opt.OptimizationMonitor(stream=None)
        history = []
        cycle_started = time.perf_counter()

        def record_step():
            nonlocal cycle_started
            x = problem.accepted.parameters
            eq = problem.equilibrium_from_x(x)
            iota, radius = problem.constraint_values(x)
            row = dict(step=problem.accepted_step, objective=problem.fun(x),
                qa=float(qs.total_state(eq.state, eq.runtime)), aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                min_abs_iota=float(iota), major_radius_m=float(radius),
                rbc00_m=float(opt.boundary_from_state(eq.state, eq.runtime)[0][inp.ntor, 0]),
                radius_error_m=float(radius-RADIUS_TARGET), iota_constraint_slack=float(iota-IOTA_FLOOR),
                radius_constraint_slack_m=float(RADIUS_TOLERANCE-abs(radius-RADIUS_TARGET)),
                optimizer_constraints_feasible=bool(np.min(inequalities((iota, radius))) >= -1e-8),
                constraints_feasible=bool(iota >= IOTA_FLOOR and abs(radius-RADIUS_TARGET) <= RADIUS_TOLERANCE),
                gradient_seconds=timings["adjoint"], predictor_seconds=timings["tangent"], solve_seconds=timings["correction"],
                step_seconds=time.perf_counter()-cycle_started, elapsed_seconds=time.perf_counter()-started,
                **{key: float(getattr(eq.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")})
            history.append(row)
            write_json("accepted_steps.json", history)
            write_json(f"checkpoint_{problem.accepted_step:04d}.json", problem.save_checkpoint(out/f"accepted_{problem.accepted_step:04d}.npz"))
            monitor.record(x, cost=row["objective"], iteration=problem.accepted_step)
            monitor.save(out / "free_boundary_scalar_objectives.csv")
            print(f"[step {problem.accepted_step}] objective {row['objective']:.6e}; QA {row['qa']:.6e}; "
                  f"gradient {row['gradient_seconds']:.2f}s, predictor {row['predictor_seconds']:.2f}s, "
                  f"correction {row['solve_seconds']:.2f}s", flush=True)
            timings.update(dict.fromkeys(timings, 0.0))
            cycle_started = time.perf_counter()

        # Production constructs its own checked LU; parity/FD experiments live
        # exclusively in verify_free_boundary_single_stage.py.
        problem.enable_matrix_free(rtol=MATRIXFREE_RTOL, restart=MATRIXFREE_RESTART,
            max_restarts=MATRIXFREE_MAX_CYCLES, rhs_batch_size=MATRIXFREE_RHS_BATCH_SIZE)
        record_step()
        phase = "optimization"
        optimization_started = time.perf_counter()
        signal.alarm(OPTIMIZATION_SECONDS)

        status, result = "iteration_budget_reached", None
        try:
            result = run_optimizer(problem, scales, inequalities, constraint_transform, args, record_step, method=method)
            if problem.accepted_step >= args.accepted_steps:
                status = "accepted_step_budget_reached"
            elif result.success:
                status = "converged"
            else:
                budget_status = 1 if method == "L-BFGS-B" else 9
                status = "iteration_or_evaluation_budget_reached" if result.status == budget_status else "optimizer_failed"
        except StopIteration as error:
            status = str(error).replace(" ", "_")
        except TimeoutError:
            status = "optimization_time_budget_reached"
        except opt.TrialRejected as error:
            # L-BFGS-B needs a gradient for every proposal. Stop cleanly at the
            # last certified accepted state when a proposal cannot supply one.
            status = "equilibrium_trial_rejected"
            write_json("rejected_trial.json", dict(error=str(error)))
        summary = dict(optimizer=method, nonlinear_constraints=method == "SLSQP", status=status, optimizer_success=status == "converged", accepted_steps=problem.accepted_step,
            optimization_seconds=time.perf_counter()-optimization_started,
            derivative_qualified=True, qualification=str(args.qualification.resolve()), initial=history[0], final=history[-1], verification="not run",
            optimizer_message=None if result is None else str(result.message))
        write_json("optimization_summary.json", summary)

        phase = "verification"
        signal.alarm(VERIFICATION_SECONDS)
        accepted_eq = problem.equilibrium_from_x(problem.accepted.parameters)
        initial_wout = vj.write_wout(out / "wout_initial.nc", initial_equilibrium.wout)
        accepted_wout = vj.write_wout(out / "wout_accepted.nc", accepted_eq.wout)
        final_coils = problem.coils_from_x(problem.accepted.parameters)
        final_coils.to_json(str(out / "coils_optimized.json"))
        rbc, zbs, _, _ = opt.boundary_from_state(accepted_eq.state, accepted_eq.runtime)
        final_input = replace(inp, rbc=np.asarray(rbc), zbs=np.asarray(zbs), ns_array=np.array([VERIFY_NS]),
                              ftol_array=np.array([VERIFY_FTOL]), niter_array=np.array([VERIFY_MAXITER]))
        final_input.to_indata(out / "input.optimized")
        problem.close()
        verification = opt.FreeBoundaryProblem.from_tuples(final_input, [(qs.residuals_state, 0, 1)],
            coils=final_coils, current_dofs=(), restart_from=accepted_wout, objective_normalization=1.0,
            solver_options=dict(device=args.device, ftol=VERIFY_FTOL, edge_force_tolerance=VERIFY_FTOL,
                                max_iterations=VERIFY_MAXITER))
        verified = verification.equilibrium_from_x(verification.x0)
        forces = {key: float(getattr(verified.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")}
        if not verified.result.converged or not all(np.isfinite(v) and v <= VERIFY_FTOL for v in forces.values()):
            raise RuntimeError(f"independent endpoint force checks failed: {forces}")
        wout_path = vj.write_wout(out / "wout_optimized.nc", verified.wout)
        surf = SurfaceRZFourier.from_wout_file(wout_path, nphi=61, ntheta=64)
        field = np.asarray(jax.vmap(BiotSavart(final_coils).B)(surf.gamma.reshape(-1, 3))).reshape(surf.gamma.shape)
        bn = np.sum(field*np.asarray(surf.unitnormal), axis=2)/np.linalg.norm(field, axis=2)
        weights = np.asarray(surf.area_element)
        rms, maximum = float(np.sqrt(np.sum(weights*bn**2)/weights.sum())), float(np.max(np.abs(bn)))
        points, surface_points = np.asarray(final_coils.gamma), np.asarray(surf.gamma).reshape(-1, 3)
        coil_distance = min(float(np.linalg.norm(points[i][:, None]-points[j][None], axis=2).min())
                            for i in range(len(points)) for j in range(i+1, len(points)))
        clearance = min(float(np.linalg.norm(p[:, None]-surface_points[None], axis=2).min()) for p in points)
        iota, radius = float(opt.min_abs_iota(verified.state, verified.runtime)), float(opt.major_radius(verified.state, verified.runtime))
        curvature = float(np.max(np.asarray(final_coils.curvature)))
        reporter = opt.EquilibriumReporter(
            ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
            ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
        reported = reporter("final verification", verified)
        lengths = np.asarray(final_coils.length[:N_COILS])
        print(f"\nObjective: {history[0]['objective']:.6e} -> {history[-1]['objective']:.6e} "
              f"in {problem.accepted_step} accepted {method} steps")
        print(f"Coil lengths = {lengths} (soft target {LENGTH_TARGET:.4f} m)")
        print(f"B.n/B: RMS = {100*rms:.3f}%, max = {100*maximum:.3f}% "
              f"(target < {100*NORMAL_FIELD_LIMIT:.1f}%)")
        print(f"Minimum coil-surface distance = {clearance:.4f} m (target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
        print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
        print(f"Maximum curvature = {curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")
        print(f"Minimum |iota| = {iota:.4f} (target >= {IOTA_FLOOR:.4f}); "
              f"aspect = {reported['aspect']:.4f} (soft target {ASPECT_TARGET:.4f})")
        print(f"Major radius = {radius:.6f} m (target {RADIUS_TARGET:.4f} +/- {RADIUS_TOLERANCE:.4f} m)")
        checks = [("minimum |iota|", iota, IOTA_FLOOR, "below"),
                  ("radius error [m]", abs(radius-RADIUS_TARGET), RADIUS_TOLERANCE, "above"),
                  ("B.n/B RMS", rms, NORMAL_FIELD_LIMIT, "above"),
                  ("B.n/B maximum", maximum, NORMAL_FIELD_LIMIT, "above"),
                  ("coil-surface distance [m]", clearance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
                  ("coil-coil distance [m]", coil_distance, COIL_DISTANCE_LIMIT, "below"),
                  ("curvature [1/m]", curvature, CURVATURE_LIMIT, "above")]
        unmet = [f"{name}: {value:.6g} {side} {limit:.6g}" for name, value, limit, side in checks
                 if not np.isfinite(value) or (value < limit if side == "below" else value > limit)]
        feasible = not unmet
        if unmet:
            print("This run did NOT meet its stated limits: " + "; ".join(unmet))
        summary.update(verification="passed numerical checks", inequalities_met=feasible, unmet=unmet,
            best_feasible_step=min((row for row in history if row["constraints_feasible"]),
                                   key=lambda row: row["objective"], default={}).get("step"),
            verified=dict(forces=forces, ns=VERIFY_NS, qa=reported["QS total"], aspect=reported["aspect"],
                          mean_iota=reported["mean iota"], magnetic_well=reported["magnetic well"], min_abs_iota=iota,
                          major_radius_m=radius, normal_field_rms=rms, normal_field_max=maximum,
                          coil_distance_m=coil_distance, coil_surface_distance_m=clearance, maximum_curvature=curvature,
                          coil_lengths_m=lengths.tolist(), aspect_target_error=reported["aspect"]-ASPECT_TARGET,
                          radius_error_m=radius-RADIUS_TARGET, iota_constraint_slack=iota-IOTA_FLOOR,
                          radius_constraint_slack_m=RADIUS_TOLERANCE-abs(radius-RADIUS_TARGET),
                          full_mesh_min_abs_iota=float(np.min(np.abs(verified.wout.iotaf)))))
        write_json("optimization_summary.json", summary)

        phase = "postprocessing"
        signal.alarm(POSTPROCESSING_SECONDS)
        postprocessing_started = time.perf_counter()
        seed_surface = SurfaceRZFourier.from_wout_file(initial_wout, nphi=60, ntheta=60)
        for label, export_surface, export_coils in (("initial", seed_surface, coils0), ("optimized", surf, final_coils)):
            export_surface.to_vtk(str(out / f"surface_{label}"), field=BiotSavart(export_coils))
            export_coils.to_vtk(str(out / f"coils_{label}"))

        # Replay saved accepted roots, never solve previous iterates again.
        points = monitor.x_history
        frame_steps = {tuple(x): step for step, x in enumerate(points)}

        @lru_cache(maxsize=1)
        def accepted_frame(step):
            checkpoint = out / f"accepted_{step:04d}.npz"
            identity = json.loads((out / f"checkpoint_{step:04d}.json").read_text())
            if sha(checkpoint) != identity["sha256"]:
                raise ValueError(f"accepted checkpoint hash mismatch at step {step}")
            with np.load(checkpoint, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["parameters"], points[step])
                state = replace(initial_equilibrium.state, **{
                    name: jnp.asarray(saved[name]) for name in
                    ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
            return state, surface(state, initial_equilibrium.runtime), chart.coils_from_x(jnp.asarray(points[step]))

        post_monitor = opt.OptimizationMonitor(stream=None)
        coil_cost_values = jax.jit(coil_costs)
        previous = initial_gamma = None
        for step, x in enumerate(points):
            _, step_surface, step_coils = accepted_frame(step)
            row = history[step]
            dofs = np.asarray(step_coils.curves.dofs).ravel()
            raw = np.asarray(step_coils.curves.dofs)/np.asarray(step_coils.curves.scaling)
            gamma, currents = np.asarray(step_coils.gamma), np.asarray(step_coils.dofs_currents_raw)
            current = dict(u=x/scales, dofs=dofs, raw=raw, gamma=gamma, currents=currents)
            if previous is None:
                previous, initial_gamma = current, gamma.copy()
            du, coil_du = current["u"]-previous["u"], (dofs-previous["dofs"])/COIL_STEP
            displacement = np.linalg.norm(gamma-previous["gamma"], axis=-1)
            coil_field = np.asarray(jax.vmap(BiotSavart(step_coils).B)(step_surface.gamma.reshape(-1, 3))).reshape(step_surface.gamma.shape)
            normal_field = np.sum(coil_field*np.asarray(step_surface.unitnormal), axis=-1)/np.linalg.norm(coil_field, axis=-1)
            area = np.asarray(step_surface.area_element)
            row.update(qa_residual_l2=float(np.sqrt(row["qa"])), aspect_error=row["aspect"]-ASPECT_TARGET,
                iota_violation=max(IOTA_FLOOR-row["min_abs_iota"], 0.0),
                step_u_l2=float(np.linalg.norm(du)), step_u_linf=float(np.max(np.abs(du))),
                coil_step_u_l2=float(np.linalg.norm(coil_du)), coil_step_u_linf=float(np.max(np.abs(coil_du))),
                coil_coefficient_step_l2_m=float(np.linalg.norm(raw-previous["raw"])),
                coil_displacement_rms_m=float(np.sqrt(np.mean(displacement**2))),
                coil_displacement_max_m=float(displacement.max()),
                coil_displacement_from_start_max_m=float(np.linalg.norm(gamma-initial_gamma, axis=-1).max()),
                current_step_max_A=float(np.max(np.abs(currents-previous["currents"]))),
                normal_field_rms=float(np.sqrt(np.sum(area*normal_field**2)/area.sum())),
                normal_field_max=float(np.max(np.abs(normal_field))))
            if not all(np.isfinite(value) for value in row.values()):
                raise ValueError(f"nonfinite accepted-step diagnostics at step {step}")
            terms = dict(quasisymmetry=0.5*row["qa"], aspect=0.5*ASPECT_WEIGHT*row["aspect_error"]**2,
                         **{"iota floor": 0.5*IOTA_WEIGHT*row["iota_violation"]**2})
            terms.update(zip(("coil length", "coil curvature", "coil separation", "coil-surface separation"),
                             map(float, np.asarray(coil_cost_values(step_coils, step_surface)))))
            np.testing.assert_allclose(sum(terms.values()), row["objective"], rtol=1e-10, atol=1e-12)
            post_monitor.record(x, cost=row["objective"], iteration=step, terms=terms)
            np.savez_compressed(out / f"step_{step:04d}.npz", **current)
            previous = current
        with (out / "accepted_steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json("accepted_steps.json", history)
        post_monitor.save(out / "free_boundary_scalar_objectives.csv")

        if not args.no_plots:
            vj.plot_optimization_objects(out / "optimization.png", ("Initial", seed_surface, coils0), ("Optimized", surf, final_coils))
            post_monitor.plot(out / "objectives.png", title="Single-stage objective terms")
            if args.movie:
                def objects_from_x(x):
                    return accepted_frame(frame_steps[tuple(x)])[1:]

                def movie_colors(x, objects):
                    state = accepted_frame(frame_steps[tuple(x)])[0]
                    data = vj.surface_field_data_from_state(inp, state, runtime=initial_equilibrium.runtime,
                                                           nphi=NPHI, ntheta=NTHETA)
                    magnitude = jnp.linalg.norm(data.B_total, axis=0)
                    if MOVIE_SURFACE_COLOR == "absB":
                        return magnitude
                    interface = vj.PlasmaVacuumInterface.from_surface_data(data, digits=4)
                    return interface.bnormal_residual(chart(jnp.asarray(x)))/magnitude

                post_monitor.movie(out / "optimization.gif", objects_from_x,
                    color_factory=movie_colors if MOVIE_SURFACE_COLOR is not None else None,
                    color_label=MOVIE_SURFACE_COLOR, cmap="jet")
            for path in vj.plot_wout(wout_path, out).values():
                print(f"Wrote {path}")
        summary.update(postprocessing="complete", postprocessing_seconds=time.perf_counter()-postprocessing_started)
        write_json("optimization_summary.json", summary)
        print(f"{status}; numerical verification passed; physical limits met: {feasible}. Results: {out}", flush=True)
        return 0 if summary["optimizer_success"] and feasible else 1
    except Exception as error:
        write_json("failure.json", dict(phase=phase, error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        for item in (verification, problem):
            if item is not None:
                item.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
