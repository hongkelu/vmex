#!/usr/bin/env python
"""Finite-beta scalar production using the same public VMEX API as vacuum.

Fixed pressure and PHIEDGE, zero prescribed plasma current, fixed coil currents.
Beta is a diagnostic around the prepared 0.5% reference. The moving plasma field
comes from virtual casing; NESTOR remains coupled to the free-boundary solve.
Production never runs derivative comparisons. Use verify_finite_beta_production.py
for independent finite differences and dense/matrix-free comparisons.
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

INPUT = HERE / "inputs/finite_beta_initial/input.fixed"
COILS = HERE / "inputs/finite_beta_initial/coils.json"
BETA_REFERENCE = 0.005
VC_DIGITS = 4
COIL_FIT_MAXITER = 200
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.5, 2.7e5
NORMAL_FIELD_WEIGHT, NORMAL_FIELD_OBJECTIVE_LIMIT, NORMAL_FIELD_LIMIT_WEIGHT = 1e3, 0.008, 2e5

# Optimization and equilibrium accuracy (different stopping criteria).
ACCEPTED_STEPS = 100
OPTIMIZER_FTOL = 1e-10
PARAMETER_BOUND = 5.0  # only the preliminary coil fit has box bounds
COIL_STEP = 0.05
MAX_TRIALS = 500
RESOLUTION = (8, 8, 51)  # MPOL, NTOR, NS
GRID = (64, 64)  # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
ROOT_TOLERANCE = 2e-6  # ordinary-root admission before coupled Newton refinement
ROOT_POLISH_TOLERANCE = 1e-12
INITIALIZATION_SECONDS = 0  # no elapsed cutoff; nonlinear iterations remain bounded
OPTIMIZATION_SECONDS = 0  # accepted-step and trial budgets still apply
VERIFICATION_SECONDS = 1800

# Same QA, aspect, iota and coil penalties as the scalar reference.
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
ASPECT_TARGET, ASPECT_WEIGHT = 5.0, 1.0
IOTA_FLOOR, IOTA_WEIGHT = 0.19, 10.0
RADIUS_TARGET, RADIUS_TOLERANCE = 1.0, 0.01
IOTA_MARGIN, RADIUS_MARGIN = 0.0005, 0.001
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 5, 256  # prepared finite-beta coils use refined quadrature
NPHI, NTHETA = 37, 32
LENGTH_TARGET, LENGTH_WEIGHT = 5.0, 1.0
CURVATURE_OBJECTIVE_LIMIT, CURVATURE_LIMIT, CURVATURE_WEIGHT = 6.9, 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.15, 1e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1e3
# Preserve the existing finite-beta total-field penalties and diagnostic limit.
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
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version("virtual-casing-jax")) < Version("0.0.8"):
        raise RuntimeError("current VMEX production requires virtual-casing-jax>=0.0.8")
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
                   load_input, make_interface_functions, common.initial_coils, common.physical_constraint),
        sources=[HERE / "benchmark.json"])

def read_qualification(args):
    """Reuse the same authenticated fitted coils and accepted seed."""
    from vmex import optimize as opt

    if args.qualification is None:
        return None
    qualified = opt.OptimizationQualification.read(args.qualification, contract=qualification_contract(args))
    if args.coils is not None and args.coils != COILS.resolve() and sha(args.coils) != qualified.report["artifacts"]["coils"]["sha256"]:
        raise ValueError("supplied coils differ from the qualified fitted coils")
    return qualified

def load_input(args, *, restore_state=True):
    """Preserve the supplied pressure, flux and zero-current physics at any grid."""
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt

    if args.input == INPUT.resolve():
        manifest = json.loads((INPUT.parent / "artifact_manifest.json").read_text())
        for name in ("input.fixed", "coils.json"):
            if sha(INPUT.parent / name) != manifest[name]:
                raise ValueError(f"prepared finite-beta input changed: {name}")
    inp = vj.VmecInput.from_file(args.input)
    if (inp.lasym or inp.ncurr != 1 or inp.curtor != 0
            or inp.pcurr_type != "power_series" or np.any(inp.ac)):
        raise ValueError("finite-beta control requires stellarator symmetry, NCURR=1, CURTOR=0 and AC=0")
    if (inp.pmass_type != "power_series" or inp.pres_scale <= 0
            or not np.isfinite(inp.pres_scale) or not np.any(inp.am)
            or not np.isclose(np.sum(inp.am), 0., atol=1e-14, rtol=0)):
        raise ValueError("finite positive pressure with a power-series profile and zero edge pressure required")
    m, n, ns = args.resolution or (inp.mpol, inp.ntor, int(inp.ns_array[-1]))
    nt, nz = args.grid or (inp.ntheta, inp.nzeta)
    inp = inp.change_resolution(mpol=m, ntor=n, ntheta=nt, nzeta=nz)
    inp = replace(inp, ns_array=np.array([ns]), ftol_array=np.array([args.ftol]), lfreeb=False)
    seed = None
    if args.wout is not None:
        wout = vj.read_wout(args.wout)
        if int(wout.nfp) != inp.nfp or bool(wout.lasym):
            raise ValueError("WOUT symmetry differs from the input")
        rbc, zbs, rbs, zbc = opt.boundary_from_wout(wout, mpol=m, ntor=n)
        inp = replace(inp, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc)
        if restore_state:
            seed = vj.state_from_wout(wout, inp=inp, ns=ns)
    return inp, seed


def make_interface_functions(inp, fixed):
    """Trace the live plasma field, with a fixed quadrature plan, as in the fixed arm."""
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.virtual_casing import plan_vc_precision
    from essos.fields import BiotSavart

    data = vj.surface_field_data_from_state(inp, fixed.state, runtime=fixed.runtime, nphi=NPHI, ntheta=NTHETA)
    precision = plan_vc_precision(data, digits=VC_DIGITS)

    def rows(state, runtime, coils, nphi=NPHI, ntheta=NTHETA):
        data = vj.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        plan = precision if (nphi, ntheta) == (NPHI, NTHETA) else plan_vc_precision(data, digits=VC_DIGITS)
        interface = vj.PlasmaVacuumInterface.from_surface_data(data, precision=plan, digits=VC_DIGITS)
        field = BiotSavart(coils)
        external = lambda xyz: jax.vmap(field.B)(xyz.reshape(-1, 3)).reshape(xyz.shape)
        total = interface.total_B_out(external)
        normal = interface.bnormal_residual(external)/jnp.linalg.norm(total, axis=0)
        jump = interface.pressure_balance_residual(external)/jnp.sum(data.B_total**2, axis=0)
        return normal, interface.weights, jump

    def metrics(state, runtime, coils, nphi=NPHI, ntheta=NTHETA):
        normal, weights, jump = rows(state, runtime, coils, nphi, ntheta)
        beta = float(opt.volume_average_beta(state, runtime))
        return dict(normal_field_rms=float(jnp.sqrt(jnp.sum(weights*normal**2))),
            normal_field_max=float(jnp.max(jnp.abs(normal))),
            pressure_jump_rms=float(jnp.sqrt(jnp.sum(weights*jump**2))),
            beta_percent=100*beta, beta_relative_deviation=beta/BETA_REFERENCE-1)

    return rows, metrics


def build_problem(args, *, event=None, qualified=None):
    """Prepare the input, stage-two coils and common scalar VMEX problem.

    Without a qualification bundle, load or fit coils and solve the seed from
    input/WOUT. A supplied bundle restores its exact checked seed, skipping
    stage-two fitting and the initial free equilibrium solve. The fixed reference
    is solved only to select VC quadrature. No derivative tests run.
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
    inp, seed = load_input(args, restore_state=qualified is None)
    # A concrete equilibrium selects VC quadrature once; the live state and its
    # plasma field remain differentiable inside every objective evaluation.
    fixed = opt.solve_equilibrium(inp, initial_state=seed, device=args.device,
        raise_on_max_iterations=True, polish_force_balance=False)
    seed = fixed.state
    interface_rows, interface_metrics = make_interface_functions(inp, fixed)

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
        return (0.5*jnp.vdot(residuals, residuals)
                + jnp.sum(coil_costs(coils, surface(state, runtime)))
                + sum(extra_objective_terms(state, runtime, coils).values()))

    def extra_objective_terms(state, runtime, coils):
        bn, weights, _ = interface_rows(state, runtime, coils)
        smooth = jax.scipy.special.logsumexp(2000*jnp.sqrt(bn**2+1e-12))/2000
        return {"total normal field": 0.5*NORMAL_FIELD_WEIGHT*jnp.sum(weights*bn**2),
                "total normal field limit": 0.5*NORMAL_FIELD_LIMIT_WEIGHT*jnp.maximum(
                    smooth-NORMAL_FIELD_OBJECTIVE_LIMIT, 0)**2}

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
            normal, weights, _ = interface_rows(fixed.state, fixed.runtime, coils)
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
    write_json(out / "physics.json", dict(beta_reference=BETA_REFERENCE, beta_policy="diagnostic only",
        phiedge=float(inp.phiedge), pressure_scale_pa=float(inp.pres_scale),
        pressure_profile=np.asarray(inp.am).tolist(), curtor_A=float(inp.curtor), ncurr=int(inp.ncurr),
        pressure_variables=False, flux_variables=False, coil_currents="fixed", auglag=False,
        gradient_checks_in_production=False))
    from importlib.metadata import version
    write_json(out / "runtime.json", dict(python=sys.version,
        packages={name: version(name) for name in ("jax", "numpy", "scipy", "essos", "virtual-casing-jax")},
        devices=[str(device) for device in jax.devices()],
        source_hashes={str(path.relative_to(HERE.parents[1])): sha(path) for path in
            (Path(__file__), Path(common.__file__), Path(free.__file__),
             HERE.parents[1] / "vmex/core/freeboundary_problem.py",
             HERE.parents[1] / "vmex/core/_freeboundary_root_polish.py")}))
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
                           constraint_transform=constraint_transform, contract=contract,
                           interface_metrics=interface_metrics, extra_objective_terms=extra_objective_terms,
                           finite_beta=True)

def configure_solver(problem, args):
    """Prepare the production solver without qualification experiments."""
    problem.enable_matrix_free(rtol=MATRIXFREE_RTOL, restart=MATRIXFREE_RESTART,
        max_restarts=MATRIXFREE_MAX_CYCLES, rhs_batch_size=MATRIXFREE_RHS_BATCH_SIZE,
        refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.accepted_steps)

def run_optimizer(stage, args, record_step, *, method):
    """Choose the optimizer and physical bounds; VMEX owns scaling and acceptance."""
    from vmex import optimize as opt

    if method != "SLSQP":
        raise ValueError("this finite-beta control uses scalar SLSQP")
    problem = stage.problem
    constraints = common.physical_constraint(problem, parameters=globals())
    return opt.minimize(problem, method="SLSQP", constraints=constraints,
        callback=lambda x: record_step(), options=dict(maxiter=args.accepted_steps, ftol=OPTIMIZER_FTOL))


def verify_endpoint(stage, args, summary, history, initial_equilibrium):

    return free.verify_endpoint(SimpleNamespace(**globals()), None, stage, args, summary, history, initial_equilibrium)


def postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified):
    return free.postprocess(SimpleNamespace(**globals()), stage, args, summary, monitor, history, initial_equilibrium, verified)


def main(argv=None, *, method="SLSQP"):
    """Prepare the case, optimize, and independently solve the endpoint."""

    return free.run(parse_args(argv), case=SimpleNamespace(**globals()), method=method)


if __name__ == "__main__":
    raise SystemExit(main())
