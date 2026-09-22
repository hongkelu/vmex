"""Public-API fast example configuration and objective parity."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/three-methods-benchmark"
spec = importlib.util.spec_from_file_location("fast_example", EXAMPLE / "free_boundary_single_stage_optimization_scalar.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.modules["free_boundary_single_stage_optimization_scalar"] = entry
verify_spec = importlib.util.spec_from_file_location("verify_fast_example", EXAMPLE / "verify_free_boundary_single_stage.py")
verify = importlib.util.module_from_spec(verify_spec)
verify_spec.loader.exec_module(verify)


def test_defaults_preserve_fast_profile():
    args = entry.parse_args([])
    assert args.ftol == 1e-15
    assert verify.GRADIENT_CHECK_FTOL == 1e-20
    assert entry.ADJOINT_RESIDUAL_RTOL == 1e-9
    assert verify.LINEARIZATION_PARITY_RTOL == 1e-6
    assert entry.MATRIXFREE_RHS_BATCH_SIZE == 3
    assert entry.VERIFY_NS == 201
    assert args.accepted_steps == 100 and args.resolution == (8, 8, 51)


def test_parameter_edits_and_cli_override(monkeypatch):
    monkeypatch.setattr(entry, "EQUILIBRIUM_FTOL", 1e-14)
    monkeypatch.setattr(entry, "ACCEPTED_STEPS", 7)
    args = entry.parse_args(["--ftol", "1e-13"])
    assert args.ftol == 1e-13 and args.accepted_steps == 7


def test_dry_run_creates_no_output(tmp_path, capsys):
    output = tmp_path / "not-created"
    assert entry.main(["--dry-run", "--output", str(output)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["arguments"]["ftol"] == 1e-15
    assert config["parameters"]["IOTA_FLOOR"] == .19
    assert not output.exists()


@pytest.mark.parametrize("args", [["--ftol", "nan"], ["--ftol", "0"],
                                  ["--accepted-steps", "0"], ["--resolution", "8", "8", "1"]])
def test_invalid_configuration_rejected(args):
    with pytest.raises(SystemExit):
        entry.parse_args(args)


def test_example_imports_public_vmex_only():
    tree = ast.parse(Path(entry.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.module.startswith(("vmex.core", "_"))
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("_anchor", "_records", "_linearization", "_HOT_CACHE")


def function(name, **namespace):
    """Exercise the actual example objective without launching an equilibrium."""
    tree = ast.parse(Path(entry.__file__).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(vars(entry), np=np, jnp=jnp, **namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(entry.__file__), "exec"), scope)
    return scope[name]


def test_constraint_margins_and_chain_rule():
    values = np.array([.23, .997])
    inequalities = function("inequalities")
    expected = np.array([[1/entry.IOTA_FLOOR, 0], [0, 1/entry.RADIUS_TOLERANCE], [0, -1/entry.RADIUS_TOLERANCE]])
    h = 1e-6
    fd = np.column_stack([(inequalities(values+h*d)-inequalities(values-h*d))/(2*h) for d in np.eye(2)])
    np.testing.assert_allclose(fd, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(inequalities([.1905, .991]), [0, 0, 1.8], atol=1e-13)


def test_coil_objective_preserves_weights_and_moving_surface():
    from types import SimpleNamespace
    costs = function("coil_costs", loss_coil_separation=lambda *a, **k: 0.02,
                     loss_coil_surface_distance=lambda coils, surf, *a, **k: surf**2)
    coils = SimpleNamespace(length=jnp.array([4., 5., 6.]), curvature=jnp.array([[6.], [7.], [8.]]))
    expected = [1., .5*10*(.1**2+1.1**2), 10., .5*1e3*.3**2]
    np.testing.assert_allclose(costs(coils, .3), expected, rtol=1e-13)
    import jax
    assert float(jax.grad(lambda s: costs(coils, s).sum())(.3)) == pytest.approx(300.)


def test_postprocessing_defaults_and_opt_outs():
    args = entry.parse_args([])
    assert not args.no_plots and args.movie
    assert entry.parse_args(["--no-plots"]).no_plots
    assert not entry.parse_args(["--no-movie"]).movie
    assert entry.MOVIE_SURFACE_COLOR == "absB"


def replay_namespace(tmp_path):
    """Run the example's actual replay block with synthetic saved equilibria."""
    from dataclasses import make_dataclass
    from types import SimpleNamespace
    import hashlib
    import jax
    from vmex import optimize as opt
    import vmex as vj

    fields = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    State = make_dataclass("State", [(name, object) for name in fields])
    state = State(*(jnp.zeros((1, 1)) for _ in fields))
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 5), np.linspace(0, 2*np.pi, 4))
    exports, colors, wout_calls = [], [], []

    class Surface:
        def __init__(self, shift=0):
            r = 1 + (.2+shift)*np.cos(theta)
            self.gamma = jnp.array(np.stack([r*np.cos(phi), r*np.sin(phi), (.2+shift)*np.sin(theta)], axis=-1))
            self.area_element = jnp.ones(theta.shape)
            self.unitnormal = jnp.ones(self.gamma.shape)/np.sqrt(3)

        def plot(self, *, ax, show):
            xyz = np.asarray(self.gamma)
            ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2])

        def to_vtk(self, path, **kwargs):
            exports.append(Path(path).name)

    class Coils:
        def __init__(self, x):
            self.gamma = jnp.array([np.asarray(Surface().gamma)[0]+x[0]])
            self.curves = SimpleNamespace(dofs=jnp.array(x).reshape(1, 1, 2), scaling=jnp.ones(2))
            self.dofs_currents_raw = jnp.array([100.])

        def plot(self, *, ax, show):
            for curve in np.asarray(self.gamma):
                ax.plot(*curve.T)

        def to_vtk(self, path):
            exports.append(Path(path).name)

    class Chart:
        def coils_from_x(self, x):
            return Coils(x)

    class Field:
        def __init__(self, coils):
            pass

        def B(self, xyz):
            return jnp.array([0., 0., 2.])

    def surface_field(inp, value, **kwargs):
        colors.append(float(value.R_cos[0, 0]))
        return SimpleNamespace(B_total=jnp.ones((3, *theta.shape))*(1+value.R_cos[0, 0]))

    def plot_wout(path, output):
        wout_calls.append((path, output))
        return {"summary": output / "wout_summary.png"}

    monitor = opt.OptimizationMonitor(stream=None)
    history = []
    for step, point in enumerate((np.array([0., 0.]), np.array([.03, -.01]))):
        cost = 1.-.5*step
        monitor.record(point, cost=cost, iteration=step)
        checkpoint = tmp_path / f"accepted_{step:04d}.npz"
        np.savez(checkpoint, parameters=point, **{name: np.full((1, 1), .03*step) for name in fields})
        (tmp_path / f"checkpoint_{step:04d}.json").write_text(json.dumps({"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
        history.append(dict(step=step, objective=cost, qa=2*cost, aspect=5., min_abs_iota=.2, constraints_feasible=True))
    namespace = dict(vars(entry), np=np, jnp=jnp, out=tmp_path, monitor=monitor, history=history,
        initial_wout=tmp_path/"initial.nc", wout_path=tmp_path/"verified.nc", coils0=Coils(np.zeros(2)),
        final_coils=Coils(np.array([.03, -.01])), surf=Surface(.03),
        SurfaceRZFourier=SimpleNamespace(from_wout_file=lambda *a, **k: Surface()),
        BiotSavart=Field, chart=Chart(), scales=np.array([.5, .2]), opt=opt, inp=object(),
        initial_equilibrium=SimpleNamespace(state=state, runtime=object()),
        surface=lambda s, rt: Surface(float(s.R_cos[0, 0])), coil_costs=lambda *a: jnp.zeros(4),
        sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        signal=SimpleNamespace(alarm=lambda *_: None), jax=SimpleNamespace(jit=lambda f: f, vmap=jax.vmap),
        args=SimpleNamespace(no_plots=False, movie=True), summary={},
        write_json=lambda name, data: (tmp_path/name).write_text(json.dumps(data)),
        vj=SimpleNamespace(plot_optimization_objects=vj.plot_optimization_objects,
                           surface_field_data_from_state=surface_field, plot_wout=plot_wout))
    tree = ast.parse(Path(entry.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    run = next(n for n in main.body if isinstance(n, ast.Try))
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Constant) and n.value.value == "postprocessing")
    code = compile(ast.Module(body=run.body[start:-2], type_ignores=[]), str(entry.__file__), "exec")
    return namespace, code, exports, colors, wout_calls


@pytest.mark.parametrize("plots,movie", [(True, True), (True, False), (False, True)])
def test_saved_replay_exports_terms_diagnostics_and_movie(tmp_path, plots, movie):
    from PIL import Image
    namespace, code, exports, colors, wout_calls = replay_namespace(tmp_path)
    namespace["args"].no_plots = not plots
    namespace["args"].movie = movie
    exec(code, namespace)
    assert set(exports) == {"surface_initial", "coils_initial", "surface_optimized", "coils_optimized"}
    assert namespace["summary"]["postprocessing"] == "complete"
    history = json.loads((tmp_path/"accepted_steps.json").read_text())
    assert history[0]["step_u_l2"] == 0 and history[1]["step_u_l2"] > 0
    assert history[1]["current_step_max_A"] == 0
    assert (tmp_path/"accepted_steps.csv").exists() and (tmp_path/"step_0001.npz").exists()
    columns = (tmp_path/"free_boundary_scalar_objectives.csv").read_text().splitlines()[0]
    assert "quasisymmetry" in columns and "coil-surface separation" in columns
    assert bool(wout_calls) == plots
    assert (tmp_path/"optimization.png").exists() == plots
    assert (tmp_path/"objectives.png").exists() == plots
    assert (tmp_path/"optimization.gif").exists() == (plots and movie)
    if plots and movie:
        with Image.open(tmp_path/"optimization.gif") as image:
            assert image.n_frames == 2
        np.testing.assert_allclose(colors, [0., .03])


def test_saved_replay_rejects_modified_checkpoint(tmp_path):
    namespace, code, *_ = replay_namespace(tmp_path)
    (tmp_path/"accepted_0001.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        exec(code, namespace)


@pytest.mark.parametrize("quantity,expected", [("absB", 5.), ("B.n/B", .5)])
def test_movie_uses_saved_total_field_and_plasma_interface(quantity, expected):
    from types import SimpleNamespace
    state, runtime, inp, field = object(), object(), object(), object()
    calls = []

    def surface_field(actual_input, actual_state, **kwargs):
        assert actual_input is inp and actual_state is state and kwargs["runtime"] is runtime
        return SimpleNamespace(B_total=jnp.array([[[3.]], [[4.]], [[0.]]]))

    def normal_residual(actual_field):
        assert actual_field is field
        calls.append("plasma interface")
        return jnp.array([[2.5]])

    color = function("movie_colors", MOVIE_SURFACE_COLOR=quantity, inp=inp,
        frame_steps={(0.,): 0}, accepted_frame=lambda step: (state, None, None),
        initial_equilibrium=SimpleNamespace(runtime=runtime), chart=lambda x: field,
        vj=SimpleNamespace(surface_field_data_from_state=surface_field,
            PlasmaVacuumInterface=SimpleNamespace(from_surface_data=lambda *a, **kw:
                SimpleNamespace(bnormal_residual=normal_residual))))
    np.testing.assert_allclose(color(np.array([0.]), ()), [[expected]])
    assert calls == (["plasma interface"] if quantity == "B.n/B" else [])


def test_verification_is_separate_and_production_requires_report(tmp_path, capsys):
    assert verify.main(["--dry-run", "--output", str(tmp_path/"unused")]) == 0
    assert json.loads(capsys.readouterr().out)["gradient_force_tolerance"] == 1e-20
    assert not (tmp_path/"unused").exists()
    with pytest.raises(ValueError, match="verify_free_boundary"):
        entry.main(["--output", str(tmp_path/"production")])
    assert not (tmp_path/"production").exists()
    tree = ast.parse(Path(entry.__file__).read_text())
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "evaluate_trial" not in calls
    enable = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "enable_matrix_free")
    assert not enable.args and "parity_rtol" not in {kw.arg for kw in enable.keywords}


def test_wout_requires_profiles_and_custom_input_preserves_resolution():
    with pytest.raises(SystemExit):
        entry.parse_args(["--wout", "wout.nc"])
    args = entry.parse_args(["--input", "input.custom", "--wout", "wout.nc"])
    assert args.resolution is None and args.grid is None


def qualification_bundle(tmp_path):
    args = entry.parse_args(["--device", "cpu"])
    (tmp_path/"coils.json").write_text("coils")
    (tmp_path/"seed.npz").write_text("checkpoint")
    report = dict(schema="vmex.single-stage-qualification/v1", passed=True,
        contract=entry.qualification_contract(args),
        configuration=dict(input=str(args.input), wout=None, resolution=args.resolution,
                           grid=args.grid, ftol=args.ftol, device=args.device),
        checks=dict(finite_difference=dict(passed=True), matrix_free=dict(passed=True)),
        artifacts={name:dict(file=file, sha256=entry.sha(tmp_path/file))
                   for name, file in (("coils", "coils.json"), ("checkpoint", "seed.npz"))})
    path = tmp_path/"qualification.json"
    path.write_text(json.dumps(report))
    return path, report


def test_qualification_matching_and_artifact_integrity(tmp_path, monkeypatch):
    path, report = qualification_bundle(tmp_path)
    args = entry.parse_args(["--qualification", str(path), "--accepted-steps", "2", "--no-plots"])
    accepted, files = entry.read_qualification(args)
    assert accepted["passed"] and files["coils"] == tmp_path/"coils.json"
    monkeypatch.setattr(entry, "IOTA_FLOOR", .21)
    with pytest.raises(ValueError, match="qualification differs"):
        entry.read_qualification(args)
    monkeypatch.undo()
    (tmp_path/"seed.npz").write_text("modified")
    with pytest.raises(ValueError, match="checkpoint hash"):
        entry.read_qualification(args)
    report["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="has not passed"):
        entry.read_qualification(args)


def test_shared_builder_fits_fixed_currents_and_restores_without_solves(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vmex import optimize as opt
    from essos.coils import Coils

    for name, value in dict(N_COILS=1, COIL_ORDER=1, N_SEGMENTS=8, NPHI=8, NTHETA=8).items():
        monkeypatch.setattr(entry, name, value)
    calls = []
    monkeypatch.setattr(opt, "solve_equilibrium", lambda inp, **kw: calls.append("fixed") or SimpleNamespace(state="fixed state"))

    def from_loss(inp, loss, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(x0=kwargs["parameterization"].x0, accepted_step=0)

    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_loss", from_loss)
    first = tmp_path/"fitted"
    first.mkdir()
    args = entry.parse_args(["--device", "cpu", "--output", str(first), "--coil-fit-maxiter", "2"])
    prepared = entry.build_problem(args)
    fit = json.loads((first/"stage_two.json").read_text())
    assert not fit["reused"] and fit["objective"] <= fit["initial_objective"]
    np.testing.assert_array_equal(prepared.chart.currents, [entry.COIL_CURRENT])
    assert prepared.chart.current_dofs == () and calls[0] == "fixed"
    assert calls[1]["restart_from"] == "fixed state"
    fitted = Coils.from_json(str(first/"coils.stage2.json"))
    np.testing.assert_array_equal(fitted.dofs_currents_raw, [entry.COIL_CURRENT])
    restored = tmp_path/"restored"
    restored.mkdir()
    args.output = restored
    calls.clear()
    stage = entry.build_problem(args, qualified=({"artifacts":{"checkpoint":{"sha256":"abc"}}},
                               {"coils": first/"coils.stage2.json", "checkpoint": first/"seed.npz"}))
    assert len(calls) == 1 and calls[0]["checkpoint"] == first/"seed.npz"
    assert "restart_from" not in calls[0]
    np.testing.assert_array_equal(stage.chart.coefficients, prepared.chart.coefficients)
    assert json.loads((restored/"stage_two.json").read_text())["reused"]


def test_shared_builder_wout_seed_uses_source_boundary_and_input_profiles(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import vmex as vj
    from vmex import optimize as opt

    source = tmp_path/"source.nc"
    source.write_bytes(b"WOUT fixture")
    coils = EXAMPLE / "coils_single_stage_scalar_fitted_1789769333906005000.json"
    args = entry.parse_args(["--input", str(EXAMPLE/"input.rotating_ellipse"), "--wout", str(source),
                            "--coils", str(coils), "--device", "cpu", "--output", str(tmp_path)])
    wout = SimpleNamespace(nfp=2, lasym=False, mpol=2, ntor=0, ns=3,
                           xm=np.array([0, 1]), xn=np.array([0, 0]),
                           rmnc=np.tile([1.2, .2], (3, 1)), zmns=np.tile([0., .2], (3, 1)))
    monkeypatch.setattr(vj, "read_wout", lambda path: wout)
    monkeypatch.setattr(vj, "state_from_wout", lambda *a, **k: "wout seed")
    monkeypatch.setattr(opt, "solve_equilibrium", lambda *a, **k: pytest.fail("WOUT must avoid the fixed solve"))
    captured = {}
    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_loss", lambda inp, loss, **kw:
                        captured.update(inp=inp, **kw) or SimpleNamespace(accepted_step=0))
    stage = entry.build_problem(args)
    assert captured["restart_from"] == "wout seed"
    assert stage.inp.rbc[stage.inp.ntor, 0] == 1.2
    assert stage.inp.mpol == 3 and stage.inp.ntor == 3 and stage.inp.ns_array[-1] == 31
    original = vj.VmecInput.from_file(args.input)
    np.testing.assert_array_equal(stage.inp.am, original.am)
    np.testing.assert_array_equal(stage.inp.ac, original.ac)
    assert stage.inp.phiedge == original.phiedge and stage.inp.curtor == original.curtor



def test_lbfgsb_entry_uses_shared_production_and_no_output_on_dry_run(tmp_path, capsys, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(EXAMPLE))
    lbfgsb = importlib.import_module("free_boundary_single_stage_optimization")
    output = tmp_path/"unused-lbfgsb"
    assert lbfgsb.main(["--dry-run", "--output", str(output)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["optimizer"] == "L-BFGS-B" and not output.exists()
    assert lbfgsb.run is entry.main
    with pytest.raises(ValueError, match="verify_free_boundary"):
        lbfgsb.main(["--output", str(output)])
    assert not output.exists()


def test_lbfgsb_backtracking_does_not_promote_gradient_probes():
    from types import SimpleNamespace

    class Rosenbrock:
        x0 = np.zeros(2)
        accepted_step = 0

        def __init__(self):
            self.accepted = SimpleNamespace(parameters=self.x0.copy())
            self.evaluated = []
            self.promoted = []

        def value_and_grad(self, x):
            self.evaluated.append((x.copy(), self.accepted.parameters.copy()))
            self.last = x.copy()
            value = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
            return value, np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

        def accept_x(self, x):
            np.testing.assert_array_equal(x, self.last)
            self.accepted = SimpleNamespace(parameters=x.copy())
            self.promoted.append(x.copy())
            self.accepted_step += 1

    problem = Rosenbrock()
    scales = np.array([.5, .3])
    recorded = []
    result = entry.run_optimizer(problem, scales, None, None, SimpleNamespace(accepted_steps=100),
                                 lambda: recorded.append(problem.accepted.parameters.copy()), method="L-BFGS-B")
    assert result.success and len(recorded) == problem.accepted_step
    assert len(problem.evaluated) > len(recorded)+1  # real line-search probes occurred
    np.testing.assert_allclose(problem.accepted.parameters, [1., 1.], atol=1e-5)
    np.testing.assert_array_equal(problem.accepted.parameters, result.x*scales)
    anchor = problem.x0
    next_accepted = iter(problem.promoted)
    accepted = next(next_accepted, None)
    for point, origin in problem.evaluated:
        np.testing.assert_array_equal(origin, anchor)
        if accepted is not None and np.array_equal(point, accepted):
            anchor, accepted = accepted, next(next_accepted, None)
