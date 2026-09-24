"""Matched entry points, file restarts and real coil fits without plasma solves."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/single-stage-benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def entries():
    return (load("single_stage_optimization_scalar"),
            load("free_boundary_single_stage_optimization_scalar"))


@pytest.mark.parametrize("flags", [[], ["--maxiter", "7"],
    ["--input", "input.custom", "--wout", "source.nc", "--initial-coils", "initial.json",
     "--resolution", "4", "3", "21", "--grid", "24", "20", "--coil-fit-maxiter", "3",
     "--device", "cpu", "--no-plots", "--no-movie", "--ftol", "1e-14", "--accepted-steps", "2"],
    ["--coils", "fitted.json", "--plots", "--movie"]])
def test_matching_options_and_defaults(entries, flags):
    fixed, free = (vars(entry.parse_args(flags)) for entry in entries)
    fixed.pop("output")
    free.pop("output")
    assert fixed == free
    for key in ("input", "wout", "coils", "initial_coils"):
        assert fixed[key] is None or fixed[key].is_absolute()


@pytest.mark.parametrize("flags", [["--wout", "alone.nc"],
    ["--coils", "a.json", "--initial-coils", "b.json"], ["--coil-fit-maxiter", "0"],
    ["--resolution", "0", "2", "11"], ["--ftol", "nan"]])
def test_matching_validation(entries, flags):
    for entry in entries:
        with pytest.raises(SystemExit):
            entry.parse_args(flags)


def test_dry_runs_and_explicit_input_defaults(entries, tmp_path, capsys, monkeypatch):
    for entry in entries:
        output = tmp_path / Path(entry.__file__).stem
        assert entry.main(["--dry-run", "--output", str(output)]) == 0
        config = json.loads(capsys.readouterr().out)
        assert config["optimizer"] == "SLSQP"
        assert config["arguments"]["resolution"] == [8, 8, 51]
        assert config["arguments"]["ftol"] == 1e-15
        assert not output.exists()
        custom = entry.parse_args(["--input", "input.custom"])
        assert custom.resolution is None and custom.grid is None
        monkeypatch.setattr(entry, "COILS", Path("default.json"))
        assert entry.parse_args(["--initial-coils", "initial.json"]).coils is None


def test_fixed_run_restores_working_directory_on_failure(entries, tmp_path, monkeypatch):
    fixed, _ = entries
    old = Path.cwd()
    output = tmp_path / "failed"
    # Restore environment changes made by main after this test.
    import os
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "JAX_ENABLE_X64",
                "JAX_PLATFORMS", "VMEX_COMPILATION_CACHE", "JAX_ENABLE_COMPILATION_CACHE", "MPLBACKEND"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    def fail(args):
        assert Path.cwd() == output
        assert args.input == old / "input.custom"
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(fixed, "run", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        fixed.main(["--device", "cpu", "--input", "input.custom", "--output", str(output)])
    assert Path.cwd() == old


@pytest.mark.parametrize("mode", ["generated", "initial-coils", "coils"])
def test_fixed_file_workflow_real_coil_fit(entries, tmp_path, monkeypatch, mode):
    """Run the fixed setup/fit, replacing only the equilibrium/optimizer entry."""
    pytest.importorskip("essos")
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    import inspect
    from vmex.core.optimize import make_problem
    from essos.coils import Coils

    fixed, free = entries
    for entry in entries:
        for name, value in dict(N_COILS=1, COIL_ORDER=1, N_SEGMENTS=8, NPHI=8, NTHETA=8).items():
            monkeypatch.setattr(entry, name, value)
    monkeypatch.delenv("VMEX_EXAMPLES_CI", raising=False)
    output = tmp_path / "fixed"
    output.mkdir()
    flags = ["--input", str(EXAMPLES / "input.rotating_ellipse"), "--device", "cpu",
             "--output", str(output), "--coil-fit-maxiter", "2"]
    inp, _ = fixed.common.load_input(fixed.parse_args(flags))
    coil_file = tmp_path / "coils.json"
    seed_coils = fixed.common.initial_coils(inp, None, parameters=vars(fixed))
    if mode != "generated":
        # Preserve user currents, including ones that differ from the default.
        seed_coils = Coils(seed_coils.curves, jnp.array([314159.]))
        seed_coils.to_json(str(coil_file))
        flags.extend([f"--{mode}", str(coil_file)])
    seed = object()
    captured = []
    real_loader = fixed.common.load_input
    monkeypatch.setattr(fixed.common, "load_input", lambda args, **kw: (real_loader(args)[0], seed))
    def boundary(x):
        return (jnp.asarray(inp.rbc).at[inp.ntor, 1].add(x[0]),
                jnp.asarray(inp.zbs).at[inp.ntor, 1].add(x[1]))

    fake_problem = SimpleNamespace(x0=np.zeros(2), scales=np.ones(2), dof_names=("r", "z"),
                                   metadata={}, boundary_from_x=boundary)
    fake_problem.with_accepted_state = lambda: fake_problem

    def plasma(inp, terms, **kw):
        inspect.signature(make_problem).bind(inp, **kw)
        captured.append(kw)
        return fake_problem

    monkeypatch.setattr(opt.VmecProblem, "from_loss", plasma)
    monkeypatch.setattr(opt.VmecProblem, "from_tuples", plasma)

    class FitFinished(Exception):
        pass

    def stop_at_optimizer(x, **kw):
        np.testing.assert_array_equal(x[:2], [0., 0.])  # Stage two freezes the boundary.
        raise FitFinished

    monkeypatch.setattr(vj.FunctionProblem, "from_functions", stop_at_optimizer)
    monkeypatch.chdir(output)
    with pytest.raises(FitFinished):
        fixed.run(fixed.parse_args(flags))
    assert len(captured) == 2 and all(call["restart_from"] is seed for call in captured)
    fitted = Coils.from_json(str(output / "coils.stage2.json"))
    np.testing.assert_array_equal(fitted.dofs_currents_raw, seed_coils.dofs_currents_raw)
    report = json.loads((output / "stage_two.json").read_text())
    assert report["reused"] == (mode == "coils")
    assert report["objective"] <= report["initial_objective"] + 1e-10
    if mode == "coils":
        np.testing.assert_array_equal(fitted.curves.dofs, seed_coils.curves.dofs)
        assert report["iterations"] == 0

    # The free entry must produce the same stage-two geometry from the same files.
    free_output = tmp_path / "free"
    free_output.mkdir()
    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_loss", lambda *a, **kw:
        SimpleNamespace(accepted_step=0, enable_root_polishing=lambda **kw: None))
    args = free.parse_args(flags)
    args.output = free_output
    stage = free.build_problem(args)
    # Scalar sums and squared residual vectors differ in floating-point reduction order.
    np.testing.assert_allclose(stage.coils.curves.dofs, fitted.curves.dofs, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(stage.coils.dofs_currents_raw, fitted.dofs_currents_raw)
