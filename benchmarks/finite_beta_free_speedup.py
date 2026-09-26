"""Isolated full-grid solver comparison and bounded pilot from authenticated roots.

The source production directory is read-only. Run under an external finite
timeout on a spare GPU, with all caches inside the new benchmark directory.
"""

from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import argparse
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples/finite-beta-bootstrap-benchmark"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fixed", "free"), default="free")
    parser.add_argument("--first-step", type=int, default=15)
    parser.add_argument("--second-step", type=int, default=16)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--baseline-results-sha256")
    parser.add_argument("--baseline-options", choices=("original", "candidate"), default="original")
    parser.add_argument("--pilot-steps", type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), "benchmark output must be new"
    out.mkdir(parents=True)
    started = time.monotonic()

    def event(**data):
        item = dict(elapsed_seconds=time.monotonic() - started, **data)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item), flush=True)

    def timed(phase, fn, *a, **kw):
        tick = time.monotonic()
        value = fn(*a, **kw)
        for leaf in jax.tree.leaves(value):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        seconds = time.monotonic() - tick
        event(event="timing", phase=phase, seconds=seconds)
        return value, seconds

    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import case
        import physics
        from linear_root import _tangent_apply, _tangent_transpose
        from vmex.core._freeboundary_dense import _compress
        from vmex.core.solver import SpectralState

        assert jax.default_backend() == "gpu" and jax.config.x64_enabled and not jax.config.jax_disable_jit
        inp, coils, seed, profiles = case.load_prepared(args.prepared, allow_code_update=True)
        model, _ = timed("model_build", physics.EquilibriumCurrentRoot, inp, coils, seed, profiles, args.arm)
        candidate = model.linear
        manifest = json.loads((args.baseline_source.parent / "source-manifest.json").read_text())
        old = args.baseline_source / "examples/finite-beta-bootstrap-benchmark/linear_root.py"
        assert case.sha(old) == manifest[str(old.relative_to(args.baseline_source))]
        spec = importlib.util.spec_from_file_location("speedup_baseline_linear_root", old)
        baseline_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_module
        spec.loader.exec_module(baseline_module)
        baseline_options = (baseline_module.Options(**asdict(candidate.options))
                            if args.baseline_options == "candidate" else baseline_module.Options(extra_newton_steps=0, adaptive_newton=False,
                                refresh_horizon=0, reuse_budget_fraction=0) if args.arm == "fixed"
                            else baseline_module.Options())
        baseline = baseline_module.LinearRoot(model.residual, valid=model.valid, invalid_reason=model.geometry_status,
                                              options=baseline_options)
        engines = {"baseline": baseline, "candidate": candidate}
        results = {}
        if args.baseline_results is not None:
            assert args.baseline_results_sha256, "reused baseline needs an explicit authenticated digest"
            assert case.sha(args.baseline_results) == args.baseline_results_sha256
            results["baseline"] = json.loads(args.baseline_results.read_text())["baseline"]
            engines.pop("baseline")
            event(event="reused_baseline", path=str(args.baseline_results), sha256=args.baseline_results_sha256)
        accepted = {
            row["step"]: row
            for row in map(json.loads, (args.checkpoint_root / "accepted_steps.jsonl").read_text().splitlines())
        }

        def load(step):
            path = args.checkpoint_root / f"accepted_{step:04d}.npz"
            assert case.sha(path) == accepted[step]["checkpoint_sha256"]
            with np.load(path, allow_pickle=False) as data:
                state = SpectralState(**{k: jnp.asarray(data[k]) for k in SpectralState.__dataclass_fields__})
                delta = model.project(jax.tree.map(lambda a, b: a - b, state, model.frozen))
                y = np.r_[np.asarray(_compress(delta, model.space)), data["y"][-model.closure.size :]]
                x = data["x"].copy()
            certificate, _ = timed(f"certificate_{step}", model.certify, y, x)
            event(event="authenticated_root", step=step, sha256=case.sha(path), **certificate)
            return y, x

        y0, x0 = load(args.first_step)
        y1, x1 = load(args.second_step)
        campaign = case.CaseRun.__new__(case.CaseRun)
        campaign.args = SimpleNamespace(accepted_steps=args.pilot_steps)
        campaign.arm, campaign.out, campaign.started = args.arm, out / "pilot", time.monotonic()
        campaign.out.mkdir()
        campaign.model, campaign.x, campaign.y = model, x0.copy(), y0.copy()
        campaign.steps, campaign.trials, campaign.cache, campaign.rejected = args.first_step, 0, {}, {}
        # Fixed-boundary normal-field loss must retain the step-zero VC grid.
        # Replanning at the replay anchor changes the objective discretization.
        initial_root = args.initial_checkpoint_root or args.checkpoint_root
        initial_rows = [json.loads(line) for line in (initial_root / "accepted_steps.jsonl").read_text().splitlines()]
        initial_record = next(row for row in initial_rows if row["step"] == 0)
        initial_path = initial_root / "accepted_0000.npz"
        assert case.sha(initial_path) == initial_record["checkpoint_sha256"]
        with np.load(initial_path, allow_pickle=False) as data:
            initial_state = SpectralState(**{k: jnp.asarray(data[k]) for k in SpectralState.__dataclass_fields__})
            delta = model.project(jax.tree.map(lambda a, b: a - b, initial_state, model.frozen))
            initial_y = np.r_[np.asarray(_compress(delta, model.space)), data["y"][-model.closure.size:]]
            initial_x = data["x"].copy()
        state, rt, _, _ = model.objects(jnp.asarray(initial_y), jnp.asarray(initial_x))
        campaign.precision = case.interface(inp, state, rt)
        event(event="initial_interface_preserved", sha256=case.sha(initial_path), checkpoint=str(initial_path))
        campaign.configure_objectives(inp)
        for step, y, x in [(args.first_step, y0, x0), (args.second_step, y1, x1)]:
            rows = np.asarray(campaign.rows(y, x))
            np.testing.assert_allclose(
                rows[:3],
                [accepted[step]["objective"], accepted[step]["min_abs_iota"], accepted[step]["major_radius_m"]],
                rtol=1e-8,
                atol=1e-10,
            )
        timed("row_derivative_compile", jax.jacrev(campaign.rows, argnums=(0, 1)), jnp.asarray(y1), jnp.asarray(x1))
        rng = np.random.default_rng(7)
        vector = jnp.asarray(rng.normal(size=len(y0)))
        for label, engine in engines.items():
            engine.event = lambda label=label, **kw: event(engine=label, **kw)
            model.linear = engine
            _, dense_seconds = timed(label + "_dense_build", engine.rebuild, y0, x0)
            timed(label + "_jvp_compile", engine._jvp, y0, x0, vector)
            timed(label + "_vjp_compile", engine._vjp, y0, x0, vector)
            timed(label + "_parameter_compile", engine._parameter_jvp, y0, x0, x1 - x0)
            pull = jax.vjp(lambda point: engine.residual(jnp.asarray(y1), point), jnp.asarray(x1))[1]
            timed(label + "_parameter_pull_compile", pull, vector)
            # This eager pullback retains a full equilibrium tape. Its warmup
            # is complete; keeping it alive inflates the device Krylov peak.
            del pull
            if label == "candidate" or getattr(engine.options, "saved_krylov_actions", False):
                tangent = engine._prepare_tangent(y0, x0)
                helpers = ((_tangent_apply, _tangent_transpose) if label == "candidate" else
                           (baseline_module._tangent_apply, baseline_module._tangent_transpose))
                for transpose, action in [(False, helpers[0]), (True, helpers[1])]:
                    actual = engine._vjp if transpose else engine._jvp
                    value, _ = timed("saved_action_compile", action, tangent, vector)
                    reference = np.asarray(actual(y0, x0, vector))
                    parity = float(np.linalg.norm(np.asarray(value) - reference) / np.linalg.norm(reference))
                    assert parity < 1e-10
                    event(event="saved_action_parity", transpose=transpose, relative_error=parity)
                    timed("saved_action_warm", action, tangent, vector)
                    # Use the actual production residual/refinement path, not raw LU.
                    value, _ = timed(
                        "checked_random_solve", engine.linear, y0, x0, np.asarray(vector), transpose=transpose
                    )
                    defect = float(
                        np.linalg.norm(np.asarray(actual(y0, x0, value)) - np.asarray(vector)) / np.linalg.norm(vector)
                    )
                    event(event="checked_random_gate", transpose=transpose, relative_residual=defect)
                    assert defect <= engine.options.gate
                cache_sizes = tuple(action._cache_size() for action in helpers)
                if hasattr(engine, "_warm"):
                    from device_root import _krylov, _direct
                    factors = jax.tree.map(jnp.asarray, engine.factors)
                    for transpose, rtol, count in [(False, engine.options.rtol, 1),
                                                   (False, engine.options.newton_rtol, 1),
                                                   (True, engine.options.rtol, 1),
                                                   (True, engine.options.rtol, engine.rhs_batch_size)]:
                        rhs = jnp.zeros((count, len(y0)))
                        timed("device_krylov_compile", _krylov, tangent, factors, rhs, transpose=transpose,
                              rtol=rtol, restart=engine.options.restart, cycles=engine.options.cycles)
                        engine._warm(_krylov, tangent, factors, rhs, transpose=transpose,
                                     rtol=rtol, restart=engine.options.restart, cycles=engine.options.cycles)
                        timed("device_direct_compile", _direct, factors, rhs, transpose=transpose)
                        engine._warm(_direct, factors, rhs, transpose=transpose)
                    del factors
                # The engine owns the current tape; do not pin an obsolete
                # root in the benchmark while correction and FD move it.
                del tangent
            predicted, prediction_seconds = timed(label + "_prediction", model.seed_at, y0.copy(), x0, x1)
            if label == "baseline" and args.arm == "fixed":
                timed("baseline_ordinary_warmup", model.ordinary_seed, predicted.copy(), x1)
            def correction():
                if label == "baseline" and args.arm == "fixed":
                    return engine.solve(model.ordinary_seed(predicted.copy(), x1), x1)
                return model.correct_trial(predicted, x1)
            corrected, correction_seconds = timed(label + "_correction", correction)
            certificate = model.certify(corrected, x1)
            (rows, gradient), derivative_seconds = timed(
                label + "_derivative", engine.derivative, campaign.rows, corrected, x1
            )
            if label == "candidate" and engine.options.saved_krylov_actions:
                final_sizes = (_tangent_apply._cache_size(), _tangent_transpose._cache_size())
                event(event="compiled_action_reuse", initial=list(cache_sizes), final=list(final_sizes))
                assert final_sizes == cache_sizes, "per-root action compilation regression"
            state_error = float(np.linalg.norm(corrected - y1) / max(np.linalg.norm(y1), 1e-20))
            np.testing.assert_allclose(
                rows[:3],
                [accepted[args.second_step]["objective"], accepted[args.second_step]["min_abs_iota"], accepted[args.second_step]["major_radius_m"]],
                rtol=1e-7,
                atol=1e-9,
            )
            assert state_error < 1e-7
            results[label] = dict(
                dense_seconds=dense_seconds,
                prediction_seconds=prediction_seconds,
                correction_seconds=correction_seconds,
                derivative_seconds=derivative_seconds,
                total_warm_step_seconds=prediction_seconds + correction_seconds + derivative_seconds,
                state_relative_error=state_error,
                certificate=certificate,
                rows=rows.tolist(),
                gradient=gradient.tolist(),
            )
            np.savez_compressed(out / f"{label}.npz", y=corrected, x=x1, rows=rows, gradient=gradient)
            case.write(out / "comparison.json", results)
            if label == "baseline":
                # Do not retain the baseline tape on the GPU while measuring
                # candidate peak memory. Keep compiled executables warm.
                engine._tangent = engine._tangent_point = None

        model.linear = candidate
        gradients = [np.asarray(results[k]["gradient"]) for k in ("baseline", "candidate")]
        errors = np.linalg.norm(gradients[1] - gradients[0], axis=1) / np.maximum(
            np.linalg.norm(gradients[0], axis=1), 1e-10
        )
        assert np.max(errors) < 1e-5, errors
        event(event="gradient_parity", per_row_relative=errors.tolist())
        device_cache_size = _krylov._cache_size() if hasattr(candidate, "_warm") else None
        direction = np.random.default_rng(17).normal(size=len(x1))
        direction /= np.linalg.norm(direction)
        ad = gradients[1] @ direction
        fd_checks = []
        for h in (1e-3, 5e-4):
            values = []
            for sign in (-1, 1):
                point = x1 + sign * h * direction
                # Independent endpoints start at the accepted root, without tangent prediction.
                root, _ = timed("independent_fd_root", candidate.solve, y1.copy(), point)
                model.certify(root, point)
                values.append(np.asarray(campaign.rows(root, point)))
            fd = (values[1] - values[0]) / (2 * h)
            relative = np.abs(fd - ad) / np.maximum(np.maximum(np.abs(fd), np.abs(ad)), 1e-7)
            check = dict(
                h=h,
                ad=ad.tolist(),
                fd=fd.tolist(),
                relative_errors=relative.tolist(),
                passed=bool(np.max(relative) < 1e-3),
            )
            fd_checks.append(check)
            event(event="independent_fd_check", **check)
            assert check["passed"], check
        if device_cache_size is not None:
            event(event="device_executable_reuse", before=device_cache_size, after=_krylov._cache_size())
            assert _krylov._cache_size() == device_cache_size
        comparison = dict(
            results=results,
            gradient_parity=errors.tolist(),
            fd_checks=fd_checks,
            warm_step_speedup=results["baseline"]["total_warm_step_seconds"]
            / results["candidate"]["total_warm_step_seconds"],
            dense_speedup=results["baseline"]["dense_seconds"] / results["candidate"]["dense_seconds"],
            source_production_modified=False,
        )
        case.write(out / "comparison-final.json", comparison)
        event(event="device_memory", statistics=jax.devices()[0].memory_stats())
        event(
            event="comparison_passed",
            warm_step_speedup=comparison["warm_step_speedup"],
            dense_speedup=comparison["dense_speedup"],
        )
        if args.pilot_steps == 0:
            case.write(out / "result.json", dict(status="comparison_passed", arm=args.arm,
                       production_derivative_qualified=False, comparison=comparison))
            return
        # A bounded new optimization branch, never a continuation of the live directory.
        campaign.x, campaign.y, campaign.steps = x1.copy(), y1.copy(), args.second_step
        campaign.trials, campaign.cache, campaign.rejected = 0, {}, {}
        campaign.started = time.monotonic()
        if hasattr(candidate, "initialize_accepted"):
            candidate.rebuild(y1, x1)
        campaign.record()
        pilot = campaign.optimize()
        event(event="pilot_complete", **pilot)
        event(event="device_memory", statistics=jax.devices()[0].memory_stats())
        case.write(
            out / "result.json",
            dict(
                comparison_passed=True,
                pilot=pilot,
                endpoint_NS201_verified=False,
                physical_feasibility_claimed=False,
                production_modified=False,
            ),
        )
    except Exception as error:
        event(event="failure", error=repr(error))
        (out / "failure.json").write_text(json.dumps(dict(error=repr(error)), indent=2))
        raise


if __name__ == "__main__":
    main()
