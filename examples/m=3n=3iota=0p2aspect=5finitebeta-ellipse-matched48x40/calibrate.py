"""Bounded fixed-boundary calibration before any finite-beta coil fit."""

from pathlib import Path
import os
import sys
import json
import time
import hashlib
import subprocess
import dataclasses  # noqa: F401

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]
OUT = ROOT / "fixed_calibration_scan"
GPU = "GPU-27f8a10c-1b18-1109-85da-247ee1927024"


def main():
    assert not OUT.exists()
    assert GPU not in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    )
    OUT.mkdir()
    for key, name in [
        ("TMPDIR", "tmp"),
        ("JAX_COMPILATION_CACHE_DIR", "jax"),
        ("MPLCONFIGDIR", "mpl"),
        ("XDG_CACHE_HOME", "xdg"),
        ("CUDA_CACHE_PATH", "cuda"),
    ]:
        p = OUT / "cache" / name
        p.mkdir(parents=True)
        os.environ[key] = str(p)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=GPU,
        JAX_PLATFORMS="cuda,cpu",
        JAX_ENABLE_X64="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    sys.path[:0] = [str(SOURCE), str(ROOT)]
    import jax
    import numpy as np
    from vmex.core.input import VmecInput
    from vmex.core.multigrid import solve_multigrid
    from vmex.core.wout import wout_from_state, write_wout, read_wout
    from initialization import verify_runtime
    from geometry import rotating_ellipse, geometry_metadata

    verify_runtime()
    contract = json.loads((ROOT / "case_contract.json").read_text())
    for n, h in json.loads((ROOT / "inputs/manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / "inputs" / n).read_bytes()).hexdigest() == h
    reference = VmecInput.from_file(ROOT / "inputs/reference_profiles.json")
    started = time.monotonic()
    history = []
    cache = {}
    failed = {}

    class FailedEquilibrium(RuntimeError):
        pass

    names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    profile_names = (
        "ncurr",
        "curtor",
        "pres_scale",
        "pmass_type",
        "am",
        "am_aux_s",
        "am_aux_f",
        "pcurr_type",
        "ac",
        "ac_aux_s",
        "ac_aux_f",
    )

    def event(phase, **data):
        line = json.dumps(dict(phase=phase, elapsed_s=time.monotonic() - started, **data), allow_nan=False)
        with (OUT / "progress.jsonl").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def evaluate(x, h, ns, ftol):
        t, flux = map(float, x)
        key = (t, flux, h, ns, ftol)
        if key in cache:
            return cache[key]
        if key in failed:
            raise FailedEquilibrium(failed[key])
        if (
            len(history) >= contract["max_equilibrium_evaluations"]
            or time.monotonic() - started > contract["max_fixed_wall_seconds"]
        ):
            raise RuntimeError("fixed target calibration budget exhausted")
        inp = rotating_ellipse(
            reference,
            major_radius=contract["major_radius_m"],
            aspect=5.0,
            t=t,
            handedness=h,
            phiedge=flux,
            ns=ns,
            ftol=ftol,
        )
        from vmex.core.solver import resolution_from_input

        for radial_ns in inp.ns_array:
            res = resolution_from_input(inp, ns=radial_ns)
            assert (res.ntheta, res.ntheta3, res.nzeta) == (48, 25, 40)
        for n in profile_names:
            assert np.array_equal(getattr(inp, n), getattr(reference, n)), n
        index = len(history)
        p = OUT / f"eval_{index:03d}"
        p.mkdir()
        inp.to_json(p / "input.json")
        history.append(dict(index=index, t=t, phiedge_Wb=flux, handedness=h, ns=ns, ftol=ftol))
        event("equilibrium_start", **history[-1])
        result = solve_multigrid(
            inp,
            device=jax.devices("gpu")[0],
            verbose=False,
            raise_on_max_iterations=False,
            jacobian_retries=0,
            coarse_grid_retry=False,
            polish_force_balance=False,
            use_fft=False,
        )
        forces = {n: float(getattr(result, n)) for n in ("fsqr", "fsqz", "fsql")}
        record = dict(**history[-1], converged=bool(result.converged), iterations=int(result.iterations), forces=forces)
        np.savez_compressed(
            p / "state.npz",
            eligible_for_stage2=np.asarray(False),
            **{n: np.asarray(getattr(result.state, n)) for n in names},
        )
        if not result.converged or any(not np.isfinite(v) or not 0 <= v <= ftol for v in forces.values()):
            (p / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
            event("equilibrium_failed", **record)
            failed[key] = f"fixed equilibrium failed at t={t}, h={h}, flux={flux}"
            raise FailedEquilibrium(failed[key])
        w = wout_from_state(inp=inp, state=result.state, niter=int(result.iterations), converged=True, **forces)
        record.update(
            mean_iota=float(np.mean(np.asarray(w.iotas)[1:])),
            b0_T=float(w.b0),
            aspect=float(w.aspect),
            major_radius_m=float(w.Rmajor_p),
            beta=float(w.betatotal),
        )
        assert abs(record["aspect"] - 5) < 1e-9 and abs(record["major_radius_m"] - contract["major_radius_m"]) < 1e-8
        assert all(np.isfinite(v) for k, v in record.items() if isinstance(v, float))
        write_wout(p / "wout.nc", w, overwrite=False)
        (p / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        history[-1] = record
        event("equilibrium_end", **record)
        cache[key] = (record, inp, result, p)
        return cache[key]

    def residual(x, h, ns, ftol):
        rec, *_ = evaluate(x, h, ns, ftol)
        return np.array([(rec["mean_iota"] - 0.2) / 0.2, (rec["b0_T"] - 5.1) / 5.1])

    status = "failed"
    try:
        h = -1
        x = np.array([0.25188920340834803, 91.09117631931035])
        event(
            "calibration_seed",
            handedness=h,
            t=float(x[0]),
            phiedge_Wb=float(x[1]),
            ntheta=48,
            nzeta=40,
            source="old calibrated geometry only; every equilibrium is solved afresh",
        )
        for ns, ftol in [(50, 1e-13)]:
            for iteration in range(10):
                rec, inp, result, p = evaluate(x, h, ns, ftol)
                f = residual(x, h, ns, ftol)
                if abs(rec["mean_iota"] - 0.2) <= 3e-6 and abs(rec["b0_T"] - 5.1) <= 3e-5:
                    break
                jac = np.empty((2, 2))
                for k, eps in enumerate([0.001, 0.08]):
                    dx = np.zeros(2)
                    dx[k] = eps
                    try:
                        f1 = residual(x + dx, h, ns, ftol)
                    except FailedEquilibrium:
                        dx[k] = -eps
                        f1 = residual(x + dx, h, ns, ftol)
                    jac[:, k] = (f1 - f) / dx[k]
                delta = np.linalg.solve(jac, -f)
                factor = min(1.0, 0.05 / max(abs(delta[0]), 1e-30), (0.05 * x[1]) / max(abs(delta[1]), 1e-30))
                delta *= factor
                accepted = False
                for trial in range(5):
                    trial_x = x + delta * (0.5**trial)
                    if not (0.01 <= trial_x[0] <= 0.9 and 50 <= trial_x[1] <= 130):
                        continue
                    try:
                        new_f = residual(trial_x, h, ns, ftol)
                    except FailedEquilibrium:
                        continue
                    if np.linalg.norm(new_f) < np.linalg.norm(f):
                        x = trial_x
                        accepted = True
                        break
                event("calibration_iteration", ns=ns, iteration=iteration, parameters=x.tolist(), accepted=accepted)
                if not accepted:
                    raise RuntimeError("no converged improving calibration step within bounded trial budget")
            rec, inp, result, p = evaluate(x, h, ns, ftol)
            event("calibration_stage_end", ns=ns, parameters=x.tolist(), mean_iota=rec["mean_iota"], b0_T=rec["b0_T"])
        passed = abs(rec["mean_iota"] - 0.2) <= 1e-5 and abs(rec["b0_T"] - 5.1) <= 1e-4 and rec["converged"]
        if not passed:
            raise RuntimeError("fixed target physical gates failed; stage two is not authorized to start")
        selected = ROOT / "fixed"
        selected.mkdir()
        inp.to_json(selected / "input.fixed.json")
        inp.to_indata(selected / "input.fixed")
        (selected / "wout.nc").write_bytes((p / "wout.nc").read_bytes())
        np.savez_compressed(
            selected / "state.npz",
            qualified_fixed_target=np.asarray(True),
            eligible_for_free_boundary_resume=np.asarray(False),
            **{n: np.asarray(getattr(result.state, n)) for n in names},
        )
        summary = dict(
            completed=True,
            qualified_fixed_target=True,
            mean_iota=rec["mean_iota"],
            b0_T=rec["b0_T"],
            aspect=rec["aspect"],
            major_radius_m=rec["major_radius_m"],
            beta=rec["beta"],
            forces=rec["forces"],
            ns=50,
            ntheta=48,
            nzeta=40,
            geometry=geometry_metadata(contract["major_radius_m"], 5.0, float(x[0]), h),
            phiedge_Wb=float(x[1]),
            profiles_preserved=True,
            source_evaluation=p.name,
            equilibrium_evaluations=len(history),
            seconds=time.monotonic() - started,
            wout_sha256=hashlib.sha256((selected / "wout.nc").read_bytes()).hexdigest(),
        )
        # The selected WOUT must retain the pure boundary and original profiles.
        loaded = read_wout(selected / "wout.nc")
        assert abs(float(loaded.aspect) - 5) < 1e-9
        (selected / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        event("fixed_target_qualified", **summary)
        status = "completed"
    finally:
        (OUT / "summary.json").write_text(
            json.dumps(
                dict(
                    status=status,
                    equilibrium_evaluations=len(history),
                    seconds=time.monotonic() - started,
                    history=history,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
