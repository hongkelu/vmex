"""Shared file handling, preparation and scalar optimization for both arms."""

from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import bootstrap_settings as S
from main_reference import REFERENCE, VACUUM_SCRIPTS, coil_modules


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    result = dict(
        schema="finite-beta-redl-benchmark/v1",
        physics={k: v for k, v in vars(S).items() if k.isupper() and k not in {"P", "IOTA_FLOOR"}},
        optimization_targets=dict(iota_floor=S.IOTA_FLOOR),
        shared_main={k: v for k, v in vars(S.P).items() if k.isupper()},
        numerical_sources={
            str(p.relative_to(HERE.parent)): sha(p)
            for p in [
                *sorted(HERE.glob("*.py")),
                REFERENCE / "parameters.py",
                REFERENCE / "_coil_constraints.py",
                REFERENCE / "_coil_resolution.py",
                *(REFERENCE / filename for filename in VACUUM_SCRIPTS.values()),
            ]
        },
        core_sources={p.name: sha(p) for p in sorted((HERE.parents[1] / "vmex/core").glob("*.py"))},
    )
    return json.loads(json.dumps(result))


def arguments(arm=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    if arm:
        parser.add_argument("--prepared", type=Path)
        parser.add_argument("--accepted-steps", type=int, default=S.P.ACCEPTED_STEPS)
        parser.add_argument("--max-trials", type=int, default=None,
                            help="total trial budget; default is at least 10 per requested accepted step")
        parser.add_argument(
            "--allow-prepared-code-update",
            action="store_true",
            help="reuse authenticated inputs across audited updates, including the pinned Chebyshev profile migration",
        )
    else:
        parser.add_argument("--input", type=Path, default=REFERENCE / "input.rotating_ellipse")
        parser.add_argument("--coils", type=Path, default=REFERENCE / "coils.initial.scalar.json")
    args = parser.parse_args(argv)
    if not args.dry_run and (args.output is None or (arm and args.prepared is None)):
        parser.error("--output and, for optimization, --prepared are required")
    if arm and args.accepted_steps < 1:
        parser.error("--accepted-steps must be positive")
    if arm and args.max_trials is not None and args.max_trials < args.accepted_steps:
        parser.error("--max-trials must be at least --accepted-steps")
    return args


def setup(args):
    out = args.output.resolve()
    if out.exists():
        raise ValueError("use a new output directory")
    out.mkdir(parents=True)
    for name in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        p = out / "cache" / name
        p.mkdir(parents=True)
        os.environ[name] = str(p)
    os.environ.update(
        JAX_ENABLE_X64="1",
        JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
        VMEX_COMPILATION_CACHE="disabled",
        JAX_ENABLE_COMPILATION_CACHE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MPLBACKEND="Agg",
    )
    import jax

    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested backend and float64 required")
    from importlib.metadata import version

    write(
        out / "runtime.json",
        dict(
            python=sys.version,
            device=str(jax.devices()[0]),
            hardware=jax.devices()[0].device_kind,
            versions={
                k: version(k)
                for k in ("jax", "jaxlib", "numpy", "scipy", "essos", "solvax", "booz-xform-jax", "virtual-casing-jax")
            },
        ),
    )
    write(out / "contract.json", contract())
    return out


def interface(inp, state, runtime, grid=S.INTERFACE_GRID):
    import vmex as vj
    from vmex.core.virtual_casing import plan_vc_precision

    data = vj.surface_field_data_from_state(inp, state, runtime=runtime, nphi=grid[0], ntheta=grid[1])
    precision = plan_vc_precision(data, digits=S.VC_DIGITS)
    return precision


def boundary_residuals(vc, external):
    """Live normal field and total-pressure jump on the same interface.

    Normalize the pressure jump by the area-mean target B^2 + 2 mu0 p,
    never by the coil field being fitted. All state/geometry dependence,
    including this normalization, remains in the implicit derivative.
    """
    import jax.numpy as jnp
    from vmex.core.profiles import MU0

    total = vc.total_B_out(external)
    bn = jnp.sum(total * vc.normal, axis=0) / jnp.linalg.norm(total, axis=0)
    target_b2 = vc.Bin_mag2 + 2 * MU0 * vc.p_edge
    reference_b2 = jnp.maximum(jnp.sum(vc.weights * target_b2), jnp.finfo(total.dtype).tiny)
    jump = (jnp.sum(total**2, axis=0) - target_b2) / reference_b2
    return bn, jump, vc.weights


def boundary_values(inp, state, rt, coils, precision, grid=S.INTERFACE_GRID):
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from essos.fields import BiotSavart

    data = vj.surface_field_data_from_state(inp, state, runtime=rt, nphi=grid[0], ntheta=grid[1])
    vc = vj.PlasmaVacuumInterface.from_surface_data(data, precision=precision, digits=S.VC_DIGITS)
    field = BiotSavart(coils)

    def external(xyz):
        return jax.vmap(field.B)(xyz.reshape(-1, 3)).reshape(xyz.shape)

    return boundary_residuals(vc, external)


def normal_values(inp, state, rt, coils, precision, grid=S.INTERFACE_GRID):
    """Normal-only adapter retained for the shared initial coil fit."""
    bn, _, weights = boundary_values(inp, state, rt, coils, precision, grid)
    return bn, weights


def normal_cost(bn, weights):
    import jax
    import jax.numpy as jnp

    peak = jax.scipy.special.logsumexp(2000 * jnp.sqrt(bn**2 + 1e-12)) / 2000
    return (
        0.5 * S.NORMAL_FIELD_WEIGHT * jnp.sum(weights * bn**2)
        + 0.5 * S.NORMAL_FIELD_EXCESS_WEIGHT * jnp.maximum(peak - S.NORMAL_FIELD_MARGIN, 0) ** 2
    )


def pressure_balance_cost(jump, weights):
    import jax
    import jax.numpy as jnp

    peak = jax.scipy.special.logsumexp(2000 * jnp.sqrt(jump**2 + 1e-12)) / 2000
    return (
        0.5 * S.PRESSURE_BALANCE_WEIGHT * jnp.sum(weights * jump**2)
        + 0.5 * S.PRESSURE_BALANCE_EXCESS_WEIGHT
        * jnp.maximum(peak - S.PRESSURE_BALANCE_MARGIN, 0) ** 2
    )


def boundary_cost(bn, jump, weights):
    return normal_cost(bn, weights) + pressure_balance_cost(jump, weights)


def boundary_diagnostics(bn, jump, weights):
    import jax.numpy as jnp

    return dict(
        normal_field_rms=float(jnp.sqrt(jnp.sum(weights * bn**2))),
        normal_field_max=float(jnp.max(jnp.abs(bn))),
        pressure_balance_rms=float(jnp.sqrt(jnp.sum(weights * jump**2))),
        pressure_balance_max=float(jnp.max(jnp.abs(jump))),
        pressure_balance_mean=float(jnp.sum(weights * jump)),
    )


def boundary_match_passed(diagnostics):
    """Dense-grid interface checks, separate from optimizer termination."""
    import math

    return (
        all(math.isfinite(value) for value in diagnostics.values())
        and diagnostics["normal_field_max"] <= S.NORMAL_FIELD_LIMIT
        and diagnostics["pressure_balance_max"] <= S.PRESSURE_BALANCE_LIMIT
    )


def prepare(args):
    if args.dry_run:
        print(
            json.dumps(dict(action="prepare common beta-axis reference and constrained coils", **contract()), indent=2)
        )
        return 0
    out = setup(args)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.solver import SpectralState
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize, NonlinearConstraint
    from physics import pressure_input, kinetic_profiles, EquilibriumCurrentRoot, bootstrap_seed

    limits, resolution = coil_modules()
    inp = vj.VmecInput.from_file(args.input)
    inp = inp.change_resolution(mpol=S.P.RESOLUTION[0], ntor=S.P.RESOLUTION[1], ntheta=S.P.GRID[0], nzeta=S.P.GRID[1])
    inp = replace(inp, ns_array=np.array([S.P.RESOLUTION[2]]), ftol_array=np.array([S.P.EQUILIBRIUM_FTOL]), gamma=0.0)
    temperature = S.TEMPERATURE_INITIAL_EV
    history = []
    seed_state = None
    try:
        for k in range(S.CALIBRATION_STEPS):
            print(json.dumps(dict(phase="beta_calibration", iteration=k, temperature_axis_eV=temperature)), flush=True)
            trial = pressure_input(inp, temperature)
            profiles = kinetic_profiles(temperature)
            if abs(inp.curtor) > 1e-8:
                trial = replace(trial, ac=inp.ac, curtor=inp.curtor)
            picard = bootstrap_seed(trial, profiles, initial_state=seed_state, device=args.device)
            inp, seed_state = picard.input, picard.equilibrium.state
            if not picard.converged:
                raise RuntimeError("initial Redl Picard closure did not converge")
            beta = float(picard.equilibrium.wout.betaxis)
            history.append(
                dict(
                    iteration=k,
                    beta_axis=beta,
                    temperature_eV=temperature,
                    plasma_current_A=float(picard.input.curtor),
                    picard=[
                        {key: (float(value) if np.isfinite(value) else None) for key, value in item.items()}
                        for item in picard.history
                    ],
                )
            )
            write(out / "calibration.json", history)
            if not np.isfinite(beta) or beta <= 0:
                raise RuntimeError("nonpositive initial axis beta")
            if abs(beta / S.BETA_AXIS_INITIAL - 1) <= S.BETA_CALIBRATION_RTOL:
                break
            temperature *= np.clip(S.BETA_AXIS_INITIAL / beta, 0.5, 2.0)
        else:
            raise RuntimeError("initial axis-beta calibration budget exhausted")
        inp, eq = picard.input, picard.equilibrium
        coils = resolution.resize_coils(Coils.from_json(str(args.coils)), S.P.COIL_ORDER, S.P.N_SEGMENTS)
        if coils.nfp != inp.nfp or coils.dofs_curves.shape[0] != S.P.N_COILS:
            raise ValueError("coil symmetry/count mismatch")
        # Polish the actual coupled fixed reference before freezing its profile.
        print(json.dumps(dict(phase="coupled_reference_polish", temperature_axis_eV=temperature)), flush=True)
        seed_dir = out / "calibrated_seed"
        seed_dir.mkdir()
        inp.to_indata(seed_dir / "input.fixed")
        np.savez_compressed(
            seed_dir / "state.npz",
            **{name: np.asarray(getattr(eq.state, name)) for name in SpectralState.__dataclass_fields__},
        )
        write(
            seed_dir / "seed.json",
            dict(
                temperature_axis_eV=temperature,
                beta_axis=float(eq.wout.betaxis),
                coupled_root_certified=False,
                contract=contract(),
                artifacts={name: sha(seed_dir / name) for name in ("input.fixed", "state.npz")},
            ),
        )

        def preparation_event(**data):
            with (out / "root_events.jsonl").open("a") as stream:
                stream.write(json.dumps(data, allow_nan=False) + "\n")
            print(json.dumps(data, allow_nan=False), flush=True)

        model = EquilibriumCurrentRoot(inp, coils, eq.state, profiles, "fixed", event=preparation_event)
        y = model.linear.solve(model.initial_y, model.x0)
        certificate = model.certify(y, model.x0)
        state, rt, _, _ = model.objects(jnp.asarray(y), jnp.asarray(model.x0))
        inp = model.input_at(y, model.x0)
        wout = vj.wout_from_state(
            inp=inp, state=state, fsqr=certificate["fsqr"], fsqz=certificate["fsqz"], fsql=certificate["fsql"]
        )
        if abs(float(wout.betaxis) / S.BETA_AXIS_INITIAL - 1) > S.BETA_CALIBRATION_RTOL:
            raise RuntimeError("coupled-root polishing moved beta outside the initial calibration tolerance")
        inp.to_indata(out / "input.fixed")
        np.savez_compressed(
            out / "state.npz", **{name: np.asarray(getattr(state, name)) for name in SpectralState.__dataclass_fields__}
        )
        write(
            out / "profiles.json",
            dict(
                temperature_axis_eV=temperature,
                density_axis_m3=S.DENSITY_AXIS_M3,
                Zeff=S.ZEFF,
                Ti_over_Te=S.TI_OVER_TE,
                beta_axis_initial=float(wout.betaxis),
                beta_volume_initial=float(wout.betatotal),
                pressure_axis_Pa=float(inp.pres_scale),
                phiedge_Wb=float(inp.phiedge),
                frozen_during_optimization=True,
            ),
        )
        print(json.dumps(dict(phase="shared_constrained_coil_fit", beta_axis=float(wout.betaxis))), flush=True)
        chart = opt.CoilParameters.from_coils(
            coils, current_dofs=(), scales=np.full(coils.dofs_curves.size, S.P.COIL_STEP)
        )
        scales = jnp.asarray(chart.scales)
        precision = interface(inp, state, rt)
        rbc, zbs, _, _ = opt.boundary_from_state(state, rt)
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=limits.SURFACE_GRID[0], ntheta=limits.SURFACE_GRID[1]
        )
        value = jax.jit(
            jax.value_and_grad(
                lambda u: normal_cost(*normal_values(inp, state, rt, chart.coils_from_x(u * scales), precision))
            )
        )

        def inequalities(u):
            c = chart.coils_from_x(u * scales)
            return jnp.r_[
                limits.coil_inequalities(c),
                (limits.surface_distance(c, surface) - S.P.COIL_SURFACE_DISTANCE_LIMIT - S.P.DISTANCE_MARGIN)
                / S.P.COIL_SURFACE_DISTANCE_LIMIT,
            ]

        fun, jac = jax.jit(inequalities), jax.jit(jax.jacrev(inequalities))

        def fit_value(u):
            cost, gradient = value(u)
            return float(cost), np.asarray(gradient)

        fit = minimize(
            fit_value,
            np.zeros(chart.size),
            jac=True,
            method="SLSQP",
            constraints=[
                NonlinearConstraint(lambda u: np.asarray(fun(u)), 0, np.inf, jac=lambda u: np.asarray(jac(u)))
            ],
            options=dict(maxiter=S.COIL_FIT_STEPS, ftol=S.P.OPTIMIZER_FTOL),
        )
        fitted = chart.coils_from_x(jnp.asarray(fit.x) * scales)
        fitted.to_json(str(out / "coils.json"))
        write(
            out / "coil_fit.json",
            dict(
                success=bool(fit.success),
                message=str(fit.message),
                iterations=int(fit.nit),
                objective=float(fit.fun),
                minimum_scaled_slack=float(jnp.min(fun(fit.x))),
            ),
        )
        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)) or float(jnp.min(fun(fit.x))) < -1e-8:
            raise RuntimeError("shared coil fit did not satisfy the sampled hard constraints")
        write(
            out / "prepared.json",
            dict(
                schema="finite-beta-redl-prepared/v1",
                contract=contract(),
                certificate=certificate,
                source_input_sha256=sha(args.input),
                source_coils_sha256=sha(args.coils),
                artifacts={
                    name: sha(out / name) for name in ("input.fixed", "coils.json", "state.npz", "profiles.json")
                },
                independent_derivative_qualified=False,
            ),
        )
        return 0
    except Exception as error:
        write(out / "failure.json", dict(phase="prepare", error=f"{type(error).__name__}: {error}"))
        raise


def validate_prepared_contract(saved, *, allow_code_update=False):
    current = contract()
    if saved == current:
        return []
    # Preserve physical settings and all kernels except the exact audited
    # additive profile update. Never inherit an old numerical qualification.
    numerical_settings = {
        # Objective-only migration reuses the authenticated prepared physics;
        # it does not inherit the old objective history or qualification.
        "BOUNDARY_OBJECTIVE_VERSION",
        "PRESSURE_BALANCE_WEIGHT",
        "PRESSURE_BALANCE_EXCESS_WEIGHT",
        "PRESSURE_BALANCE_MARGIN",
        "PRESSURE_BALANCE_LIMIT",
        "BOUNDARY_SPECTRAL_ALPHA",
        "BOUNDARY_MAX_DISPLACEMENT_M",  # legacy fixed-only cap, now removed
        "NEWTON_EXTRA_STEPS",
        "TRIAL_DENSE_REBUILDS",
        "NEWTON_STAGNATION_STEPS",
        "LINEAR_REFRESH_HORIZON",
        "LINEAR_REUSE_BUDGET_FRACTION",
        "CONTINUATION_INITIAL_FREE",  # legacy name for the shared setting
        "CONTINUATION_INITIAL",
        "CONTINUATION_MIN_STEP",
        "CONTINUATION_MAX_ATTEMPTS",
        "FREE_DENSE_BATCH_SIZE",
        "FREE_SAVED_LINEARIZATION",
        "FREE_SAVED_KRYLOV_ACTIONS",
        "FREE_NEWTON_RTOL",
        "FREE_NEWTON_GATE",
        "FREE_DEVICE_SOLVES",
        "FREE_DENSE_ASSEMBLY",
        "FREE_ADJOINT_RHS_BATCH_SIZE",
        "DENSE_BATCH_SIZE",
        "SAVED_LINEARIZATION",
        "SAVED_KRYLOV_ACTIONS",
        "NEWTON_RTOL",
        "NEWTON_GATE",
        "DEVICE_SOLVES",
        "DENSE_ASSEMBLY",
        "ADJOINT_RHS_BATCH_SIZE",
    }
    code_files = {
        "finite-beta-bootstrap-benchmark/" + name
        for name in (
            "case.py",
            "physics.py",
            "linear_root.py",
            "device_root.py",
            "bootstrap_settings.py",
            "optimizer_driver.py",
            "main_reference.py",
            "fixed_boundary_single_stage.py",
            "free_boundary_single_stage.py",
            "production.py",
            "verify_gradients.py",
            "fit_initial_coils.py",
        )
    }
    if not allow_code_update or saved.get("schema") != current["schema"]:
        raise ValueError("prepared bundle differs from this physics/code contract")
    previous_core = saved.get("core_sources", {})
    latest_core = current["core_sources"]
    core_changed = sorted(
        k for k in previous_core.keys() | latest_core.keys() if previous_core.get(k) != latest_core.get(k)
    )
    profile_migration = (
        core_changed == ["profiles.py"]
        and previous_core["profiles.py"] == "7cbc5b1e62d73163e51e70ee3f43a402db1803608d51e2b5cd30bb93558688a5"
        and latest_core["profiles.py"] == "f157243b48635debd4d46698e5d3c2b29cb837d1728f5b9811e2dbb35fcd6612"
    )
    if (core_changed and not profile_migration) or saved.get("shared_main") != current["shared_main"]:
        raise ValueError("prepared core or shared constraints changed")

    def physics(c):
        return {k: v for k, v in c["physics"].items() if k not in numerical_settings}

    if physics(saved) != physics(current):
        raise ValueError("prepared physics or existing numerical settings changed")
    previous = saved["numerical_sources"]
    latest = current["numerical_sources"]
    # Older contracts did not hash the reference entry points. Allow adding
    # them, never silently accept a change to an already authenticated source.
    code_files.update(
        "coil-constraints-benchmarks/" + name
        for name in VACUUM_SCRIPTS.values()
        if "coil-constraints-benchmarks/" + name not in previous
    )
    changed = sorted(k for k in previous.keys() | latest.keys() if previous.get(k) != latest.get(k))
    if not set(changed) <= code_files:
        raise ValueError("prepared source changes exceed the audited trial-recovery update")
    return changed + ["vmex/core/" + name for name in core_changed]


def load_prepared(path, *, allow_code_update=False):
    import numpy as np
    import jax.numpy as jnp
    import vmex as vj
    from vmex.core.solver import SpectralState
    from essos.coils import Coils
    from physics import kinetic_profiles

    root = path.resolve()
    report = json.loads((root / "prepared.json").read_text())
    if report["schema"] != "finite-beta-redl-prepared/v1":
        raise ValueError("prepared bundle differs from this physics/code contract")
    validate_prepared_contract(report["contract"], allow_code_update=allow_code_update)
    for name, digest in report["artifacts"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or sha(p) != digest:
            raise ValueError("prepared artifact authentication failed")
    from importlib.metadata import version

    saved_runtime = json.loads((root / "runtime.json").read_text())
    if any(version(name) != value for name, value in saved_runtime["versions"].items()):
        raise ValueError("prepared dependency versions differ from the active runtime")
    profile = json.loads((root / "profiles.json").read_text())
    with np.load(root / "state.npz", allow_pickle=False) as data:
        state = SpectralState(**{name: jnp.asarray(data[name]) for name in SpectralState.__dataclass_fields__})
    return (
        vj.VmecInput.from_file(root / "input.fixed"),
        Coils.from_json(str(root / "coils.json")),
        state,
        kinetic_profiles(profile["temperature_axis_eV"]),
    )


class CaseRun:
    def __init__(self, args, arm, out):
        import jax.numpy as jnp
        from physics import EquilibriumCurrentRoot

        self.args, self.arm, self.out = args, arm, out
        self.started = time.monotonic()
        self.steps = 0
        self.trials = 0
        self.trial_budget = (
            getattr(args, "max_trials", None) or max(S.MAX_TRIALS, 10 * args.accepted_steps)
        )
        self.cache = {}
        self.rejected = {}
        allow_update = getattr(args, "allow_prepared_code_update", False)
        inp, coils, seed, profiles = load_prepared(args.prepared, allow_code_update=allow_update)
        prepared_report = json.loads((args.prepared / "prepared.json").read_text())
        changed = validate_prepared_contract(prepared_report["contract"], allow_code_update=allow_update)
        write(
            out / "prepared_lineage.json",
            dict(
                prepared=str(args.prepared.resolve()),
                prepared_manifest_sha256=sha(args.prepared / "prepared.json"),
                changed_numerical_sources=changed,
                physical_settings_unchanged=True,
                core_unchanged=prepared_report["contract"]["core_sources"] == contract()["core_sources"],
                current_representation="chebyshev_ip",
                previous_iota_floor=prepared_report["contract"]
                .get("optimization_targets", {})
                .get("iota_floor", prepared_report["contract"]["shared_main"]["IOTA_FLOOR"]),
                requested_iota_floor=S.IOTA_FLOOR,
                previous_boundary_objective_version=prepared_report["contract"]["physics"].get(
                    "BOUNDARY_OBJECTIVE_VERSION", "normal-only"
                ),
                boundary_objective_version=S.BOUNDARY_OBJECTIVE_VERSION,
                historical_derivative_qualification_inherited=False,
            ),
        )
        self.model = EquilibriumCurrentRoot(inp, coils, seed, profiles, arm, event=self.event)
        self.x = self.model.x0.copy()
        self.y = self.model.linear.solve(self.model.initial_y, self.x)
        self.model.certify(self.y, self.x)
        state, rt, _, _ = self.model.objects(jnp.asarray(self.y), jnp.asarray(self.x))
        self.precision = interface(inp, state, rt)
        self.configure_objectives(inp)
        self.record()

    def configure_objectives(self, inp):
        """Shared objective setup for fresh runs and authenticated benchmark replay."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        from vmex import optimize as opt
        from essos.surfaces import surfacerzfourier_from_boundary

        self.limits, _ = coil_modules()
        qs = opt.QuasisymmetryRatioResidual(np.asarray(S.QA_SURFACES), 1, 0)
        self.qs = qs

        def interface_rows(y, x):
            state, rt, _, coils = self.model.objects(y, x)
            bn, jump, weights = boundary_values(inp, state, rt, coils, self.precision)
            return jnp.array([normal_cost(bn, weights), pressure_balance_cost(jump, weights)])

        def rows(y, x):
            state, rt, _, coils = self.model.objects(y, x)
            residual = qs.residuals_state(state, rt)
            aspect = opt.aspect_ratio(state, rt)
            iota = opt.min_abs_iota(state, rt)
            loss = 0.5 * jnp.vdot(residual, residual) + 0.5 * S.P.ASPECT_WEIGHT * (aspect - S.P.ASPECT_TARGET) ** 2
            loss += 0.5 * S.P.IOTA_WEIGHT * jnp.maximum(S.IOTA_FLOOR - iota, 0.0) ** 2
            if self.arm == "fixed":
                bn, jump, weights = boundary_values(inp, state, rt, coils, self.precision)
                loss += boundary_cost(bn, jump, weights)
            rbc, zbs, _, _ = opt.boundary_from_state(state, rt)
            surface = surfacerzfourier_from_boundary(
                rbc, zbs, inp.nfp, nphi=self.limits.SURFACE_GRID[0], ntheta=self.limits.SURFACE_GRID[1]
            )
            return jnp.array([loss, iota, opt.major_radius(state, rt), self.limits.surface_distance(coils, surface)])

        self.rows = jax.jit(rows)
        # Used only by the separate derivative-audit command, not production.
        self.interface_rows = jax.jit(interface_rows)
        self.coil_rows = jax.jit(
            lambda x: self.limits.coil_inequalities(self.model.chart.coils_from_x(self.model.design(x)[1]))
        )
        self.coil_jac = jax.jit(jax.jacrev(self.coil_rows))

    def event(self, **data):
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(elapsed_seconds=time.monotonic() - self.started, **data), allow_nan=False) + "\n"
            )

    def evaluate(self, x):
        import numpy as np
        from linear_root import RootFailure

        # SciPy can ask for both objective and constraints at the same failed
        # probe. Do not repeat expensive continuation from the same anchor.
        key = np.asarray(x, dtype=float).tobytes()
        rejected = getattr(self, "rejected", {})
        if key in rejected:
            raise RootFailure(rejected[key])
        try:
            return self._evaluate(x)
        except RootFailure as error:
            if hasattr(self.model.linear, "discard_trial"):
                self.model.linear.discard_trial()
            if len(rejected) >= 4:
                rejected.pop(next(iter(rejected)))
            rejected[key] = str(error)
            self.rejected = rejected
            raise

    def _evaluate(self, x):
        import numpy as np
        from linear_root import RootFailure

        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        if np.array_equal(x, self.x):
            if hasattr(self.model.linear, "activate_trial"):
                self.model.linear.activate_trial(x)
            y = self.y.copy()
        else:
            self.trials += 1
            if self.trials > getattr(self, "trial_budget", S.MAX_TRIALS):
                raise RootFailure("trial budget reached")
            if hasattr(self.model.linear, "begin_trial"):
                self.model.linear.begin_trial(x)
            y = self.continue_root(x)
        certificate = self.model.certify(y, x)
        rows = np.asarray(self.rows(y, x))
        if not np.all(np.isfinite(rows)):
            raise RootFailure("nonfinite scalar objective or constraints")
        record = dict(y=y, x=x.copy(), rows=rows, jac=None, certificate=certificate)
        if len(self.cache) >= 2:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = record
        if hasattr(self.model.linear, "retain_trial"):
            self.model.linear.retain_trial(x)
        return record

    def continue_root(self, x):
        """Advance certified substeps from the accepted root without promotion.

        A failed segment is retried from its last certified predecessor. Only
        the optimizer can promote the complete endpoint to an accepted state.
        Reject proposals needing smaller than quarter segments, so the outer
        optimizer can reduce the design move instead of a long internal walk.
        This policy is identical for fixed and free boundary.
        """
        from linear_root import RootFailure

        y, previous = self.y.copy(), self.x.copy()
        fraction = 0.0
        step = getattr(self, "continuation_step", S.CONTINUATION_INITIAL)
        last_error = "continuation attempt budget exhausted"
        for attempt in range(S.CONTINUATION_MAX_ATTEMPTS):
            width = min(step, 1.0 - fraction)
            target = min(1.0, fraction + width)
            point = self.x + target * (x - self.x)
            try:
                # Prediction is evaluated at the certified predecessor. Retain
                # any strict recovery there when the speculative corrector fails.
                seed = self.model.seed_at(y, previous, point)
                segment = getattr(self.model.linear, "continuation_segment", nullcontext)
                with segment():
                    candidate = self.model.correct_trial(seed, point)
                    certificate = self.model.certify(candidate, point)
            except RootFailure as error:
                last_error = str(error)
                self.event(
                    event="continuation_retry",
                    attempt=attempt + 1,
                    from_fraction=fraction,
                    target_fraction=target,
                    error=last_error,
                )
                step = width / 2
                if step < S.CONTINUATION_MIN_STEP:
                    break
                continue
            y, previous, fraction = candidate.copy(), point.copy(), target
            info = self.model.linear.last_solve
            self.event(
                event="continuation_certified",
                fraction=fraction,
                width=width,
                root_residual=certificate["root_residual"],
                newton_iterations=info["iterations"],
            )
            if fraction == 1.0:
                # A numerical step-size hint only; never a new accepted root.
                self.continuation_step = min(1.0, 2 * width) if info["iterations"] <= 4 else width
                return y
            if info["iterations"] <= 4 and info["minimum_alpha"] >= 0.5:
                step = min(1.0, 2 * width)
            elif info["iterations"] > S.NEWTON_STEPS:
                step = max(S.CONTINUATION_MIN_STEP, width / 2)
            else:
                step = width
        raise RootFailure("bounded continuation from accepted state failed: " + last_error)

    def jacobian(self, x):
        record = self.evaluate(x)
        if record["jac"] is None:
            if hasattr(self.model.linear, "activate_trial"):
                self.model.linear.activate_trial(x)
            _, record["jac"] = self.model.linear.derivative(self.rows, record["y"], record["x"])
            if hasattr(self.model.linear, "retain_trial"):
                self.model.linear.retain_trial(x)
        return record["jac"]

    def accept(self, x):
        import numpy as np

        if np.array_equal(x, self.x):
            return
        record = self.evaluate(x)
        if hasattr(self.model.linear, "accept_trial"):
            record["jac"] = self.model.linear.accept_trial(record["y"], record["x"], self.rows, self.jacobian(x))
        self.x, self.y = record["x"].copy(), record["y"].copy()
        # Cached roots from a previous accepted seed cannot bypass continuation.
        self.cache = {self.x.tobytes(): record}
        self.rejected = {}
        self.steps += 1
        self.record()

    def record(self):
        import numpy as np
        import jax.numpy as jnp
        import vmex as vj
        from vmex import optimize as opt

        state, rt, _, coils = self.model.objects(jnp.asarray(self.y), jnp.asarray(self.x), vacuum=True)
        cert = self.model.certify(self.y, self.x)
        inp = self.model.input_at(self.y, self.x)
        w = vj.wout_from_state(inp=inp, state=state, fsqr=cert["fsqr"], fsqz=cert["fsqz"], fsql=cert["fsql"])
        values = np.asarray(self.rows(self.y, self.x))
        bn, jump, weights = boundary_values(inp, state, rt, coils, self.precision)
        row = dict(
            step=self.steps,
            elapsed_seconds=time.monotonic() - self.started,
            objective=float(values[0]),
            loss_qa=0.5 * float(self.qs.total_state(state, rt)),
            loss_aspect=0.5 * S.P.ASPECT_WEIGHT * (float(opt.aspect_ratio(state, rt)) - S.P.ASPECT_TARGET) ** 2,
            loss_iota=0.5 * S.P.IOTA_WEIGHT * max(S.IOTA_FLOOR - float(values[1]), 0.0) ** 2,
            loss_normal=float(normal_cost(bn, weights)) if self.arm == "fixed" else 0.0,
            loss_pressure_balance=float(pressure_balance_cost(jump, weights)) if self.arm == "fixed" else 0.0,
            boundary_objective_version=S.BOUNDARY_OBJECTIVE_VERSION,
            iota_floor=S.IOTA_FLOOR,
            qa=float(self.qs.total_state(state, rt)),
            min_abs_iota=float(values[1]),
            major_radius_m=float(values[2]),
            clearance_m=float(values[3]),
            coil_scaled_slack=float(jnp.min(self.coil_rows(self.x))),
            beta_axis_percent=100 * float(w.betaxis),
            beta_volume_percent=100 * float(w.betatotal),
            **boundary_diagnostics(bn, jump, weights),
            pressure_axis_Pa=float(inp.pres_scale),
            phiedge_Wb=float(inp.phiedge),
            **cert,
        )
        checkpoint = self.out / f"accepted_{self.steps:04d}.npz"
        np.savez_compressed(
            checkpoint,
            x=self.x,
            y=self.y,
            **{name: np.asarray(getattr(state, name)) for name in state.__dataclass_fields__},
        )
        row["checkpoint_sha256"] = sha(checkpoint)
        with (self.out / "accepted_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def optimize(self):
        from optimizer_driver import optimize

        report = optimize(self)
        write(self.out / "optimization.json", report)
        return report

    def verify(self):
        """Independent NS201 equilibrium/current root and finer geometry/interface checks."""
        import numpy as np
        import jax.numpy as jnp
        import vmex as vj
        from vmex import optimize as opt
        from essos.surfaces import surfacerzfourier_from_boundary

        state, rt, _, coils = self.model.objects(jnp.asarray(self.y), jnp.asarray(self.x))
        inp = self.model.input_at(self.y, self.x)
        rbc, zbs, rbs, zbc = opt.boundary_from_state(state, rt)
        inp = replace(
            inp,
            rbc=np.asarray(rbc),
            zbs=np.asarray(zbs),
            rbs=np.asarray(rbs),
            zbc=np.asarray(zbc),
            lfreeb=False,
            ns_array=np.array([S.P.VERIFY_NS]),
            ftol_array=np.array([S.P.VERIFY_FTOL]),
        )
        from physics import verify_high_resolution
        from vmex.core.multigrid import interpolate_state

        seed = interpolate_state(state, ns_fine=S.P.VERIFY_NS, modes=rt.modes)
        inp, state, rt, cert = verify_high_resolution(
            inp, coils, self.model.closure.profiles, self.arm, initial_state=seed, event=self.event
        )
        rbc, zbs, _, _ = opt.boundary_from_state(state, rt)
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=self.limits.VERIFY_SURFACE_GRID[0], ntheta=self.limits.VERIFY_SURFACE_GRID[1]
        )
        geometry = self.limits.verify(coils, surface, self.out / "geometry_verification.json")
        precision = interface(inp, state, rt, S.VERIFY_INTERFACE_GRID)
        bn, jump, weights = boundary_values(inp, state, rt, coils, precision, S.VERIFY_INTERFACE_GRID)
        interface_report = boundary_diagnostics(bn, jump, weights)
        w = vj.wout_from_state(inp=inp, state=state, fsqr=cert["fsqr"], fsqz=cert["fsqz"], fsql=cert["fsql"])
        vj.write_wout(self.out / "wout_verified.nc", w)
        coils.to_json(str(self.out / "coils_final.json"))
        inp.to_indata(self.out / "input.final")
        iota = float(opt.min_abs_iota(state, rt))
        radius = float(opt.major_radius(state, rt))
        report = dict(
            resolution=[S.P.RESOLUTION[0], S.P.RESOLUTION[1], S.P.VERIFY_NS],
            **cert,
            qa=float(self.qs.total_state(state, rt)),
            beta_axis_percent=100 * float(w.betaxis),
            beta_volume_percent=100 * float(w.betatotal),
            min_abs_iota=iota,
            major_radius_m=radius,
            **interface_report,
            boundary_matching_passed=boundary_match_passed(interface_report),
            boundary_objective_version=S.BOUNDARY_OBJECTIVE_VERSION,
            geometry=geometry,
            physical_plasma_limits_met=iota >= S.IOTA_FLOOR and abs(radius - S.P.RADIUS_TARGET) <= S.P.RADIUS_TOLERANCE,
            independent_derivative_qualified=False,
        )
        report["all_reported_checks_pass"] = bool(
            report["physical_plasma_limits_met"]
            and geometry["all_reported_checks_pass"]
            and report["normal_field_max"] <= S.NORMAL_FIELD_LIMIT
            and (self.arm != "fixed" or report["boundary_matching_passed"])
        )
        write(self.out / "verification.json", report)
        return report


def run(arm, args):
    if args.dry_run:
        print(json.dumps(dict(arm=arm, **contract()), indent=2))
        return 0
    out = setup(args)
    try:
        campaign = CaseRun(args, arm, out)
        if getattr(args, "verify_gradients", False):
            from verify_gradients import verify

            verify(campaign)
            return 0
        optimization = campaign.optimize()
        verification = campaign.verify()
        return 0 if optimization["optimizer_success"] and verification["all_reported_checks_pass"] else 2
    except Exception as error:
        write(out / "failure.json", dict(error=f"{type(error).__name__}: {error}"))
        raise
