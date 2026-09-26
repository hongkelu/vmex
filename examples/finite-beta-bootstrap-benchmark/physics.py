"""One equilibrium/Redl residual for both boundary formulations.

Uses VMEX's force, NESTOR and Redl kernels. The fitted I'(s) closure follows
bootstrap.self_consistent_bootstrap's smooth current reconstruction, represented
in a conditioned Chebyshev chart. Its coordinates are root unknowns, not design
variables; current feedback is included in every coupled adjoint.
"""

from dataclasses import replace
import time
import jax
import jax.numpy as jnp
import numpy as np
from vmex import optimize as opt
from vmex.core import bootstrap as bs, implicit as im, freeboundary_implicit as fbi
from vmex.core import statephysics as sp
from vmex.core._freeboundary_dense import _active_space, _compress, _expand
from vmex.core.solver import evaluate_forces
from linear_root import LinearRoot, Options, RootFailure, SlowProgress
import bootstrap_settings as S


def kinetic_profiles(temperature):
    if S.ZEFF != 1 or S.TI_OVER_TE <= 0 or S.DENSITY_AXIS_M3 <= 0 or temperature <= 0:
        raise ValueError("This case assumes a pure singly charged ion species, Zeff=1, positive n and T")
    return bs.KineticProfiles(
        S.DENSITY_AXIS_M3 * np.asarray(S.DENSITY_SHAPE),
        temperature * np.asarray(S.TEMPERATURE_SHAPE),
        S.TI_OVER_TE * temperature * np.asarray(S.TEMPERATURE_SHAPE),
        S.ZEFF,
    )


def pressure_input(inp, temperature):
    """p=e(ne Te+ni Ti), ni=ne for this Zeff=1 case; pressure in Pa."""
    profiles = kinetic_profiles(temperature)
    pressure = bs.ELEMENTARY_CHARGE * np.polynomial.polynomial.polymul(
        profiles.ne_coeffs, profiles.Te_coeffs + profiles.Ti_coeffs
    )
    am = np.zeros(max(len(inp.am), len(pressure)))
    am[: len(pressure)] = pressure / pressure[0]
    ac = np.zeros(max(21, len(inp.ac)))
    ac[0] = 1.0
    return replace(
        inp,
        pmass_type="power_series",
        am=am,
        pres_scale=float(pressure[0]),
        ncurr=1,
        pcurr_type="power_series",
        ac=ac,
        curtor=0.0,
        lfreeb=False,
    )


class CurrentClosure:
    def __init__(self, inp, ns, profiles):
        self.profiles = profiles
        self.s = jnp.linspace(0, 1, ns)
        self.surfaces = jnp.asarray(S.REDL_SURFACES)
        degree = min(S.CURRENT_DEGREE, ns - 2)
        self.size = degree + 1
        self.vander = jnp.asarray(np.polynomial.chebyshev.chebvander(2 * np.asarray(self.s) - 1, degree))
        self.fit = jnp.asarray(np.linalg.pinv(np.asarray(self.vander)))
        self.ac_size = len(inp.ac)
        ints = np.zeros(degree + 1)
        for k in range(degree + 1):
            c = np.polynomial.Chebyshev.basis(k, domain=[0, 1])
            ints[k] = c.integ()(1) - c.integ()(0)
        self.ints = jnp.asarray(ints)
        self.dds = jnp.asarray(bs._picard_dds_matrix(ns, 1 / (ns - 1)))

    def coordinates(self, inp):
        kind = inp.pcurr_type.strip().lower()
        if kind == "chebyshev_ip":
            c = np.polynomial.Chebyshev(np.asarray(inp.ac), domain=[0, 1])
            norm = c.integ()(1) - c.integ()(0)
            values = c(np.asarray(self.s))
        elif kind == "power_series":
            norm = np.sum(np.asarray(inp.ac) / (np.arange(len(inp.ac)) + 1))
            values = np.polynomial.polynomial.polyval(np.asarray(self.s), np.asarray(inp.ac))
        else:
            raise ValueError("current chart requires power_series or chebyshev_ip input")
        if not np.isfinite(norm) or abs(norm) < 1e-30 or abs(inp.curtor) < 1e-8:
            raise ValueError("A nonzero converged Redl seed is required for this current chart")
        # Avoid even a fitting round trip for native coefficients of this order.
        if kind == "chebyshev_ip" and not np.any(np.asarray(inp.ac)[self.size :]):
            return np.asarray(inp.ac)[: self.size] * inp.curtor / norm / S.CURRENT_SCALE_A
        return np.asarray(self.fit) @ (values * inp.curtor / norm / S.CURRENT_SCALE_A)

    def params(self, base, q):
        ac = jnp.pad(S.CURRENT_SCALE_A * q, (0, self.ac_size - self.size))
        return replace(base, ac=ac, curtor=S.CURRENT_SCALE_A * jnp.dot(self.ints, q))

    def input(self, inp, q):
        """Use the same native profile for forward solves, AD, and saved decks."""
        native = replace(inp, pcurr_type="chebyshev_ip")
        return im.input_with_params(native, self.params(im.params_from_input(native), jnp.asarray(q)))

    def target(self, state, rt):
        hm = bs._half_mesh_fields(state, rt)
        geom = bs._geometry_from_half(hm, self.surfaces, n_lambda=S.REDL_N_LAMBDA)
        jr, _ = bs.j_dot_B_redl(self.profiles, geom, S.HELICITY_N)
        full = jnp.interp(self.s, jnp.r_[0.0, self.surfaces, 1.0], jnp.r_[0.0, jr, 0.0])
        b2 = jnp.interp(self.s, self.surfaces, geom.fsa_B2)
        dp = jnp.interp(self.s, hm.s_half, jnp.gradient(hm.p_int, 1 / (len(self.s) - 1)))
        flux = 2 * jnp.pi * hm.phi_edge
        matrix = (b2[:, None] * self.dds + jnp.diag(dp)) / flux
        matrix = matrix.at[0, :].set(0.0).at[0, 0].set(1.0)
        enclosed = jnp.linalg.solve(matrix, full.at[0].set(0.0))
        derivative = (full * flux - enclosed * dp) / b2
        fit = self.fit @ (derivative / S.CURRENT_SCALE_A)
        total = jnp.trapezoid(derivative, self.s) / S.CURRENT_SCALE_A
        return fit * total / jnp.dot(self.ints, fit)

    def diagnostics(self, state, rt, q):
        target = self.target(state, rt)
        relative = float(
            jnp.max(jnp.abs(self.vander @ (q - target))) / jnp.maximum(jnp.max(jnp.abs(self.vander @ target)), 1e-30)
        )
        hm = bs._half_mesh_fields(state, rt)
        surfaces = jnp.linspace(0.1, 0.9, 25)  # independent of current-fit nodes
        geom = bs._geometry_from_half(hm, surfaces, n_lambda=S.REDL_N_LAMBDA)
        redl, _ = bs.j_dot_B_redl(self.profiles, geom, S.HELICITY_N)
        vmec = bs._jv_from_half(hm, surfaces)
        mismatch = float(jnp.linalg.norm(vmec - redl) / jnp.maximum(jnp.linalg.norm(redl), 1e-30))
        return dict(
            redl_closure_relative=relative,
            redl_interior_relative=mismatch,
            plasma_current_A=float(S.CURRENT_SCALE_A * jnp.dot(self.ints, q)),
        )


def bootstrap_seed(inp, profiles, *, initial_state=None, device="gpu"):
    """Warm-started Picard preparation on the same current map as the root.

    This is a seed at the looser ordinary-force tolerance, never an accepted
    optimization root. No cold fallback is allowed after an equilibrium exists.
    """
    from types import SimpleNamespace

    closure = CurrentClosure(inp, int(inp.ns_array[-1]), profiles)
    q = closure.coordinates(inp) if abs(inp.curtor) > 1e-8 else np.zeros(closure.size)
    inp = closure.input(inp, q)
    base = im.params_from_input(inp)
    state = initial_state
    history = []
    for iteration in range(S.PICARD_STEPS):
        deck = inp if not np.any(q) else im.input_with_params(inp, closure.params(base, jnp.asarray(q)))
        eq = opt.solve_equilibrium(
            deck,
            initial_state=state,
            device=device,
            raise_on_max_iterations=True,
            forward_ftol=S.SEED_FORCE_TOLERANCE,
            forward_max_iterations=S.SEED_MAX_ITERATIONS,
            polish_force_balance=False,
            verbose=state is None,
        )
        state = eq.state
        target = np.asarray(closure.target(state, eq.runtime))
        scale = max(float(np.max(np.abs(np.asarray(closure.vander) @ target))), 1e-30)
        delta = float(np.max(np.abs(np.asarray(closure.vander) @ (q - target)))) / scale
        if not np.all(np.isfinite(target)) or not np.isfinite(delta):
            raise RootFailure("nonfinite bootstrap preparation update")
        item = dict(
            iteration=iteration,
            curtor=float(deck.curtor),
            delta=delta,
            fsqr=float(eq.result.fsqr),
            fsqz=float(eq.result.fsqz),
            fsql=float(eq.result.fsql),
        )
        history.append(item)
        print("[bootstrap_seed] " + str(item), flush=True)
        if delta <= S.PICARD_RTOL:
            return SimpleNamespace(input=deck, equilibrium=eq, converged=True, history=history)
        q = 0.5 * q + 0.5 * target
    raise RootFailure("warm-started bootstrap preparation did not converge within its Picard budget")


class EquilibriumCurrentRoot:
    def __init__(self, inp, coils, seed, profiles, arm, *, event=lambda **kw: None):
        if arm not in ("fixed", "free") or not jax.config.x64_enabled:
            raise ValueError("fixed/free formulation and float64 required")
        ns = int(inp.ns_array[-1])
        self.closure = CurrentClosure(inp, ns, profiles)
        inp = self.closure.input(inp, self.closure.coordinates(inp))
        self.arm, self.inp, self.event = arm, inp, event
        self.chart = opt.CoilParameters.from_coils(
            coils, current_dofs=(), scales=np.full(coils.dofs_curves.size, S.P.COIL_STEP)
        )
        self.nb = 0 if arm == "free" else len(opt.pack_boundary(inp, S.MAX_BOUNDARY_MODE, vary_major_radius=True))
        boundary = np.empty(0) if arm == "free" else opt.pack_boundary(inp, S.MAX_BOUNDARY_MODE, vary_major_radius=True)
        self.origin = np.r_[boundary, self.chart.x0]
        boundary_scales = (
            S.BOUNDARY_STEP_M
            * opt._ess_scale(inp, S.MAX_BOUNDARY_MODE, S.BOUNDARY_SPECTRAL_ALPHA, vary_major_radius=True)
            if self.nb
            else np.empty(0)
        )
        self.scales = np.r_[boundary_scales, self.chart.scales]
        self.x0 = np.zeros(len(self.origin))
        self.cfg = None
        ns = int(inp.ns_array[-1])
        if arm == "free":
            self.inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
            self.cfg = fbi.make_free_boundary_config(
                self.inp,
                self.chart(jnp.asarray(self.chart.x0)),
                field_from_parameters=self.chart,
                ns=ns,
                device=str(jax.default_backend()),
                include_edge_in_convergence=True,
                edge_force_tolerance=S.P.EQUILIBRIUM_FTOL,
                ftol=S.P.EQUILIBRIUM_FTOL,
                max_iterations=12000,
                adjoint_fail="error",
            )
            stage = fbi._solve_free_boundary_stage(
                self.inp,
                external_field=self.chart(jnp.asarray(self.chart.x0)),
                resolution=self.cfg.resolution,
                ftol=S.P.EQUILIBRIUM_FTOL,
                max_iterations=12000,
                initial_state=seed,
                include_edge_in_convergence=True,
                edge_force_tolerance=S.P.EQUILIBRIUM_FTOL,
                error_on_no_convergence=True,
                jacobian_retries=0,
                allow_initial_axis_reguess=False,
                use_fft=False,
            )
            self.icfg = self.cfg.implicit
            _, self.mask, self.rcon, self.zcon = fbi._linearization_from_stage(
                self.cfg, im.params_from_input(self.inp), stage, inp=self.inp
            )
            self.frozen = stage.result.state
        else:
            self.icfg = im.make_config(inp, ns=ns, ftol=S.P.EQUILIBRIUM_FTOL, device=jax.devices()[0])
            rt = im.runtime_from_params(im.params_from_input(inp), self.icfg)
            self.mask = im._fixed_boundary_dof_mask(self.icfg)
            self.frozen = seed
            self.rcon, self.zcon = rt.rcon0, rt.zcon0
        self.mask = jax.tree.map(jnp.asarray, self.mask)
        self.space = jax.tree.map(jnp.asarray, _active_space(self.icfg, self.mask, S.MAX_ROOT_DOFS))
        self.nstate = len(self.space.left)
        self.project = im._dof_projector(self.icfg, self.mask)
        self.base = im.params_from_input(self.inp)
        self.runtime_params(self.closure.coordinates(inp), jnp.asarray(self.x0))  # prime runtime before AD
        self.initial_y = np.r_[np.zeros(self.nstate), self.closure.coordinates(inp)]
        root_type, extra = LinearRoot, {}
        if S.DEVICE_SOLVES:
            from device_root import DeviceRoot
            root_type = DeviceRoot
            extra = dict(rhs_batch_size=S.ADJOINT_RHS_BATCH_SIZE, dense_assembly=S.DENSE_ASSEMBLY)
        self.linear = root_type(
            self.residual,
            valid=self.valid,
            invalid_reason=self.geometry_status,
            options=Options(
                S.ROOT_ATOL,
                S.NEWTON_STEPS,
                S.LINEAR_RTOL,
                S.LINEAR_GATE,
                S.LINEAR_RESTART,
                S.LINEAR_CYCLES,
                S.DENSE_BATCH_SIZE,
                S.MAX_ROOT_DOFS,
                extra_newton_steps=S.NEWTON_EXTRA_STEPS,
                adaptive_newton=True,
                refresh_horizon=S.LINEAR_REFRESH_HORIZON,
                reuse_budget_fraction=S.LINEAR_REUSE_BUDGET_FRACTION,
                separate_action_warmup=True,
                saved_linearization=S.SAVED_LINEARIZATION,
                saved_krylov_actions=S.SAVED_KRYLOV_ACTIONS,
                newton_rtol=S.NEWTON_RTOL,
                newton_gate=S.NEWTON_GATE,
                trial_dense_rebuilds=S.TRIAL_DENSE_REBUILDS,
                stagnation_steps=S.NEWTON_STAGNATION_STEPS,
            ),
            event=event,
            **extra,
        )

    def design(self, x):
        physical = jnp.asarray(self.origin) + jnp.asarray(self.scales) * x
        return physical[: self.nb], physical[self.nb :]

    def runtime_params(self, q, x):
        boundary, _ = self.design(x)
        params = self.base
        if self.nb:
            rbc, zbs = opt.boundary_arrays_from_x(self.inp, boundary, S.MAX_BOUNDARY_MODE, vary_major_radius=True)
            params = replace(params, rbc=rbc, zbs=zbs)
        params = self.closure.params(params, q)
        rt = replace(im.runtime_from_params(params, self.icfg), rcon0=self.rcon, zcon0=self.zcon)
        if self.arm == "free":
            rt = replace(
                rt,
                lfreeb=True,
                jmax=len(self.closure.s),
                presf_ns_scale=fbi._presf_ns_scale_traceable(params, self.inp, len(self.closure.s)),
            )
        return rt, params

    def objects(self, y, x, *, vacuum=False):
        q = y[self.nstate :]
        rt, params = self.runtime_params(q, x)
        delta = _expand(y[: self.nstate], self.frozen, self.space)
        z = jax.tree.map(jnp.add, self.frozen, delta)
        state = z if self.arm == "free" else im._assemble(z, rt, self.frozen, self.project, im._edge_mask(self.icfg))
        _, c = self.design(x)
        if vacuum and self.arm == "free":
            rt = replace(rt, bsqvac_edge=self.cfg.vacuum_program.bsq(state, rt, self.chart(c)))
        return state, rt, params, self.chart.coils_from_x(c)

    def residual(self, y, x):
        state, rt, _, _ = self.objects(y, x, vacuum=True)
        force = evaluate_forces(state, rt)[0]
        target = self.closure.target(state, rt)
        return jnp.r_[_compress(self.project(force), self.space), y[self.nstate :] - target]

    def valid(self, y, x):
        return self.geometry_status(y, x)["valid"]

    def geometry_status(self, y, x):
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
            return dict(valid=False, reason="nonfinite_state_or_design")
        if abs(float(S.CURRENT_SCALE_A * jnp.dot(self.closure.ints, jnp.asarray(y[self.nstate :])))) < 1e-8:
            return dict(valid=False, reason="zero_total_current")
        state, rt, _, _ = self.objects(jnp.asarray(y), jnp.asarray(x))
        _, jac, _, _, _ = sp._field_chain(state, rt)
        if not bool(jnp.all(jnp.isfinite(jac.sqrt_g))):
            return dict(valid=False, reason="nonfinite_jacobian")
        if bool(jac.jacobian_sign_changed):
            tau = np.asarray(jac.tau)[1:]
            return dict(valid=False, reason="jacobian_sign_changed", tau_min=float(tau.min()), tau_max=float(tau.max()))
        return dict(valid=True, reason=None)

    def transport_seed(self, y, previous_x, trial_x):
        """Transport the accepted interior to the imposed edge, without solving.

        Keep the axis and current seed. Mode-dependent radial powers preserve
        regularity for high poloidal harmonics; the active projection preserves
        the m=1 constraints. The exact root and force gates still decide success.
        """
        if self.arm == "free":
            return np.array(y, copy=True)
        previous, _, _, _ = self.objects(jnp.asarray(y), jnp.asarray(previous_x))
        trial, rt, _, _ = self.objects(jnp.asarray(y), jnp.asarray(trial_x))
        powers = jnp.maximum(jnp.abs(rt.modes.m) / 2.0, 1.0)
        weights = jnp.asarray(rt.setup.s_full)[:, None] ** powers[None, :]
        changes = {}
        for name in previous.__dataclass_fields__:
            before = getattr(previous, name)
            changes[name] = (
                weights * (getattr(trial, name)[-1] - before[-1])[None, :]
                if name.startswith(("R_", "Z_"))
                else jnp.zeros_like(before)
            )
        delta = self.project(type(previous)(**changes))
        transported = np.array(y, copy=True)
        transported[: self.nstate] += np.asarray(_compress(delta, self.space))
        return transported

    def seed_at(self, y, previous_x, trial_x):
        """Choose a valid low-residual seed derived from the certified prior root."""
        candidates = [("accepted", np.array(y, copy=True))]
        if self.arm == "fixed":
            candidates.append(("transport", self.transport_seed(y, previous_x, trial_x)))
        try:
            prediction = self.linear.predict(y, previous_x, trial_x)
            damping = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125) if self.arm == "free" else (1.0,)
            for alpha in damping:
                name = "accepted_tangent" if alpha == 1 else f"accepted_tangent_{alpha:g}"
                candidates.append((name, np.asarray(y) + alpha * (prediction - y)))
        except RootFailure as error:
            self.event(event="predictor_rejected", error=str(error))
        viable = []
        for name, candidate in candidates:
            if self.valid(candidate, trial_x):
                defect = np.asarray(self.linear.residual(jnp.asarray(candidate), jnp.asarray(trial_x)))
                if np.all(np.isfinite(defect)):
                    viable.append((float(np.linalg.norm(defect)), name, candidate))
        if not viable:
            raise RootFailure("no valid trial seed from accepted equilibrium")
        norm, name, result = min(viable, key=lambda item: item[0])
        self.event(
            event="trial_seed",
            method=name,
            root_residual=norm,
            candidate_residuals={name: norm for norm, name, _ in viable},
        )
        return result

    def forward_seed(self, y, x):
        """Warm ordinary VMEX correction, using the same solvers as main.

        The predicted current is held only for this seed solve. It remains a
        root unknown in the subsequent coupled Newton polish and total adjoint.
        Never use the vacuum problem's prescribed-current derivative here.
        """
        state, _, params, _ = self.objects(jnp.asarray(y), jnp.asarray(x))
        deck = im.input_with_params(self.inp, params)
        if self.arm == "free":
            stage = fbi._solve_free_boundary_stage(
                deck,
                external_field=self.chart(jnp.asarray(self.design(x)[1])),
                resolution=self.cfg.resolution,
                ftol=S.P.EQUILIBRIUM_FTOL,
                max_iterations=S.SEED_MAX_ITERATIONS,
                initial_state=state,
                constraint_continuation=(self.rcon, self.zcon),
                include_edge_in_convergence=True,
                edge_force_tolerance=S.P.EQUILIBRIUM_FTOL,
                error_on_no_convergence=True,
                jacobian_retries=0,
                allow_initial_axis_reguess=False,
                use_fft=False,
            )
            corrected = stage.result.state
        else:
            corrected = opt.solve_equilibrium(
                deck,
                initial_state=state,
                device=jax.default_backend(),
                forward_ftol=S.P.EQUILIBRIUM_FTOL,
                forward_max_iterations=S.SEED_MAX_ITERATIONS,
                raise_on_max_iterations=True,
                polish_force_balance=False,
                jacobian_retries=0,
                coarse_grid_retry=False,
            ).state
        delta = self.project(jax.tree.map(lambda a, b: a - b, corrected, self.frozen))
        result = np.array(y, dtype=float, copy=True)
        result[: self.nstate] = np.asarray(_compress(delta, self.space))
        return result

    def correct_trial(self, seed, x):
        """Both arms: coupled Newton first, ordinary seed only for hard failure."""
        candidate = np.array(seed, dtype=float, copy=True)
        try:
            return self.linear.solve(candidate.copy(), x)
        except SlowProgress:
            raise  # request a smaller shared continuation step
        except RootFailure as error:
            failure = error
            self.event(event="coupled_correction_recovery", reason=str(error))
        # A failed solver must not corrupt the last certified seed used here.
        forward = self.ordinary_seed(candidate, x)
        if np.array_equal(forward, candidate):
            raise failure
        # No ordinary solve can bypass full current feedback and polishing.
        return self.linear.solve(forward, x)

    def ordinary_seed(self, candidate, x):
        """Return a warm ordinary seed only when it improves the coupled norm."""
        from vmex.core.errors import VmecError

        started = time.monotonic()
        try:
            forward = self.forward_seed(candidate, x)
            if self.valid(forward, x):
                before = float(jnp.linalg.norm(self.linear.residual(candidate, x)))
                after = float(jnp.linalg.norm(self.linear.residual(forward, x)))
                selected = np.isfinite(after) and after < before
                self.event(
                    event="ordinary_correction",
                    seconds=time.monotonic() - started,
                    before=before,
                    after=after if np.isfinite(after) else None,
                    selected=bool(selected),
                )
                if selected:
                    return forward
            else:
                self.event(
                    event="ordinary_correction_rejected", seconds=time.monotonic() - started, reason="invalid geometry"
                )
        except (VmecError, RootFailure) as error:
            self.event(event="ordinary_correction_rejected", seconds=time.monotonic() - started, reason=str(error))
        return candidate

    def certify(self, y, x):
        if not self.valid(y, x):
            raise RootFailure("geometry certificate failed")
        state, rt, params, coils = self.objects(jnp.asarray(y), jnp.asarray(x), vacuum=True)
        _, forces, diag = evaluate_forces(state, rt)
        names = ("fsqr", "fsqz", "fsql", "fedge") if self.arm == "free" else ("fsqr", "fsqz", "fsql")
        values = {n: float(getattr(forces, n)) for n in names}
        root = float(jnp.linalg.norm(self.linear.residual(jnp.asarray(y), jnp.asarray(x))))
        redl = self.closure.diagnostics(state, rt, jnp.asarray(y[self.nstate :]))
        if (
            not all(np.isfinite(v) and 0 <= v <= S.P.EQUILIBRIUM_FTOL for v in values.values())
            or not np.isfinite(root)
            or root > S.ROOT_ATOL
            or not all(np.isfinite(v) for v in redl.values())
            or redl["redl_closure_relative"] > S.REDL_CLOSURE_RTOL
            or redl["redl_interior_relative"] > S.REDL_INTERIOR_MISMATCH_LIMIT
        ):
            raise RootFailure(f"force/current certification failed: {values}, root={root}, {redl}")
        return dict(**values, root_residual=root, **redl)

    def input_at(self, y, x):
        _, _, params, _ = self.objects(jnp.asarray(y), jnp.asarray(x))
        return im.input_with_params(self.inp, params)


def verify_high_resolution(inp, coils, profiles, arm, *, initial_state,
                           initial_baselines=None, event=lambda **kw: None):
    """Independent Picard/ordinary-force verification; no NS201 dense adjoint.

    This checks force and current tolerances independently. It does not certify
    an NS201 coupled Newton root or an NS201 derivative. Production derivatives
    are always computed on the stricter coupled NS51 root instead.

    For a free solve seeded from an accepted optimization state, carry its
    matching spectral-constraint baselines on this radial grid. Recomputing
    them from an optimized boundary changes the startup force operator and
    can prevent vacuum activation. These are numerical reference arrays,
    not coil constraints; the free solver applies its normal decay to them.
    """
    if initial_baselines is not None and arm != "free":
        raise ValueError("initial_baselines currently applies only to free verification")
    closure = CurrentClosure(inp, int(inp.ns_array[-1]), profiles)
    q = closure.coordinates(inp)
    inp = closure.input(inp, q)
    base = im.params_from_input(inp)
    state = initial_state
    baselines = initial_baselines
    chart = opt.CoilParameters.from_coils(coils, current_dofs=())
    field = chart(jnp.asarray(chart.x0))
    if arm == "free":
        inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
        cfg = fbi.make_free_boundary_config(
            inp,
            field,
            field_from_parameters=chart,
            device=jax.default_backend(),
            include_edge_in_convergence=True,
            edge_force_tolerance=S.P.VERIFY_FTOL,
            ftol=S.P.VERIFY_FTOL,
            max_iterations=12000,
        )
    for iteration in range(60):
        params = closure.params(base, jnp.asarray(q))
        deck = im.input_with_params(inp, params)
        if arm == "free":
            stage = fbi._solve_free_boundary_stage(
                deck,
                external_field=field,
                resolution=cfg.resolution,
                ftol=S.P.VERIFY_FTOL,
                max_iterations=12000,
                initial_state=state,
                constraint_continuation=baselines,
                include_edge_in_convergence=True,
                edge_force_tolerance=S.P.VERIFY_FTOL,
                error_on_no_convergence=False,
                jacobian_retries=0,
                allow_initial_axis_reguess=False,
                use_fft=False,
            )
            result = stage.result
            forward = dict(
                ns=len(closure.s), iteration=iteration,
                converged=bool(result.converged), iterations=int(result.iterations),
                **{name: float(getattr(result, name)) for name in ("fsqr", "fsqz", "fsql", "fedge")},
                vacuum_active=bool(stage.vacuum.turned_on),
                vacuum_calls=int(stage.vacuum.vacuum_calls),
            )
            event(event="verification_forward", **forward)
            if not result.converged:
                raise RootFailure(f"independent free forward solve did not converge: {forward}")
            state = stage.result.state
            baselines = (stage.rcon0, stage.zcon0)
            rt = replace(
                im.runtime_from_params(params, cfg.implicit),
                rcon0=stage.rcon0,
                zcon0=stage.zcon0,
                lfreeb=True,
                jmax=len(closure.s),
                presf_ns_scale=fbi._presf_ns_scale_traceable(params, inp, len(closure.s)),
            )
            rt = replace(rt, bsqvac_edge=cfg.vacuum_program.bsq(state, rt, field))
        else:
            eq = opt.solve_equilibrium(
                deck,
                initial_state=state,
                device=jax.default_backend(),
                forward_max_iterations=12000,
                raise_on_max_iterations=True,
                polish_force_balance=False,
            )
            state, rt = eq.state, eq.runtime
        _, force, diag = evaluate_forces(state, rt)
        names = ("fsqr", "fsqz", "fsql", "fedge") if arm == "free" else ("fsqr", "fsqz", "fsql")
        values = {name: float(getattr(force, name)) for name in names}
        if bool(diag.jacobian_sign_changed) or not all(
            np.isfinite(v) and 0 <= v <= S.P.VERIFY_FTOL for v in values.values()
        ):
            raise RootFailure("independent NS201 force certificate failed")
        redl = closure.diagnostics(state, rt, jnp.asarray(q))
        if not all(np.isfinite(value) for value in redl.values()):
            raise RootFailure("nonfinite NS201 current certificate")
        event(event="verification_picard", iteration=iteration, **values, **redl)
        if redl["redl_closure_relative"] <= S.REDL_CLOSURE_RTOL:
            if redl["redl_interior_relative"] > S.REDL_INTERIOR_MISMATCH_LIMIT:
                raise RootFailure("independent NS201 physical Redl mismatch failed")
            return (
                deck,
                state,
                rt,
                dict(
                    **values,
                    **redl,
                    method="independent Picard plus ordinary force convergence",
                    coupled_newton_root_certified=False,
                    iterations=iteration + 1,
                ),
            )
        target = np.asarray(closure.target(state, rt))
        if not np.all(np.isfinite(target)):
            raise RootFailure("nonfinite NS201 current update")
        q = 0.5 * q + 0.5 * target
    raise RootFailure("independent NS201 current convergence budget exhausted")
