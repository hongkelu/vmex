"""Derivative certificates for the coupled free-boundary root."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input
from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridField, read_mgrid
from vmex.core.freeboundary_implicit import (
    make_free_boundary_config,
    solve_free_boundary_implicit,
    solve_free_boundary_implicit_status,
)
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import AdjointSolveError, VmecJacobianError


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_free_boundary_status_callback_turns_trial_error_into_status(monkeypatch):
    """An invalid optimizer trial returns status 1 instead of crossing JAX."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]),
    )
    field = lasym_free_field()
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=20,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
        device="cpu",
    )
    assert cfg.implicit.device.platform == "cpu"

    def fail(*_args, **_kwargs):
        raise VmecJacobianError("invalid trial")

    monkeypatch.setattr(fbi, "_host_solve_and_mask", fail)
    _state, status, fsq, ratio = solve_free_boundary_implicit_status(
        im.params_from_input(inp), field.extcur, cfg)
    assert status == 1
    assert np.isinf(fsq) and np.isinf(ratio)
    assert cfg.resolution.ns == 8

    monkeypatch.setattr(fbi, "_host_solve_and_mask",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("implementation bug")))
    with pytest.raises(RuntimeError, match="implementation bug"):
        fbi._host_solve_and_mask_status(cfg, im.params_from_input(inp), field.extcur)


def test_free_boundary_config_rejects_fixed_boundary_input():
    inp = dataclasses.replace(lasym_free_input(DATA), lfreeb=False)
    with pytest.raises(ValueError, match="LFREEB"):
        make_free_boundary_config(inp, lasym_free_field())


def test_free_boundary_config_validates_adjoint_solver():
    inp, field = lasym_free_input(DATA), lasym_free_field()
    with pytest.raises(ValueError, match="'boundary_schur' or 'coupled_gcrot'"):
        make_free_boundary_config(inp, field, adjoint_solver="dense")


def test_free_boundary_config_validates_adjoint_fail():
    """The adjoint failure policy is opt-in and defaults to raising.

    ``best_effort`` returns a stalled Krylov solution so that one bad trial
    in an optimization is a poor search direction rather than a dead run; a
    typo must not silently select it.
    """
    inp, field = lasym_free_input(DATA), lasym_free_field()
    assert make_free_boundary_config(inp, field).adjoint_fail == "error"
    assert make_free_boundary_config(
        inp, field, adjoint_fail="best_effort").adjoint_fail == "best_effort"
    with pytest.raises(ValueError, match="'error' or 'best_effort'"):
        make_free_boundary_config(inp, field, adjoint_fail="warn")


def test_host_adjoint_best_effort_warns_instead_of_raising(monkeypatch):
    """A stalled Krylov solve is a warning under the opt-in policy, not a stop.

    The stall arrives inside the VJP, where an optimizer's own
    ``lax.cond`` on the solve status cannot catch it, so a raise ends the
    whole run over one bad line-search trial.  Under ``best_effort`` the
    inaccurate direction is returned instead and the line search rejects it.
    """
    cfg = SimpleNamespace(adjoint_tol=1.0e-10, adjoint_maxiter=5,
                          adjoint_gcrot_m=2, adjoint_gcrot_k=1)

    def residual(z, params, field, base, rcon, zcon):
        return 2.0 * z

    z_star, rhs = jnp.arange(4.0), jnp.ones(4)
    # A Krylov solve that returns a deliberately wrong answer, so the true
    # residual check that follows it cannot pass.
    monkeypatch.setattr(fbi, "gcrotmk", lambda *args, **kwargs: (np.zeros(4), 0))
    def call(**kwargs):
        return fbi._host_adjoint(
            residual, z_star, None, None, z_star, None, None, rhs, cfg, **kwargs)

    with pytest.raises(AdjointSolveError, match="did not converge"):
        call()
    with pytest.warns(RuntimeWarning, match="best_effort"):
        solution = call(fail="best_effort")
    np.testing.assert_allclose(np.asarray(solution), np.zeros(4))


@pytest.mark.parametrize("backend", ["coupled_gcrot", "reverse_gcrot"])
def test_host_adjoint_prepared_transpose_tracks_changed_root_data(backend):
    """Executable reuse must not reuse a previous root's numerical tape."""
    matrix = jnp.array([[4.0, 0.3, -0.2], [0.1, 3.0, 0.4], [0.2, -0.1, 5.0]])

    def residual(z, params, field, base, rcon, zcon):
        coefficient = params + field + base + rcon + zcon
        return matrix @ z + 0.5 * coefficient * z**2

    cfg = SimpleNamespace(adjoint_tol=1.0e-11, adjoint_maxiter=10,
                          adjoint_gcrot_m=3, adjoint_gcrot_k=1)
    values = [jnp.array([0.2, 0.3, 0.4]) + 0.1 * i for i in range(6)]
    rhs = jnp.array([0.4, -0.7, 1.2])
    reference = None
    # Change each dynamic input independently, keeping all shapes unchanged.
    for changed in (None, 0, 1, 2, 3, 4, 5):
        args = list(values)
        if changed is not None:
            args[changed] = args[changed] + 0.25
        z, params, field, base, rcon, zcon = args
        jacobian = matrix + jnp.diag((params + field + base + rcon + zcon) * z)
        expected = np.linalg.solve(np.asarray(jacobian.T), np.asarray(rhs))
        if backend == "reverse_gcrot":
            tape = fbi._prepare_reverse_transpose(*args, residual=residual)
            result = fbi._solve_prepared_reverse_adjoint(tape, rhs, cfg)
        else:
            result = fbi._host_adjoint(residual, *args, rhs, cfg)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-11)
        if changed is None:
            reference = expected
        else:
            assert np.linalg.norm(expected - reference) > 1e-4


@pytest.mark.parametrize("backend", ["coupled_gcrot", "reverse_gcrot"])
def test_multi_rhs_pullback_matches_nonsymmetric_analytic_response(monkeypatch, backend):
    """All rows, including a zero row, match an independent dense derivative."""
    matrix = jnp.array([[4., .3, -.2], [.1, 3., .4], [.2, -.1, 5.]])
    profile_map = jnp.array([[1., 2.], [-1., .2], [.5, -.3]])
    field_map = jnp.array([[.3], [1.], [-.4]])

    def residual(z, params, field, base, rcon, zcon):
        return matrix @ z + .2*z**2 - profile_map @ params['drive'] - field_map @ field['scale']

    cfg = SimpleNamespace(adjoint_tol=1e-11, adjoint_maxiter=10,
                          adjoint_gcrot_m=3, adjoint_gcrot_k=1)
    z = jnp.array([.2, .4, -.1])
    params, field = {'drive':jnp.array([.1, .2])}, {'scale':jnp.array([.3])}
    rhs = jnp.array([[1., 0., 0.], [0., 1., 1.], [1., -2., .2], [0., 0., 0.]])
    counts = {'state':0, 'parameter':0}
    state_name = "_prepare_reverse_transpose" if backend == "reverse_gcrot" else "_prepare_host_transpose"
    prepare_state, prepare_parameter = getattr(fbi, state_name), fbi._prepare_parameter_pullback
    def counted_state(*args, **kwargs):
        counts['state'] += 1
        return prepare_state(*args, **kwargs)
    def counted_parameter(*args, **kwargs):
        counts['parameter'] += 1
        return prepare_parameter(*args, **kwargs)
    monkeypatch.setattr(fbi, state_name, counted_state)
    monkeypatch.setattr(fbi, '_prepare_parameter_pullback', counted_parameter)
    pb, fb = fbi._host_pullback_multi_rhs(residual,z,params,field,z,None,None,rhs,cfg,backend=backend)
    jacobian = np.asarray(matrix + jnp.diag(.4*z))
    np.testing.assert_allclose(pb['drive'], np.asarray(rhs) @ np.linalg.solve(jacobian,profile_map), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(fb['scale'], np.asarray(rhs) @ np.linalg.solve(jacobian,field_map), rtol=1e-9, atol=1e-11)
    assert counts == {'state':1, 'parameter':1}


def test_multi_rhs_pullback_rejects_one_failed_row(monkeypatch):
    """Successful neighboring rows cannot hide a false-success middle row."""
    cfg = SimpleNamespace(adjoint_tol=1e-11, adjoint_maxiter=10,
                          adjoint_gcrot_m=3, adjoint_gcrot_k=1)
    calls = []
    native = fbi.gcrotmk
    def faulty(matrix, rhs, **kwargs):
        calls.append(rhs)
        return (np.zeros_like(rhs), 0) if len(calls) == 2 else native(matrix,rhs,**kwargs)
    monkeypatch.setattr(fbi, 'gcrotmk', faulty)
    def residual(z, p, field, *_):
        return 2*z-p-field
    with pytest.raises(AdjointSolveError, match='row 1'):
        fbi._host_pullback_multi_rhs(residual,jnp.zeros(3),jnp.zeros(3),jnp.zeros(3),
                                    None,None,None,jnp.eye(3),cfg)
    assert len(calls) == 2


@pytest.mark.parametrize('cotangents', [jnp.zeros((0,3)), jnp.zeros((2,4)), jnp.zeros(3)])
def test_multi_rhs_pullback_rejects_bad_batch_shape(cotangents):
    cfg = SimpleNamespace(adjoint_solver='coupled_gcrot')
    with pytest.raises(ValueError, match='leading RHS axis'):
        fbi.free_boundary_state_pullback_multi_rhs(None,None,cfg,jnp.zeros(3),
            jnp.ones(3),cotangents,rcon0=None,zcon0=None)


@pytest.mark.parametrize("backend", ["coupled_gcrot", "reverse_gcrot", "forward_dense", "forward_dense_jax"])
def test_multi_rhs_public_projection_and_root_gate(monkeypatch, backend):
    import vmex
    assert vmex.free_boundary_state_pullback_multi_rhs is fbi.free_boundary_state_pullback_multi_rhs
    controls = SimpleNamespace(device=None, lconm1=False, adjoint_tol=1e-11, adjoint_maxiter=10,
                               adjoint_gcrot_m=3, adjoint_gcrot_k=1)
    cfg = SimpleNamespace(implicit=controls,adjoint_solver=backend,adjoint_fail='error',
                          adjoint_dense_batch_size=2,adjoint_dense_max_dofs=100)
    def residual(z, p, field, *_):
        return 2*z-p-field
    monkeypatch.setattr(fbi, '_projected_residual', lambda *_:residual)
    monkeypatch.setattr(im, '_dof_projector', lambda _,mask:lambda value:value*mask)
    monkeypatch.setattr(im, 'runtime_from_params', lambda *_:None)
    zero, mask, rhs = jnp.zeros(3), jnp.array([1.,0.,1.]), jnp.eye(3)
    pb, fb = fbi.free_boundary_state_pullback_multi_rhs(zero,zero,cfg,zero,mask,rhs,rcon0=None,zcon0=None)
    np.testing.assert_allclose(pb,np.asarray(rhs*mask/2),atol=1e-12)
    np.testing.assert_allclose(fb,np.asarray(rhs*mask/2),atol=1e-12)
    with pytest.raises(ValueError, match='root residual'):
        fbi.free_boundary_state_pullback_multi_rhs(zero,zero,cfg,jnp.ones(3),mask,rhs,rcon0=None,zcon0=None)


def test_free_boundary_warm_failure_retries_once_from_cold(monkeypatch):
    """A bad cached state is discarded, but implementation errors are not."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(inp, field, ns=8, ftol=1.0e-6,
                                    max_iterations=20)
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    # These are module-level caches keyed on (resolution, lconm1, ncurr), so
    # monkeypatch.setitem restores both entries at teardown; a bare assignment
    # would hand the all-zero mask to the next free-boundary case sharing that
    # key, which reaches the Schur lane as an edge basis with no columns.
    seed = object()
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, seed)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
    calls = []

    def solve(*_args, initial_state=None, **_kwargs):
        calls.append(initial_state)
        if initial_state is seed:
            raise VmecJacobianError("bad warm state")
        result = SimpleNamespace(state=state, fsqr=0.0, fsqz=0.0, fsql=0.0,
                                 converged=True)
        return SimpleNamespace(result=result, continuation_state=state,
                               rcon0=runtime.rcon0, zcon0=runtime.zcon0)

    monkeypatch.setattr(fbi, "_solve_free_boundary_stage", solve)
    solved, *_ = fbi._host_solve_and_mask(cfg, im.params_from_input(inp), field)
    assert calls == [seed, None]
    np.testing.assert_allclose(solved.R_cos, state.R_cos)


def test_free_boundary_host_adjoint_rejects_a_false_solver_success(monkeypatch):
    """The true transpose residual, not SciPy's info flag, certifies a gradient."""
    cfg = SimpleNamespace(adjoint_tol=1.0e-10, adjoint_gcrot_m=3,
                          adjoint_gcrot_k=1, adjoint_maxiter=2)
    monkeypatch.setattr(fbi, "gcrotmk", lambda *_args, **_kwargs: (np.zeros(2), 0))
    def residual(z, *_args):
        return z

    with pytest.raises(AdjointSolveError, match="host GCROT"):
        fbi._host_adjoint(residual, jnp.zeros(2), None, None, None, None, None,
                          jnp.ones(2), cfg)

    monkeypatch.setattr(
        fbi, "gcrotmk", lambda *_args, **_kwargs: (np.full(2, np.nan), 0))
    with pytest.raises(AdjointSolveError, match="host GCROT"):
        fbi._host_adjoint(residual, jnp.zeros(2), None, None, None, None, None,
                          jnp.ones(2), cfg)


@pytest.mark.full
def test_free_boundary_current_gradient_matches_resolve_finite_difference():
    """The implicit coil-current response agrees with two independent solves."""
    inp = lasym_free_input(DATA)
    inp = dataclasses.replace(
        inp, ns_array=np.array([16]), ftol_array=np.array([1.0e-7]),
        niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-8, adjoint_maxiter=100,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = field.extcur
    strict_state = solve_free_boundary_implicit(params, current, cfg)
    assert bool(jnp.all(jnp.isfinite(strict_state.R_cos)))
    # Exercise the strict public transform as well; the status lane below is
    # the optimizer-facing path and shares this hot forward state.
    _strict_value, _unused_pullback = jax.vjp(
        lambda value: jnp.mean(solve_free_boundary_implicit(
            params, value, cfg).R_cos[-1] ** 2), current)
    derivative = jax.grad(objective)(current)[0]
    step = 2.0e-4
    finite_difference = (
        objective(current + step) - objective(current - step)
    ) / (2.0 * step)
    # The host solve uses adaptive vacuum cadence and stops at finite force
    # tolerance; the independent re-solve FD therefore has percent-level
    # noise on this deliberately coarse nightly case.
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=2.0e-2, atol=2.0e-4
    )


@pytest.mark.full
def test_ncsx_free_boundary_current_gradient_matches_resolve_finite_difference():
    """PF5 coil-current response on the NCSX c09r00 family vs cold re-solves.

    Second-family spot certificate for the coil-current adjoint: the
    CTH-like case above is nfp=5 with a generated single-channel field;
    this is the committed nfp=3 NCSX c09r00 mgrid with ten coil groups
    (``tests/test_ncsx_free_boundary_parity.py`` documents the family).
    PF5 carries the largest boundary response of the ten channels
    (|dJ/dI| ~ 3e-8 per A); central-differencing it with a 300 A step
    keeps the FD signal well above the deterministic solver-endpoint
    floor that dominates the weaker channels.  Measured (Apple Silicon
    CPU): adjoint -3.0101e-8 vs central FD -2.9994e-8, relative
    difference 3.6e-3; 280 s wall for the whole certificate cold with
    compilation (the adjoint dominates), 4-5 s per warm-compiled FD
    re-solve.
    """
    inp = dataclasses.replace(
        VmecInput.from_file(DATA / "input.ncsx_c09r00_free_lowres"),
        ns_array=np.array([15]), ftol_array=np.array([1.0e-9]),
        niter_array=np.array([4000]),
    )
    mgrid_path = DATA / "mgrid_ncsx_c09r00_small.nc"
    if not mgrid_path.exists():
        pytest.skip("mgrid_ncsx_c09r00_small.nc not fetched (tools/fetch_assets.py)")
    data = read_mgrid(mgrid_path)
    field = MgridField.from_mgrid_data(
        data, extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur])
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=15, ftol=1.0e-9, max_iterations=4000,
        adjoint_tol=1.0e-8, adjoint_maxiter=200,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = np.asarray(field.extcur, dtype=float)
    pf5 = 7  # EXTCUR(8), the PF5 ring pair at 3.01e4 A
    derivative = float(jax.grad(objective)(jnp.asarray(current))[pf5])
    step = 300.0
    values = []
    for sign in (1.0, -1.0):
        # Independent cold re-solves prevent continuation history from
        # manufacturing agreement with the implicit derivative.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        perturbed = current.copy()
        perturbed[pf5] += sign * step
        values.append(float(objective(jnp.asarray(perturbed))))
    finite_difference = (values[0] - values[1]) / (2.0 * step)

    assert abs(finite_difference) > 1.0e-9
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-2)


@pytest.mark.full
def test_free_boundary_pressure_gradient_matches_resolve_finite_difference():
    """The solved objective retains the edge-pressure normalization response."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-9]), niter_array=np.array([4000]))
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-9, max_iterations=4000,
        adjoint_tol=1.0e-7, adjoint_maxiter=150,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")

    def objective(relative_am0):
        trial = dataclasses.replace(
            params, am=params.am.at[0].set(params.am[0] * (1.0 + relative_am0)))
        state, _, _, _ = solve_free_boundary_implicit_status(
            trial, field.extcur, cfg)
        return jnp.mean(state.R_cos[-1]**2 + state.Z_sin[-1]**2)

    derivative = jax.grad(objective)(0.0)
    step = 1.0e-2
    values = []
    for sign in (-1.0, 1.0):
        # Independent cold re-solves prevent continuation history from
        # manufacturing agreement with the implicit derivative.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        values.append(objective(sign * step))
    finite_difference = (values[1] - values[0]) / (2.0 * step)

    assert abs(float(finite_difference)) > 1.0e-7
    # This ns=8 campaign is limited by the independently reconverged nonlinear
    # roots. The missing presf_ns_scale term changes the response by O(1).
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=1.0e-1, atol=1.0e-7)


@pytest.mark.full
def test_boundary_schur_adjoint_reproduces_the_coupled_gcrot_gradient():
    """Both adjoint solvers invert the same converged plasma-vacuum Jacobian.

    ``coupled_gcrot`` is matrix-free on the full coupled transpose;
    ``boundary_schur`` eliminates the block-tridiagonal bulk exactly and
    solves only NESTOR's edge response (``(I + U.T A.T^-1 E.T U) mu = ...``).
    Agreement between them on one converged root certifies the radial
    elimination itself; the nightly ``full`` test below anchors both against
    an independent re-solve finite difference.

    Both lanes are certified only to ``10 x adjoint_tol x ||rhs||``, so they
    are compared at the percent level, which a wrong Schur complement (rather
    than a differently converged Krylov solve) would miss by far more.
    """
    # The nightly FD anchor retains mpol=12. This cross-solver identity uses
    # the same DIII-D equilibrium at mpol=10, which preserves the 2% agreement
    # while reducing the measured call from 791 s to 128 s.
    base = lasym_free_input(DATA).change_resolution(
        mpol=10, ntor=0, ntheta=30, nzeta=4)
    inp = dataclasses.replace(
        base, ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([2500]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=8, ftol=1.0e-8, max_iterations=2500,
            adjoint_tol=1.0e-5, adjoint_maxiter=100, adjoint_solver=solver,
            schur_probe_chunk_size=4,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current),
            device="cpu")

    def gradient(cfg):
        def objective(current):
            state, _, _, _ = solve_free_boundary_implicit_status(
                params, current, cfg)
            return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

        return np.asarray(jax.grad(objective)(field.extcur))

    coupled_cfg, schur_cfg = configure("coupled_gcrot"), configure("boundary_schur")
    coupled = gradient(coupled_cfg)
    # Re-enter the same root warm: the comparison is about the adjoint, and a
    # second cold ladder would only add solver noise to it.
    fbi._FREE_HOT_CACHE[schur_cfg] = fbi._FREE_HOT_CACHE[coupled_cfg]
    schur = gradient(schur_cfg)

    assert np.all(np.isfinite(coupled)) and np.max(np.abs(coupled)) > 0.0
    np.testing.assert_allclose(schur, coupled, rtol=2.0e-2, atol=1.0e-8)


@pytest.mark.full
def test_boundary_schur_current_gradient_matches_resolve_finite_difference():
    """The reduced adjoint retains a nontrivial external-field derivative."""
    base = lasym_free_field()
    zeros = np.zeros_like(base.br)
    field = dataclasses.replace(
        base, br=np.concatenate((zeros, base.br)),
        bp=np.concatenate((base.bp, zeros)),
        bz=np.concatenate((base.bz, zeros)), extcur=np.ones(2))
    inp = dataclasses.replace(
        lasym_free_input(DATA), extcur=np.ones(2), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([6000]))
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-8, max_iterations=6000,
        adjoint_tol=1.0e-7, adjoint_maxiter=100,
        adjoint_solver="boundary_schur", schur_probe_chunk_size=4,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(
            params, current, cfg)
        return jnp.mean(state.R_cos[-1]**2 + state.Z_sin[-1]**2)

    derivative = jax.grad(objective)(field.extcur)[1]
    step = 1.0e-3
    direction = jnp.array([0.0, 1.0])
    values = []
    for sign in (-1.0, 1.0):
        # Independent cold re-solves avoid continuation-history hysteresis in
        # this deliberately coarse nonlinear free-boundary certificate.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        values.append(objective(field.extcur + sign * step * direction))
    finite_difference = (values[1] - values[0]) / (2.0 * step)
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=5.0e-2, atol=3.0e-4)


def test_presf_ns_scale_is_differentiated_in_the_adjoint_lanes():
    """``presf_ns_scale`` tracks ``am``; the adjoint lanes may not freeze it.

    ``funct3d.f``'s edge force carries ``presf_ns_scale * pressure[-1]``, and
    the ratio ``pmass(1)/pmass(hs*(ns-1.5))`` is a smooth function of ``am``,
    which is a differentiated parameter.  Taking the host float computed from
    the reference input gives the right value at the reference point and no
    derivative at all -- the same failure mode as the frozen lasym ``delta``
    branch, and invisible to any check that compares the traceable lane with
    itself.  This deck is ``power_series`` with a live derivative; a
    ``two_power`` deck has ``p(1) = 0`` and could not show it.
    """
    from vmex.core.freeboundary import (
        _presf_ns_scale, _presf_ns_scale_traceable,
    )

    inp, ns = lasym_free_input(DATA), 9
    assert inp.pmass_type == "power_series"
    params = im.params_from_input(inp)
    host = _presf_ns_scale(inp, ns)
    np.testing.assert_allclose(
        float(_presf_ns_scale_traceable(params, inp, ns)), host,
        rtol=1e-14, atol=0.0)

    am = np.asarray(inp.am, dtype=float)
    active = int(np.max(np.nonzero(am)[0])) + 1
    grad = jax.grad(lambda p: _presf_ns_scale_traceable(p, inp, ns))(params)
    analytic = np.asarray(grad.am, dtype=float)[:active]

    def shifted(k, delta):
        return dataclasses.replace(
            inp, am=np.where(np.arange(am.size) == k, am + delta, am))

    finite = np.empty(active)
    for k in range(active):
        step = 1.0e-6 * max(abs(am[k]), 1.0)
        finite[k] = (_presf_ns_scale(shifted(k, step), ns)
                     - _presf_ns_scale(shifted(k, -step), ns)) / (2.0 * step)

    # The frozen host float reported exactly zero for all of these, so the
    # tolerance only has to separate a live derivative from a missing one.
    # This deck's am coefficients reach 5e7 against a ratio of 0.32, which
    # caps central differences on the host float at a few times 1e-4.
    assert np.max(np.abs(finite)) > 1e-6, f"probe is degenerate: {finite}"
    np.testing.assert_allclose(analytic, finite, rtol=1e-3, atol=0.0)

    # pres_scale cancels in the ratio, so its derivative is genuinely zero.
    assert float(np.asarray(grad.pres_scale)) == 0.0


def _flat(tree):
    """Flatten a state-like pytree to one vector for inner products."""
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])


@pytest.mark.full
def test_free_boundary_gradient_is_certified_factor_by_factor():
    """A certificate whose tolerances come from arithmetic, not solver noise.

    The re-solve certificates in this module central-difference two cold
    forward solves, so their percent-level gates are set by where the solver
    stops rather than by the adjoint, and a disagreement cannot be attributed:
    an FD floor and a genuinely wrong adjoint look the same.  This test
    factors the implicit gradient at a frozen root,

        dJ/dp = -(dF/dp)^T (dF/dz)^-T (dJ/dz),

    and certifies each factor separately, with no cold re-solve anywhere:

    1.  ``dF/dp`` — the AD Jacobian-vector product against a central
        difference *of the residual itself*.  Nothing is solved, so the only
        error is FD truncation and the gate is 1e-6 relative.
    2.  ``(dF/dz)^T`` — the transpose identity ``<v, J u> == <J^T v, u>`` on
        random tangents, to 1e-11 relative.  This is what makes the
        ``jax.vjp`` operator the adjoint of the forward linearization and not
        merely something with the right shape.
    3.  The adjoint linear solve — by its own transpose residual
        ``||J^T lam - rhs|| / ||rhs||``, against the configured tolerance.
    4.  The assembly, by forward-adjoint duality within the one root:
        ``<dJ/dz, dz>`` where ``(dF/dz) dz = -(dF/dp) dp`` from a *forward*
        linear solve, against ``-<lam, (dF/dp) dp>`` from the adjoint.  These
        are the same number computed through opposite sides of the identity,
        so agreeing certifies the whole assembly without a second forward
        solve anywhere.

    Part 4 deliberately does not compare against ``jax.grad``.  Every call to
    the transformed lane re-enters the forward solve, which warm-starts from
    ``_FREE_HOT_CACHE``, so a second call lands on a *different* root -- 4.7e-4
    apart in relative state norm here, which moves ``dJ/dI`` by 1.9e-3.  Both
    roots have exact adjoints; they are just not the same root, and a test that
    compared across them would be measuring root reproducibility while claiming
    to measure the adjoint.  That amplification is worth its own gate, and has
    one below.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    adjoint_tol = 1.0e-10
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=adjoint_tol, adjoint_maxiter=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )
    current = jnp.asarray(field.extcur)

    # One root for everything.  jax.grad below must differentiate the same
    # solve the factors are built from, so the host solve runs first and the
    # transformed lane picks it up from the hot cache.
    state, mask, rcon0, zcon0 = fbi._host_solve_and_mask(cfg, params, current)
    state = jax.tree.map(jnp.asarray, state)
    mask = jax.tree.map(jnp.asarray, mask)
    rcon0, zcon0 = jnp.asarray(rcon0), jnp.asarray(zcon0)
    project = im._dof_projector(cfg.implicit, mask)
    residual = fbi._projected_residual(cfg, mask)
    frozen = jax.lax.stop_gradient(state)
    z_star = project(state)

    def force_of_current(p):
        return residual(z_star, params, p, frozen, rcon0, zcon0)

    def force_of_state(z):
        return residual(z, params, current, frozen, rcon0, zcon0)

    # 1. dF/dp against a central difference of the residual.  The step is
    # relative to the coil current, and the residual is smooth in it.
    direction = jnp.zeros_like(current).at[0].set(1.0)
    _, jvp_p = jax.jvp(force_of_current, (current,), (direction,))
    step = 1.0e-3 * float(jnp.abs(current[0]))
    fd_p = jax.tree.map(
        lambda plus, minus: (plus - minus) / (2.0 * step),
        force_of_current(current + step * direction),
        force_of_current(current - step * direction),
    )
    ad_vec, fd_vec = _flat(jvp_p), _flat(fd_p)
    scale = float(jnp.linalg.norm(fd_vec))
    assert scale > 0.0
    assert float(jnp.linalg.norm(ad_vec - fd_vec)) / scale < 1.0e-6

    # 2. Transpose identity on the state linearization.
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    leaves, treedef = jax.tree.flatten(z_star)
    def random_like(key):
        parts = jax.random.split(key, len(leaves))
        return jax.tree.unflatten(treedef, [
            jax.random.normal(k, leaf.shape, leaf.dtype)
            for k, leaf in zip(parts, leaves)])
    u, v = project(random_like(keys[0])), project(random_like(keys[1]))
    _, jvp_z = jax.jvp(force_of_state, (z_star,), (u,))
    _, pullback = jax.vjp(force_of_state, z_star)
    jt_v = pullback(v)[0]
    left = float(jnp.dot(_flat(v), _flat(jvp_z)))
    right = float(jnp.dot(_flat(jt_v), _flat(u)))
    assert abs(left) > 0.0
    assert abs(left - right) / abs(left) < 1.0e-11

    # 3. The adjoint solve, by its own transpose residual.
    state_bar = jax.grad(
        lambda z: jnp.mean(z.R_cos[-1] ** 2 + z.Z_sin[-1] ** 2))(state)
    rhs = project(state_bar)
    lam = im._adjoint_solve_gcrot(
        lambda cotangent: pullback(cotangent)[0], rhs, cfg.implicit)[0]
    solve_residual = jax.tree.map(
        jnp.subtract, pullback(lam)[0], rhs)
    assert (float(jnp.linalg.norm(_flat(solve_residual)))
            / float(jnp.linalg.norm(_flat(rhs)))) < 1.0e3 * adjoint_tol

    # 4. Forward-adjoint duality on the same root.
    _, parameter_pullback = jax.vjp(
        lambda p: residual(z_star, params, p, frozen, rcon0, zcon0), current)
    assembled = parameter_pullback(jax.tree.map(jnp.negative, lam))[0]
    assert float(jnp.linalg.norm(assembled)) > 0.0
    adjoint_side = float(jnp.dot(assembled, direction))

    forward_rhs = jax.tree.map(jnp.negative, jvp_p)      # -(dF/dp) dp
    delta_z = im._adjoint_solve_gcrot(
        lambda tangent: jax.jvp(force_of_state, (z_star,), (tangent,))[1],
        forward_rhs, cfg.implicit)[0]
    forward_side = float(jnp.dot(_flat(rhs), _flat(delta_z)))
    assert abs(adjoint_side) > 0.0
    assert abs(forward_side - adjoint_side) / abs(adjoint_side) < 1.0e-6


@pytest.mark.full
def test_free_boundary_root_reproducibility_bounds_the_gradient():
    """Two entry points, two roots, and the gradient amplifies the gap.

    ``solve_free_boundary_implicit_status`` and ``_host_solve_and_mask`` solve
    the same problem, and the certificate above shows the adjoint of either
    root is exact to 3.5e-12.  They do not return the same root: measured
    4.7e-4 in relative state norm at ``ftol = 1e-7``, which moves
    ``dJ/dI`` by 1.9e-3 -- a 4x amplification.

    That is the accuracy limit of the free-boundary gradient, and it is the
    same class the fixed-boundary lane fixed by refining the returned state
    before linearizing: ``ftol`` gates a sum of squares, so a converged solve
    stops with ``|F| ~ sqrt(ftol)``, and where ``dF/dz`` has a small singular
    value that is a real displacement.  This test pins the amplification so a
    regression in either direction is visible; it is deliberately not a tight
    gate on the difference itself.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-10, adjoint_maxiter=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )
    current = jnp.asarray(field.extcur)

    status_root, *_ = solve_free_boundary_implicit_status(params, current, cfg)
    host_root, *_ = fbi._host_solve_and_mask(cfg, params, current)
    host_root = jax.tree.map(jnp.asarray, host_root)
    gap = float(jnp.linalg.norm(_flat(
        jax.tree.map(jnp.subtract, status_root, host_root))))
    scale = float(jnp.linalg.norm(_flat(status_root)))
    relative_gap = gap / scale
    # Both are "converged" by the same ftol; neither is wrong.
    assert 1.0e-6 < relative_gap < 1.0e-2, relative_gap


@pytest.mark.parametrize("bad", [0., float("nan")])
def test_reverse_gcrot_independently_rejects_false_success(monkeypatch, bad):
    """A claimed Krylov success must not hide an incorrect or nonfinite row."""
    cfg = SimpleNamespace(adjoint_tol=1e-11, adjoint_maxiter=10,
                          adjoint_gcrot_m=3, adjoint_gcrot_k=1)
    def residual(z, *args):
        return 2*z
    tape = fbi._prepare_reverse_transpose(jnp.zeros(3),None,None,None,None,None,residual=residual)
    native = fbi._solvax_gcrot
    def faulty(action, rhs, **kw):
        return native(action,rhs,**kw)._replace(x=jnp.full_like(rhs,bad),converged=jnp.array(True))
    fbi._reverse_gcrot_core.clear_cache()
    monkeypatch.setattr(fbi, '_solvax_gcrot', faulty)
    try:
        with pytest.raises(AdjointSolveError, match='reverse GCROT row 1'):
            fbi._solve_prepared_reverse_adjoint(tape,jnp.ones(3),cfg,row=1)
        if np.isfinite(bad):
            with pytest.warns(RuntimeWarning, match='best-effort'):
                fbi._solve_prepared_reverse_adjoint(tape,jnp.ones(3),cfg,fail='best_effort')
        else:
            with pytest.raises(AdjointSolveError):
                fbi._solve_prepared_reverse_adjoint(tape,jnp.ones(3),cfg,fail='best_effort')
        traced = jax.jit(lambda rhs:fbi._solve_prepared_reverse_adjoint(tape,rhs,cfg))(jnp.ones(3))
        assert np.all(np.isnan(traced))
    finally:
        fbi._reverse_gcrot_core.clear_cache()


@pytest.mark.parametrize("backend", ["forward_dense", "forward_dense_jax"])
@pytest.mark.parametrize("compiled", [False,True])
def test_dense_adjoint_dynamic_nonsymmetric_and_shared_factorization(backend, compiled, monkeypatch):
    from vmex.core import _freeboundary_dense as dense
    matrix = jnp.array([[4., .3, -.2], [.1, 3., .4], [.2, -.1, 5.]])
    cfg = SimpleNamespace(implicit=SimpleNamespace(lconm1=False,adjoint_tol=1e-11,
        adjoint_gcrot_m=3,adjoint_gcrot_k=1,adjoint_maxiter=10),adjoint_solver=backend,
        adjoint_fail='error',adjoint_dense_batch_size=2,adjoint_dense_max_dofs=100)
    def residual(z,p,f,base,rc,zc):return matrix@z+.5*(p+f+base+rc+zc)*z**2
    if compiled:residual=jax.jit(residual)
    values = [jnp.array([.2,.3,.4])+.1*i for i in range(6)]
    rhs=jnp.array([[1.,0.,0.],[.1,.5,1.],[0.,0.,0.]])
    native=dense._factor_solve;calls=[]
    def measured(*args):calls.append(1);return native(*args)
    monkeypatch.setattr(dense,'_factor_solve',measured)
    for changed in (None,0,1,2,3,4,5):
        args=list(values)
        if changed is not None:args[changed]=args[changed]+.25
        z,p,f,base,rc,zc=args
        jac=matrix+jnp.diag((p+f+base+rc+zc)*z)
        actual=dense.solve_dense_adjoint(residual,*args,rhs,jnp.ones(3),cfg)
        expected=np.linalg.solve(np.asarray(jac.T),np.asarray(rhs).T).T
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    assert len(calls)==7  # One factorization per root, not per RHS.


@pytest.mark.parametrize("lasym", [False,True])
def test_dense_active_basis_matches_main_projector_and_pair_signs(monkeypatch,lasym):
    from vmex.core import _freeboundary_dense as dense
    from vmex.core.solver import SpectralState
    cfg=SimpleNamespace(lconm1=True,resolution=SimpleNamespace(ntor=1,lasym=lasym))
    monkeypatch.setattr(im,'_m1_pair_columns',lambda _:(np.array([1]),np.array([2])))
    leaves=[jnp.ones((2,3)) for _ in range(6)]
    leaves[0]=leaves[0].at[0,0].set(0)
    mask=SpectralState(*leaves)
    space=dense._active_space(cfg,mask,100)
    _,unravel=jax.flatten_util.ravel_pytree(mask)
    vector=unravel(jnp.arange(36,dtype=jnp.float64))
    compressed=dense._compress(vector,space)
    expanded=dense._expand(compressed,mask,space)
    expected=im._dof_projector(cfg,mask)(vector)
    for a,b in zip(jax.tree.leaves(expanded),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(dense._compress(expanded,space),compressed,rtol=1e-14,atol=1e-14)
    assert space.left.size==35-(4 if lasym else 2)
    bad=dataclasses.replace(mask,Z_sin=mask.Z_sin.at[0,1].set(0))
    with pytest.raises(ValueError,match='unequal'):
        dense._active_space(cfg,bad,100)


@pytest.mark.parametrize("backend", ["forward_dense", "forward_dense_jax"])
@pytest.mark.parametrize("failure", ['singular','false_success','nonfinite'])
def test_dense_failure_policy(backend,failure,monkeypatch):
    from vmex.core import _freeboundary_dense as dense
    cfg=SimpleNamespace(implicit=SimpleNamespace(lconm1=False,adjoint_tol=1e-11,
        adjoint_gcrot_m=3,adjoint_gcrot_k=1,adjoint_maxiter=10),adjoint_solver=backend,
        adjoint_fail='error',adjoint_dense_batch_size=2,adjoint_dense_max_dofs=100)
    def residual(z,*args):return (0. if failure=='singular' else 2.)*z
    if failure!='singular':
        monkeypatch.setattr(dense,'_factor_solve',lambda matrix,rhs,backend:
                            jnp.full_like(rhs,jnp.nan if failure=='nonfinite' else 0.))
    def call():return dense.solve_dense_adjoint(residual,jnp.zeros(3),None,None,None,None,None,
                                              jnp.ones((2,3)),jnp.ones(3),cfg)
    with pytest.raises(AdjointSolveError):call()
    cfg.adjoint_fail='best_effort'
    if failure=='false_success':
        with pytest.warns(RuntimeWarning,match='best-effort'):call()
    else:
        with pytest.raises(AdjointSolveError):call()


def test_dense_dimension_and_mask_guards():
    from vmex.core import _freeboundary_dense as dense
    cfg=SimpleNamespace(lconm1=False)
    for mask in (jnp.zeros(3),jnp.ones(4)):
        with pytest.raises(ValueError,match='active dimension'):
            dense._active_space(cfg,mask,3)
    with pytest.raises(ValueError,match='binary'):
        dense._active_space(cfg,jnp.array([1.,.5]),3)


@pytest.mark.parametrize("name", ['adjoint_dense_batch_size','adjoint_dense_max_dofs'])
@pytest.mark.parametrize("value", [0,True,1.5])
def test_dense_config_rejects_invalid_controls(name,value):
    with pytest.raises(ValueError,match=name):
        make_free_boundary_config(lasym_free_input(DATA),lasym_free_field(),**{name:value})
