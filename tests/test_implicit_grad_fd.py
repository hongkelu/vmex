"""Implicit-gradient validations split from ``test_implicit_grad.py``.

They live in their own module only so the pull-request lanes stay inside
their time cap; helpers, fixtures, and the FD cache are shared with the
parent module, whose docstring explains the FD step choices.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tests import test_implicit_grad as _base
from tests.test_implicit_grad import CASES, DATA_DIR, _fd
from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.statephysics import mean_iota

# Shared fixtures register here by name.  Assignment rather than import keeps
# the test parameters that request them from shadowing an imported name.
_release_jax_caches_in_full_matrix = _base._release_jax_caches_in_full_matrix
solovev = _base.solovev


# ---------------------------------------------------------------------------
# 5. one 3D case: li383, d(wb)/d(boundary coefficient) vs FD
# ---------------------------------------------------------------------------


def test_li383_boundary_gradient_vs_fd():
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])
    fd = _fd(name, inp, cfg, p0, "rbc", (ntor, 1), 4e-4)["wb"]
    rel = abs(ad / fd - 1.0)
    print(f"\n[{name}] d(wb)/d(RBC(0,1)) h=4e-4: AD={ad:+.10e} FD={fd:+.10e} "
          f"rel={rel:.2e} (FD noise floor ~3e-5)")
    assert rel <= 2e-4


# ---------------------------------------------------------------------------
# 6. iteration-policy independence + memory sanity (informational)
# ---------------------------------------------------------------------------


def test_gradient_independent_of_iteration_policy(solovev):
    """Only the fixed point defines the derivative; RSS prints are the
    O(1)-memory sanity (backward = a few residual linearizations, no tape)."""
    import resource

    name, inp, cfg, p0, _, _, _ = solovev
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # noqa: E731

    grads = {}
    r0 = rss()
    for cap in (500, 5000):
        grads[cap] = jax.grad(
            lambda p: im.run(inp, p, ftol=cfg.ftol, max_iterations=cap).wb)(p0)
        print(f"\n[{name}] grad(wb) with max_iterations={cap}: "
              f"peak RSS delta so far = {rss() - r0:.0f} MB")

    a = np.asarray(grads[500].rbc)
    b = np.asarray(grads[5000].rbc)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-14)
    assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# 7. solver-sensitive metric (iota_edge, ncurr=1): the implicit adjoint equals
#    the FROZEN-PATH FD; a naive full re-solve FD is NOT a valid reference.
# ---------------------------------------------------------------------------


def test_iota_edge_gradient_vs_frozen_path_fd():
    """``d(iota_edge)/d(boundary)`` on the 3D ``ncurr=1`` case: the implicit
    adjoint must equal the frozen-path central FD to solver accuracy.  A naive
    re-solve FD is deliberately NOT the reference (measured, ``h=1e-4``):

        RBC(n=-1,m=1):  adjoint -0.77343,  frozen-path FD -0.77343,  naive FD +0.04543  (sign flip)
        RBC(n=+1,m=1):  adjoint -1.26567,  frozen-path FD -1.26567,  naive FD -2.14135  (69% off)
    """
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    assert int(inp.ncurr) == 1  # derived-iota case where the effect is largest

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).iota_edge)(p0)
    zero = jax.tree.map(jnp.zeros_like, p0)
    print(f"\n[{name}] d(iota_edge)/d(RBC): implicit adjoint vs frozen-path FD")
    for n in (ntor - 1, ntor + 1):          # m=1, n=-1 (naive-FD sign flip) and n=+1
        ad = float(np.asarray(grad.rbc)[n, 1])
        tangent = dataclasses.replace(zero, rbc=zero.rbc.at[n, 1].set(1.0))
        fd, info = im.frozen_path_directional_fd(
            p0, cfg, im.iota_edge, tangent, h=1e-4)
        res = max(info["newton_res"])
        rel = abs(ad / fd - 1.0) if fd else abs(ad - fd)
        print(f"  RBC(n={n - ntor:+d},m=1): AD={ad:+.9e}  frozen-FD={fd:+.9e}  "
              f"rel={rel:.2e}  (Newton res {res:.0e})")
        assert res < 1e-8, f"n={n - ntor}: frozen solve not converged (res {res:.1e})"
        assert rel <= 3e-4, (
            f"RBC(n={n - ntor},1): adjoint {ad:.5e} vs frozen-path FD {fd:.5e} "
            f"(rel {rel:.2e}) — the implicit gradient must match the frozen path")


def _li383_mean_iota_grad(m1_complement):
    """``d(mean iota)/d(RBC(1,1), ZBS(-1,1))`` (implicit) and the refined state on li383_low_res."""
    inp = VmecInput.from_file(str(DATA_DIR / "input.li383_low_res"))
    cfg = im.make_config(inp, **CASES["li383_low_res"], m1_complement=m1_complement)
    p0 = im.params_from_input(inp)
    grad = jax.grad(lambda p: mean_iota(im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg)))(p0)
    ntor = int(inp.ntor)
    x, _ = im.solve_implicit_with_aux(p0, cfg)
    return np.array([float(grad.rbc[ntor + 1, 1]), float(grad.zbs[ntor - 1, 1])]), np.asarray(x.Z_sin)


def test_m1_policy_is_part_of_the_config_identity():
    """K -> frozen -> K in one process, no cache clearing: each config solves and differentiates its own equations
    (the policy is a field of ImplicitConfig, so compiled residuals and memoized solves cannot cross over), and the
    frozen values equal a fresh process's."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    k1, zk1 = _li383_mean_iota_grad("solved")
    f1, zf1 = _li383_mean_iota_grad("frozen")
    k2, zk2 = _li383_mean_iota_grad("solved")
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(zk1, zk2)
    assert np.max(np.abs(f1 / k1 - 1.0)) > 1e-3  # different equations, different derivatives (0.9% / 2.5%)
    script = ("import sys; sys.path.insert(0, 'tests'); import numpy as np; "
              "import json; from test_implicit_grad_fd import _li383_mean_iota_grad; "
              "g, z = _li383_mean_iota_grad('frozen'); print('grad', json.dumps(g.tolist()))")
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, check=True)
    fresh = np.array(json.loads(done.stdout.split("grad")[-1].strip()))
    np.testing.assert_allclose(f1, fresh, rtol=1e-9)
    with pytest.raises(ValueError, match="m1_complement"):
        im.make_config(VmecInput.from_file(str(DATA_DIR / "input.li383_low_res")), m1_complement="gauge")


def test_li383_mean_iota_matches_resolve_fd_with_the_m1_complement_solved():
    """Mean iota: with the m=1 complement solved at the root (``ImplicitConfig.m1_complement``,
    the default) no entry is frozen at a path-dependent value, so the implicit
    derivative IS an independent re-solve FD's -- the gap the frozen treatment
    leaves (1.0% / 2.8%, next test) closes without any correction."""
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    grad = jax.grad(lambda p: mean_iota(
        im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg)))(p0)
    x0, mask = im.solve_implicit_with_aux(p0, cfg)
    project = im._dof_projector(cfg, mask)
    edge = im._edge_mask(cfg)

    def frozen_part(x):
        return jax.tree.map(lambda a, b, e: (a - b) * (1.0 - e), x, project(x), edge)

    h = 5e-4
    zero = jax.tree.map(jnp.zeros_like, p0)
    for field, n in (("rbc", 1), ("zbs", -1)):
        tangent = dataclasses.replace(zero, **{field: getattr(zero, field).at[ntor + n, 1].set(1.0)})
        implicit = float(sum(jnp.vdot(a, b) for a, b in zip(
            jax.tree.leaves(grad), jax.tree.leaves(tangent))))
        values = []
        for sign in (1.0, -1.0):
            p = jax.tree.map(lambda a, d, s=sign: a + s * h * d, p0, tangent)
            x, _ = im.solve_implicit_with_aux(p, cfg)
            # nothing outside the evolved dofs and the edge moves (to round-off)
            assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(
                jax.tree.leaves(frozen_part(x)), jax.tree.leaves(frozen_part(x0)))) < 1e-14
            values.append(float(mean_iota(x, im.runtime_from_params(p, cfg))))
        resolve = (values[0] - values[1]) / (2 * h)
        gap = abs(implicit / resolve - 1.0)
        print(f"\n[{name}] d(mean iota)/d({field.upper()}(n={n:+d},m=1)): "
              f"implicit {implicit:+.6e} re-solve {resolve:+.6e} (gap {gap:.1e})")
        assert gap < 1e-4


def test_li383_mean_iota_resolve_fd_gap_is_the_m1_constrained_family():
    """Mean iota: the implicit derivative against an independent re-solve FD,
    with the m=1 complement frozen as VMEC2000 leaves it (``m1_complement=
    "frozen"``).

    The two differ, and by more than solver accuracy (1.0% on ``RBC(1,1)``,
    2.8% on ``ZBS(-1,1)`` at ftol 1e-13).  The whole gap is the released
    m=1 ``Z_sin`` pair combination (``residue.f90`` zeroes its force once
    ``fsqz < 1e-6``).  The linearization holds it at its converged value,
    while each cold re-solve freezes it wherever its own path left it, at a
    rate of about 0.5 per unit boundary change.  Iota depends on that
    angle-gauge coordinate only through truncation error, so neither answer
    is wrong.  The implicit value is the exact derivative with the gauge held
    fixed, and the re-solve adds the gauge drift of the solver's path.

    Checked here as a closure: the re-solve FD must equal the implicit
    derivative plus the iota response to moving only that family by the
    re-solves' own drift.  Measured closures: 2.3e-6 and 1.6e-5.
    """
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name], m1_complement="frozen")
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    assert int(inp.ncurr) == 1

    def metric(state, runtime):
        return mean_iota(state, runtime)

    grad = jax.grad(lambda p: metric(
        im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg)))(p0)
    x0, mask = im.solve_implicit_with_aux(p0, cfg)
    project = im._dof_projector(cfg, mask)
    edge = im._edge_mask(cfg)

    def constrained(x):
        # Entries that are neither evolved nor set by the boundary.
        return jax.tree.map(
            lambda a, b, e: (a - b) * (1.0 - e), x, project(x), edge)

    def norm(tree):
        return float(jnp.sqrt(sum(jnp.vdot(v, v) for v in jax.tree.leaves(tree))))

    def root_with(frozen):
        # Newton on the frozen residual at p0, only the constrained family moved.
        residual = im.residual_fn(cfg, frozen, mask)
        z = project(x0)
        for _ in range(20):
            fz = residual(z, p0)
            if norm(fz) <= 1e-12:
                break
            _, jvp = jax.linearize(lambda zz: residual(zz, p0), z)
            delta, _ = im._adjoint_solve_gcrot(jvp, fz, cfg, enforce=False)
            z = jax.tree.map(jnp.subtract, z, delta)
        assert norm(residual(z, p0)) < 1e-10
        runtime = im.runtime_from_params(p0, cfg)
        return float(metric(im._assemble(z, runtime, frozen, project, edge), runtime))

    h = 5e-4
    zero = jax.tree.map(jnp.zeros_like, p0)
    for field, n in (("rbc", 1), ("zbs", -1)):
        column = getattr(zero, field).at[ntor + n, 1].set(1.0)
        tangent = dataclasses.replace(zero, **{field: column})
        implicit = float(sum(jnp.vdot(a, b) for a, b in zip(
            jax.tree.leaves(grad), jax.tree.leaves(tangent))))
        states, values = [], []
        for sign in (1.0, -1.0):
            p = jax.tree.map(lambda a, d, s=sign: a + s * h * d, p0, tangent)
            x, _ = im.solve_implicit_with_aux(p, cfg)
            states.append(x)
            values.append(float(metric(x, im.runtime_from_params(p, cfg))))
        resolve = (values[0] - values[1]) / (2 * h)
        drift = jax.tree.map(lambda a, b: (a - b) / (2 * h),
                             constrained(states[0]), constrained(states[1]))
        # The re-solves move the constrained family and nothing else frozen.
        assert norm(drift) > 0.1
        assert norm(drift) == pytest.approx(float(jnp.linalg.norm(drift.Z_sin)))
        gauge = (root_with(jax.tree.map(lambda a, d: a + h * d, x0, drift))
                 - root_with(jax.tree.map(lambda a, d: a - h * d, x0, drift))) / (2 * h)
        gap = abs(implicit / resolve - 1.0)
        closure = abs((implicit + gauge) / resolve - 1.0)
        print(f"\n[{name}] d(mean iota)/d({field.upper()}(n={n:+d},m=1)): "
              f"implicit {implicit:+.6e} re-solve {resolve:+.6e} (gap {gap:.1e}), "
              f"m=1 family term {gauge:+.6e}, closure {closure:.1e}")
        assert closure < 1e-4


# ---------------------------------------------------------------------------
# 8. typed errors through pure_callback (zero-crash policy)
# ---------------------------------------------------------------------------


def test_typed_error_through_pure_callback():
    """A failing host solve raises the SHORT typed exception, not callback
    noise: the ``_HOST_ERROR`` relay stashes the typed exception in the host
    callback and re-raises it at the ``pure_callback`` site with ``from
    None`` (previously a ~3.7 KB raw ``JaxRuntimeError``)."""
    from vmex.core.errors import VmecConvergenceError

    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    with pytest.raises(VmecConvergenceError) as excinfo:
        im.run(inp, ftol=1e-14, max_iterations=3)
    exc = excinfo.value
    assert len(str(exc)) < 200, f"typed message must stay short: {len(str(exc))} chars"
    assert "MORE ITERATIONS REQUIRED" in str(exc)
    assert exc.__cause__ is None and exc.__suppress_context__  # noise killed
    assert exc.ftol == 1e-14  # diagnostics preserved

    # the custom-vjp forward rule call site relays the typed exception too
    p0 = im.params_from_input(inp)
    with pytest.raises(VmecConvergenceError):
        jax.grad(lambda p: im.run(inp, p, ftol=1e-14, max_iterations=3).wb)(p0)


# ---------------------------------------------------------------------------
# 9. multigrid implicit gradient vs frozen-path FD
# ---------------------------------------------------------------------------


def test_multigrid_gradient_vs_frozen_path_fd():
    """``im.run(multigrid=True)`` through a genuine ns 5 -> 11 solovev ladder:
    the adjoint must match the frozen-path central FD of ``wb`` along
    ``RBC(0,1)`` (only the final fixed point defines the derivative)."""
    inp0 = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    inp = dataclasses.replace(
        inp0,
        ns_array=np.array([5, 11]),
        ftol_array=np.array([1e-10, 1e-14]),
        niter_array=np.array([1000, 2000]),
    )
    cfg = im.make_config(inp, multigrid=True, ftol=1e-14)
    assert cfg.multigrid and int(cfg.resolution.ns) == 11
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, multigrid=True, ftol=1e-14).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])

    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent = dataclasses.replace(zero, rbc=zero.rbc.at[ntor, 1].set(1.0))
    fd, info = im.frozen_path_directional_fd(p0, cfg, lambda s, rt:
                                             im.mhd_energy(s, rt)[0],
                                             tangent, h=3e-5)
    res = max(info["newton_res"])
    rel = abs(ad / fd - 1.0)
    print(f"\n[solovev multigrid ns=5->11] d(wb)/d(RBC(0,1)) h=3e-5: "
          f"AD={ad:+.10e}  frozen-FD={fd:+.10e}  rel={rel:.2e} "
          f"(Newton res {res:.0e})")
    assert res < 1e-8, f"frozen solve not converged (res {res:.1e})"
    assert rel <= 1e-6


def test_status_certifies_the_refined_root_not_the_host_solve(monkeypatch):
    """The status lane certifies the returned state in the config's equations (the solved m=1 complement
    included): a successful refinement is status 0 with its residual recorded under refine_tol; a refinement that
    returns the host state (a moderate stall: the host solve is converged, the expanded equations are not) and a
    non-finite refined state are status 4, not a certified derivative point."""
    inp = VmecInput.from_file(str(DATA_DIR / "input.li383_low_res"))
    cfg = im.make_config(inp, **CASES["li383_low_res"])
    params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), im.params_from_input(inp))

    def status():
        im._LAST_REFINED.pop(cfg, None)
        im._LAST_REFINEMENT_CORRECTION.pop(cfg, None)
        return int(im._host_solve_and_mask_status(cfg, params_np)[2])

    assert status() == 0
    assert im._LAST_REFINED[cfg][3] <= cfg.refine_tol
    original = im._refined_state
    monkeypatch.setattr(im, "_refined_state", lambda cfg, params, state, mask, **kw: state)
    assert status() == 4  # the frozen host root does not solve the complement's equation
    assert im._LAST_REFINED[cfg][3] > cfg.refine_tol
    monkeypatch.setattr(im, "_refined_state", lambda cfg, params, state, mask, **kw: jax.tree.map(
        lambda a: a * np.nan if jnp.issubdtype(a.dtype, jnp.inexact) else a, original(cfg, params, state, mask)))
    assert status() == 4
    assert im._LAST_REFINED[cfg][3] == np.inf
    with pytest.raises(ValueError, match="finite refine_tol"):
        im.make_config(inp, refine_tol=np.inf)
    im.make_config(inp, refine_tol=np.inf, m1_complement="frozen")  # the unrefined host root is the frozen one


def test_li383_mean_iota_matches_resolve_fd_with_the_m1_complement_gauged():
    """Mean iota with the m=1 complement gauged (``m1_complement="gauged"``: fixed to ``sqrt(s)`` times its edge
    value, a function of the boundary): the refined state depends on the parameters alone, so the implicit derivative
    is an independent re-solve FD's, as with ``"solved"`` -- without solving the complement's near-singular force
    (``"solved"`` stalls at 12 x 12)."""
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name], m1_complement="gauged")
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    grad = jax.grad(lambda p: mean_iota(
        im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg)))(p0)
    x0, _ = im.solve_implicit_with_aux(p0, cfg)
    pos, neg = im._m1_pair_columns(cfg)
    edge_anti = 0.5 * (np.asarray(x0.Z_sin)[-1, pos] - np.asarray(x0.Z_sin)[-1, neg])
    anti = 0.5 * (np.asarray(x0.Z_sin)[:, pos] - np.asarray(x0.Z_sin)[:, neg])
    s = np.linspace(0.0, 1.0, anti.shape[0])
    np.testing.assert_allclose(anti, np.sqrt(s)[:, None] * edge_anti[None, :], atol=1e-14)
    h = 5e-4
    zero = jax.tree.map(jnp.zeros_like, p0)
    for field, n in (("rbc", 1), ("zbs", -1)):
        tangent = dataclasses.replace(zero, **{field: getattr(zero, field).at[ntor + n, 1].set(1.0)})
        implicit = float(sum(jnp.vdot(a, b) for a, b in zip(
            jax.tree.leaves(grad), jax.tree.leaves(tangent))))
        values = []
        for sign in (1.0, -1.0):
            p = jax.tree.map(lambda a, d, s=sign: a + s * h * d, p0, tangent)
            x, _ = im.solve_implicit_with_aux(p, cfg)
            values.append(float(mean_iota(x, im.runtime_from_params(p, cfg))))
        resolve = (values[0] - values[1]) / (2 * h)
        gap = abs(implicit / resolve - 1.0)
        print(f"\n[{name}, gauged] d(mean iota)/d({field.upper()}(n={n:+d},m=1)): "
              f"implicit {implicit:+.6e} re-solve {resolve:+.6e} (gap {gap:.1e})")
        assert gap < 1e-4
