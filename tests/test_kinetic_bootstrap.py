"""DKX per-surface bootstrap rows (:mod:`vmex.core.kinetic_bootstrap`): the surface value is DKX's row at the same
E_r, its gradient (held E_r and implicit ion root) matches central differences, the rows' custom VJP matches a
traced reference, and remote workers return what local ones do."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

pytest.importorskip("dkx")
pytest.importorskip("booz_xform_jax")

from vmex.core import bootstrap as bs
from vmex.core import kinetic_bootstrap as kb
from vmex.core import optimize as opt
from vmex.core.boozer_tables import boozer_input_tables
from vmex.core.input import VmecInput

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
GRID = (dict(Ntheta=5, Nzeta=5, Nxi=8, NL=2, Nx=3), 4)
PROFILES = bs.KineticProfiles(ne_coeffs=2.0e20 * np.array([1, 0, 0, 0, 0, -1.0]),
                              Te_coeffs=4.0e3 * np.array([1, -1.0]), Ti_coeffs=4.0e3 * np.array([1, -1.0]))
OPTIONS = dict(grid=GRID, collision_operator=1, minor_radius=0.17, tol=1e-11, er_tol=1e-7)


@pytest.fixture(scope="module")
def eq():
    inp = VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres")
    inp = dataclasses.replace(inp, ns_array=np.array([16, 31]), ftol_array=np.array([1e-10, 1e-12]),
                              niter_array=np.array([2000, 4000]))
    equilibrium = opt.solve_equilibrium(inp)
    assert float(equilibrium.result.fsqr) < 1e-11
    return equilibrium


def _surface(eq, s, **kwargs):
    ns = int(np.asarray(eq.runtime.setup.s_full).size)
    surface = kb.KineticSurface(PROFILES, s, ns, **{**OPTIONS, **kwargs})
    table = boozer_input_tables(eq.state, eq.runtime, surface.row)
    surface.bind(table["xm"], table["xn"], int(eq.runtime.resolution.nfp), int(np.sign(eq.runtime.setup.signgs)))
    psi = jnp.abs(jnp.sum(jnp.asarray(eq.runtime.setup.phipf)[1:]) / (ns - 1))
    return surface, {k: jnp.asarray(table[k]) for k in kb.SPECTRUM}, psi


def test_held_value_is_dkx_row_at_that_field(eq):
    from dkx.bootstrap import KineticBootstrapMismatch

    surface, spectrum, psi = _surface(eq, 0.5, er=-1.5, hold=True)
    row = KineticBootstrapMismatch(PROFILES, [surface.s_row], resolution=GRID[0], mboz=GRID[1], nboz=GRID[1],
                                   collision_operator=1, minor_radius=0.17, er_kV_per_m=-1.5, tol=1e-11)
    reference = float(row.current_profiles(eq.state, eq.runtime)[2][0])
    value, _ = surface.evaluate(spectrum, psi)
    assert value == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("hold", [True, False])
def test_gradient_matches_central_differences(eq, hold):
    surface, spectrum, psi = _surface(eq, 0.5, er=-1.0, hold=hold)
    if not hold and surface.refresh(spectrum, psi)["hold"]:
        pytest.skip("no ion root on this grid")
    er = surface.er
    value, (g_spectrum, g_psi) = surface.evaluate(spectrum, psi, gradient=True)
    rng = np.random.default_rng(3)
    direction = {k: jnp.asarray(rng.standard_normal(np.shape(v)) * 1e-3 * (np.abs(v) + 1e-3))
                 for k, v in spectrum.items()}
    d_psi = 1e-3 * float(psi)
    predicted = sum(float(jnp.vdot(g_spectrum[k], direction[k])) for k in kb.SPECTRUM) + g_psi * d_psi

    def at(t):
        surface.er = er  # the same warm start on both sides
        return surface.evaluate({k: v + t * direction[k] for k, v in spectrum.items()}, psi + t * d_psi)[0]

    h = 1e-2
    differenced = (at(h) - at(-h)) / (2 * h)
    assert predicted == pytest.approx(differenced, rel=2e-4, abs=1e-6 * abs(value))


class _Identity:
    """Surfaces whose "kinetic" <j.B> is a smooth function of the spectrum: sum(bmnc^2) psi - c."""

    def __init__(self, s):
        self.specs = [dict(s=float(v)) for v in s]

    def bind(self, *args):
        pass

    def call(self, method, spectra, psi, gradient):
        out = {}
        for i in range(len(self.specs)):
            b = np.asarray(spectra["bmnc"][i])
            grads = {k: np.zeros_like(np.asarray(v[i])) for k, v in spectra.items()}
            grads["bmnc"] = 2 * b * psi
            out[i] = (float(np.sum(b * b) * psi), (grads, float(np.sum(b * b))) if gradient else None)
        return out


def test_rows_custom_vjp_matches_traced_reference(eq):
    ns = int(np.asarray(eq.runtime.setup.s_full).size)
    s = (np.array([3, 6, 9]) - 0.5) / (ns - 1)
    rows = kb.KineticCurrentRows(_Identity(s), ns, reference=1e4)

    def reference(state):
        spectra, psi = rows.spectra(state, eq.runtime)
        jk = jnp.sum(spectra["bmnc"] ** 2, axis=1) * psi
        hm = bs._half_mesh_fields(state, eq.runtime)
        jv = bs._jv_from_half(hm, hm.s_half)
        n = hm.s_half.shape[0]
        return (bs._current_derivative(hm, jv) - bs._current_derivative(hm, jnp.zeros(n).at[rows.index].set(jk))
                )[rows.index] / 1e4

    np.testing.assert_allclose(np.asarray(jax.jit(rows)(eq.state, eq.runtime)), np.asarray(reference(eq.state)),
                               rtol=1e-12)
    got = jax.jit(jax.jacrev(lambda st: rows(st, eq.runtime)))(eq.state)
    want = jax.jacrev(reference)(eq.state)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))


def test_rows_vanish_at_the_equilibrium_current(eq):
    """With the "kinetic" current the equilibrium's own <J.B>, the rows are zero."""
    ns = int(np.asarray(eq.runtime.setup.s_full).size)
    s = (np.array([4, 8]) - 0.5) / (ns - 1)
    hm = bs._half_mesh_fields(eq.state, eq.runtime)
    jv = np.asarray(bs._jv_from_half(hm, hm.s_half))
    rows = kb.KineticCurrentRows(_Identity(s), ns, reference=1e4)
    rows.delta = jv[rows.index] - rows._host(*_spectra(rows, eq), False)[0]
    np.testing.assert_allclose(np.asarray(rows(eq.state, eq.runtime)), 0.0, atol=1e-9)


def _spectra(rows, eq):
    spectra, psi = rows.spectra(eq.state, eq.runtime)
    return {k: np.asarray(v) for k, v in spectra.items()}, np.asarray(psi)


def test_remote_workers_match_local(eq, tmp_path):
    ns = int(np.asarray(eq.runtime.setup.s_full).size)
    specs = [dict(s=float((j - 0.5) / (ns - 1)), er=-1.0, hold=True) for j in (5, 9, 12)]
    options = dict(profiles=PROFILES, ns=ns, surface=OPTIONS)
    local = kb.KineticCurrentRows(kb.KineticSurfaces.local(specs, options), ns, reference=1e4)
    want = np.asarray(local(eq.state, eq.runtime))
    rendezvous = tmp_path / "rendezvous.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    workers = [subprocess.Popen([sys.executable, "-m", "vmex.core.kinetic_bootstrap", str(rendezvous)], env=env)
               for _ in range(2)]
    try:
        surfaces = kb.KineticSurfaces.remote(specs, options, rendezvous, workers=2)
        remote = kb.KineticCurrentRows(surfaces, ns, reference=1e4)
        np.testing.assert_allclose(np.asarray(remote(eq.state, eq.runtime)), want, rtol=1e-10)
        got = jax.jacrev(lambda st: remote(st, eq.runtime))(eq.state)
        ref = jax.jacrev(lambda st: local(st, eq.runtime))(eq.state)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(ref)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-12 * np.max(np.abs(b)))
        assert [r["er"] for r in surfaces.records()] == [-1.0] * 3
        surfaces.stop()
    finally:
        for w in workers:
            w.wait(timeout=60)
