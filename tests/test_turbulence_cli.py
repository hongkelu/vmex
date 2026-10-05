"""``vmex --turbulence``: helper identities (no gkx) and a tiny end-to-end smoke run (gkx)."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from vmex.core import gk_run
from vmex.core.cli import main

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_saturated_mean_is_window_mean_with_batch_error():
    t = np.linspace(0.0, 100.0, 401)
    q = np.where(t < 50.0, 0.0, 2.0 + 0.1 * np.sin(t))
    mean, err, t0 = gk_run.saturated_mean(t, q)
    assert t0 == 50.0
    assert mean == pytest.approx(q[t >= 50.0].mean())
    assert 0.0 < err < 0.1
    assert gk_run.saturated_mean(t, q, t_start=80.0)[2] == 80.0


def test_ky_scan_points_sit_on_the_gkx_grid(tmp_path):
    cfg = gk_run.TurbulenceSettings(ky_min=0.1, ky_max=1.0, nky=8)
    ky = gk_run.ky_grid(cfg)
    assert ky[0] == pytest.approx(0.1) and ky[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(ky / 0.1, np.rint(ky / 0.1), atol=1e-12)
    deck = tomllib.loads(gk_run.write_deck(tmp_path / "d.toml", tmp_path / "wout_x.nc", cfg,
                                           nonlinear=False).read_text())
    assert deck["grid"]["y0"] == pytest.approx(10.0)
    assert (deck["grid"]["Ny"] - 1) // 3 >= 10  # highest scan ky is a resolved mode
    assert deck["geometry"]["torflux"] == 0.5 and deck["physics"]["adiabatic_electrons"] is True
    kinetic = dataclasses.replace(cfg, kinetic_electrons=True)
    deck = tomllib.loads(gk_run.write_deck(tmp_path / "k.toml", tmp_path / "wout_x.nc", kinetic,
                                           nonlinear=True).read_text())
    assert [s["name"] for s in deck["species"]] == ["ion", "electron"]
    assert deck["physics"]["nonlinear"] is True and deck["physics"]["adiabatic_electrons"] is False


def test_turbulence_cli_smoke(tmp_path, capsys):
    pytest.importorskip("gkx", minversion="2.5.0")
    import vmex
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(DATA_DIR / "input.shaped_tokamak_pressure")
    inp = dataclasses.replace(inp, ns_array=np.array([13]), ftol_array=np.array([1e-12]),
                              niter_array=np.array([2000]))
    wout = vmex.write_wout(tmp_path / "wout_smoke.nc", opt.solve_equilibrium(inp).wout)
    rc = main([str(wout), "--turbulence", "--outdir", str(tmp_path),
               "--turbulence-ky", "0.2", "0.4", "2", "--turbulence-grid", "8", "8", "16",
               "--turbulence-moments", "2", "2", "--turbulence-tmax", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Linear ky scan" in out and "Turbulence summary" in out
    summary = json.loads((tmp_path / "smoke_turbulence.json").read_text())
    assert summary["ky"] == pytest.approx([0.2, 0.4])
    assert np.all(np.isfinite(summary["gamma"])) and np.all(np.isfinite(summary["heat_flux"]))
    for suffix in ("", "_geometry", "_fluxes"):
        assert (tmp_path / f"smoke_turbulence{suffix}.png").stat().st_size > 10_000
