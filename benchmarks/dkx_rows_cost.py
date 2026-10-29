"""Cost and memory of one in-loop DKX surface (``vmex.core.kinetic_bootstrap.KineticSurface``) on one GPU.

    COIL_CASE=qi6-beta-A8 python benchmarks/dkx_rows_cost.py DECK_OR_WOUT GRID... [--surfaces S...] [--out FILE]

GRID is ``NTHETA NZETA NXI NL NX MBOZ`` (e.g. 15 25 48 4 5 12).  For each surface: the ion-root refresh (compile
and Newton), a warm value (Newton + one solve), the value and gradient (one adjoint solve) and a central difference
of the value along a random spectrum direction against that gradient, with the device's peak memory after each.
The kinetic profiles are the campaign driver's (``--profiles stellaris``) for the deck's pressure.
"""

import argparse
import importlib.util
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location(
        "driver", ROOT / "examples/optimization/single_stage_free_boundary_optimization_three_term.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROFILES = "stellaris"
    return module


def peak():
    return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path, help="input.* deck or wout.nc (its boundary, pressure and current)")
    parser.add_argument("grid", type=int, nargs=6)
    parser.add_argument("--surfaces", type=float, nargs="+", default=[0.29, 0.54, 0.88])
    parser.add_argument("--er", type=float, nargs="+", help="E_r seeds per surface [kV/m] (default -2)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.boozer_tables import boozer_input_tables
    from vmex.core.kinetic_bootstrap import SPECTRUM, KineticSurface

    d = driver()
    deck = (vj.VmecInput.from_wout(str(args.deck)) if args.deck.suffix == ".nc"
            else vj.VmecInput.from_file(str(args.deck)))
    started = time.monotonic()
    eq = opt.solve_equilibrium(replace(deck, lfreeb=False), device="gpu", polish_force_balance=False)
    ns = int(np.asarray(eq.runtime.setup.s_full).size)
    print(f"equilibrium ns {ns} in {time.monotonic() - started:.0f} s", flush=True)
    profiles = d.redl_profiles(deck)[0]
    grid = (dict(zip(("Ntheta", "Nzeta", "Nxi", "NL", "Nx"), args.grid[:5])), args.grid[5])
    psi = jnp.abs(jnp.sum(jnp.asarray(eq.runtime.setup.phipf)[1:]) / (ns - 1))
    records = []
    for k, s in enumerate(args.surfaces):
        surface = KineticSurface(profiles, s, ns, grid=grid, er=(args.er or [-2.0] * len(args.surfaces))[k],
                                 collision_operator=d.DKX_COLLISION_OPERATOR,
                                 minor_radius=d.RADIUS_TARGET / np.mean(d.ASPECT_RANGE), er_tol=1e-3)
        table = boozer_input_tables(eq.state, eq.runtime, surface.row)
        surface.bind(table["xm"], table["xn"], int(eq.runtime.resolution.nfp),
                     int(np.sign(eq.runtime.setup.signgs)))
        spectrum = {name: jnp.asarray(table[name]) for name in SPECTRUM}
        record = dict(s=float(surface.s_row), grid=args.grid)
        t = time.monotonic()
        refreshed = surface.refresh(spectrum, psi)
        record.update(refresh_s=time.monotonic() - t, er=surface.er, held=surface.held, peak_refresh_gib=peak())
        er = surface.er
        t = time.monotonic()
        value = surface.evaluate(spectrum, psi)[0]
        record.update(value_s=time.monotonic() - t, value=value)
        for label in ("first", "warm"):
            surface.er = er
            t = time.monotonic()
            value, (g, g_psi) = surface.evaluate(spectrum, psi, gradient=True)
            record[f"gradient_{label}_s"] = time.monotonic() - t
        record["peak_gradient_gib"] = peak()
        rng = np.random.default_rng(k)
        direction = {n: jnp.asarray(rng.standard_normal(np.shape(v)) * 1e-3 * (np.abs(v) + 1e-6))
                     for n, v in spectrum.items()}
        predicted = sum(float(jnp.vdot(g[n], direction[n])) for n in SPECTRUM)

        def at(h):
            surface.er = er
            return surface.evaluate({n: v + h * direction[n] for n, v in spectrum.items()}, psi)[0]

        fd = {h: (at(h) - at(-h)) / (2 * h) for h in (1e-1, 3e-2)}
        record.update(gradient_directional=predicted, difference=fd,
                      relative={h: abs(predicted - v) / abs(v) for h, v in fd.items()},
                      scan=refreshed.get("scan"))
        records.append(record)
        print(json.dumps({key: v for key, v in record.items() if key != "scan"}), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(records, indent=1) + "\n")


if __name__ == "__main__":
    main()
