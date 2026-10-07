#!/usr/bin/env python
"""WOUT figures and alpha losses of a three-term single-stage run's final equilibrium.

    COIL_CASE=qa4-beta python benchmarks/three_term_postprocess.py <run dir> [--ns 201] [--bootstrap] [--trace-args "..."]

A run of ``examples/optimization/single_stage_free_boundary_optimization_three_term.py``. Its final boundary is the
sheet-current-free free boundary of its final coils; this re-solves it as a fixed boundary at ``--ns`` (the campaign's
dense radial resolution) with the run's PHIEDGE, pressure and current -- not with VMEC + NESTOR, which would leave that
branch. Writes into ``<run>/postprocess``: ``wout_dense.nc``, ``dense.json`` (QA, min/max |iota|, aspect, major radius,
bootstrap mismatch), the ``vmex.plot_wout`` figures and, unless ``--no-trace``, ``trace/`` (``vmex --trace`` alpha
losses, as ``coil_constraints_postprocess.py --trace``).
"""

import argparse
import json
import os
import subprocess
import shlex
import sys
from dataclasses import replace
from pathlib import Path

# The examples' COIL_CASE values this script reads: the target and the iota rows, and the Redl profiles.
CASE = os.environ.get("COIL_CASE", "ellipse5")
if CASE not in ("ellipse5", "ellipse5-beta7", "qa3", "qh", "qi", "qa4-beta", "qa4-beta-tok", "qi6-beta", "qi6-beta-tok"):
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
HELICITY = (1, -1) if CASE == "qh" else None if CASE.startswith("qi") else (1, 0)  # None: the constructed QI residual
IOTA_AXIS = CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta")  # iota rows include the extrapolated axis and edge
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
RESOLUTION = (8, 8, 51)            # the Redl surfaces are the half-grid surfaces at this NS
B0 = 1.0                           # T
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA = 32


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``HELICITY``, or constructed QI."""
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if HELICITY is None:
        return ConstructedQIResidual(np.asarray(QI_SURFACES), **QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), *HELICITY)


def redl_surfaces():
    """Surfaces of the Redl self-consistency check: ``REDL_SURFACES``, or by default every VMEC
    half-grid surface s = (j - 1/2) / (ns - 1) at the run's ns, as simsopt's ``RedlGeomVmec``."""
    import numpy as np

    if REDL_SURFACES is not None:
        return np.asarray(REDL_SURFACES)
    ns = int(RESOLUTION[2])
    return (np.arange(1, ns) - 0.5) / (ns - 1)


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    import numpy as np
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0, b0 = float(inp.rbc[inp.ntor, 0]), B0
    t0 = REACTOR_T0 * (b0 / REACTOR_B0) ** (2 / 3) * (r0 / REACTOR_R0) ** (1 / 3)
    n0 = REACTOR_N0 * (b0 / REACTOR_B0) ** (4 / 3) * (REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, 0, redl_surfaces(), n_lambda=REDL_N_LAMBDA)


def abs_iota(state, runtime):
    """|iota| bounded by the floor and ceiling rows.

    The half-mesh surfaces (axis slot excluded), as ``opt.min_abs_iota``; with
    ``IOTA_AXIS`` also VMEC's extrapolated axis and edge values (wout
    ``iotaf``), iotaf[0] = 1.5 iotas[1] - 0.5 iotas[2] and likewise at the edge.
    """
    import jax.numpy as jnp
    from vmex.core.statephysics import _iotas_half  # private: opt exposes only the half-mesh minimum

    half = _iotas_half(state, runtime)[1:]
    if IOTA_AXIS:
        half = jnp.concatenate([1.5 * half[:1] - 0.5 * half[1:2], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]])
    return jnp.abs(half)


def min_abs_iota(state, runtime):
    """Smallest |iota| of ``abs_iota``: ``opt.min_abs_iota``, or with ``IOTA_AXIS`` including the axis."""
    import jax.numpy as jnp

    return jnp.min(abs_iota(state, runtime))


def max_abs_iota(state, runtime):
    """Largest |iota| of ``abs_iota``, the counterpart of ``min_abs_iota``."""
    import jax.numpy as jnp

    return jnp.max(abs_iota(state, runtime))


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""
    import numpy as np

    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    return current_from_wout(replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)


def current_from_wout(inp, w):
    """``inp`` with ``w``'s CURTOR and current-profile coefficients, when ``inp`` prescribes the current."""
    import numpy as np

    if int(inp.ncurr) == 1:
        spline = "spline" in str(inp.pcurr_type)
        field_name, values = ("ac_aux_f", w.ac_aux_f) if spline else ("ac", w.ac)
        inp = replace(inp, curtor=float(w.ctor),
                      **{field_name: np.asarray(values, dtype=float)[: np.size(getattr(inp, field_name))]})
    return inp


def trace(wout, out, extra):
    """``vmex <wout> --trace`` in a fresh process, so it can give JAX one CPU device per core."""
    command = [sys.executable, "-m", "vmex", str(wout), "--trace", "--outdir", str(out / "trace"), *shlex.split(extra)]
    subprocess.run(command, check=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))


p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--ns", type=int, default=201)
p.add_argument("--bootstrap", action="store_true", help="report the Redl mismatch of the run's profiles")
p.add_argument("--no-trace", action="store_true")
p.add_argument("--trace-args", default="")
args = p.parse_args()
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np  # noqa: E402
import vmex  # noqa: E402
from vmex import optimize as opt  # noqa: E402

out = args.run.resolve() / "postprocess"
out.mkdir(exist_ok=True)
inp = restart_input(args.run)
deck = replace(inp, ns_array=np.array([args.ns]), ftol_array=np.array([1e-13]), niter_array=np.array([60000]))
eq = opt.solve_equilibrium(deck, initial_state=vmex.state_from_wout(vmex.read_wout(args.run / "wout.nc"), inp=deck,
                                                                     ns=args.ns))
w = eq.wout
path = out / "wout_dense.nc"
vmex.write_wout(str(path), w)
q = np.asarray(target_residual().residuals_state(eq.solution, eq.solver_context))
report = dict(ns=args.ns, fsq=float(w.fsqr + w.fsqz + w.fsql), qa=float(q @ q),
              min_abs_iota=float(min_abs_iota(eq.solution, eq.solver_context)),
              max_abs_iota=float(max_abs_iota(eq.solution, eq.solver_context)), aspect=float(w.aspect),
              major_radius_m=float(w.Rmajor_p), betatotal=float(w.betatotal))
if args.bootstrap:
    report["redl_mismatch"] = float(redl_profiles(deck)[1].total_state(eq.solution, eq.solver_context))
report["figures"] = [str(f) for f in vmex.plot_wout(str(path), out, name="three_term").values()]
(out / "dense.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if not args.no_trace:
    trace(path, out, args.trace_args)
print(f"wrote {out}", flush=True)
