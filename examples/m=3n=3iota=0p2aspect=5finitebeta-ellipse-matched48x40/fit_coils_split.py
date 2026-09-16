"""Finite-beta coil fit: fixed WOUT, virtual casing, and the original 111 DOFs."""

from pathlib import Path
import os
import sys
import json
import hashlib
import time
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]
CASE = ROOT
DEPENDENCIES = Path("/root/autodl-tmp/vmex-finitebeta-ellipse-matched48x40-lhk3-6-20260914T174815Z/dependencies")
ESSOS = "/root/autodl-tmp/vmex-m3n3-iota02-aspect5-angular48x40-20260914/external/ESSOS"
GPU = "GPU-0b009025-c499-c2ad-030f-91e852095542"
OUT = ROOT / "coils_split"
assert SOURCE.is_dir() and not OUT.exists()
assert json.loads((ROOT / "fixed/summary.json").read_text())["qualified_fixed_target"]
assert GPU not in subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
)
OUT.mkdir()
for key, name in [
    ("TMPDIR", "tmp"),
    ("MPLCONFIGDIR", "mpl"),
    ("XDG_CACHE_HOME", "xdg"),
    ("JAX_COMPILATION_CACHE_DIR", "jax"),
    ("CUDA_CACHE_PATH", "cuda"),
]:
    p = OUT / "cache" / name
    p.mkdir(parents=True)
    os.environ[key] = str(p)
os.environ.update(
    CUDA_VISIBLE_DEVICES=GPU,
    JAX_ENABLE_X64="1",
    JAX_PLATFORMS="cuda,cpu",
    PYTHONDONTWRITEBYTECODE="1",
    XLA_PYTHON_CLIENT_PREALLOCATE="false",
)
sys.path[:0] = [str(SOURCE), str(CASE), ESSOS, str(DEPENDENCIES / "vendor")]
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from case import Full111FieldBuilder, DirectCoilField
from initialization import verify_runtime
from essos.coils import Coils, Curves
from vmex.core.wout import read_wout
from vmex.core.input import VmecInput
from vmex.core.virtual_casing import surface_field_data_from_wout, plasma_field_on_boundary, MU0  # noqa: F401

verify_runtime()
for n, h in json.loads((DEPENDENCIES / "manifest.json").read_text())["files"].items():
    assert hashlib.sha256((DEPENDENCIES / "vendor" / n).read_bytes()).hexdigest() == h
for n, h in json.loads((ROOT / "inputs/manifest.json").read_text()).items():
    assert hashlib.sha256((ROOT / "inputs" / n).read_bytes()).hexdigest() == h
fixed = json.loads((ROOT / "fixed/summary.json").read_text())
wpath = ROOT / "fixed/wout.nc"
assert hashlib.sha256(wpath.read_bytes()).hexdigest() == fixed["wout_sha256"]
w = read_wout(wpath)
assert w.ns == 50 and abs(float(w.aspect) - 5.0) <= 1e-9
assert abs(float(w.Rmajor_p) - 11.067033897730942) <= 1e-8
assert abs(float(np.mean(np.asarray(w.iotas)[1:])) - 0.2) <= 1e-5
assert abs(float(w.b0) - 5.1) <= 1e-4
inp = VmecInput.from_file(ROOT / "fixed/input.fixed.json")
assert (inp.ntheta, inp.nzeta) == (48, 40) and (fixed["ntheta"], fixed["nzeta"]) == (48, 40)
raw = json.loads((ROOT / "inputs/coils_seed.json").read_text())
builder = Full111FieldBuilder(jnp.asarray(raw["dofs_curves"]), jnp.asarray(raw["dofs_currents"]))
R = float(w.Rmajor_p)
scales = jnp.asarray(
    np.r_[
        np.full(3, 0.25),
        np.tile(
            [
                0.02 * R,
                0.02 * R,
                0.02 * R,
                0.02 * R / 4,
                0.02 * R / 4,
                0.02 * R / 9,
                0.02 * R / 9,
                0.02 * R / 16,
                0.02 * R / 16,
            ],
            12,
        ),
    ]
)
assert scales.shape == (111,)
pressure_edge = float(inp.pres_scale * np.sum(inp.am))
started = time.monotonic()


def write(name, data):
    with (OUT / name).open("x") as f:
        f.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def event(phase, **data):
    line = json.dumps(dict(phase=phase, elapsed_s=time.monotonic() - started, **data), allow_nan=False)
    with (OUT / "progress.jsonl").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def coil_set(u, segments=128):
    params = jnp.asarray(u) * scales
    return Coils(Curves(builder.curve_dofs_at(params), segments, 2, True), builder.base_currents_at(params))


def field_cart(coils, sd):
    xyz = jnp.moveaxis(sd.gamma, 0, -1)
    phi = sd.phi[:, None]
    field = DirectCoilField(coils.gamma, coils.gamma_dash, coils.currents)
    br, bp, bz = field.b_cyl(jnp.linalg.norm(xyz[..., :2], axis=-1), phi, xyz[..., 2])
    return jnp.stack([br * jnp.cos(phi) - bp * jnp.sin(phi), br * jnp.sin(phi) + bp * jnp.cos(phi), bz])


def target(grid, digits, quad_nt, quad_np):
    assert grid == 48
    assert quad_nt % (2 * 40) == 0 and quad_np % 48 == 0
    event("virtual_casing_start", grid=grid, digits=digits)
    sd = surface_field_data_from_wout(w, nphi=40, ntheta=48)
    from virtual_casing_jax import VirtualCasingExteriorField, ExteriorFieldConfig

    vc = VirtualCasingExteriorField(
        sd,
        ExteriorFieldConfig(
            digits=digits, levels=((48, 40), (96, 80)), chunk_size=128, target_chunk_size=8, dtype="float64"
        ),
    )
    plasma = vc._vc.compute_internal_B(
        vc.B_total, digits=digits, quad_nt=quad_nt, quad_np=quad_np, chunk_size=128, target_chunk_size=8
    )
    assert np.isfinite(np.asarray(plasma)).all()
    area = jnp.linalg.norm(sd.area_vector, axis=0)
    weights = area / jnp.sum(area)
    bin2 = jnp.sum(sd.B_total**2, axis=0)
    assert float(jnp.min(bin2)) > 0
    tangent = float(jnp.max(jnp.abs(jnp.sum(sd.B_total * sd.normal, axis=0)) / jnp.sqrt(bin2)))
    assert tangent < 1e-10
    np.savez_compressed(
        OUT / f"target_g{grid:03d}_d{digits}_q{quad_nt}_{quad_np}.npz",
        gamma=np.asarray(sd.gamma),
        normal=np.asarray(sd.normal),
        B_total=np.asarray(sd.B_total),
        B_plasma=np.asarray(plasma),
        weights=np.asarray(weights),
        theta=np.asarray(sd.theta),
        phi=np.asarray(sd.phi),
        pressure_edge_Pa=pressure_edge,
    )
    event(
        "virtual_casing_end",
        grid=grid,
        digits=digits,
        plasma_Bn_rms_T=float(jnp.sqrt(jnp.sum(weights * jnp.sum(plasma * sd.normal, axis=0) ** 2))),
        interior_tangency_max=tangent,
    )
    return sd, plasma, weights, bin2


sd, plasma, weights, bin2 = target(48, 5, 800, 144)
points = jnp.moveaxis(sd.gamma, 0, -1)[::2, ::2].reshape(-1, 3)
zero = jnp.zeros(111)
orientation_scores = []
for reflected in (False, True):
    curve = np.asarray(raw["dofs_curves"]).copy()
    if reflected:
        curve[:, 1, :] *= -1
    builder = Full111FieldBuilder(jnp.asarray(curve), jnp.asarray(raw["dofs_currents"]))
    coil = coil_set(zero)
    total = field_cart(coil, sd) + plasma
    score = float(
        jnp.sum(weights * jnp.sum(total * sd.normal, axis=0) ** 2 / bin2)
        + 0.25
        * jnp.sum(
            weights
            * ((jnp.sum(total**2, axis=0) - bin2 - 2 * MU0 * pressure_edge) / (bin2 + 2 * MU0 * pressure_edge)) ** 2
        )
    )
    assert np.isfinite(score)
    orientation_scores.append((score, reflected))
reflected = min(orientation_scores)[1]
curve = np.asarray(raw["dofs_curves"]).copy()
if reflected:
    curve[:, 1, :] *= -1
builder = Full111FieldBuilder(jnp.asarray(curve), jnp.asarray(raw["dofs_currents"]))
seed = coil_set(zero)
write(
    "seed_orientation.json",
    dict(reflected_y=reflected, scores=[dict(score=s, reflected_y=r) for s, r in orientation_scores]),
)
write(
    "coils_seed_selected.json",
    dict(dofs_curves=curve.tolist(), dofs_currents=raw["dofs_currents"], n_segments=256, nfp=2, order=4, stellsym=True),
)


DISTANCE_SMOOTHING_M = 1e-3


def sampled_gaps(coils, *, smooth=True):
    from jax.scipy.special import logsumexp

    g = coils.gamma
    d = jnp.sqrt(jnp.sum((g[:, :, None, None, :] - g[None, None, :, :, :]) ** 2, axis=-1) + 1e-24)
    cp_d = jnp.sqrt(jnp.sum((g[:, :, None, :] - points[None, None, :, :]) ** 2, axis=-1) + 1e-24)
    if smooth:
        pair = -DISTANCE_SMOOTHING_M * logsumexp(-d / DISTANCE_SMOOTHING_M, axis=(1, 3))
        cp = -DISTANCE_SMOOTHING_M * logsumexp(-cp_d / DISTANCE_SMOOTHING_M, axis=(1, 2))
    else:
        pair = jnp.min(d, axis=(1, 3))
        cp = jnp.min(cp_d, axis=(1, 2))
    pair = pair + jnp.eye(g.shape[0]) * 1e6
    return pair, cp


cc0, cp0 = sampled_gaps(seed)
limits = dict(
    max_length_m=71.58777355100058,
    max_curvature_per_m=0.6856318365509855,
    min_coil_coil_m=1.1558637264840923,
    min_coil_plasma_m=2.3171028264479103,
)
assert min(limits.values()) > 0
settings = dict(
    distance_smoothing_m=DISTANCE_SMOOTHING_M,
    distance_penalty="Conservative unnormalized logsumexp soft minimum; exact sampled minima retained for acceptance",
    geometry_penalty_sampling=128,
    geometry_guard_margins=dict(coil_coil_m=0.25, coil_plasma_m=0.15, length_factor=0.99, curvature_factor=0.98),
    dofs=111,
    shape_dofs=108,
    current_groups=[1, 2, 3],
    fixed_base_current_0_A=float(builder.nominal_base_currents[0]),
    fourier_order=4,
    nfp=2,
    stellsym=True,
    base_coils=4,
    physical_coils=16,
    training_grid=[48, 40],
    training_quadrature=[800, 144],
    validation_quadrature=[1600, 288],
    training_coil_segments=128,
    validation_grid=[48, 40],
    validation_coil_segments=[75, 256],
    pressure_edge_Pa=pressure_edge,
    parameter_scales=np.asarray(scales).tolist(),
    scaled_bounds=dict(current=[-2, 4], shape=[-20, 20]),
    max_iterations=2000,
    max_function_evaluations=60000,
    max_wall_seconds=3000,
    objective="1e6*(area_mean((Bout.n)^2/Bin^2) + 0.25*area_mean((Bout^2-Bin^2-2mu0*p_edge)^2/(Bin^2+2mu0*p_edge)^2) + 1.0*geometry_penalty + 1e-10*mean(u^2))",
    geometry_limits=limits,
    source_commit=json.loads((Path(__file__).resolve().parent / "runtime_manifest.json").read_text())["commit"],
    wout_sha256=fixed["wout_sha256"],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    dependency_manifest_sha256=hashlib.sha256((DEPENDENCIES / "manifest.json").read_bytes()).hexdigest(),
    acceptance=dict(normal_errors_are_diagnostics=True, relative_pressure_jump=0.01),
    free_boundary_validated=False,
)
write("settings.json", settings)


def geometry_components(u, *, smooth=True):
    coils = coil_set(u)
    pair, cp = sampled_gaps(coils, smooth=smooth)
    return 1e6 * jnp.array(
        [
            jnp.mean(jax.nn.relu(coils.length / (0.99 * limits["max_length_m"]) - 1) ** 2),
            jnp.mean(jax.nn.relu(coils.curvature / (0.98 * limits["max_curvature_per_m"]) - 1) ** 2),
            jnp.mean(jax.nn.relu(1 - pair / (limits["min_coil_coil_m"] + 0.25)) ** 2),
            jnp.mean(jax.nn.relu(1 - cp / (limits["min_coil_plasma_m"] + 0.15)) ** 2),
        ]
    )


# The plasma remains frozen; only the 111 external-coil parameters vary.
import dataclasses
from vmex.core import implicit as im, freeboundary_implicit as fbi
from vmex.core.freeboundary import _vacuum_scalars
from vmex.core.solver import SpectralState

free_inp = dataclasses.replace(
    inp, lfreeb=True, mgrid_file="DIRECT_ESSOS_BIOT_SAVART", ns_array=(50,), ftol_array=(1e-13,), niter_array=(12000,)
)
names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
with np.load(ROOT / "fixed/state.npz", allow_pickle=False) as data:
    assert bool(data["qualified_fixed_target"])
    fixed_arrays = {n: data[n].copy() for n in names}
nestor_state = SpectralState(*(jnp.asarray(fixed_arrays[n]) for n in names))
nestor_solver = fbi.make_free_boundary_config(
    free_inp,
    builder(zero),
    field_from_parameters=builder,
    device=jax.devices("gpu")[0],
    ftol=1e-10,
    include_edge_in_convergence=False,
)
assert (nestor_solver.resolution.ns, nestor_solver.resolution.ntheta, nestor_solver.resolution.nzeta) == (50, 48, 40)
nparams = im.params_from_input(free_inp)
nrt = im.runtime_from_params(nparams, nestor_solver.implicit)
nrt = dataclasses.replace(
    nrt, lfreeb=True, jmax=50, presf_ns_scale=fbi._presf_ns_scale_traceable(nparams, free_inp, 50)
)
_, _, _, _, nbsq3, npres = _vacuum_scalars(nestor_state, nrt)
nweights = jnp.asarray(nrt.trig.wint)
assert nweights.shape == (25, 40)


def nestor_values(u):
    output = nestor_solver.vacuum_program.full(nestor_state, nrt, builder(u * scales))
    difference = output["bsqvac"] + npres * nrt.presf_ns_scale - nbsq3
    l2 = jnp.sum(nweights * (difference / nbsq3) ** 2) / jnp.sum(nweights)
    return l2, output["delbsq_num"] / output["delbsq_den"]


def base_components(u):
    coils = coil_set(u)
    total = field_cart(coils, sd) + plasma
    bn = jnp.sum(total * sd.normal, axis=0)
    dp = (jnp.sum(total**2, axis=0) - bin2 - 2 * MU0 * pressure_edge) / (bin2 + 2 * MU0 * pressure_edge)
    return jnp.r_[
        1e6 * jnp.sum(weights * bn**2 / bin2),
        1e6 * 0.25 * jnp.sum(weights * dp**2),
        geometry_components(u),
        1e-4 * jnp.mean(u * u),
    ]


# Evaluate the two independently verified derivative paths separately.
base_values = jax.jit(base_components)
base_jacobian = jax.jit(jax.jacrev(base_components))
base_vg = jax.jit(jax.value_and_grad(lambda u: jnp.sum(base_components(u))))
nestor_vg = jax.jit(jax.value_and_grad(lambda u: 1e6 * nestor_values(u)[0]))


def objective_components(u):
    nv, _ = nestor_vg(u)
    return np.r_[np.asarray(base_values(u)), float(nv)]


def objective_jacobian(u):
    _, ng = nestor_vg(u)
    return np.vstack([np.asarray(base_jacobian(u)), np.asarray(ng)])


def fun(u):
    x = jnp.asarray(u)
    bv, bg = base_vg(x)
    nv, ng = nestor_vg(x)
    value = float(bv) + float(nv)
    grad = np.asarray(bg) + np.asarray(ng)
    assert np.isfinite(value) and np.isfinite(grad).all()
    return value, grad


with np.load(ROOT / "coils_smooth/fit_result.npz", allow_pickle=False) as prior:
    assert np.array_equal(prior["parameter_scales"], np.asarray(scales))
    initial_u = prior["scaled_parameters"].copy()
assert json.loads((ROOT / "coils_smooth/seed_orientation.json").read_text())["reflected_y"] == reflected
ninitial = nestor_values(jnp.asarray(initial_u))
initial_delbsq = float(ninitial[1])
expected = json.loads((ROOT / "frozen_nestor_smooth/summary.json").read_text())
assert abs(initial_delbsq - expected["delbsq"]) < 1e-12
write(
    "nestor_objective.json",
    dict(
        initial_delbsq=initial_delbsq,
        weight=1.0,
        metric="angular weighted mean squared relative NESTOR pressure mismatch",
        ns=50,
        ntheta=48,
        nzeta=40,
        mf=4,
        nf=3,
        coil_segments=75,
        frozen_state=True,
        ordinary_equilibrium_solves=0,
        source_coil_fit_sha256=hashlib.sha256((ROOT / "coils_smooth/fit_result.npz").read_bytes()).hexdigest(),
    ),
)
write(
    "restart_provenance.json",
    dict(
        source="inputs/coils_seed.json",
        sha256=hashlib.sha256((ROOT / "inputs/coils_seed.json").read_bytes()).hexdigest(),
        changes="Restart from coils_smooth scaled parameters in the same chart; add the actual frozen-state NESTOR pressure residual. Preserve the fixed WOUT, profiles, all 111 DOFs, first current, and acceptance gates.",
    ),
)
# Diagnose the original geometric subterms and qualify all corrected terms.
x = jnp.asarray(initial_u)
rng = np.random.default_rng(20260914)
direction = rng.normal(size=111)
direction /= np.linalg.norm(direction)
raw = lambda u: geometry_components(u, smooth=False)  # noqa: E731
rv = jax.jit(raw)
rj = jax.jit(jax.jacrev(raw))
raw_ad = np.asarray(rj(x)) @ direction
eps = 1e-6
raw_fd = (np.asarray(rv(x + eps * direction)) - np.asarray(rv(x - eps * direction))) / (2 * eps)
write(
    "original_geometry_diagnosis.json",
    dict(
        components=["length", "curvature", "coil_coil", "coil_plasma"],
        ad=raw_ad.tolist(),
        central_fd=raw_fd.tolist(),
        difference=(raw_ad - raw_fd).tolist(),
        step=eps,
    ),
)
cv = objective_components
cj = objective_jacobian
jac = np.asarray(cj(x))
checks = []
passed = True
for label, d in [
    ("all", direction),
    ("currents", np.r_[direction[:3], np.zeros(108)]),
    ("shape", np.r_[np.zeros(3), direction[3:]]),
]:
    d = d / np.linalg.norm(d)
    ad = jac @ d
    rows = []
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        fd = (np.asarray(cv(x + eps * d)) - np.asarray(cv(x - eps * d))) / (2 * eps)
        component_error = np.abs(ad - fd) / np.maximum(1, np.maximum(np.abs(ad), np.abs(fd)))
        total_error = abs(np.sum(ad) - np.sum(fd)) / max(1, abs(np.sum(ad)), abs(np.sum(fd)))
        error = max(float(np.max(component_error)), float(total_error))
        rows.append(
            dict(step=eps, relative_error=error, component_errors=component_error.tolist(), passed=error < 1e-5)
        )
    direction_pass = any(a["passed"] and b["passed"] for a, b in zip(rows, rows[1:]))
    passed = passed and direction_pass
    checks.append(dict(direction=label, checks=rows, passed=direction_pass))
write(
    "gradient_check.json",
    dict(
        checks=checks,
        passed=passed,
        criterion="All eight components and their sum agree in current, shape, and mixed directions at two adjacent FD step sizes below 1e-5",
        seed=20260914,
    ),
)
event("gradient_check", checks=checks, passed=passed)
assert passed, "Corrected coil objective failed componentwise derivative gate"
value0, gradient0 = fun(initial_u)
np.testing.assert_allclose(gradient0, np.sum(jac, axis=0), rtol=1e-9, atol=1e-7)
np.testing.assert_allclose(value0, np.sum(cv(x)), rtol=1e-12, atol=1e-10)
error = max(min(r["relative_error"] for r in d["checks"]) for d in checks)
event("fit_start", objective=value0, gradient_check_error=error, limits=limits)
count = 0


def callback(u):
    global count
    if time.monotonic() - started > 3000:
        raise TimeoutError("bounded coil-fit walltime")
    count += 1
    if count % 25 == 0:
        event(
            "fit_progress", iteration=count, objective=fun(u)[0], frozen_delbsq=float(nestor_values(jnp.asarray(u))[1])
        )
        np.savez_compressed(OUT / "latest_fit.npz", scaled_parameters=u, parameters=u * np.asarray(scales))


result = minimize(
    fun,
    initial_u,
    jac=True,
    method="L-BFGS-B",
    bounds=[(-2, 4)] * 3 + [(-20, 20)] * 108,
    callback=callback,
    options=dict(maxiter=2000, maxfun=60000, maxls=30, ftol=1e-16, gtol=1e-8, maxcor=30),
)
u = jnp.asarray(result.x)
fitted = coil_set(u, 256)
assert float(fitted.base_currents[0]) == float(builder.nominal_base_currents[0])
write(
    "coils_candidate.json",
    dict(
        dofs_curves=np.asarray(fitted.dofs_curves).tolist(),
        dofs_currents=np.asarray(fitted.base_currents).tolist(),
        n_segments=256,
        nfp=2,
        order=4,
        stellsym=True,
    ),
)
np.savez_compressed(
    OUT / "fit_result.npz",
    scaled_parameters=np.asarray(u),
    parameters=np.asarray(u * scales),
    parameter_scales=np.asarray(scales),
)
event(
    "fit_end",
    iterations=int(result.nit),
    objective=float(result.fun),
    optimizer_success=bool(result.success),
    message=str(result.message),
)


def metrics(coils, target_data):
    surf, bp, wt, b2 = target_data
    total = field_cart(coils, surf) + bp
    relative = jnp.sum(total * surf.normal, axis=0) / jnp.sqrt(b2)
    pressure_jump = jnp.sum(wt * jnp.abs(jnp.sum(total**2, axis=0) - b2 - 2 * MU0 * pressure_edge)) / jnp.sum(
        wt * (b2 + 2 * MU0 * pressure_edge)
    )
    g = np.asarray(coils.gamma)
    points = np.asarray(jnp.moveaxis(surf.gamma, 0, -1)).reshape(-1, 3)
    cc = min(
        float(np.linalg.norm(g[i, :, None, :] - g[j, None, :, :], axis=-1).min())
        for i in range(len(g))
        for j in range(i + 1, len(g))
    )
    cp = min(float(np.linalg.norm(curve[:, None, :] - points[None, :, :], axis=-1).min()) for curve in g)
    return dict(
        normal_rms=float(jnp.sqrt(jnp.sum(wt * relative**2))),
        normal_max=float(jnp.max(jnp.abs(relative))),
        relative_pressure_jump=float(pressure_jump),
        max_length_m=float(jnp.max(coils.length)),
        max_curvature_per_m=float(jnp.max(coils.curvature)),
        min_coil_coil_m=cc,
        min_coil_plasma_m=cp,
    )


validation_coarse = target(48, 6, 800, 144)
validation = target(48, 6, 1600, 288)
quad_diff = jnp.linalg.norm(validation_coarse[1] - validation[1], axis=0) / jnp.sqrt(validation[3])
quad_rms = float(jnp.sqrt(jnp.sum(validation[2] * quad_diff**2)))
write(
    "virtual_casing_quadrature_check.json",
    dict(relative_field_rms_difference=quad_rms, passed=quad_rms <= 2e-5, rms_limit=2e-5),
)
vc_difference = jnp.linalg.norm(plasma - validation[1], axis=0) / jnp.sqrt(bin2)
vc_rms = float(jnp.sqrt(jnp.sum(weights * vc_difference**2)))
vc_max = float(jnp.max(vc_difference))
vc_pass = vc_rms <= 2e-5 and quad_rms <= 2e-5
write(
    "virtual_casing_resolution_check.json",
    dict(
        relative_field_rms_difference=vc_rms,
        relative_field_max_difference=vc_max,
        rms_limit=2e-5,
        passed=bool(vc_pass),
        comparison="Same 48x40 target nodes; digits5 vs digits6 and doubled VC singular quadrature",
    ),
)
coarse = metrics(coil_set(zero, 256), validation)
fine = metrics(fitted, validation)
production = metrics(coil_set(u, 75), validation)
field_pass = all(m["relative_pressure_jump"] <= 0.01 for m in (fine, production))
geometry_pass = all(
    m["max_length_m"] <= limits["max_length_m"]
    and m["max_curvature_per_m"] <= limits["max_curvature_per_m"]
    and m["min_coil_coil_m"] >= limits["min_coil_coil_m"]
    and m["min_coil_plasma_m"] >= limits["min_coil_plasma_m"]
    for m in (fine, production)
)
# Actual frozen-state NESTOR gate must be run separately before selection.
assert all(np.array_equal(np.asarray(getattr(nestor_state, n)), fixed_arrays[n]) for n in names)
summary = dict(
    frozen_nestor_delbsq=float(nestor_values(u)[1]),
    fixed_state_unchanged=True,
    ordinary_equilibrium_solves=0,
    completed=True,
    virtual_casing_pass=bool(vc_pass),
    qualified_for_free_boundary_trial=False,
    eligible_for_frozen_nestor_check=bool(field_pass and geometry_pass and vc_pass),
    optimizer_success=bool(result.success),
    optimizer_message=str(result.message),
    iterations=int(result.nit),
    objective_initial=value0,
    objective_final=float(result.fun),
    field_gates_passed=bool(field_pass),
    geometry_gates_passed=bool(geometry_pass),
    initial_validation=coarse,
    final_validation_256=fine,
    production_validation_75=production,
    seconds=time.monotonic() - started,
    free_boundary_validated=False,
    base_currents_A=np.asarray(fitted.base_currents).tolist(),
)
write("summary.json", summary)
write(
    "qualification.json",
    dict(
        virtual_casing_pass=bool(vc_pass),
        field_pass=bool(field_pass),
        geometry_pass=bool(geometry_pass),
        selected=False,
        requires_actual_frozen_nestor_check=True,
    ),
)
event("finished", **summary)
