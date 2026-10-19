"""JVP cost of each piece of the three-term interface rows, per 8 state directions, on the seed.

    COIL_CASE=qa4-beta python extra/bench_rows.py <seed run dir>     (cwd: vmex tree, GPU)
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path.cwd() / "examples" / "optimization"))
import single_stage_free_boundary_optimization_three_term as tt  # noqa: E402
import vmex as vj  # noqa: E402
from essos.coils import Coils  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

run = Path(sys.argv[1])
inp = dataclasses.replace(vj.VmecInput.from_file(run / "input.run"), lfreeb=False)
coils = Coils.from_json(str(run / "coils.initial.json"))
chart = tt.CoilChart(coils, current_dofs=())
field = chart(jnp.asarray(chart.x0))
n = 48
t = time.perf_counter()
model = fvc.ThreeTermFreeBoundaryModel(inp, nphi=n, ntheta=n, quadrature=(4 * int(inp.nfp) * n, 2 * n),
                                       bootstrap=tt.redl_profiles(inp)[0], bootstrap_helicity=0, trial_ftol=1e-11)
print(f"model {time.perf_counter() - t:.0f} s", flush=True)
state, mask = model.seed
im, cfg = model._im, model.cfg
params = model.params0
runtime = im.runtime_from_params(params, cfg)
rng = np.random.default_rng(0)
dirs = jax.tree.map(lambda *a: jnp.stack(a), *[jax.tree.map(lambda a: jnp.asarray(1e-6 * rng.normal(size=a.shape)),
                                                             state) for _ in range(8)])
data0 = vc.surface_field_data_from_state(model.fixed, state, runtime=runtime, nphi=n, ntheta=n)
precision = vc.plan_vc_precision(data0, digits=4, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)


def timed(label, fn, *args):
    out = jax.block_until_ready(fn(*args))
    t = time.perf_counter()
    for _ in range(3):
        out = jax.block_until_ready(fn(*args))
    dt = (time.perf_counter() - t) / 3
    print(f"{label:46s} {dt * 1e3:9.1f} ms per 8 directions ({dt / 8 * 1e3:6.1f} ms/col)", flush=True)
    return out


def jvp_map(f, x, ts):
    return jax.lax.map(lambda t: jax.jvp(f, (x,), (t,))[1], ts, batch_size=8)


surf = lambda s: vc.surface_field_data_from_state(model.fixed, s, runtime=runtime, nphi=n, ntheta=n)  # noqa: E731
timed("JVP: surface data from state", jax.jit(lambda s, ts: jvp_map(lambda z: (lambda d: (d.gamma, d.B_total, d.normal, d.area_vector))(surf(z)), s, ts)),
      state, dirs)
timed("JVP: surface data + virtual casing",
      jax.jit(lambda s, ts: jvp_map(lambda z: vc.PlasmaVacuumInterface.from_surface_data(
          surf(z), digits=4, precision=precision).B_plasma, s, ts)), state, dirs)
timed("JVP: surface data + coil field",
      jax.jit(lambda s, ts: jvp_map(lambda z: vc.external_B_cartesian(field, surf(z).gamma), s, ts)), state, dirs)
timed("primal: virtual casing", jax.jit(lambda s: vc.PlasmaVacuumInterface.from_surface_data(
    surf(s), digits=4, precision=precision).B_plasma), state)
for src in (1024, 4096, 9216):
    def vc_src(z, src=src):
        d = surf(z)
        f = vc.VirtualCasingExteriorField(d, vc.ExteriorFieldConfig(digits=4, chunk_size=src, target_chunk_size=8,
                                                                     dtype="float64"))
        return f._vc.compute_internal_B(f.B_total, digits=4, chunk_size=src, quad_nt=precision.quad_nt,
                                        quad_np=precision.quad_np, patch_dim0=precision.patch_dim0,
                                        patch_idx=precision.patch_idx)
    timed(f"JVP: virtual casing, source chunk {src}", jax.jit(lambda s, ts: jvp_map(vc_src, s, ts)), state, dirs)
