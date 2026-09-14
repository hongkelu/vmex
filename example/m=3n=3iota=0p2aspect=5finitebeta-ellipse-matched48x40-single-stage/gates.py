"""Fresh DEL-BSQ acceptance, separate from FTOL and projected edge force."""
import math


def pressure_diagnostic(numerator, denominator, limit):
    numerator, denominator, limit = map(float, (numerator, denominator, limit))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('DEL-BSQ limit must be finite and positive')
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or numerator < 0 or denominator <= 0):
        raise ValueError('invalid fresh DEL-BSQ numerator or denominator')
    ratio = numerator / denominator
    if not math.isfinite(ratio):
        raise ValueError('nonfinite fresh DEL-BSQ ratio')
    return dict(fresh_delbsq=ratio, fresh_delbsq_percent=100*ratio,
                delbsq_limit=limit, passed=ratio <= limit,
                delbsq_numerator=numerator, delbsq_denominator=denominator)


def fresh_pressure_gate(solver, params, parameters, state, rcon0, zcon0, limit):
    """Recompute NESTOR at the candidate; never use a cadence-cached value."""
    import dataclasses
    import jax.numpy as jnp
    from vmex.core import implicit as im, freeboundary_implicit as fbi
    from vmex.core.errors import VmecError
    rt = im.runtime_from_params(params, solver.implicit)
    rt = dataclasses.replace(rt, rcon0=rcon0, zcon0=zcon0, lfreeb=True,
        jmax=int(solver.resolution.ns),
        presf_ns_scale=fbi._presf_ns_scale_traceable(
            params, solver.implicit.inp, int(solver.resolution.ns)))
    out = solver.vacuum_program.full(
        state, rt, solver.field_from_parameters(jnp.asarray(parameters)))
    try:
        data = pressure_diagnostic(out['delbsq_num'], out['delbsq_den'], limit)
    except ValueError as exc:
        raise VmecError(str(exc)) from exc
    if not data['passed']:
        raise VmecError(f"fresh DEL-BSQ {data['fresh_delbsq']:.9e} exceeds {limit:.9e}")
    return data
