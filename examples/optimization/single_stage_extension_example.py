"""Example ``--extension`` file for the single-stage drivers: extra constraint rows and objective terms.

    python single_stage_free_boundary_optimization_three_term.py ... --extension single_stage_extension_example.py
    python single_stage_optimization_coil_constraints.py ... --extension single_stage_extension_example.py

Every function takes the equilibrium ``state`` and its ``runtime`` and returns a JAX scalar or array, as the
``vmex.optimize`` quantities do; the drivers differentiate it with the same implicit adjoint as their built-in rows.
Stability models (ballooning, finite-n) plug in the same way, from a file kept outside the drivers.

``ROWS = {name: (f, lower, upper, scale)}``: hard constraints ``lower <= f <= upper`` on every element of ``f``, with
``scale`` the row's unit (use ``-inf``/``inf`` for one-sided rows).  ``TERMS = {name: (f, weight)}``: the objective
adds ``weight / 2 |f|^2``.  Every value is logged in ``metrics.jsonl`` as ``extension_<name>``.
"""
import math

import jax.numpy as jnp

from vmex import optimize as opt

ROWS = {
    # a favorable vacuum magnetic well, (V'(0) - V'(1)) / V'(0) >= 1%
    "magnetic_well": (opt.magnetic_well, 0.01, math.inf, 0.01),
}

TERMS = {
    # a mild push towards boundary elongation 3 or below (the term is zero below 3)
    "elongation": (lambda state, runtime: jnp.maximum(opt.max_elongation(state, runtime) - 3.0, 0.0), 1e-3),
}
