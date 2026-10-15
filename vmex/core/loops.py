"""Loops over derivative computations, in a form XLA:GPU compiles correctly.

No VMEC2000 counterpart: JAX infrastructure shared by the implicit, polish,
virtual-casing and free-boundary modules.

**Why this module exists.**  On GPU (jaxlib 0.11.2, CUDA, and the latest
release), XLA can miscompile a *bare* ``jax.lax.map`` / ``lax.scan`` /
``lax.fori_loop`` whose body takes a forward or reverse derivative: the
while-loop body returns wrong tangents while every primal is right.  Measured
on the three-terms free boundary (LP QA, beta 2.5 %, 8 x 8): the
plain-moments Jacobian columns came out 55-84 % off resolved differences,
the tangent of ``B_out . n`` wrong while those of ``B_out`` and ``n`` were
right.  The same loop is exact on CPU, eagerly, unrolled, and whenever the
loop body carries a batch axis.  The failure is silent and depends on the
program, so a small test case can pass while the production size fails.

Rule: a loop whose body differentiates (``jvp``, ``vjp``, ``linearize``,
``grad``, ``jacfwd``, ``jacrev``), or that is itself differentiated, goes
through :func:`map_columns` (or ``lax.map`` with an explicit
``batch_size``), never a bare ``lax.map``.  ``tests/test_loops.py`` enforces
this statically over ``vmex/`` and checks the helper against finite
differences on the device the suite runs on.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ["map_columns"]


def _leading(xs) -> int:
    sizes = {int(jnp.shape(a)[0]) for a in jax.tree.leaves(xs)}
    if len(sizes) != 1:
        raise ValueError(f"map_columns: leaves disagree on the mapped axis length {sorted(sizes)}")
    return sizes.pop()


def map_columns(function: Callable[[Any], Any], columns: Any, *, batch_size: int | None = 1) -> Any:
    """``jax.lax.map(function, columns)`` with a batch axis in the loop body.

    ``batch_size`` columns are evaluated together, as one ``vmap`` inside the
    loop: ``1`` (the default) keeps one column's temporaries live at a time;
    ``None`` maps every column in one ``vmap`` with no loop at all.  With
    ``batch_size = k > 1`` the last ``n mod k`` columns are one extra
    ``vmap`` after the loop (no padding is computed).

    Unlike ``lax.map(..., batch_size=k)`` this accepts outputs with zero-size
    leaves (an empty parameter's cotangent).  The batch axis is what keeps
    XLA:GPU from miscompiling the loop body; see the module docstring.
    """
    if batch_size is not None and (isinstance(batch_size, bool) or int(batch_size) < 1):
        raise ValueError("batch_size must be a positive integer or None")
    count = _leading(columns)
    if batch_size is None or int(batch_size) >= count:
        return jax.vmap(function)(columns)
    size = int(batch_size)
    steps, main = count // size, (count // size) * size

    def block(chunk):
        return jax.vmap(function)(chunk)

    head = jax.tree.map(lambda a: a[:main].reshape((steps, size) + a.shape[1:]), columns)
    out = jax.lax.map(block, head)
    out = jax.tree.map(lambda a: a.reshape((main,) + a.shape[2:]), out)
    if main == count:
        return out
    tail = jax.vmap(function)(jax.tree.map(lambda a: a[main:], columns))
    return jax.tree.map(lambda a, b: jnp.concatenate((a, b)), out, tail)
