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

The defect is in the unbatched derivative program itself (a single jitted
column, with no loop, is equally wrong); a bare loop body is exactly that
program.  A batch of one is not robust either: exact in-process, wrong again
under a multi-platform ``jax.export`` (paper reproducers, scan_repro6-8).
Batches of two or more were exact in every form tried.

Rule: a loop whose body differentiates (``jvp``, ``vjp``, ``linearize``,
``grad``, ``jacfwd``, ``jacrev``), or that is itself differentiated, goes
through :func:`map_columns` (batches of at least two), or ``lax.scan``
unrolled, never a bare ``lax.map``.  ``tests/test_loops.py`` enforces
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


def map_columns(function: Callable[[Any], Any], columns: Any, *, batch_size: int | None = 2) -> Any:
    """``jax.lax.map(function, columns)`` with a batch axis of at least two in every compiled body.

    ``batch_size`` columns are evaluated together, as one ``vmap`` per loop
    step; ``None`` maps every column in one ``vmap`` with no loop.  A batch
    is never smaller than two: a one-column batch is not robust (exact
    in-process, wrong again under a multi-platform ``jax.export``), and an
    unbatched body is exactly the program XLA:GPU miscompiles.  When the
    batch does not divide the column count, the steps are evened out
    (``ceil(n / ceil(n / batch_size))`` columns each) and the last one is
    padded by repeating the last column, the padding sliced off: no
    unbatched or one-column remainder is ever compiled.

    Unlike ``lax.map(..., batch_size=k)`` this accepts outputs with zero-size
    leaves (an empty parameter's cotangent).  See the module docstring.
    """
    if batch_size is not None and (isinstance(batch_size, bool) or int(batch_size) < 1):
        raise ValueError("batch_size must be a positive integer or None")
    count = _leading(columns)
    if count == 0:
        return jax.vmap(function)(columns)
    steps = 1 if batch_size is None else -(-count // max(2, int(batch_size)))
    size = max(2, -(-count // steps))  # the fewest padded columns for that many steps
    pad = steps * size - count

    def padded(a):
        if pad:
            a = jnp.concatenate((a, jnp.repeat(a[-1:], pad, axis=0)))
        return a.reshape((steps, size) + a.shape[1:])

    blocks = jax.tree.map(padded, columns)
    if steps == 1:
        out = jax.tree.map(lambda a: a[None], jax.vmap(function)(jax.tree.map(lambda a: a[0], blocks)))
    else:
        out = jax.lax.map(jax.vmap(function), blocks)
    return jax.tree.map(lambda a: a.reshape((steps * size,) + a.shape[2:])[:count], out)
