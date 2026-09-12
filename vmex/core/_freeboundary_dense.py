"""Host-controlled dense adjoints of the canonical projected free-boundary root."""
from __future__ import annotations

import functools
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import scipy.linalg as sl
from jax.flatten_util import ravel_pytree

from . import implicit as im


class _Space(NamedTuple):
    left: object
    right: object
    sign: object
    weight: object


def _active_space(cfg, mask, max_dofs):
    """Sparse orthonormal Q with QQ.T equal to main's DOF projector.

    Equal Z_sin pairs and opposite Z_cos pairs each supply one coordinate.
    The mask must already have equal support within each constrained pair.
    """
    flat = np.asarray(ravel_pytree(mask)[0])
    if not np.all(np.isfinite(flat)) or not np.all((flat == 0) | (flat == 1)):
        raise ValueError("dense adjoints require a finite binary DOF mask")
    active = flat.astype(bool).copy()
    pairs = []
    if bool(cfg.lconm1) and int(cfg.resolution.ntor) > 0:
        pos, neg = im._m1_pair_columns(cfg)
        offset = 0
        for name in im._STATE_FIELDS:
            shape = getattr(mask, name).shape
            sign = 1 if name == 'Z_sin' else -1
            if name == 'Z_sin' or (name == 'Z_cos' and cfg.resolution.lasym):
                for row in range(shape[0]):
                    for p, n in zip(pos, neg):
                        left, right = offset + row*shape[1] + int(p), offset + row*shape[1] + int(n)
                        if active[left] != active[right]:
                            raise ValueError("dense adjoint mask has unequal m=1 pair support")
                        if active[left]:
                            active[left] = active[right] = False
                            pairs.append((left, right, sign))
            offset += int(np.prod(shape))
    singles = np.flatnonzero(active)
    size = len(singles) + len(pairs)
    if not 0 < size <= max_dofs:
        raise ValueError(f"dense adjoint active dimension {size} must be in [1, {max_dofs}]; "
                         "adjust adjoint_dense_max_dofs explicitly for a larger matrix")
    left = np.asarray([*singles, *(p[0] for p in pairs)], dtype=np.int32)
    right = np.asarray([*singles, *(p[1] for p in pairs)], dtype=np.int32)
    sign = np.asarray([0]*len(singles) + [p[2] for p in pairs], dtype=np.float64)
    weight = np.asarray([1.]*len(singles) + [1/np.sqrt(2.)]*len(pairs), dtype=np.float64)
    return _Space(left, right, sign, weight)


def _expand(value, template, space):
    flat, unravel = ravel_pytree(template)
    weighted = value * space.weight
    result = jnp.zeros_like(flat).at[space.left].set(weighted)
    result = result.at[space.right].add(weighted * space.sign)
    return unravel(result)


def _compress(value, space):
    flat = ravel_pytree(value)[0]
    return space.weight * (flat[space.left] + space.sign * flat[space.right])


def _prepare_tangent(z, params, field, frozen, rcon, zcon, *, residual):
    # Main's residual is already compiled. Keep linearize host-eager: wrapping
    # the returned JVP closure in another jit can leak nested force-kernel
    # tracers through its static callable data on supported JAX versions.
    return jax.linearize(lambda x:residual(x, params, field, frozen, rcon, zcon), z)[1]


@jax.jit
def _columns(tangent, template, space, indices):
    size = space.left.shape[0]
    def column(index):
        seed = jax.nn.one_hot(index, size, dtype=ravel_pytree(template)[0].dtype)
        return _compress(tangent(_expand(seed, template, space)), space)
    return jax.vmap(column)(indices)


@functools.partial(jax.jit, static_argnames=('batch_size',))
def _assemble_device(tangent, template, space, *, batch_size):
    size = space.left.shape[0]
    chunks = (size + batch_size - 1)//batch_size
    def chunk(index):
        return _columns(tangent, template, space, index*batch_size+jnp.arange(batch_size))
    rows = jax.lax.map(chunk, jnp.arange(chunks)).reshape((-1, size))[:size]
    return rows.T


def _assemble_host(tangent, template, space, batch_size):
    size = space.left.shape[0]
    matrix = np.empty((size, size), dtype=np.asarray(ravel_pytree(template)[0]).dtype)
    for start in range(0, size, batch_size):
        count = min(batch_size, size-start)
        matrix[:,start:start+count] = np.asarray(
            _columns(tangent,template,space,jnp.arange(batch_size)+start))[:count].T
    return matrix


def _factor_solve(matrix, rhs, backend):
    """One factorization, all transpose RHS columns, without assuming symmetry."""
    if backend == 'forward_dense':
        with warnings.catch_warnings():
            warnings.simplefilter('error', sl.LinAlgWarning)
            return sl.lu_solve(sl.lu_factor(matrix), np.asarray(rhs).T, trans=1).T
    return jsl.lu_solve(jsl.lu_factor(matrix), rhs.T, trans=1).T


def solve_dense_adjoint(residual, z, params, field, frozen, rcon, zcon,
                        rhs_batch, mask, cfg):
    """Forward assembly and certified transpose solve for scalar or shared callers.

    This interface is host-eager even for the JAX matrix backend: the runtime
    mask determines the active dimension. No Krylov fallback is substituted.
    """
    values = (z,params,field,frozen,rcon,zcon,rhs_batch,mask)
    if any(isinstance(x,jax.core.Tracer) for x in jax.tree.leaves(values)):
        raise ValueError("dense free-boundary adjoints require host-eager inputs; "
                         "do not wrap the scalar gradient in jax.jit")
    dtype = ravel_pytree(z)[0].dtype
    if np.dtype(dtype) != np.dtype(np.float64):
        raise TypeError("dense free-boundary adjoints require float64; enable JAX x64")
    space = _active_space(cfg.implicit, mask, cfg.adjoint_dense_max_dofs)
    # Numerical arrays follow the caller's selected device; nothing is cached
    # across roots, masks, or changes in the constraint baselines.
    space = jax.tree.map(jnp.asarray, space)
    batch_size = min(cfg.adjoint_dense_batch_size, space.left.size)
    tangent = _prepare_tangent(z,params,field,frozen,rcon,zcon,residual=residual)
    if cfg.adjoint_solver == 'forward_dense':
        matrix = _assemble_host(tangent,z,space,batch_size)
    else:
        matrix = _assemble_device(tangent,z,space,batch_size=batch_size)
    rhs = jax.vmap(lambda value:_compress(value,space))(rhs_batch)
    def reject(row, norm, tolerance):
        im._raise_adjoint_unconverged(cfg.implicit,iterations=1,
            residual_norm=float(norm),tolerance=float(tolerance),
            method=f'{cfg.adjoint_solver} row {row}')
    if not bool(jnp.all(jnp.isfinite(matrix))):
        reject(0,np.inf,0.)
    try:
        solution = _factor_solve(matrix,rhs,cfg.adjoint_solver)
    except (np.linalg.LinAlgError, sl.LinAlgWarning):
        reject(0,np.inf,0.)
    # Check the actual dense equation after the solve, independently of the
    # factorization's status. Singular/nonfinite solutions always fail closed.
    if cfg.adjoint_solver == 'forward_dense':
        norms = np.linalg.norm(np.asarray(solution) @ matrix - np.asarray(rhs),axis=1)
    else:
        norms = jnp.linalg.norm(solution @ matrix - rhs,axis=1)
    tolerances = im._adjoint_acceptance(cfg.implicit,jnp.linalg.norm(rhs,axis=1))
    for row in range(rhs.shape[0]):
        finite = bool(jnp.isfinite(norms[row]) & jnp.all(jnp.isfinite(solution[row])))
        if not finite or float(norms[row]) > float(tolerances[row]):
            if cfg.adjoint_fail != 'best_effort' or not finite:
                reject(row,norms[row],tolerances[row])
            warnings.warn(f'{cfg.adjoint_solver} row {row} residual exceeds acceptance; '
                          "returning an inaccurate best-effort adjoint",RuntimeWarning,stacklevel=2)
    return jax.vmap(lambda value:_expand(value,z,space))(jnp.asarray(solution))
