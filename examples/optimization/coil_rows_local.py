"""Memory-lean, exact coil-separation and coil-force rows for reverse-mode differentiation.

A minimum (separation) or a maximum (force) passes its gradient only through the active element. Stage 1 locates the
active segment pairs / peak-force points at full resolution under ``stop_gradient`` (forward memory only); stage 2
recomputes, with gradient, only a window of WINDOW segments around each of the TOPK best candidates. The window
contains the global minimizer / maximizer, so values and gradients equal the all-pairs versions, while the reverse
pass stores O(TOPK * WINDOW^2) instead of O(n_pairs * N^2) residuals (145 GB at N = 1024 for 20 coils).
"""
import jax
import jax.numpy as jnp

WINDOW, TOPK, TOPK_FORCE = 6, 6, 6


def seg_dist(a0, a1, b0, b1):
    """Distances between segments [a0_i, a1_i] and [b0_j, b1_j], interiors included: (len(a0), len(b0))."""
    u, v = a1 - a0, b1 - b0
    w = a0[:, None] - b0[None, :]
    aa, bb = jnp.sum(u * u, axis=-1)[:, None], jnp.sum(v * v, axis=-1)[None, :]
    uv = jnp.einsum("ik,jk->ij", u, v)
    uw, vw = jnp.sum(u[:, None] * w, axis=-1), jnp.sum(v[None, :] * w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa * bb - uv * uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv * vw - bb * uw) / safe, (aa * vw - uv * uw) / safe

    def distance(s, t):
        q = w + s[..., None] * u[:, None] - t[..., None] * v[None, :]
        return jnp.sum(q * q, axis=-1)

    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw / bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw + uv) / bb, 0, 1)),
                  distance(jnp.clip(-uw / aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv - uw) / aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), distance(s, t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0) + 1e-30)


def closed(a):
    return a, jnp.roll(a, -1, axis=0)


def separations_local(points):
    """(min inter-coil, min nonadjacent self) segment distance of closed polygons ``points`` (n_coils, N, 3)."""
    n_c, N = points.shape[0], points.shape[1]
    pairs = jnp.asarray([(i, j) for i in range(n_c) for j in range(i + 1, n_c)], dtype=jnp.int32)
    sp = jax.lax.stop_gradient(points)
    offs = jnp.arange(-WINDOW, WINDOW + 1)

    def pmin(ij):
        d = seg_dist(*closed(sp[ij[0]]), *closed(sp[ij[1]])).reshape(-1)
        k = jnp.argmin(d)
        return d[k], k

    dmin, karg = jax.lax.map(pmin, pairs)
    top = jnp.argsort(dmin)[:min(TOPK, pairs.shape[0])]

    def inter_local(p):
        ij, k = pairs[p], karg[p]
        ia, jb = (k // N + offs) % N, (k % N + offs) % N
        A, B = points[ij[0]], points[ij[1]]
        return jnp.min(seg_dist(A[ia], A[(ia + 1) % N], B[jb], B[(jb + 1) % N]))

    idx = jnp.arange(N)
    gap = jnp.abs(idx[:, None] - idx[None, :])
    nonadj = jnp.minimum(gap, N - gap) > 1

    def smin(c):
        d = jnp.where(nonadj, seg_dist(*closed(sp[c]), *closed(sp[c])), jnp.inf).reshape(-1)
        k = jnp.argmin(d)
        return d[k], k

    dself, kself = jax.lax.map(smin, jnp.arange(n_c))
    topc = jnp.argsort(dself)[:min(TOPK, n_c)]

    def self_local(c):
        k = kself[c]
        ia, jb = (k // N + offs) % N, (k % N + offs) % N
        P = points[c]
        d = seg_dist(P[ia], P[(ia + 1) % N], P[jb], P[(jb + 1) % N])
        g = jnp.abs(ia[:, None] - jb[None, :])
        return jnp.min(jnp.where(jnp.minimum(g, N - g) > 1, d, jnp.inf))

    return jnp.min(jax.vmap(inter_local)(top)), jnp.min(jax.vmap(self_local)(topc))


def b_reg_self_at(gamma, gd, gdd, quad, current, reg, idx):
    """ESSOS B_regularized_pure (regularized self field of a filament) evaluated only at the points ``idx``."""
    from essos.objective_functions import B_regularized_singularity_term

    phi = quad * 2 * jnp.pi
    rc, rp, rpp = gamma, gd / 2 / jnp.pi, gdd / 4 / jnp.pi**2
    dphi = 2 * jnp.pi / phi.shape[0]
    analytic = B_regularized_singularity_term(rp[idx], rpp[idx], reg)
    dr = rc[idx][:, None] - rc[None, :]
    first = jnp.cross(rp[None, :], dr) / ((jnp.sum(dr * dr, axis=2) + reg) ** 1.5)[:, :, None]
    cos_fac = 2 - 2 * jnp.cos(phi[None, :] - phi[idx][:, None])
    den2 = cos_fac * jnp.sum(rp[idx] * rp[idx], axis=1)[:, None] + reg
    second = jnp.cross(rpp[idx], rp[idx])[:, None, :] * (0.5 * cos_fac / den2**1.5)[:, :, None]
    return current * 1e-7 * (analytic + dphi * jnp.sum(first + second, 1))


def coil_forces_local(g, gd, gdd, currents, quad, n_base, reg):
    """Max |I t x B| per base coil (N/m); mutual field of every other coil + regularized self field."""
    from essos.fields import BiotSavart_from_gamma

    n_total = g.shape[0]
    others = jnp.asarray([[j for j in range(n_total) if j != i] for i in range(n_base)], dtype=jnp.int32)
    sg, sgd, sgdd, sI = (jax.lax.stop_gradient(x) for x in (g, gd, gdd, currents))
    all_idx = jnp.arange(g.shape[1])

    def force_at(i, other, idx, g, gd, gdd, I):
        mutual = BiotSavart_from_gamma(g[other], gd[other], gdd[other], I[other])
        b = jax.vmap(mutual.B)(g[i][idx]) + b_reg_self_at(g[i], gd[i], gdd[i], quad, I[i], reg, idx)
        t = gd[i][idx] / jnp.linalg.norm(gd[i][idx], axis=1)[:, None]
        return jnp.linalg.norm(jnp.cross(I[i] * t, b), axis=1)

    f_all = jax.lax.map(lambda a: force_at(a[0], a[1], all_idx, sg, sgd, sgdd, sI), (jnp.arange(n_base), others))
    top = jnp.argsort(-f_all, axis=1)[:, :TOPK_FORCE]
    return jax.vmap(lambda i, o, idx: jnp.max(force_at(i, o, idx, g, gd, gdd, currents)))(jnp.arange(n_base), others, top)
