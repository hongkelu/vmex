"""Where are the Redl rows not C1?  Probe the |B| extrema and the Redl <J.B> on a solved equilibrium.

usage: python redl_kink_probe.py DECK PROFILES_JSON [HELICITY_N [asym]]   (one JSON line per surface group, then a summary)

The deck is solved fixed-boundary as written. On its half-mesh |B| (the in-solve Redl lane's input) reported per
surface:
- the two largest local maxima (and two smallest minima) of the grid's trigonometric interpolant, sampled on a
  4x refined grid, and their gap relative to Bmax - Bmin: a gap near 0 means the global extremum can switch
  between two peaks under a small change of the field, a kink of Bmax (Bmin) and of everything downstream;
- one-sided directional derivatives of Bmax, Bmin and Redl's <J.B> along a smooth random |B| perturbation (stellarator symmetric unless ``asym``: under the
  symmetry the global extremum's twin peaks stay equal) at
  steps t = 1e-4 .. 1e-7 (relative): for a C1 function the forward and backward ones agree to O(t).
"""
import json
import sys

import numpy as np


def main():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import bootstrap as bs

    inp = vj.VmecInput.from_file(sys.argv[1])
    prof = json.load(open(sys.argv[2]))
    profiles = bs.KineticProfiles(**{k: np.asarray(v, float) for k, v in prof.items()})
    helicity = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    asym = len(sys.argv) > 4 and sys.argv[4] == "asym"
    eq = opt.solve_equilibrium(inp)
    hm = bs._half_mesh_fields(eq.state, eq.runtime)
    B = np.asarray(hm.bmag)
    nsurf, nt, nz = B.shape

    def refined(Bs, k=4):
        F = np.fft.fft2(Bs)
        Fp = np.zeros((k * nt, k * nz), complex)
        ht, hz = nt // 2, nz // 2
        Fp[:ht, :hz], Fp[-ht:, :hz], Fp[:ht, -hz:], Fp[-ht:, -hz:] = F[:ht, :hz], F[-ht:, :hz], F[:ht, -hz:], F[-ht:, -hz:]
        return np.real(np.fft.ifft2(Fp)) * k * k

    def peaks(Bf, sign):
        f = sign * Bf
        nb = [np.roll(np.roll(f, a, 0), c, 1) for a in (-1, 0, 1) for c in (-1, 0, 1) if (a, c) != (0, 0)]
        mask = np.all([f > g for g in nb], axis=0)
        return np.sort(sign * f[mask])[::-1] * 1.0 if sign > 0 else np.sort(Bf[mask])

    rows = []
    for j in range(nsurf):
        Bf = refined(B[j])
        mx, mn = peaks(Bf, 1.0), peaks(Bf, -1.0)
        span = Bf.max() - Bf.min()
        rows.append(dict(s=float(np.asarray(hm.s_half)[j]), n_max=int(mx.size), n_min=int(mn.size),
                         max_gap=float((mx[0] - mx[1]) / span) if mx.size > 1 else None,
                         min_gap=float((mn[1] - mn[0]) / span) if mn.size > 1 else None,
                         eps=float(span / (Bf.max() + Bf.min()))))

    # Directional one-sided derivatives along a smooth perturbation of |B| (low Fourier modes, fixed seed)
    rng = np.random.default_rng(0)
    th = 2 * np.pi * np.arange(nt) / nt
    ze = 2 * np.pi * np.arange(nz) / nz
    d = np.zeros_like(B)
    for m in range(4):
        for n in range(-3, 4):
            phase = rng.uniform(0, 2 * np.pi) if asym else 0.0  # phase 0: stellarator symmetric, as every design
            d += rng.normal() * np.cos(m * th[:, None] + n * ze[None, :] + phase)[None]
    d *= np.abs(B).mean() * 1e-2 / np.abs(d).max()

    def outs(t):
        h = hm._replace(bmag=jnp.asarray(B + t * d))
        geom = bs._geometry_from_half(h, h.s_half, n_lambda=bs.N_LAMBDA, refine_extrema=True)
        jr = bs.j_dot_B_redl(profiles, geom, helicity)[0]
        return np.asarray(geom.Bmax), np.asarray(geom.Bmin), np.asarray(geom.f_t), np.asarray(jr)

    base = outs(0.0)
    kinks = {}
    for t in (1e-4, 1e-5, 1e-6, 1e-7):
        p, m = outs(t), outs(-t)
        for name, a, b0, c in zip(("Bmax", "Bmin", "f_t", "jdotB"), p, base, m):
            fwd, bwd = (a - b0) / t, (b0 - c) / t
            scale = np.maximum(np.abs(fwd) + np.abs(bwd), 1e-300)
            kinks.setdefault(name, {})[f"{t:.0e}"] = dict(
                rel_jump_max=float(np.max(np.abs(fwd - bwd) / scale)),
                worst_surface=int(np.argmax(np.abs(fwd - bwd) / scale)))
    for r in rows:
        print(json.dumps(r))
    gaps = np.array([r["max_gap"] if r["max_gap"] is not None else np.inf for r in rows])
    print(json.dumps(dict(summary=True, nsurf=nsurf, grid=[nt, nz], max_gap_min=float(gaps.min()),
                          surfaces_max_gap_lt_1e2=int(np.sum(gaps < 1e-2)), kinks=kinks)), flush=True)


if __name__ == "__main__":
    main()
