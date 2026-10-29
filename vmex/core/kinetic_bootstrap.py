"""DKX's drift-kinetic bootstrap current as per-surface constraint rows, each surface solved on a GPU of its own.

The kinetic counterpart of the Redl rows (``RedlBootstrapMismatch``, ``HalfMeshCurrent``): on every DKX surface
``s_j`` (a half-mesh row) the row is

    ``(I'(s_j) - I'_DKX(s_j)) / I'_ref``,

``I'`` the equilibrium's own enclosed-current derivative [A] and ``I'_DKX`` the one that would make its ``<J.B>``
DKX's (``bootstrap.current_derivative``, the MHD identity ``<B^2> dI/ds + I dp/ds = mu0 Phi_a <J.B>``).  With the
current's spline knots as design variables (``CoilDesign(plasma_current=..., plasma_current_spline=True)``) an
optimizer holding the rows near zero keeps the current self-consistent with DKX at every step, and the rows' design
gradients see how the current responds to the coils.

Each surface's ``<j.B>`` is DKX's ion-root value, a smooth function of the surface's Boozer spectrum:

- E_r is the root of ``J_r(E_r) = 0`` (DKX's damped Newton, warm-started), differentiated implicitly: the row value
  ``<j.B> - (d<j.B>/dE) J_r / (dJ_r/dE)`` at the root has the gradient ``d<j.B> - (d<j.B>/dE) dJ_r / (dJ_r/dE)``,
  one adjoint (transposed) solve, no derivative through the Newton iterations;
- a surface with no root on the ion side (``E_r <= ion_side``: the edge, where J_r touches zero between E x B
  poles, or a flat-J_r axis surface that wanders to the electron side) holds, for the round, the E_r of smallest
  ``|J_r|`` near its previous one (``refresh``), and its row is ``<j.B>`` at that fixed E_r;
- ``<j.B>`` is linear in the drift-kinetic solution, so its gradient is one transposed solve on the surface's own
  grid: DKX's differentiable solve (``solvax.implicit.linear_solve``), memory near the forward solve's.

The cut between VMEX and DKX is the surface's Boozer-transform input (``boozer_input_tables``: a few kB): the
VMEX side stays one traced program differentiated by the three-term problem, and the rows' gradients come back as
cotangents of those tables, so a surface can live on any GPU of any process (``KineticSurfaces.local`` threads,
``KineticSurfaces.remote`` worker processes, e.g. one per GPU under ``srun``).  Multi-fidelity: the in-loop grid
is coarse (g15), and a frozen ``delta = <j.B>_fine - <j.B>_coarse`` per surface (``correct``) carries the fine
grid; re-measure it between rounds and certify the endpoint with a fine Picard solve.

Needs ``dkx`` (>= 2.12).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing.connection import Client, Listener
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .bootstrap import _current_derivative, _half_mesh_fields, _jv_from_half

__all__ = ["KineticSurface", "KineticSurfaces", "KineticCurrentRows", "serve_kinetic_surfaces"]

#: Boozer-transform inputs of a surface (``boozer_input_tables``) that DKX reads.
SPECTRUM = ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")

_KERNELS = {}  # (options, geometry) -> the jitted programs, shared by a process's surfaces


def _kernel(dkx, xm, xn, nfp, signgs):
    """The jitted DKX programs of one grid and mode set: the template is an argument, so the surfaces share them."""
    key = (dkx.mboz, dkx.tol, dkx.collision_operator, dkx.spectral_angles, json.dumps(dkx.resolution, sort_keys=True),
           xm.tobytes(), xn.tobytes(), nfp, signgs)
    if key in _KERNELS:
        return key, _KERNELS[key]
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from dkx.er import operator_at_er
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve
    from dkx.units import PARALLEL_CURRENT
    from dkx.workflows.geometry_adapters import PSI_A_HAT_LEAVES, kinetic_operator_on_boozer_surface

    plan = dkx._plan(xm, xn, nfp)

    def operator_of(template, spectrum, psi):  # DKX's KineticBootstrapMismatch chain, one surface
        booz = booz_xform_jax_impl(**{k: v[None] for k, v in spectrum.items()}, xm=jnp.asarray(xm),
                                   xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm), xn_nyq=jnp.asarray(xn),
                                   constants=plan.constants, grids=plan.grids, plan=plan)
        return kinetic_operator_on_boozer_surface(
            replace(template, **{leaf: getattr(template, leaf) / psi for leaf in PSI_A_HAT_LEAVES}),
            bmnc_b=booz["bmnc_b"][0], ixm_b=np.asarray(plan.grids.xm_b), ixn_b=np.asarray(plan.grids.xn_b), nfp=nfp,
            iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0], signgs=signgs)

    def evaluate(op):  # (J_r, <j.B> [A T/m^2]) from one solve
        solved = solve(op, op.rhs(), method="auto", tol=dkx.tol, differentiable=True, tier1_keep_lowest=op.n_xi,
                       emit=None)
        moments = profile_moments_from_operator(op, solved.x.reshape(-1))
        return (jnp.dot(jnp.asarray(op.z_s), moments["particleFlux_vm_psiHat"]),
                moments["FSABjHat"].reshape(()) * PARALLEL_CURRENT)

    def fields(template, spectrum, psi, er):
        # the template is built at E_r = 1 kV/m: its E_r leaf is the drive per kV/m (and moves with psi)
        op = operator_of(template, spectrum, psi)
        return evaluate(operator_at_er(op, er, dphi_per_er=op.dphi_hat_dpsi_hat_kinetic))

    def newton(template, spectrum, psi, er0):
        return dkx._newton(evaluate, operator_of(template, spectrum, psi), er0)

    def corrected(template, spectrum, psi, er, d_r, d_b):
        j_r, j_b = fields(template, spectrum, psi, er)
        return j_b - d_b / d_r * j_r  # at a root: <j.B>, with the implicit-function gradient

    _KERNELS[key] = dict(fields=jax.jit(fields), newton=jax.jit(newton), value=jax.jit(corrected),
                         value_and_grad=jax.jit(jax.value_and_grad(corrected, argnums=(1, 2))))
    return key, _KERNELS[key]


def _closest(fields, j_r, k):
    """Parabola vertex of ``|J_r|`` through the grid minimum ``k`` and its neighbours (the minimum at an end)."""
    if 0 < k < len(fields) - 1:
        y = np.abs(j_r[k - 1:k + 2])
        curvature = y[0] - 2 * y[1] + y[2]
        if curvature > 0:
            return float(fields[k] + 0.5 * (fields[k + 1] - fields[k]) * (y[0] - y[2]) / curvature)
    return float(fields[k])


class KineticSurface:
    """One DKX surface: its ion-root (or held) ``<j.B>`` from its Boozer spectrum, value and gradient.

    ``s`` is moved to the nearest half-mesh row of an ``ns`` grid (``row``, ``s_row``).  ``grid`` is
    ``(resolution, mboz)`` (DKX ``Ntheta Nzeta Nxi NL Nx`` keys); the other keywords are
    ``KineticBootstrapMismatch``'s.  ``er`` [kV/m] is the warm start (or the held field with ``hold``),
    ``ion_side`` the largest E_r accepted as an ion root.
    """

    def __init__(self, profiles, s, ns, *, grid, er=-2.0, hold=False, ion_side=0.0, edge_window=10.0,
                 collision_operator=0, minor_radius=1.0, spectral_angles=True, tol=1e-10, er_tol=0.05,
                 er_max_step=2.0):
        from .._compat import require_optional

        require_optional("dkx", "the kinetic bootstrap rows")
        from dkx.bootstrap import KineticBootstrapMismatch

        resolution, mboz = grid
        self.dkx = KineticBootstrapMismatch(
            profiles, [float(s)], ambipolar=True, resolution=dict(resolution), mboz=int(mboz), nboz=int(mboz),
            collision_operator=collision_operator, spectral_angles=spectral_angles, minor_radius=minor_radius,
            tol=tol, er_tol=er_tol, er_max_step=er_max_step)
        self.ns = int(ns)
        (self.row,), (self.s_row,) = self.dkx.rows(self.ns)
        self.er, self.hold = float(er), bool(hold)
        self.ion_side, self.edge_window = float(ion_side), float(edge_window)
        self.unconverged = 0  # calls whose root search failed (the row then used the last E_r, held)
        self._template, self._kernel = None, None

    def bind(self, xm, xn, nfp, signgs):
        """Build the surface's operator template and programs for this mode set."""
        self._geometry = (xm, xn, nfp, signgs)
        self._template = self.dkx._setup(self.ns)[2][0]
        self._key, self._kernel = _kernel(self.dkx, np.asarray(xm), np.asarray(xn), int(nfp), int(signgs))

    def _fields(self, spectrum, psi, er):
        return tuple(float(v) for v in self._kernel["fields"](self._template, spectrum, psi, jnp.float64(er)))

    def _root(self, spectrum, psi):
        """``(E_r, dJ_r/dE, d<j.B>/dE)`` of the ion root from the current E_r, or ``None``."""
        e, d_r, d_b, ok = (np.asarray(v) for v in self._kernel["newton"](self._template, spectrum, psi,
                                                                       jnp.float64(self.er)))
        if bool(ok) and np.isfinite(e) and float(e) <= self.ion_side and float(d_r) != 0.0:
            return float(e), float(d_r), float(d_b)
        return None

    def refresh(self, spectrum, psi):
        """Start of a round: follow the ion root if there is one, else hold the E_r of smallest ``|J_r|``.

        The scan runs on a 1 kV/m grid within ``edge_window`` of the current E_r, capped at ``ion_side``, inside
        the stretch between sign changes of J_r (E x B poles, or noise) that holds the current E_r, and is refined
        by a parabola (or interpolated to the crossing where that minimum sits against a small sign change).
        """
        root = None if self.hold else self._root(spectrum, psi)
        if root is not None:
            self.er = root[0]
            return dict(er=self.er, hold=False)
        guess = self.er
        fields = np.arange(np.floor(guess) - self.edge_window, min(np.floor(guess) + self.edge_window,
                                                                   self.ion_side) + 0.5, 1.0)
        j_r = np.array([self._fields(spectrum, psi, f)[0] for f in fields])
        breaks = np.flatnonzero(np.sign(j_r[1:]) != np.sign(j_r[:-1])) + 1
        home = np.searchsorted(breaks, np.argmin(np.abs(fields - guess)), side="right")
        lo, hi = [0, *breaks][home], [*breaks, fields.size][home]
        k = int(np.argmin(np.abs(j_r[lo:hi])))
        field = _closest(fields[lo:hi], j_r[lo:hi], k)
        for end, across in ((0, lo - 1), (hi - lo - 1, hi)):  # the minimum against a sign change: a root
            if k == end and 0 <= across < fields.size and abs(j_r[across]) < 3 * abs(j_r[lo + k]):
                a, b = lo + k, across
                field = float(fields[a] - j_r[a] * (fields[b] - fields[a]) / (j_r[b] - j_r[a]))
        self.er, self.hold = min(field, self.ion_side), True
        return dict(er=self.er, hold=True, scan=[fields.tolist(), j_r.tolist()])

    def evaluate(self, spectrum, psi, gradient=False):
        """``(<j.B> [A T/m^2], (d/d spectrum, d/d psi) or None)`` at the root (or the held E_r)."""
        root = None if self.hold else self._root(spectrum, psi)
        if root is None:
            self.unconverged += not self.hold
            er, d_r, d_b = self.er, 1.0, 0.0
        else:
            (er, d_r, d_b), self.er = root, root[0]
        args = (self._template, spectrum, psi, jnp.float64(er), jnp.float64(d_r), jnp.float64(d_b))
        if not gradient:
            return float(self._kernel["value"](*args)), None
        value, (g_spectrum, g_psi) = self._kernel["value_and_grad"](*args)
        return float(value), ({k: np.asarray(v) for k, v in g_spectrum.items()}, float(g_psi))

    def current(self, spectrum, psi, grid):
        """``<j.B>`` at this surface's E_r on another ``grid`` (one solve, no gradient): the fine grid of
        ``KineticCurrentRows.correct``."""
        fine = KineticSurface(self.dkx.profiles, self.s_row, self.ns, grid=grid,
                              collision_operator=self.dkx.collision_operator, minor_radius=self.dkx.minor_radius,
                              spectral_angles=self.dkx.spectral_angles, tol=self.dkx.tol)
        fine.bind(*self._geometry)
        value = fine._fields(spectrum, psi, self.er)[1]
        _KERNELS.pop(fine._key, None)  # the fine grid's programs and buffers go with it
        return value

    def record(self):
        return dict(s=float(self.s_row), er=self.er, hold=self.hold, unconverged=self.unconverged)


class _Worker:
    """The surfaces one device (in this process) owns, served one at a time."""

    def __init__(self, device=None):
        self.device, self.surfaces = device, {}

    def handle(self, method, items):
        out = {}
        with jax.default_device(self.device) if self.device is not None else _nullcontext():
            if method == "init":
                options, specs = items
                for i, spec in specs.items():
                    self.surfaces[i] = KineticSurface(options["profiles"], spec["s"], options["ns"],
                                                      er=spec.get("er", -2.0), hold=spec.get("hold", False),
                                                      **options["surface"])
                return {i: None for i in specs}
            for i, args in items.items():
                surface = self.surfaces[i]
                if method == "bind":
                    surface.bind(*args)
                    out[i] = None
                elif method == "record":
                    out[i] = surface.record()
                else:
                    spectrum, psi, *rest = args
                    spectrum = {k: jnp.asarray(v) for k, v in spectrum.items()}
                    out[i] = getattr(surface, method)(spectrum, jnp.float64(psi), *rest)
        return out


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Remote:
    """A worker process at the other end of a connection."""

    def __init__(self, connection):
        self.connection = connection

    def send(self, method, items):
        self.connection.send((method, items))

    def receive(self):
        result = self.connection.recv()
        if isinstance(result, BaseException):
            raise result
        return result


def serve_kinetic_surfaces(rendezvous, timeout=1800.0):
    """Worker loop: connect to the master named in the ``rendezvous`` file and serve its requests on this
    process's default device (``python -m vmex.core.kinetic_bootstrap RENDEZVOUS``, one process per GPU)."""
    path, waited = Path(rendezvous), 0.0
    while not path.exists():
        if waited > timeout:
            raise TimeoutError(f"no master at {path}")
        time.sleep(2.0)
        waited += 2.0
    info = json.loads(path.read_text())
    connection = Client((info["host"], int(info["port"])), authkey=bytes.fromhex(info["authkey"]))
    worker = _Worker()
    while True:
        method, items = connection.recv()
        if method == "stop":
            connection.close()
            return
        try:
            connection.send(worker.handle(method, items))
        except Exception as error:  # the master re-raises it
            connection.send(RuntimeError(f"{socket.gethostname()}: {type(error).__name__}: {error}"))


class KineticSurfaces:
    """A run's DKX surfaces, shared out over workers (one per GPU): ``local`` threads or ``remote`` processes.

    ``specs`` is a list of ``dict(s=..., er=..., hold=...)``; ``options`` holds ``profiles``, ``ns`` and the
    ``KineticSurface`` keywords under ``surface``.  Surfaces go round-robin to the workers, and a call runs every
    worker's surfaces at once.
    """

    def __init__(self, workers, specs, options):
        self.workers, self.specs, self.options = workers, list(specs), options
        self.owner = {i: workers[i % len(workers)] for i in range(len(self.specs))}
        self.bound = None
        self._call("init", {i: None for i in range(len(self.specs))})

    @classmethod
    def local(cls, specs, options, devices=None):
        """Every surface in this process, round-robin over ``devices`` (default: every visible one), a thread each."""
        return cls([_Worker(d) for d in (jax.devices() if devices is None else devices)], specs, options)

    @classmethod
    def remote(cls, specs, options, rendezvous, workers):
        """Listen for ``workers`` processes running :func:`serve_kinetic_surfaces` on ``rendezvous`` (a file both
        sides see, written owner-only: it holds the connection key)."""
        authkey = secrets.token_bytes(32)
        listener = Listener(("0.0.0.0", 0), authkey=authkey)
        path = Path(rendezvous)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.touch(mode=0o600)
        tmp.write_text(json.dumps(dict(host=os.environ.get("VMEX_DKX_HOST", socket.gethostname()), port=listener.address[1], authkey=authkey.hex())))
        tmp.replace(path)
        remotes = [_Remote(listener.accept()) for _ in range(int(workers))]
        path.unlink()
        return cls(remotes, specs, options)

    def _call(self, method, items):
        """``{i: result}``: each worker's share of ``items`` at once."""
        shares = {}
        for i, args in items.items():
            shares.setdefault(id(self.owner[i]), (self.owner[i], {}))[1][i] = args
        payload = lambda w, share: ((self.options, {i: self.specs[i] for i in share})  # noqa: E731
                                    if method == "init" else share)
        remote = [(w, share) for w, share in shares.values() if isinstance(w, _Remote)]
        for w, share in remote:
            w.send(method, payload(w, share))
        local = [(w, share) for w, share in shares.values() if not isinstance(w, _Remote)]
        out = {}
        with ThreadPoolExecutor(max(1, len(local))) as pool:
            for part in pool.map(lambda item: item[0].handle(method, payload(*item)), local):
                out.update(part)
        for w, _ in remote:
            out.update(w.receive())
        return out

    def bind(self, xm, xn, nfp, signgs):
        geometry = (np.asarray(xm), np.asarray(xn), int(nfp), int(signgs))
        key = (geometry[0].tobytes(), geometry[1].tobytes(), geometry[2], geometry[3])
        if self.bound != key:
            self._call("bind", {i: geometry for i in range(len(self.specs))})
            self.bound = key

    def call(self, method, spectra, psi, *args):
        """``method`` of every surface, surface ``i`` on ``{k: spectra[k][i]}``."""
        return self._call(method, {i: ({k: np.asarray(v[i]) for k, v in spectra.items()}, float(psi), *args)
                                   for i in range(len(self.specs))})

    def records(self):
        records = self._call("record", {i: None for i in range(len(self.specs))})
        return [records[i] for i in range(len(self.specs))]

    def stop(self):
        for w in self.workers:
            if isinstance(w, _Remote):
                w.connection.send(("stop", None))
                w.connection.close()


class KineticCurrentRows:
    """``(state, runtime) -> (I'(s_j) - I'_DKX(s_j)) / reference`` on the DKX surfaces (traceable).

    ``surfaces`` is a :class:`KineticSurfaces`; their ``s`` must be half-mesh rows of the run's ``ns``.
    ``reference`` [A] is the fixed ``I'_ref``; ``delta`` [A T/m^2] the frozen fine-minus-coarse ``<j.B>`` per surface
    (``correct``).  With ``basis`` (``I'`` at the surfaces per unit spline-knot value, surfaces x knots) the rows
    are instead the least-squares knot correction ``pinv(basis) (I' - I'_DKX) / reference``: one row per knot, which
    an 8-knot spline can zero even where DKX's surface values jitter.

    Inside a traced program the DKX values come from a host callback with a custom VJP: the forward pass solves each
    surface at its root, the reverse pass asks each surface for its gradient once per state (cached) and contracts it
    with the cotangent, so a row Jacobian costs one adjoint solve per surface however many rows are pulled back.
    """

    name = "dkx_current_rows"

    def __init__(self, surfaces, ns, *, reference, delta=None, basis=None):
        self.surfaces, self.ns = surfaces, int(ns)
        self.index = np.rint(np.array([r["s"] for r in surfaces.specs]) * (self.ns - 1) - 0.5).astype(int)
        if not np.allclose((self.index + 0.5) / (self.ns - 1), [r["s"] for r in surfaces.specs]):
            raise ValueError("the DKX surfaces must be half-mesh rows of ns")
        self.reference = float(reference)
        self.delta = np.zeros(self.index.size) if delta is None else np.asarray(delta, dtype=float)
        self.projection = None if basis is None else np.linalg.pinv(np.asarray(basis, dtype=float))
        self._cache = {}  # state key -> (values, gradients or None)
        self.calls = dict(values=0, gradients=0, seconds=0.0)
        self._kinetic = self._custom_vjp()

    # -- host side ---------------------------------------------------------------------------------------------------

    def _key(self, spectra, psi):
        digest = hashlib.sha1(np.float64(psi).tobytes())
        for k in SPECTRUM:
            digest.update(np.ascontiguousarray(spectra[k], dtype=np.float64).tobytes())
        return digest.hexdigest()

    def _host(self, spectra, psi, gradient):
        key = self._key(spectra, psi)
        cached = self._cache.get(key)
        if cached is not None and (cached[1] is not None or not gradient):
            return cached
        started = time.monotonic()
        out = self.surfaces.call("evaluate", spectra, psi, gradient)
        values = np.array([out[i][0] for i in range(self.index.size)])
        grads = None
        if gradient:
            grads = ({k: np.stack([out[i][1][0][k] for i in range(self.index.size)]) for k in SPECTRUM},
                     np.array([out[i][1][1] for i in range(self.index.size)]))
        self.calls["gradients" if gradient else "values"] += 1
        self.calls["seconds"] += time.monotonic() - started
        self._cache = {key: (values, grads)}  # one state at a time: the latest trial or linearization
        return values, grads

    def _custom_vjp(self):
        n = self.index.size
        shape = jax.ShapeDtypeStruct((n,), jnp.float64)

        def values(spectra, psi):
            return self._host({k: np.asarray(v) for k, v in spectra.items()}, np.asarray(psi), False)[0]

        def gradients(spectra, psi):
            g_spectrum, g_psi = self._host({k: np.asarray(v) for k, v in spectra.items()}, np.asarray(psi), True)[1]
            return g_spectrum, g_psi

        @jax.custom_vjp
        def kinetic(spectra, psi):
            return jax.pure_callback(values, shape, spectra, psi, vmap_method="sequential")

        def forward(spectra, psi):
            return kinetic(spectra, psi), (spectra, psi)

        def backward(residuals, cotangent):
            spectra, psi = residuals
            shapes = ({k: jax.ShapeDtypeStruct(v.shape, jnp.float64) for k, v in spectra.items()}, shape)
            g_spectrum, g_psi = jax.pure_callback(gradients, shapes, spectra, psi, vmap_method="sequential")
            return ({k: g * cotangent.reshape((n,) + (1,) * (g.ndim - 1)) for k, g in g_spectrum.items()},
                    jnp.sum(g_psi * cotangent))

        kinetic.defvjp(forward, backward)
        return kinetic

    def spectra(self, state, rt):
        """``({name: stacked table}, psi_a)`` of the surfaces (traceable), and binds the surfaces to the mode set."""
        from .boozer_tables import boozer_input_tables

        tables = [boozer_input_tables(state, rt, int(j) + 1) for j in self.index]
        self.surfaces.bind(tables[0]["xm"], tables[0]["xn"], int(rt.resolution.nfp), int(np.sign(rt.setup.signgs)))
        # |psi_a| = |phi_edge| / (2 pi) [Wb/rad], as DKX's row forms it
        psi = jnp.abs(jnp.sum(jnp.asarray(rt.setup.phipf)[1:]) / (self.ns - 1))
        return {k: jnp.stack([jnp.asarray(t[k]) for t in tables]) for k in SPECTRUM}, psi

    # -- traced rows -------------------------------------------------------------------------------------------------

    def mismatch(self, state, rt):
        """``(I' - I'_DKX, I'_DKX)`` [A] on the surfaces (traceable)."""
        spectra, psi = self.spectra(state, rt)
        jk = self._kinetic(spectra, psi) + jnp.asarray(self.delta)
        hm = _half_mesh_fields(state, rt)
        n = hm.s_half.shape[0]
        jv = _jv_from_half(hm, hm.s_half)
        target = _current_derivative(hm, jnp.zeros(n).at[self.index].set(jk))[self.index]
        return _current_derivative(hm, jv)[self.index] - target, target

    def __call__(self, state, rt):
        difference = self.mismatch(state, rt)[0] / self.reference
        return difference if self.projection is None else jnp.asarray(self.projection) @ difference

    # -- between rounds ----------------------------------------------------------------------------------------------

    def refresh(self, state, rt):
        """Re-pick every surface's E_r at a concrete state (root, or held minimum ``|J_r|``); returns the records."""
        spectra, psi = self.spectra(state, rt)
        self.surfaces.call("refresh", {k: np.asarray(v) for k, v in spectra.items()}, float(psi))
        self._cache = {}
        return self.surfaces.records()

    def correct(self, state, rt, grid):
        """Measure ``delta = <j.B>_grid - <j.B>`` per surface at a concrete state and E_r, and freeze it."""
        spectra, psi = self.spectra(state, rt)
        spectra = {k: np.asarray(v) for k, v in spectra.items()}
        fine = self.surfaces.call("current", spectra, float(psi), grid)
        coarse = self._host(spectra, np.asarray(psi), False)[0]
        self.delta = np.array([fine[i] for i in range(self.index.size)]) - coarse
        return self.delta

    def record(self):
        """The round's state, for the next round: ``reference``, ``delta`` and every surface's E_r and hold."""
        return dict(reference=self.reference, delta=self.delta.tolist(), surfaces=self.surfaces.records(),
                    calls=dict(self.calls))


if __name__ == "__main__":
    import sys

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    jax.config.update("jax_enable_x64", True)
    serve_kinetic_surfaces(sys.argv[1])
