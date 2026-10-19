"""Run the three-term driver with the model's Jacobian phases timed: tangent solves and interface-row pushes.

    COIL_CASE=... python extra/time_linearize.py <driver args...>     (cwd: vmex tree, GPU)
"""
import runpy
import sys
import time

import jax

from vmex.core import freeboundary_vc as fvc

init = fvc.ThreeTermFreeBoundaryModel.__init__


def timed(name, fn):
    def inner(*a, **k):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*a, **k))
        print(f"    [timing] {name:12s} {time.perf_counter() - t:7.2f} s", flush=True)
        return out
    return inner


def patched(self, *a, **k):
    init(self, *a, **k)
    self._tangents = timed("tangents", self._tangents)
    self._push_rows = timed("push rows", self._push_rows)
    self.pullback = timed("pullback", self.pullback)


fvc.ThreeTermFreeBoundaryModel.__init__ = patched
problem_init = fvc.ThreeTermFreeBoundaryProblem.__init__


def problem_patched(self, *a, **k):
    names = ("_field_vjp", "_direct", "_pullback", "_coordinates", "_field_jvp")
    original = {}

    def setattr_hook(obj, name, value):
        object.__setattr__(obj, name, timed(name, value) if name in names and callable(value) else value)

    cls = type(self)
    cls.__setattr__ = setattr_hook
    try:
        problem_init(self, *a, **k)
    finally:
        del cls.__setattr__


fvc.ThreeTermFreeBoundaryProblem.__init__ = problem_patched
linearize = fvc.ThreeTermFreeBoundaryModel.linearize


def logged_linearize(self, x, aux, field, extra=None):
    import hashlib
    import numpy as np
    key = hashlib.sha1(np.asarray(x, dtype=float).tobytes()).hexdigest()[:10]
    pkey = hashlib.sha1(b"".join(np.asarray(a).tobytes() for a in jax.tree.leaves(aux[2]))).hexdigest()[:10]
    print(f"    [linearize] x {key} params {pkey} tight {aux[3]} extra {extra is not None}", flush=True)
    return timed("linearize", linearize)(self, x, aux, field, extra)


fvc.ThreeTermFreeBoundaryModel.linearize = logged_linearize
solve = fvc.ThreeTermFreeBoundaryModel.solve
fvc.ThreeTermFreeBoundaryModel.solve = timed("solve", solve)
sys.argv = ["single_stage_free_boundary_optimization_three_term.py", *sys.argv[1:]]
runpy.run_path("examples/optimization/single_stage_free_boundary_optimization_three_term.py", run_name="__main__")
