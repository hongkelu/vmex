"""pytest plugin: never release the constrained m=1 Z force (solver and implicit residual)."""
from vmex.core import implicit as _im
from vmex.core import solver as _solver

_solver.zero_m1_z_force = lambda force, enabled: force
_im.zero_m1_z_force = lambda force, enabled: force
