"""Run a script with the constrained m = 1 Z force never released (solver and implicit residual)."""
import runpy
import sys

from vmex.core import implicit as _im
from vmex.core import solver as _solver

_solver.zero_m1_z_force = lambda force, enabled: force
_im.zero_m1_z_force = lambda force, enabled: force
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
