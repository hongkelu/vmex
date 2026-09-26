"""Load the maintained coil benchmark without replacing its global modules."""

from pathlib import Path
import importlib.util
import sys
from functools import lru_cache

REFERENCE = Path(__file__).resolve().parents[1] / "coil-constraints-benchmarks"
VACUUM_SCRIPTS = {
    "fixed": "single_stage_optimization_scalar.py",
    "free": "free_boundary_single_stage_optimization_scalar.py",
}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("_redl_main_parameters", "parameters.py")


@lru_cache(maxsize=2)
def vacuum_script(arm):
    """Import an unchanged vacuum entry point without leaking its import setup.

    Only call its optimizer routine: its equilibrium factory assumes prescribed
    current and its coil normal-field objective assumes vacuum.
    """
    previous_path = sys.path.copy()
    previous = sys.modules.get("parameters")
    try:
        sys.modules["parameters"] = P
        return load(f"_redl_vacuum_{arm}", VACUUM_SCRIPTS[arm])
    finally:
        sys.path[:] = previous_path
        if previous is None:
            sys.modules.pop("parameters", None)
        else:
            sys.modules["parameters"] = previous


def coil_modules():
    previous = sys.modules.get("parameters")
    try:
        sys.modules["parameters"] = P
        limits = load("_redl_main_coil_limits", "_coil_constraints.py")
    finally:
        if previous is None:
            sys.modules.pop("parameters", None)
        else:
            sys.modules["parameters"] = previous
    return limits, load("_redl_main_coil_resolution", "_coil_resolution.py")
