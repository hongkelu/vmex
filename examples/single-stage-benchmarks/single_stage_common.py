"""Compatibility import for the shared single-stage interface."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from single_stage_support.common import initial_coils, load_input, parse_options, physical_constraint
__all__ = ["initial_coils", "load_input", "parse_options", "physical_constraint"]
