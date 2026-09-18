"""Make the standalone case helpers available during test collection."""

from pathlib import Path
import sys


# Some tests load helpers at module scope, before fixtures can adjust imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
