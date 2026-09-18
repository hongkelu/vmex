#!/usr/bin/env python3
"""Launch the maintained 48x40 vacuum example using the public problem API.

Run with --output-dir NEW_DIRECTORY and an explicit bounded --target-step.
Use --help for initialization, device and authenticated checkpoint options.
Objectives and constraints live in the maintained case's readable script.
"""

from pathlib import Path
import runpy
import sys


def main(argv=None):
    case = Path(__file__).resolve().parent / "free_boundary_qa"
    previous_path = sys.path.copy()
    try:
        sys.path.insert(0, str(case))
        example = runpy.run_path(str(case / "single_stage_free_boundary_optimization.py"))
        return example["main"](argv)
    finally:
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
