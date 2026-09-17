#!/usr/bin/env python3
"""Compatibility entry point for the maintained fast single-stage optimizer.

All runs delegate to the same dense-factor-reuse implementation. There is no
independent legacy optimizer loop or GCROT predictor fallback in this workflow.
"""
from single_stage_free_boundary_optimization import main


if __name__ == "__main__":
    main()
