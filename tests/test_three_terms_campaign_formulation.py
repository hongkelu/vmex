"""The three-terms driver's switches to the earlier campaign's formulation (e3313c0f): aggregate coil rows."""
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("essos")

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "optimization"
MODULE = "single_stage_free_boundary_optimization_three_terms"
CAMPAIGN = {"COIL_POINT_ROWS": False, "PLASMA_SURFACE_ROWS": False, "SLSQP_RESUME_AFTER_REJECTION": False,
            "SEGMENT_SCALES": False, "MERCIER_FLOOR": 0.0, "MERCIER_MARGIN": 0.02}


def _import(overrides):
    """The driver for COIL_CASE qa4-beta with ``overrides`` (COIL_CASE_OVERRIDES), environment restored after."""
    saved = {key: os.environ.get(key) for key in ("COIL_CASE", "COIL_CASE_OVERRIDES")}
    os.environ.update(COIL_CASE="qa4-beta", COIL_CASE_OVERRIDES=json.dumps(overrides))
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        sys.modules.pop(MODULE, None)
        return importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop(MODULE, None)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_defaults_are_the_current_formulation():
    driver = _import({})
    assert driver.COIL_POINT_ROWS and driver.PLASMA_SURFACE_ROWS
    assert driver.SLSQP_RESUME_AFTER_REJECTION and driver.SEGMENT_SCALES
    assert driver.MERCIER_MARGIN == 0.0


def test_campaign_coil_rows_are_one_per_coil_and_kind():
    import jax

    jax.config.update("jax_enable_x64", True)
    from essos.coils import Coils

    driver = _import(CAMPAIGN)
    assert not driver.COIL_POINT_ROWS and not driver.PLASMA_SURFACE_ROWS and driver.MERCIER_MARGIN == 0.02
    coils = driver.resize_coils(Coils.from_json(str(driver.COILS_FILE)), driver.COIL_ORDER, driver.N_SEGMENTS)
    nbase = coils.dofs_curves.shape[0]
    chart = driver.CoilChart(coils, current_dofs=(), scales=np.ones(np.asarray(coils.dofs_curves).size))
    rows = driver.coil_constraint(chart.coils_from_x, None, 64)
    values = rows.fun(chart.x0)
    # length, peak curvature, MSC and minimum speed of each base coil, one coil-coil and one self distance
    assert values.shape == (4 * nbase + 2,)
    np.testing.assert_allclose(values, np.asarray(driver.coil_inequalities(coils)), rtol=0, atol=1e-12)
    assert rows.jac(chart.x0).shape == (values.size, chart.x0.size)
