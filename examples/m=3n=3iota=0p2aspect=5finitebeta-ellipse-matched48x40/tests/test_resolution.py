import dataclasses
import json  # noqa: F401
from pathlib import Path
import numpy as np
from geometry import rotating_ellipse
from vmex.core.input import VmecInput
from vmex.core.solver import resolution_from_input
from vmex.core.freeboundary import free_boundary_resolution

ROOT = Path(__file__).resolve().parents[1]


def test_all_radial_stages_and_free_boundary_keep_48x40():
    ref = VmecInput.from_file(ROOT / "inputs/reference_profiles.json")
    inp = rotating_ellipse(
        ref, major_radius=11.067033897730942, aspect=5, t=0.25188920340834803, handedness=-1, phiedge=91.09117631931035
    )
    assert (inp.mpol, inp.ntor, inp.nfp) == (3, 3, 2)
    for ns in inp.ns_array:
        r = resolution_from_input(inp, ns=ns)
        assert (r.ntheta, r.ntheta3, r.nzeta) == (48, 25, 40)
    free = dataclasses.replace(inp, lfreeb=True, mgrid_file="DIRECT_ESSOS_BIOT_SAVART", ns_array=(50,))
    r = free_boundary_resolution(free, None)
    assert free.lfreeb and (r.ns, r.ntheta, r.ntheta3, r.nzeta) == (50, 48, 25, 40)
    for n in (
        "ncurr",
        "curtor",
        "pres_scale",
        "pmass_type",
        "am",
        "am_aux_s",
        "am_aux_f",
        "pcurr_type",
        "ac",
        "ac_aux_s",
        "ac_aux_f",
    ):
        np.testing.assert_array_equal(getattr(inp, n), getattr(ref, n))
    assert np.count_nonzero(inp.rbc) == 3 and np.count_nonzero(inp.zbs) == 2
