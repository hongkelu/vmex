#!/bin/bash
# The production boundary-Jacobian check (driver --check-jacobian) on a frozen fixture, without a free-boundary fit.
# usage: check_jacobian_fixture.sh CASE OUT N [driver arguments, e.g. --jacobian-at RECORD.npz --labelling start]
# Run from the root of a vmex tree on a GPU node, with the environment of paper/env.sh.
set -euo pipefail
CASE=$1; OUT=$2; N=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
FIX=$(cd "$HERE/../fixtures/$CASE" && pwd)
(cd "$FIX" && sha256sum -c SHA256SUMS --quiet) || { echo "fixture $CASE fails its checksums"; exit 1; }
read -r COIL_CASE DRIVER_ARGS < <(python3 -c "
import json; d = json.load(open('$FIX/fixture.json')); print(d['coil_case'], d['driver_args'])")
export COIL_CASE PYTHONPATH=$PWD
# shellcheck disable=SC2086
exec "${PY:-python}" -u examples/optimization/single_stage_free_boundary_optimization_three_term.py --output "$OUT" \
    $DRIVER_ARGS --seed-run "$FIX" --steps 0 --check-jacobian "$N" "$@"
