#!/bin/bash
# Reproduce a frozen fixture's seed solve and its complete diagnostic record in a fresh process.
#
# usage: reproduce_fixture.sh CASE OUT [labelling arguments]
#   CASE: qh25 | qa25 | qi25 | qavac | qhvac | qivac (diag/three_term/perlmutter/fixtures/CASE)
#   OUT:  a new directory (fbrecord.py refuses an existing one)
#   labelling arguments: default "--labelling equal_arc --label-weight 1e-3"
# Run from the root of a vmex tree on a GPU node, with the environment of paper/env.sh.  Extra environment
# (POLISH_ROUNDS, POLISH_MAX_NFEV, STALL_STEPS, MAX_MODE) is passed through and recorded in OUT/manifest.json.
set -euo pipefail
CASE=$1; OUT=$2; shift 2
LABEL=${*:-"--labelling equal_arc --label-weight 1e-3"}
HERE=$(cd "$(dirname "$0")" && pwd)
FIX=$(cd "$HERE/../fixtures/$CASE" && pwd)
(cd "$FIX" && sha256sum -c SHA256SUMS --quiet) || { echo "fixture $CASE fails its checksums"; exit 1; }
read -r COIL_CASE DRIVER_ARGS < <(python3 -c "
import json; d = json.load(open('$FIX/fixture.json')); print(d['coil_case'], d['driver_args'])")
export COIL_CASE PYTHONPATH=$PWD
# shellcheck disable=SC2086
exec "${PY:-python}" -u "$HERE/fbrecord.py" "$OUT" $DRIVER_ARGS --seed-run "$FIX" --steps 0 --save-every 1 $LABEL
