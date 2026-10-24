#!/bin/bash
# usage: launch_round.sh <jobid> <tag>
# QH calibration runs (6) + the 8-way test suite at the head of vmex-pr121fix, one process per GPU of the 4 nodes.
set -euo pipefail
JOB=$1; TAG=$2
P=/pscratch/sd/h/hongkelu/freeboundary-single-stage
H=$(git -C "$P/vmex-pr121fix" rev-parse --short=8 HEAD)
T=$P/vmex-qhrun-$H
TT=$P/vmex-pin-$H
[ -d "$T" ] && [ -d "$TT" ] || { echo "missing trees for $H"; exit 1; }
D=$P/paper/runs/pr121fix
cd "$D"
R=$D/qh; Q=$D/qhprobe
ARGS="--modes 8 8 --ns 51 --save-every 1 --coil-point-rows --beta 0.025 --bootstrap --design-step-scale 1.0 --max-iterations 20000 --seed-run $R/fixture-qh25 --steps 5"
mk() { echo "$Q/$1.log cd $T && CALIBRATE=1 NORMAL_TOL=$2 PYTHONPATH=$T COIL_CASE=qh4-beta-A8-merc \$PY -u $Q/qhprobe.py $Q/$1 $ARGS $3 --output $Q/$1-run"; }
{ mk "nt3$TAG-C" 1e-3 "--labelling equal_arc --label-weight 1e-3"
  mk "nt3$TAG-NL" 1e-3 "--labelling start --label-weight 0"
  mk "nt3$TAG-C2" 1e-3 "--labelling equal_arc --label-weight 1e-2"
  mk "nt5$TAG-C" 5e-4 "--labelling equal_arc --label-weight 1e-3"; } > "$D/round$TAG.A"
{ mk "nt5$TAG-NL" 5e-4 "--labelling start --label-weight 0"
  mk "nt5$TAG-C2" 5e-4 "--labelling equal_arc --label-weight 1e-2"; } > "$D/round$TAG.B"
N=$D/suite-$H; mkdir -p "$N"
source "$P/paper/env.sh"
(cd "$TT" && JAX_PLATFORMS=cpu RUN_FULL=1 PYTHONPATH=$TT $PY -m pytest --collect-only -q tests/test_freeboundary_vc.py tests/test_bootstrap.py \
  -k "not nestor and not desc and not test_checkpoint_restores_the_accepted_equilibrium" 2>/dev/null | grep "::" > "$N/ids.txt")
echo "$(wc -l < "$N/ids.txt") test ids at $H"
for g in 0 1 2 3 4 5 6 7; do awk -v g=$g 'NR%8==g' "$N/ids.txt" > "$N/ids.$g"; done
line() { echo "$N/log.$1 cd $TT && PYTHONPATH=$TT RUN_FULL=1 \$PY -m pytest -q -p no:cacheprovider \$(cat $N/ids.$1)${2:-}"; }
{ line 0; line 1; } >> "$D/round$TAG.B"
: > "$D/round$TAG.C"; for g in 2 3 4 5; do line $g >> "$D/round$TAG.C"; done
{ line 6; line 7 "; XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$TT RUN_FULL=1 \$PY -m pytest -q -p no:cacheprovider tests/test_freeboundary_vc.py::test_checkpoint_restores_the_accepted_equilibrium"; } > "$D/round$TAG.D"
for spec in "nid200333 A" "nid200393 B" "nid200396 C" "nid002804 D"; do
  set -- $spec
  NODE=$1 setsid nohup ./onnode.sh "$JOB" "$D/round$TAG.$2" 0 > "$D/round$TAG.$2.out" 2>&1 < /dev/null &
done
echo "launched round $TAG at $H"
