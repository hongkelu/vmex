#!/bin/bash
# Gil-convention force/B.n metrics of the chosen coils (coils/ours_ff*.json from prep_ours_list.py) -> ours_<run> keys in
# runs/force_front_metrics.json.   bash ours_force_metrics.sh
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh >/dev/null 2>&1; export JAX_PLATFORMS=cpu MPICH_GPU_SUPPORT_ENABLED=0
args=""; for f in coils/ours_ff*.json; do n=$(basename $f .json); args="$args $n=$f"; done
echo "$(echo $args | wc -w) runs"
$PY force_front_metrics.py runs/force_front_metrics.json $args 2>&1 | grep -E "^(ours_|wrote|Trace|  File|[A-Za-z]+Error)"
