#!/bin/bash
# Gil-convention force/B.n metrics of the latest saved coils of every force-front run (ours_<run> keys in
# runs/force_front_metrics.json), for analysis/fig_force_front.py.   bash ours_force_metrics.sh
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh >/dev/null 2>&1; export JAX_PLATFORMS=cpu MPICH_GPU_SUPPORT_ENABLED=0
args=""
for name in $(ls -d runs/batch/ff*/ | sed 's:runs/batch/::; s:/$::' | sed -E 's/-c[0-9]+$//' | sort -u); do
  d=$(ls -d runs/batch/$name/ runs/batch/$name-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2)
  f=$(ls $d/coils.step*.json 2>/dev/null | sed -E 's/.*step([0-9]+)\.json/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -n "$f" ] && args="$args ours_$name=$f"
done
echo "$(echo $args | wc -w) runs"
$PY force_front_metrics.py runs/force_front_metrics.json $args 2>&1 | grep -E "^(ours_|wrote|Trace|  File|[A-Za-z]+Error)"
