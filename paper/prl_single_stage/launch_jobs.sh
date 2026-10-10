#!/bin/bash
# Launch the jobs of a jobs file into a running allocation (fresh starts or prepared restart dirs only):
#   bash launch_jobs.sh <jobid> <jobs.txt>
set -u
J=$1; JOBS=$2
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
OPT=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization
T=$OPT/single_stage_free_boundary_optimization_three_term.py; F=$OPT/single_stage_optimization_coil_constraints.py
while IFS='|' read -r name drv case ovr extra src; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  out=runs/batch/$name; n=0; while [ -e "$out" ]; do n=$((n + 1)); out=runs/batch/$name-c$n; done
  args=""; [ "$extra" != "-" ] && args="$extra"; [ "$src" != "-" ] && args="$args --restart $src"
  script=$T; [ "$drv" = "F" ] && script=$F
  echo "$(date +%H:%M) launch $name ($drv $case) -> $out : $args"
  env COIL_CASE=$case ${ovr:+"COIL_CASE_OVERRIDES=$ovr"} srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0 \
    $PY $script --output $out --steps 1000 --save-every 1 $args < /dev/null > $out.log 2>&1 &
  sleep 2
done < "$JOBS"
wait
