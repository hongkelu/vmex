#!/bin/bash
# One job per GPU across the whole allocation.  Run INSIDE salloc:  salloc -N4 ... bash run_batch.sh <jobs.txt> [old_jobid]
# jobs.txt lines:  name|T or F|COIL_CASE|overrides-json or -|extra driver args or -|restart source run dir or -
# A restart source is turned into a --restart directory from its latest saved step (make_restart.py);
# if old_jobid is given it is cancelled once all restart directories are built.
set -u
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
export JAX_PLATFORMS=cuda,cpu
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
OPT=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization
T=$OPT/single_stage_free_boundary_optimization_three_term.py
F=$OPT/single_stage_optimization_coil_constraints.py
JOBS=$1; OLD=${2:-}
JOBID=${SLURM_JOB_ID:?run inside salloc or with SLURM_JOB_ID=<jobid> set}
mkdir -p runs/batch
echo "allocation $SLURM_JOB_ID on $SLURM_NODELIST, $(date), jobs from $JOBS"

# pass 1: restart directories
declare -A RESTART
while IFS='|' read -r name drv case ovr extra src; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [ "$src" != "-" ]; then
    if [ -e "$src/wout.nc" ] && [ -e "$src/coils.json" ] && [ -z "$(ls $src/wout.step*.nc 2>/dev/null)" ]; then
      RESTART[$name]=$src      # an already prepared --restart directory
    else
      rdir=runs/batch/restart_${name}_$JOBID
      $PY make_restart.py "$src" "$rdir" && RESTART[$name]=$rdir || echo "restart build failed for $name"
    fi
  fi
done < "$JOBS"
[ -n "$OLD" ] && { scancel "$OLD"; echo "cancelled old allocation $OLD"; sleep 5; }

# pass 2: launch
while IFS='|' read -r name drv case ovr extra src; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  out=runs/batch/$name; n=0
  while [ -e "$out" ]; do n=$((n + 1)); out=runs/batch/$name-c$n; done
  args=""; [ "$extra" != "-" ] && args="$extra"
  [ -n "${RESTART[$name]:-}" ] && args="$args --restart ${RESTART[$name]}"
  script=$T; [ "$drv" = "F" ] && script=$F
  env_ovr=""; [ "$ovr" != "-" ] && env_ovr="COIL_CASE_OVERRIDES=$ovr"
  echo "launch $name ($drv $case) -> $out : $args ${env_ovr:+[$env_ovr]}"
  env COIL_CASE=$case ${env_ovr:+"$env_ovr"} srun --jobid=$JOBID --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0 \
    $PY $script --output $out --steps 1000 --save-every 1 $args < /dev/null > $out.log 2>&1 &
  sleep 2
done < "$JOBS"
wait
echo "allocation ended $(date)"
