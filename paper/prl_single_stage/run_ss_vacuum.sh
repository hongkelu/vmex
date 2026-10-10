#!/bin/bash
# Vacuum free-boundary single-stage runs on the LP QA benchmark, one case per GPU.
# Run INSIDE an interactive allocation:  salloc ... bash run_ss_vacuum.sh CASE[:restart_dir] ...
# Each case writes runs/ss_vac/<CASE>[-cN]/ ; a killed run is continued with make_restart.py.
set -u
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
export JAX_PLATFORMS=cuda,cpu
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
T=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization/single_stage_free_boundary_optimization_three_term.py
mkdir -p runs/ss_vac
echo "allocation $SLURM_JOB_ID on $SLURM_NODELIST, $(date): $*"
gpu=0
for spec in "$@"; do
  case=${spec%%:*}
  restart=""
  [[ "$spec" == *:* ]] && restart=${spec#*:}
  out=runs/ss_vac/$case
  n=0
  while [ -e "$out" ]; do n=$((n + 1)); out=runs/ss_vac/$case-c$n; done
  extra=""
  if [ -n "$restart" ]; then
    rdir=runs/ss_vac/restart_$(basename $out)
    $PY make_restart.py "$restart" "$rdir" || { echo "restart build failed for $case"; continue; }
    extra="--restart $rdir"
  fi
  echo "launch $case -> $out on GPU $gpu $extra"
  CUDA_VISIBLE_DEVICES=$gpu COIL_CASE=$case srun --exact -N1 -n1 -c16 --gpus-per-task=1 --gpu-bind=none --mem=0 \
    $PY $T --output $out --steps 1000 --save-every 1 $extra > $out.log 2>&1 &
  gpu=$((gpu + 1))
done
wait
echo "allocation ended $(date)"
