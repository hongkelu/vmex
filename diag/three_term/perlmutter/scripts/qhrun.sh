#!/bin/bash
# usage: qhrun.sh <jobid> <tree> <name> <extra driver args...>   (one GPU of the allocation)
JOB=$1; TREE=$2; NAME=$3; shift 3
R=/pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/runs/pr121fix/qh
srun --jobid=$JOB --overlap -N1 -n1 --gpus=1 bash -c "source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh; cd $TREE; \
  export PYTHONPATH=$TREE COIL_CASE=qh4-beta-A8-merc XLA_PYTHON_CLIENT_PREALLOCATE=false; \
  echo TREE \$(git rev-parse --short=8 HEAD) \$(date); \
  \$PY -u examples/optimization/single_stage_free_boundary_optimization_three_term.py --modes 8 8 --ns 51 --save-every 1 \
    --coil-point-rows --beta 0.025 --bootstrap --design-step-scale 1.0 --max-iterations 20000 $* --output $R/$NAME" \
  > $R/$NAME.log 2>&1
echo END $? $(date) >> $R/$NAME.log
