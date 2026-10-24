#!/bin/bash
# usage: onnode.sh <jobid> <cmdfile>   -- runs each line of <cmdfile> on its own GPU (CUDA_VISIBLE_DEVICES=line index)
# of the allocation's node, inside one srun step that owns all 4 GPUs. Lines: "<logfile> <command...>".
JOB=$1; CMDS=$2; FIRST=${3:-0}
srun --jobid=$JOB --overlap -N1 -n1 --gpus=4 ${NODE:+-w $NODE} bash -c '
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
i='"$FIRST"'
while read -r log cmd; do
  [ -z "$log" ] && continue
  CUDA_VISIBLE_DEVICES=$i bash -c "$cmd; echo EXIT \$?" > $log 2>&1 &
  i=$((i+1))
done < '"$CMDS"'
wait'
