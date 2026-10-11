#!/bin/bash
# Second (2-node, 16-GPU, 4 h) interactive allocation for the finite-beta force front C, requested as soon as the user's
# other interactive job has ended (interactive QOS: 2 jobs per user).  14 driver runs + 2 evaluation lanes (extras_second.sh).
#   setsid nohup bash second_alloc.sh <other_jobid_to_wait_for> > runs/second_alloc.log 2>&1 &
set -u
OTHER=${1:-}
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
if [ -n "$OTHER" ]; then
  while squeue -h -j $OTHER -o "%T" 2>/dev/null | grep -q -E "RUNNING|PENDING|COMPLETING"; do sleep 120; done
fi
echo "$(date): requesting the second allocation (front C)"
salloc -q interactive --constraint=gpu -A m4656_g --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
  -t 04:00:00 bash /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/run_batch2.sh \
  /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/jobs_second.txt > runs/second_salloc.log 2>&1 < /dev/null
echo "$(date): second allocation ended"
