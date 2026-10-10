#!/bin/bash
# Wait for the running allocation to end, then request the next one with every run continued.
#   bash chain_next.sh <jobid> <jobs_prev.txt> <tag>
# Keeps exactly one allocation for this project.  Logs to runs/chain_<tag>.log
set -u
JOBID=$1; PREV=$2; TAG=$3
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
while squeue -h -j $JOBID -o "%T" 2>/dev/null | grep -q -E "RUNNING|PENDING|COMPLETING"; do sleep 120; done
echo "$(date): allocation $JOBID ended; building restarts"
sleep 30
/global/homes/h/hongkelu/.conda/envs/uwplasma-env/bin/python make_batch_restart.py $PREV jobs_$TAG.txt
salloc -q interactive --constraint=gpu -A m4656_g --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
  -t 04:00:00 bash /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/run_batch2.sh \
  /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/jobs_$TAG.txt > runs/${TAG}_salloc.log 2>&1 < /dev/null
echo "$(date): allocation for $TAG ended"
