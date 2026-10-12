#!/bin/bash
# Self-re-arming chain (v2): optionally wait for a running allocation first; preflight before every round.
#   setsid nohup bash chain_loop2.sh <name> <jobs.txt> "<RES:LIST:N:ORDER ...>" [wait_jobid] > runs/chain_<name>.log 2>&1 &
# Stop with: touch runs/STOP_<name>
set -u
NAME=$1; PREV=$2; LANES=${3:-}; WAITJOB=${4:-}; i=0
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh
if [ -n "$WAITJOB" ]; then
  while squeue -h -j $WAITJOB -o %T 2>/dev/null | grep -q -E 'RUNNING|PENDING|COMPLETING'; do sleep 60; done
  echo "$(date): allocation $WAITJOB ended"; sleep 30
fi
while [ ! -e runs/STOP_$NAME ]; do
  i=$((i + 1)); TAG=${NAME}_$(date +%m%d%H%M); NEXT=jobs_$TAG.txt
  $PY make_batch_restart.py $PREV $NEXT
  JAX_PLATFORMS=cpu timeout 1800 $PY preflight.py $NEXT > runs/preflight_$TAG.log 2>&1 || echo "$(date): preflight reports infeasible starts, see runs/preflight_$TAG.log"
  for spec in $LANES; do IFS=: read -r R L N O <<< "$spec"
    (RES=$R LIST=$L setsid nohup bash extras_lanes.sh runs/${TAG}_salloc.log $N 1 $O > runs/extras_${TAG}_$R.log 2>&1 < /dev/null &)
  done
  echo "$(date): round $i -> $TAG"
  salloc -q interactive --constraint=gpu -A m4656_g --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
    -t 04:00:00 bash $PWD/run_batch2.sh $PWD/$NEXT > runs/${TAG}_salloc.log 2>&1 < /dev/null
  echo "$(date): $TAG ended: $(grep -a -m1 -E 'error' runs/${TAG}_salloc.log)"
  grep -a -q '^launch' runs/${TAG}_salloc.log && PREV=$NEXT || sleep 300
done
