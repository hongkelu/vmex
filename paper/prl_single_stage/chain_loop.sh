#!/bin/bash
# Self-re-arming chain: forever (until runs/STOP_<name> exists) rebuild restarts from the previous jobs file, start lanes, salloc.
#   setsid nohup bash chain_loop.sh <name> <jobs_start.txt> "<lane spec>" > runs/chain_<name>.log 2>&1 &
# lane spec: space-separated RES:LIST:N:ORDER entries, e.g. "12x12:runs/envelope_candidates.txt:3:forward"
set -u
NAME=$1; PREV=$2; LANES=${3:-}; i=0
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh
while [ ! -e runs/STOP_$NAME ]; do
  i=$((i + 1)); TAG=${NAME}_$(date +%m%d%H%M); NEXT=jobs_$TAG.txt
  $PY make_batch_restart.py $PREV $NEXT
  for spec in $LANES; do IFS=: read -r R L N O <<< "$spec"
    (RES=$R LIST=$L setsid nohup bash extras_lanes.sh runs/${TAG}_salloc.log $N 1 $O > runs/extras_${TAG}_$R.log 2>&1 < /dev/null &)
  done
  echo "$(date): round $i -> $TAG"
  salloc -q interactive --constraint=gpu -A m4656_g --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
    -t 04:00:00 bash $PWD/run_batch2.sh $PWD/$NEXT > runs/${TAG}_salloc.log 2>&1 < /dev/null
  echo "$(date): $TAG ended: $(grep -a -m1 -E 'error' runs/${TAG}_salloc.log)"
  grep -a -q '^launch' runs/${TAG}_salloc.log && PREV=$NEXT || sleep 300   # failed request: retry with the same list
done
