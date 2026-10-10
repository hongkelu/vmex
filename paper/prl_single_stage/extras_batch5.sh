#!/bin/bash
# Wait for the allocation-4 chain to launch its last driver job, then inject the stage-two / scoring tasks.
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until grep -q "^launch vac-3coil-exact" runs/batch5_salloc.log 2>/dev/null; do sleep 60; done
J=$(grep -m1 "^allocation" runs/batch5_salloc.log | awk '{print $2}')
echo "$(date): injecting extras into $J"; sleep 60
bash stage_two_batch5.sh $J
