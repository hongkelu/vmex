#!/bin/bash
# Two evaluation lanes on the 2 spare GPUs of the second allocation (front C uses 6 of 8): continues the published-set
# list from the end (the batch-9 lanes work from the start), skipping rows already done.
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until [ "$(grep -c '^launch' runs/second_salloc.log 2>/dev/null)" -ge 14 ]; do sleep 60; done
J=$(grep -m1 "^allocation" runs/second_salloc.log | awk '{print $2}'); echo "$(date): extras into second allocation $J"; sleep 30
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
OUT=runs/lpqa_vacuum_eval_12x12
ev() { [ -s $OUT/$1/row.json ] && return; [ -e $OUT/$1.lock ] && return; touch $OUT/$1.lock; echo "$(date +%H:%M) eval $1"; $SR $PY eval_coilsets_vacuum.py coils/$1.json $OUT/$1 --modes 12 12 --vc-grid 64 < /dev/null > $OUT/$1.log 2>&1; rm -f $OUT/$1.lock; }
until [ -s runs/front_eval_list.txt ]; do sleep 30; done
mapfile -t SETS < <(tac runs/front_eval_list.txt)
for lane in 0 1; do ( for ((i = lane; i < ${#SETS[@]}; i += 2)); do ev ${SETS[$i]}; done ) & sleep 3; done
wait; echo "$(date): extras_second done"
