#!/bin/bash
# Generic evaluation lanes for a chained allocation: wait until <salloc_log> shows >= <nlaunch> launches, then run
# <lanes> lanes of eval_coilsets_vacuum.py (12x12 / VC 64) over runs/front_eval_list.txt, skipping rows already done
# or locked by another allocation's lanes.   setsid nohup bash extras_lanes.sh runs/second2_salloc.log 2 14 [reverse] &
set -u
LOG=$1; LANES=$2; NL=$3; ORDER=${4:-forward}
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until [ "$(grep -c '^launch' $LOG 2>/dev/null)" -ge "$NL" ] 2>/dev/null; do sleep 60; done
J=$(grep -m1 "^allocation" $LOG | awk '{print $2}'); echo "$(date): $LANES lanes into $J ($ORDER)"; sleep 30
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
OUT=runs/lpqa_vacuum_eval_12x12
ev() { [ -s $OUT/$1/row.json ] && return; [ -e $OUT/$1.lock ] && return; touch $OUT/$1.lock; echo "$(date +%H:%M) eval $1"
      $SR $PY eval_coilsets_vacuum.py coils/$1.json $OUT/$1 --modes 12 12 --vc-grid 64 < /dev/null > $OUT/$1.log 2>&1; rm -f $OUT/$1.lock; }
if [ "$ORDER" = "reverse" ]; then mapfile -t SETS < <(tac runs/front_eval_list.txt); else mapfile -t SETS < runs/front_eval_list.txt; fi
for ((lane = 0; lane < LANES; lane++)); do ( for ((i = lane; i < ${#SETS[@]}; i += LANES)); do ev ${SETS[$i]}; done ) & sleep 3; done
wait; echo "$(date): lanes done"
