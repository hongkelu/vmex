#!/bin/bash
# remove stale evaluation locks: a lock whose row.json exists, or whose set has no running evaluator step.  bash clean_locks.sh
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
for L in runs/lpqa_vacuum_eval_8x8/*.lock runs/lpqa_vacuum_eval_12x12/*.lock; do
  [ -e "$L" ] || continue; d=${L%.lock}; n=$(basename $d)
  if [ -s $d/row.json ]; then rm -f "$L"; echo "cleared (done) $n"; continue; fi
  pgrep -f "eval_coilsets_vacuum.py coils/$n.json" >/dev/null || { rm -f "$L"; echo "cleared (no evaluator) $n"; }
done
