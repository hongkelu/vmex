#!/bin/bash
# Relaunch the two finite-beta restarts whose override env was word-split, then vac-L24-feas2 after the scorer.
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
J=59624380; T=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization/single_stage_free_boundary_optimization_three_term.py
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
launch() { local name=$1 case=$2 out=$3 ovr=$4; shift 4
  echo "$(date +%H:%M) launch $name -> $out : $* [${ovr}]"
  env COIL_CASE=$case ${ovr:+"COIL_CASE_OVERRIDES=$ovr"} $SR $PY $T --output $out --steps 1000 --save-every 1 "$@" < /dev/null > $out.log 2>&1 &
  sleep 2; }
mv runs/batch/qa4-free-vac031-c2.log runs/batch/stale3_qa4-free-vac031-c2.failedlaunch.log; mv runs/batch/qa6-free-c2.log runs/batch/stale3_qa6-free-c2.failedlaunch.log
launch qa4-free-vac031 qa4-beta runs/batch/qa4-free-vac031-c2 '{"VACUUM_IOTA_FLOOR": 0.31}' --beta 0.025 --bootstrap --design-step-scale 0.2 --restart runs/batch/restart_qa4-free-vac031_fix3
launch qa6-free qa4-beta runs/batch/qa6-free-c2 '{"SEED": [2, 6.0, 0.5], "ASPECT_RANGE": [5.9, 6.1]}' --beta 0.025 --bootstrap --design-step-scale 0.2 --restart runs/batch/restart_qa6-free_fix3
while pgrep -f "[e]val_steps_freeboundary.py runs/batch/qa4-fixed-c1" > /dev/null; do sleep 30; done
echo "$(date +%H:%M) scorer finished ($(wc -l < runs/fig2/qa4-fixed-c1/steps.jsonl) rows)"
launch vac-L24-feas2 lpqa-L24 runs/batch/vac-L24-feas2 "$(cat runs/ovr_L24feas2.json)" --design-step-scale 0.2
wait; echo "$(date): fix_batch3b all steps ended"
