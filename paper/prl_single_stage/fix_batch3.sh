#!/bin/bash
# Replace the four stale-checkpoint steps of allocation 59624380 and launch the job that failed to parse.
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh
J=59624380
OPT=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization
T=$OPT/single_stage_free_boundary_optimization_three_term.py
echo "$(date): cancelling steps .0 .2 .3 .5"; scancel $J.0 $J.2 $J.3 $J.5; sleep 15
for d in vac-L18-c2 vac-L20-c3 qa4-free-vac031-c2 qa6-free-c2; do mv runs/batch/$d runs/batch/stale3_$d; mv runs/batch/$d.log runs/batch/stale3_$d.log; done
export JAX_PLATFORMS=cpu
for p in "vac-L18 vac-L18-c1" "vac-L20 vac-L20-c2" "qa4-free-vac031 qa4-free-vac031-c1" "qa6-free qa6-free-c1"; do set -- $p
  $PY make_restart.py runs/batch/$2 runs/batch/restart_${1}_fix3 2>&1 | grep "restart from"; done
export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
launch() { # name case out overrides extra...
  local name=$1 case=$2 out=$3 ovr=$4; shift 4
  echo "launch $name -> $out : $* [${ovr}]"
  env COIL_CASE=$case ${ovr:+"COIL_CASE_OVERRIDES=$ovr"} $SR $PY $T --output $out --steps 1000 --save-every 1 "$@" < /dev/null > $out.log 2>&1 &
  sleep 2; }
launch vac-L18 lpqa-L18 runs/batch/vac-L18-c2 "" --design-step-scale 0.2 --restart runs/batch/restart_vac-L18_fix3
launch vac-L20 lpqa-L20 runs/batch/vac-L20-c3 "" --design-step-scale 0.2 --restart runs/batch/restart_vac-L20_fix3
launch qa4-free-vac031 qa4-beta runs/batch/qa4-free-vac031-c2 '{"VACUUM_IOTA_FLOOR": 0.31}' --beta 0.025 --bootstrap --design-step-scale 0.2 --restart runs/batch/restart_qa4-free-vac031_fix3
launch qa6-free qa4-beta runs/batch/qa6-free-c2 '{"SEED": [2, 6.0, 0.5], "ASPECT_RANGE": [5.9, 6.1]}' --beta 0.025 --bootstrap --design-step-scale 0.2 --restart runs/batch/restart_qa6-free_fix3
( echo "$(date): scorer qa4-fixed-c1"; mkdir -p runs/fig2
  $SR $PY eval_steps_freeboundary.py runs/batch/qa4-fixed-c1 runs/fig2/qa4-fixed-c1 --every 10 < /dev/null > runs/fig2/qa4-fixed-c1.log 2>&1
  echo "$(date): scorer done, launching vac-L24-feas2"
  launch vac-L24-feas2 lpqa-L24 runs/batch/vac-L24-feas2 "$(cat runs/ovr_L24feas2.json)" --design-step-scale 0.2
  wait ) &
wait; echo "$(date): fix_batch3 all steps ended"
