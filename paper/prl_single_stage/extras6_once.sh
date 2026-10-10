#!/bin/bash
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
J=59644067; SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
latest_dir() { ls -d runs/batch/$1/ runs/batch/$1-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2; }
# stage two on the last SAVED stage-one step (c3 had none at injection time)
( S1=runs/batch/qa4-stage1-c2; N=$(ls $S1/wout.step*.nc | sed -E 's/.*step([0-9]+)\.nc/\1/' | sort -n | tail -1); O=runs/stage_two/qa4-stage1-c2-step$N
  echo "$(date +%H:%M) stage two on $S1 step $N"
  COIL_CASE=qa4-beta $SR $PY stage_two_fit.py --wout $S1/wout.step$N.nc --input $S1/input.run --coils $DATA/ESSOS_coils_qa4_beta.json --out $O --weights 1e3,1e5 --maxiter 1500 --slsqp 400 < /dev/null > $O.log 2>&1 && \
  COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1
  COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $S1 runs/fig2/qa4-stage1-c2 --last < /dev/null > runs/fig2/qa4-stage1-c2.log 2>&1 ) &
sleep 2
( for r in vac-L18-fair vac-3coil-fair vac-L20-fair vac-L24-wech-fair vac-L18-exact vac-L20-exact vac-3coil-exact vac-L18-c2 vac-L20-c3 vac-3coil-c2 vac-L24-wech-feas2 vac-L24-feas2; do
    d=$(latest_dir $r); [ -z "$d" ] && continue
    grep -q "own LCFS (v3)" $d.gilscore.log 2>/dev/null && grep -q "^done" $d.gilscore.log 2>/dev/null && continue
    $SR bash score_run_gil.sh $d < /dev/null > $d.gilscore.log 2>&1; done ) &
sleep 2
( while timeout 30 sacct -j $J -n -o JobID,State 2>/dev/null | grep RUNNING | awk '{print $1}' | sed 's/.*\.//' | awk '$1 ~ /^[0-9]+$/ && $1>=12' | grep -q .; do sleep 60; done
  echo "$(date +%H:%M) leftover extras step finished; sweep"; bash eval_sweep_batch4.sh $J 12x12 --modes 12 12 --vc-grid 64 > runs/eval_sweep_12x12.b6.log 2>&1 ) &
wait; echo "$(date): extras6_once done"
