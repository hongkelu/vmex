#!/bin/bash
# Extras injected into allocation 6:  bash stage_two_batch6.sh <jobid>   (3 GPUs)
#  1: remaining 12x12 rescoring (skips rows already done)
#  2: stage two (penalty + SLSQP hard rows) on the pure stage-one boundary (qa4-stage1, latest step) -> the equilibrium it makes
#  3: Gil-pipeline scores still missing (score_run_gil.sh with the QFM seeded from the run's LCFS)
#  4/5: Fig-2 scorers of the newest fixed-arm and free-arm continuations
set -u
J=$1
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
latest_dir() { ls -d runs/batch/$1/ runs/batch/$1-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2; }
latest_step() { ls $1/wout.step*.nc 2>/dev/null | sed -E 's/.*step([0-9]+)\.nc/\1/' | sort -n | tail -1; }
( bash eval_sweep_batch4.sh $J 12x12 --modes 12 12 --vc-grid 64 > runs/eval_sweep_12x12.b5.log 2>&1 ) &
sleep 2
( S1=$(latest_dir qa4-stage1); N=$(latest_step $S1); O=runs/stage_two/qa4-stage1-slsqp; echo "stage one: $S1 step $N"
  if [ -n "$N" ]; then
    COIL_CASE=qa4-beta $SR $PY stage_two_fit.py --wout $S1/wout.step$N.nc --input $S1/input.run --coils $DATA/ESSOS_coils_qa4_beta.json \
      --out $O --weights 1e3,1e5 --maxiter 1500 --slsqp 400 < /dev/null > $O.log 2>&1 && \
    COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1
    # and the stage-one coils the F driver carried along (its own fit under the same rows), for reference
    COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $S1 runs/fig2/$(basename $S1) --last < /dev/null > runs/fig2/$(basename $S1).log 2>&1
  fi ) &
sleep 2
( for p in runs/published/*; do grep -q '^done' $p.gilscore.log 2>/dev/null || $SR bash score_run_gil.sh $p < /dev/null > $p.gilscore.log 2>&1; done
  for r in vac-L18-fair vac-3coil-fair vac-L20-fair vac-L24-wech-fair vac-L18-exact vac-L20-exact vac-3coil-exact vac-L18-c2 vac-L20-c3 vac-3coil-c2 vac-L24-wech-feas2 vac-L24-feas2; do
    d=$(latest_dir $r); [ -z "$d" ] && continue
    grep -q "own LCFS (v3)" $d.gilscore.log 2>/dev/null && grep -q "^done" $d.gilscore.log 2>/dev/null && continue
    $SR bash score_run_gil.sh $d < /dev/null > $d.gilscore.log 2>&1; done ) &
sleep 2
wait; echo "$(date): stage_two_batch6 done"
