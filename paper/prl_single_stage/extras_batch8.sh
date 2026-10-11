#!/bin/bash
# Allocation 8 extras (after the chain's 11 launches): 2 Pareto points at Gil's exact lengths, the Wechsung cloud
# evaluation (32 sets at 12x12, two GPUs), Gil-pipeline v3 scoring (one GPU), stage two on the latest stage-one step (one GPU).
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until grep -q "^launch pareto-L24" runs/batch8_salloc.log 2>/dev/null; do sleep 60; done
J=$(grep -m1 "^allocation" runs/batch8_salloc.log | awk '{print $2}'); echo "$(date): extras into $J"; sleep 60
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
latest_dir() { ls -d runs/batch/$1/ runs/batch/$1-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2; }
latest_step() { ls $1/wout.step*.nc 2>/dev/null | sed -E 's/.*step([0-9]+)\.nc/\1/' | sort -n | tail -1; }
# 1. extra Pareto points
> jobs_pareto_extra_resolved.txt
while IFS='|' read -r name drv case ovr extra src; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  d=$(latest_dir $src); r=runs/batch/restart_${name}_$J
  JAX_PLATFORMS=cpu $PY make_restart.py $d $r 2>&1 | grep "restart from"
  echo "$name|$drv|$case|$ovr|$extra|$r" >> jobs_pareto_extra_resolved.txt
done < jobs_pareto_extra.txt
bash launch_jobs.sh $J jobs_pareto_extra_resolved.txt > runs/launch_pareto_extra.log 2>&1 &
sleep 10
# 2. Wechsung cloud at 12x12 (skips rows already done)
OUT=runs/lpqa_vacuum_eval_12x12
ev() { [ -s $OUT/$1/row.json ] && return; echo "$(date +%H:%M) eval $1"; $SR $PY eval_coilsets_vacuum.py coils/$1.json $OUT/$1 --modes 12 12 --vc-grid 64 < /dev/null > $OUT/$1.log 2>&1; }
HUR=($(ls coils/hurwitz_*.json 2>/dev/null | sed 's:.*/::; s:\.json$::'))   # Hurwitz 2024 lower envelope (5 coils/hfp), after the Wechsung cloud
( for L in 18 22; do for k in 0 1 2 3 4 5 6 7; do ev wech_cloud_L${L}_ig$k; done; done
  for ((i = 0; i < ${#HUR[@]}; i += 2)); do ev ${HUR[$i]}; done ) &
( for L in 20 24; do for k in 0 1 2 3 4 5 6 7; do ev wech_cloud_L${L}_ig$k; done; done
  for ((i = 1; i < ${#HUR[@]}; i += 2)); do ev ${HUR[$i]}; done ) &
sleep 5
# 3. Gil-pipeline scoring (v3 convention) of published sets, fair twins and the Pareto points' latest coils
( for p in runs/published/*/; do p=${p%/}; grep -q "own LCFS (v3)" $p.gilscore.log 2>/dev/null && grep -q "^done" $p.gilscore.log 2>/dev/null && continue
    $SR bash score_run_gil.sh $p < /dev/null > $p.gilscore.log 2>&1; done
  for r in vac-L18-fair vac-L20-fair vac-L24-wech-fair vac-3coil-fair2 pareto-L16 pareto-L17 pareto-L18 pareto-L19 pareto-L20 pareto-L21 pareto-L22 pareto-L24; do
    d=$(latest_dir $r); [ -z "$d" ] && continue
    grep -q "own LCFS (v3)" $d.gilscore.log 2>/dev/null && grep -q "^done" $d.gilscore.log 2>/dev/null && continue
    $SR bash score_run_gil.sh $d < /dev/null > $d.gilscore.log 2>&1; done ) &
sleep 5
# 4. stage two on the latest saved stage-one step
( S1=$(latest_dir qa4-stage1); N=$(latest_step $S1); [ -z "$N" ] && { S1=runs/batch/qa4-stage1-c3; N=$(latest_step $S1); }
  O=runs/stage_two/$(basename $S1)-step$N; echo "stage two on $S1 step $N"
  COIL_CASE=qa4-beta $SR $PY stage_two_fit.py --wout $S1/wout.step$N.nc --input $S1/input.run --coils $DATA/ESSOS_coils_qa4_beta.json --out $O --weights 1e3,1e5 --maxiter 1500 --slsqp 400 < /dev/null > $O.log 2>&1 && \
  COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1
  COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $S1 runs/fig2/$(basename $S1) --last < /dev/null > runs/fig2/$(basename $S1).log 2>&1 ) &
wait; echo "$(date): extras_batch8 done"
