#!/bin/bash
# Extras injected into allocation 4:  bash stage_two_batch4.sh <jobid>
#  GPU 1: LBD22 A6 stage one at R0 = 1 m (vmex solve) -> stage-two coils -> the equilibrium they make
#  GPU 2: Gil-pipeline scores of the benchmark rows' latest coils (sequential)
#  GPU 3: Fig-2 scorer of the fixed arm's second continuation (qa4-fixed-c2, complete by then)
set -u
J=$1
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh
export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
OVR6='{"SEED": [2, 6.0, 0.5], "ASPECT_RANGE": [5.9, 6.1]}'
mkdir -p runs/stage_two runs/fig2
( O=runs/stage_two/lbd-qa6; mkdir -p $O
  $SR $PY -m vmex decks/input.lbd_qa6_R1_8x8 --outdir $O --device gpu < /dev/null > $O/solve.log 2>&1
  W=$(ls $O/wout*.nc | head -1); echo "lbd wout: $W"
  COIL_CASE=qa4-beta COIL_CASE_OVERRIDES="$OVR6" $SR $PY stage_two_fit.py --wout $W --input decks/input.lbd_qa6_R1_8x8 \
    --coils $DATA/ESSOS_coils_qa4_beta.json --out $O < /dev/null > $O.log 2>&1 && \
  COIL_CASE=qa4-beta COIL_CASE_OVERRIDES="$OVR6" $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1 ) &
sleep 2
( for r in vac-L18-exact vac-L20-exact vac-3coil-exact vac-L18-c2 vac-L20-c3 vac-3coil-c2 vac-L24-wech-hires12-feas-c3 vac-L24-wech-feas2 vac-L24-feas2; do
    $SR bash score_run_gil.sh runs/batch/$r < /dev/null > runs/batch/$r.gilscore.log 2>&1; done ) &
sleep 2
( COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py runs/batch/qa4-fixed-c2 runs/fig2/qa4-fixed-c2 --every 10 < /dev/null > runs/fig2/qa4-fixed-c2.log 2>&1 ) &
( COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py runs/batch/qa4-free-vac029-s1-c1 runs/fig2/qa4-free-vac029-s1-c1 --every 10 < /dev/null > runs/fig2/qa4-free-vac029-s1-c1.log 2>&1
  W=$(ls runs/batch/qa4-free-vac029-s1-c1/wout.step*.nc | sort -V | tail -1); mkdir -p runs/alpha/qa4-free; cp $W runs/alpha/qa4-free/wout_qa4_free.nc
  $SR $PY -m vmex --trace runs/alpha/qa4-free/wout_qa4_free.nc --trace-tmax 0.2 --trace-particles 1000 --trace-s 0.25 --outdir runs/alpha/qa4-free --device gpu < /dev/null > runs/alpha/qa4-free/trace.log 2>&1 ) &
sleep 2
( bash eval_sweep_batch4.sh $J 12x12 --modes 12 12 --vc-grid 64 > runs/eval_sweep_12x12.log 2>&1 ) &
wait; echo "$(date): stage_two_batch4 done"
