#!/bin/bash
# After the penalty-fit LBD evaluation frees its GPU: stage two with the SLSQP hard-constraint polish, then evaluate.
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
J=59628428; SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
OVR6='{"SEED": [2, 6.0, 0.5], "ASPECT_RANGE": [5.9, 6.1]}'
until grep -q "ROW\|Traceback\|Error" runs/stage_two/lbd-qa6.eval.log 2>/dev/null; do sleep 60; done; sleep 20
O=runs/stage_two/lbd-qa6-slsqp; W=runs/stage_two/lbd-qa6/wout_lbd_qa6_R1_8x8.nc
echo "$(date): lbd slsqp stage two"
COIL_CASE=qa4-beta COIL_CASE_OVERRIDES="$OVR6" $SR $PY stage_two_fit.py --wout $W --input decks/input.lbd_qa6_R1_8x8 \
  --coils $DATA/ESSOS_coils_qa4_beta.json --out $O --weights 1e3,1e5 --maxiter 1500 --slsqp 400 < /dev/null > $O.log 2>&1 && \
COIL_CASE=qa4-beta COIL_CASE_OVERRIDES="$OVR6" $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1
echo "$(date): lbd slsqp done"
