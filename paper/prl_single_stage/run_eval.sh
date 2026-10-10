#!/bin/bash
# Vacuum free-boundary evaluation of the seven published LP QA coil sets.
# Run INSIDE an interactive allocation (salloc ... bash run_eval.sh): one node, 4 GPUs.
set -u
PY=$HOME/.conda/envs/uwplasma-env/bin/python
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
OUT=runs/lpqa_vacuum_eval
mkdir -p $OUT
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
export JAX_PLATFORMS=cuda,cpu
SETS=(wechsung_L18 gil_L18 wechsung_L20 gil_L20 wechsung_L24 gil_L24 gil_3coil_L18)

run_one() {  # $1 = set name, $2 = GPU index
  CUDA_VISIBLE_DEVICES=$2 srun --exact -N1 -n1 -c16 --gpus-per-task=1 --gpu-bind=none --mem=0 \
    $PY eval_coilsets_vacuum.py coils/lpqa_$1.json $OUT/$1 --check coils/lpqa_$1_check.npz \
    > $OUT/$1.log 2>&1
  echo "done $1 rc=$?"
}

echo "allocation $SLURM_JOB_ID on $SLURM_NODELIST, $(date)"
# first set alone: it also writes the shared target.json
run_one ${SETS[0]} 0
# the remaining six, three at a time on GPUs 1-3, then 0-2
i=0
for s in "${SETS[@]:1}"; do
  run_one $s $(( (i % 3) + 1 )) &
  i=$((i + 1))
  if (( i % 3 == 0 )); then wait; fi
done
wait
echo "all done $(date)"
grep -h "^ROW\|^TARGET" $OUT/*.log
