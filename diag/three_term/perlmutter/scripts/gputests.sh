#!/bin/bash
# usage: gputests.sh <jobid> <tree> <outdir> <ngpu> <pytest args...>
# Collects test ids in <tree>, splits them round-robin over <ngpu> GPUs of the allocation, one srun per GPU.
JOB=$1; TREE=$2; OUT=$3; NG=$4; shift 4
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
mkdir -p $OUT; cd $TREE
export PYTHONPATH=$TREE RUN_FULL=1
JAX_PLATFORMS=cpu $PY -m pytest --collect-only -q "$@" 2>/dev/null | grep "::" > $OUT/ids.txt
for g in $(seq 0 $((NG-1))); do awk -v g=$g -v n=$NG 'NR%n==g' $OUT/ids.txt > $OUT/ids.$g; done
for g in $(seq 0 $((NG-1))); do
  [ -s $OUT/ids.$g ] || continue
  srun --jobid=$JOB --overlap -N1 -n1 --gpus-per-task=4 bash -c "export CUDA_VISIBLE_DEVICES=$g; source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh; cd $TREE; export PYTHONPATH=$TREE RUN_FULL=1; \$PY -m pytest -q -p no:cacheprovider \$(cat $OUT/ids.$g) > $OUT/log.$g 2>&1; echo EXIT \$? >> $OUT/log.$g" &
done
wait
grep -h "EXIT\|passed\|failed\|error" $OUT/log.* | tail -20
