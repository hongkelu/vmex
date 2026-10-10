#!/bin/bash
# After the sequential Gil-scoring loop ends: rescore the rows that were scored before the QFM seeding fix.
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
J=59628428; SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
sleep 120; while pgrep -f "[s]core_run_gil.sh" > /dev/null; do sleep 60; done
for r in vac-L18-exact vac-L20-exact vac-3coil-exact vac-L18-c2 vac-L20-c3 vac-3coil-c2 vac-L24-wech-hires12-feas-c3 vac-L24-wech-feas2 vac-L24-feas2; do
  grep -q "own LCFS (v3)" runs/batch/$r.gilscore.log 2>/dev/null && continue
  echo "$(date +%H:%M) rescore $r"; $SR bash score_run_gil.sh runs/batch/$r < /dev/null > runs/batch/$r.gilscore.log 2>&1; done
echo "$(date): rescoring done"
