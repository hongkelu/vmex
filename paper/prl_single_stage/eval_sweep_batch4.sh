#!/bin/bash
# Common-resolution rescoring of every benchmark coil set (published + our finals): bash eval_sweep_batch4.sh <jobid> <tag> <extra eval args>
#   e.g.  bash eval_sweep_batch4.sh 123 12x12 --modes 12 12 --vc-grid 64      (one GPU, sequential)
set -u
J=$1; TAG=$2; shift 2
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
OUT=runs/lpqa_vacuum_eval_$TAG; mkdir -p $OUT
EXTRA=("$@")
latest() { ls $1/coils.step*.json 2>/dev/null | sed -E 's/.*step([0-9]+)\.json/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
ev() { local name=$1 coils=$2; [ -z "$coils" ] && { echo "no coils for $name"; return; }
  [ -s $OUT/$name/row.json ] && return
  echo "$(date +%H:%M) eval $name <- $coils"; $SR $PY eval_coilsets_vacuum.py $coils $OUT/$name "${EXTRA[@]}" < /dev/null > $OUT/$name.log 2>&1; }
( for s in gil_L18 wechsung_L18 gil_L20 wechsung_L20 gil_L24 wechsung_L24 gil_3coil_L18; do ev $s coils/lpqa_$s.json; done
  for r in vac-L18-c2 vac-L18-wech-c2 vac-L20-c3 vac-3coil-c2 vac-L24-wech-c1 vac-L24-c2; do ev ours_$r "$(latest runs/batch/$r)"; done
  for r in vac-L18-exact vac-L20-exact vac-3coil-exact vac-L24-feas2 vac-L24-wech-feas2 vac-L24-wech-o24-feas vac-L24-wech-hires12-feas-c3; do ev ours_$r "$(latest runs/batch/$r)"; done
  # fair twins (plasma pinned to the target): latest continuation, re-evaluated whenever the sweep is re-run
  for r in vac-L18-fair vac-3coil-fair vac-L20-fair vac-L24-wech-fair; do d=$(ls -d runs/batch/$r/ runs/batch/$r-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2); rm -rf $OUT/ours_$r; ev ours_$r "$(latest $d)"; done ) &
wait; echo "$(date): eval sweep $TAG done"
