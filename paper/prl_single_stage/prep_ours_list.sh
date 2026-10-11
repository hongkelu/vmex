#!/bin/bash
# Symlink the latest saved coils of every force-front run to coils/ours_<run>.json and write runs/ours_eval_list.txt
# (G families first).   bash prep_ours_list.sh
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
: > runs/ours_eval_list.txt
for name in $(ls -d runs/batch/ff*/ | sed 's:runs/batch/::; s:/$::' | grep -v '^restart_' | sed -E 's/-c[0-9]+$//' | sort -u); do
  d=$(ls -d runs/batch/$name/ runs/batch/$name-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2)
  f=$(ls $d/coils.step*.json 2>/dev/null | sed -E 's/.*step([0-9]+)\.json/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -n "$f" ] || continue
  ln -sfn "$PWD/$f" coils/ours_$name.json
  rm -rf runs/lpqa_vacuum_eval_12x12/ours_$name   # always re-solve the newest step
  echo ours_$name >> runs/ours_eval_list.txt
done
# G families first, then the rest
{ grep -E '^ours_ff(25|30)G' runs/ours_eval_list.txt; grep -v -E '^ours_ff(25|30)G' runs/ours_eval_list.txt; } > runs/ours_eval_list.tmp && mv runs/ours_eval_list.tmp runs/ours_eval_list.txt
echo "$(wc -l < runs/ours_eval_list.txt) sets"
