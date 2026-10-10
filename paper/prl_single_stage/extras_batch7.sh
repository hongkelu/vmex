#!/bin/bash
# Allocation 7: after the chain's launches, start the Pareto length scan (restarts from the latest fair twins) and the extras.
set -u
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until grep -q "^launch qa4-stage1" runs/batch7_salloc.log 2>/dev/null; do sleep 60; done
J=$(grep -m1 "^allocation" runs/batch7_salloc.log | awk '{print $2}'); echo "$(date): pareto + extras into $J"; sleep 60
source env.sh
latest_dir() { ls -d runs/batch/$1/ runs/batch/$1-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2; }
# resolve each start name to a prepared --restart directory from its latest saved step
> jobs_pareto_resolved.txt
while IFS='|' read -r name drv case ovr extra src; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  d=$(latest_dir $src); r=runs/batch/restart_${name}_$J
  JAX_PLATFORMS=cpu $PY make_restart.py $d $r 2>&1 | grep "restart from"
  echo "$name|$drv|$case|$ovr|$extra|$r" >> jobs_pareto_resolved.txt
done < jobs_pareto.txt
bash launch_jobs.sh $J jobs_pareto_resolved.txt > runs/launch_pareto.log 2>&1 &
sleep 30
bash stage_two_batch6.sh $J > runs/stage_two_batch7.log 2>&1
wait
