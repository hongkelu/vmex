#!/bin/bash
# Cancel only the named runs on an allocation and relaunch them from <list>: bash restart_named.sh <jobid> <list>
set -u; J=$1; L=$2; cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
names=$(cut -d'|' -f1 $L | grep -v '^#' | tr '\n' ' ')
timeout 240 srun --jobid=$J --overlap -N4 -n4 --ntasks-per-node=1 --mem=0 bash -c 'for p in $(pgrep -u $USER -x python); do n=$(tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -o "runs/batch/[A-Za-z0-9.-]*" | head -1); [ -n "$n" ] || continue; s=$(tr "\0" "\n" </proc/$p/environ 2>/dev/null | sed -n "s/^SLURM_STEP_ID=//p"); echo "$s $n"; done' 2>/dev/null | sort -u > runs/steps_$J.txt
for nm in $names; do for s in $(awk -v n="runs/batch/$nm" '$2==n || $2 ~ "^"n"-c[0-9]+$" {print $1}' runs/steps_$J.txt); do scancel $J.$s && echo "cancelled $nm (step $s)"; done; done
sleep 15; bash launch_jobs.sh $J $L
