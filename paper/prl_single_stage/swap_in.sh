#!/bin/bash
# Replace the runs of a live allocation: cancel every step except the keeper, drop queued launches, launch <new_jobs>.
#   bash swap_in.sh <jobid> <keeper_regex> <new_jobs.txt>
set -u
J=$1; KEEP=$2; NEW=$3
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
timeout 240 srun --jobid=$J --overlap -N4 -n4 --ntasks-per-node=1 --mem=0 bash -c 'for p in $(pgrep -u $USER -x python); do n=$(tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -o "runs/batch/[A-Za-z0-9.-]*\|coils/[A-Za-z0-9_.-]*json" | head -1); [ -n "$n" ] || continue; s=$(tr "\0" "\n" </proc/$p/environ 2>/dev/null | sed -n "s/^SLURM_STEP_ID=//p"); echo "$s $n"; done' 2>/dev/null | sort -n | uniq > runs/steps_$J.txt
echo "running steps:"; cat runs/steps_$J.txt | tr '\n' ';'; echo
for s in $(awk -v k="$KEEP" '$2 !~ k {print $1}' runs/steps_$J.txt); do scancel $J.$s; done
for p in $(pgrep -u $USER -x srun); do c=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null); echo "$c" | grep -q -- "--jobid=$J" || continue; echo "$c" | grep -q -E -- "$KEEP" && continue; kill $p; done
sleep 15
echo "kept: $(awk -v k="$KEEP" '$2 ~ k' runs/steps_$J.txt | tr '\n' ' ')"
bash launch_jobs.sh $J $NEW
