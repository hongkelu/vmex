#!/bin/bash
# GPU health of the two allocations every 5 min: HIGH MEM (> 73 GB), UNUSED (< 2 GB), allocation loss. Prints events only.
while true; do
  for J in $(squeue -u $USER -h -o "%i %q %T" | awk '$2=="gpu_interactive" && $3=="RUNNING" {print $1}'); do
    timeout 100 srun --jobid=$J --overlap -N4 -n4 --ntasks-per-node=1 --gpus-per-task=4 --mem=0 \
      bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed "s/^/$(hostname) /"' 2>/dev/null \
      | awk -v J=$J -v T="$(date +%H:%M)" -F'[ ,]+' '$3>73000 {print T" "J" HIGH MEM "$1" gpu"$2" "$3" MiB"} $3<2000 {print T" "J" UNUSED "$1" gpu"$2" "$3" MiB"}'
  done
  n=$(squeue -u $USER -h -o "%q %T" | awk '$1=="gpu_interactive" && $2=="RUNNING"' | wc -l); [ "$n" -lt 2 ] && echo "$(date +%H:%M) only $n interactive allocations running"
  sleep 300
done
