#!/bin/bash
# Rewrite STATUS.md every 15 min (cheap: squeue + log tails). setsid nohup bash status_snapshot.sh > /dev/null 2>&1 &
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
while [ ! -e runs/STOP_status ]; do
  { echo "# STATUS $(date '+%Y-%m-%d %H:%M') (auto, status_snapshot.sh; decisions in RUNLOG.md)"; echo
    echo "## Allocations"; echo '```'; squeue -u $USER -h -o "%i %T %M %L %q %j" | grep -v gpu_shared; echo '```'
    echo "## Loops and helpers (this login node: $(hostname))"; echo '```'
    ps -u $USER -o pid=,etime=,args= | awk '$3=="bash" && ($4 ~ /chain_loop|extras_lanes|status_snapshot/)' | cut -c1-120 | sort -k4 | uniq -c -f3; echo '```'
    echo "## Runs (newest dir per name): steps | step-0 QS | best QS | latest QS | slack | last step time | end message"; echo '```'
    for f in $(ls runs/batch/*.log 2>/dev/null | grep -E '/(GRc|GRcF|P25A|P30A|P25F)[0-9]*-F[0-9.]+(-c[0-9]+)?\.log$' | sed -E 's/(-c[0-9]+)?\.log$//' | sort -u); do
      l=$(ls -t ${f}.log ${f}-c*.log 2>/dev/null | head -1); n=$(grep -a -c '^\[step' $l)
      s0=$(grep -a -m1 '^\[step' $l | grep -o 'QA=[0-9.e+-]*' | cut -d= -f2)
      best=$(grep -a '^\[step' $l | grep -o 'QA=[0-9.e+-]*' | cut -d= -f2 | sort -g | head -1)
      last=$(grep -a '^\[step' $l | tail -1); q=$(echo "$last" | grep -o 'QA=[0-9.e+-]*' | cut -d= -f2)
      sl=$(echo "$last" | grep -o 'coil_slack=[-0-9.e]*' | cut -d= -f2); t=$(echo "$last" | grep -o '[0-9.]*s (trials' | cut -d's' -f1)
      m=$(grep -a -o '"message": "[^"]*"' $l | tail -1 | cut -c12-60); e=$(grep -a -q Traceback $l && echo TRACEBACK)
      printf "%-22s %4s  %-12s %-12s %-12s %-8s %6s  %s %s\n" "$(basename $l .log)" "$n" "$s0" "$best" "$q" "$sl" "$t" "$m" "$e"
    done; echo '```'
    echo "## Convergence ladders"; echo '```'; timeout 120 $HOME/.conda/envs/uwplasma-env/bin/python analysis/conv_report.py 2>&1 | grep -v -i warn; echo '```'
  } > STATUS.md.tmp 2>&1 && mv STATUS.md.tmp STATUS.md
  sleep 900
done
