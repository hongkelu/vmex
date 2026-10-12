#!/bin/bash
# Event stream for the production runs (newest dir per name): milestones, early ends, tracebacks, and two-step drift alarms.
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
declare -A seen
while true; do
  for l in $(ls -t runs/batch/{GRc,GRcF,P25A,P30A,P25F}*-F*.log 2>/dev/null | sed -E 's/(-c[0-9]+)?\.log$//' | awk '!s[$0]++'); do
    f=$(ls -t ${l}.log ${l}-c*.log 2>/dev/null | head -1); n=$(grep -a -c '^\[step' $f); b=$(basename $f .log)
    for k in 2 5 10 20 40 80; do if [ "$n" -ge $k ] && [ -z "${seen[$f.$k]:-}" ]; then
      echo "$b step$((k-1)): $(grep -a '^\[step' $f | sed -n ${k}p | grep -o 'QA=[0-9.e+-]*\|coil_slack=[-0-9.e]*\|aspect=[0-9.]*\|[0-9.]*s (trials [0-9]*' | tr '\n' ' ')"; seen[$f.$k]=1; fi; done
    if [ "$n" -ge 3 ] && [ -z "${seen[$f.drift$n]:-}" ]; then
      read q1 q2 q3 s1 s2 s3 <<< $(grep -a '^\[step' $f | tail -3 | awk '{for(i=1;i<=NF;i++){if($i~/^QA=/)q[NR]=substr($i,4); if($i~/^coil_slack=/)s[NR]=substr($i,12)}} END{print q[1],q[2],q[3],s[1],s[2],s[3]}')
      awk -v a=$q1 -v b=$q2 -v c=$q3 -v x=$s1 -v y=$s2 -v z=$s3 'BEGIN{exit !((c>b && b>a) || (z<y && y<x && z<-0.001))}' && echo "$b DRIFT at step $((n-1)): QS $q1 $q2 $q3 slack $s1 $s2 $s3"
      seen[$f.drift$n]=1; fi
    m=$(grep -a -o '"message": "[^"]*"' $f | tail -1); [ -n "$m" ] && [ -z "${seen[$f.m]:-}" ] && { echo "$b ENDED at step $((n-1)): $m"; seen[$f.m]=1; }
    grep -a -q 'Traceback' $f && [ -z "${seen[$f.e]:-}" ] && { echo "$b TRACEBACK: $(grep -a 'Error' $f | tail -1 | cut -c1-110)"; seen[$f.e]=1; }
  done; sleep 90; done
