#!/bin/bash
# wrapper kept for the armed lane helper: best step of every force-front run -> coils/ours_*.json + runs/ours_eval_list.txt
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh >/dev/null 2>&1
$PY prep_ours_list.py "$@"
