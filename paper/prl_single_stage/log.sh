#!/bin/bash
# append a timestamped line to RUNLOG.md: bash log.sh "message"
echo "- $(date '+%m-%d %H:%M') $*" >> /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/RUNLOG.md
