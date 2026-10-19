#!/bin/bash
# setup.sh: the old campaign's Python environment and code, offline.
#
#   WHEELHOUSE=<dir with the 65 wheels> PYTHON=python3.12 campaign/setup.sh
#
# The wheels are not in git (3.2 GB). They are in env/wheelhouse/ of the full copy, e.g. on Perlmutter
# /pscratch/sd/h/hongkelu/freeboundary-single-stage-old/env/wheelhouse.
# venv/: Python 3.12 with the frozen wheels (hash-checked, essos 0.19.3).
# code/vmex-<commit>/: a worktree of this repository per commit in COMMITS.txt, each with the run inputs linked
#   where the commands of RUNS.md expect them (seeds/, runs/seed-qa4, runs/smoke-c2-qa4, prod/<restart>).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${WHEELHOUSE:?set WHEELHOUSE to the directory holding the 65 wheels}"
PY=${PYTHON:-python3.12}
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "need Python 3.12 (the runs used 3.12.3), got " + sys.version'
[ -d "$here/venv" ] || "$PY" -m venv "$here/venv"
"$here/venv/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" --require-hashes \
    -r "$here/env/requirements.txt"
"$here/venv/bin/python" -m pip check

while read -r sha _; do
  tree="$here/code/vmex-${sha:0:8}"
  [ -d "$tree" ] || git -C "$here" worktree add --quiet --detach "$tree" "$sha"
  mkdir -p "$tree/runs" "$tree/prod"
  ln -sfn "$here/inputs/seeds" "$tree/seeds"
  ln -sfn "$here/inputs/seeds/seed-qa4" "$tree/runs/seed-qa4"
  ln -sfn "$here/inputs/seeds/smoke-c2-qa4" "$tree/runs/smoke-c2-qa4"
  for restart in "$here"/inputs/restarts/*; do ln -sfn "$restart" "$tree/prod/$(basename "$restart")"; done
done < "$here/COMMITS.txt"
echo "ready: venv/ and $(ls -d "$here"/code/vmex-*/ | wc -l) code trees; see RUNS.md and run.sh"
