#!/bin/bash
# Allocation 9 extras (after the chain's launches): free-boundary 12x12 evaluation of the published force-vs-B.n Pareto
# sets (Gil Fig 1 / Hurwitz 2024 front under Gil's filter + stratified cloud sample + Gil's 25/30 m points), N lanes;
# stage two on the best pure-stage-one step; Gil-pipeline v3 scoring of the new 3-coil run.
#   bash extras_batch9.sh [lanes]
set -u
LANES=${1:-3}
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
until grep -q "^launch " runs/batch9_salloc.log 2>/dev/null && [ $(grep -c "^launch " runs/batch9_salloc.log) -ge $(grep -c "^[^#]" jobs_batch9.txt) ]; do sleep 60; done
J=$(grep -m1 "^allocation" runs/batch9_salloc.log | awk '{print $2}'); echo "$(date): extras into $J ($LANES evaluation lanes)"; sleep 30
source env.sh; export JAX_PLATFORMS=cuda,cpu
SR="srun --jobid=$J --exact -N1 -n1 -c4 --gpus-per-task=1 --mem=0"
DATA=/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/data
latest_dir() { ls -d runs/batch/$1/ runs/batch/$1-c*/ 2>/dev/null | sed 's:/$::' | awk '{n=0; if (match($0,/-c[0-9]+$/)) n=substr($0,RSTART+2); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2; }
OUT=runs/lpqa_vacuum_eval_12x12
ev() { [ -s $OUT/$1/row.json ] && return; echo "$(date +%H:%M) eval $1"; $SR $PY eval_coilsets_vacuum.py coils/$1.json $OUT/$1 --modes 12 12 --vc-grid 64 < /dev/null > $OUT/$1.log 2>&1; }
# evaluation list: Gil's points first, then the Hurwitz front (Gil filter, B.n < 4e-3) from low to high force, then the cloud sample
python3 - > runs/front_eval_list.txt <<'EOF'
import json
gil = [r["name"] for r in json.load(open("coils/gil_pareto_index.json")) if r["max_kappa"] < 20]
fr = sorted((p for p in json.load(open("../references/hurwitz2024/gil_filter_front.json")) if p["bn"] < 4e-3), key=lambda p: p["force"])
cl = [p for p in json.load(open("../references/hurwitz2024/gil_filter_cloud_sample.json")) if p["bn"] < 4e-3 and 7000 <= p["force"] <= 16000]
seen = set(gil); out = list(gil)
# 8-hour budget: every 3rd front member by force (36) and every 5th cloud sample (14) first; the rest follow if time allows
first = fr[::3] + cl[::5]
for p in first + fr + cl:
    n = "hurfront_" + p["uuid"][:8]
    if n not in seen: seen.add(n); out.append(n)
print("\n".join(out))
EOF
mapfile -t SETS < runs/front_eval_list.txt; echo "${#SETS[@]} sets to evaluate"
for ((lane = 0; lane < LANES; lane++)); do
  ( for ((i = lane; i < ${#SETS[@]}; i += LANES)); do ev ${SETS[$i]}; done ) &
  sleep 3
done
# stage two on the best saved pure-stage-one step (lowest QS among saved steps of the latest two continuations)
# (deferred to batch 10: all 16 GPUs of batch 9 go to the force fronts, the showcase and the 3 evaluation lanes)
[ "${STAGE_TWO:-0}" = "1" ] && ( S=$(python3 - <<'EOF'
import json, re, glob
best = None
for d in sorted(glob.glob("runs/batch/qa4-stage1*"))[-3:]:
    if not glob.glob(d + "/wout.step*.nc"): continue
    rows = {r["step"]: r["qa"] for r in (json.loads(l) for l in open(d + "/metrics.jsonl") if l.strip())}
    for w in glob.glob(d + "/wout.step*.nc"):
        n = int(re.search(r"step(\d+)", w).group(1))
        if n in rows and (best is None or rows[n] < best[0]): best = (rows[n], d, n)
print(f"{best[1]} {best[2]}")
EOF
); set -- $S; S1=$1; N=$2; O=runs/stage_two/$(basename $S1)-step$N; echo "stage two on $S1 step $N"
  [ -s $O/eval/steps.jsonl ] || { COIL_CASE=qa4-beta $SR $PY stage_two_fit.py --wout $S1/wout.step$N.nc --input $S1/input.run --coils $DATA/ESSOS_coils_qa4_beta.json --out $O --weights 1e3,1e5 --maxiter 1500 --slsqp 400 < /dev/null > $O.log 2>&1 && \
  COIL_CASE=qa4-beta $SR $PY eval_steps_freeboundary.py $O $O/eval --steps 0 < /dev/null > $O.eval.log 2>&1; }
  d=$(latest_dir vac-3coil-fair2); grep -q "own LCFS (v3)" $d.gilscore.log 2>/dev/null && grep -q "^done" $d.gilscore.log 2>/dev/null || $SR bash score_run_gil.sh $d < /dev/null > $d.gilscore.log 2>&1 ) &
wait; echo "$(date): extras_batch9 done"
