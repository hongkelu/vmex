#!/bin/bash
# The 18-run labelling comparison (A, C, NL x QA/QH/QI x vacuum/2.5%), each from step 0 to 2000 steps or SLSQP
# convergence, on venv-tt1-v2 and this checkout's code.  usage: abc18.sh HOST [smoke]
#   HOST: lhk0-7 | lhk14-19 | lhk8-13 (runs whose seeds or slots are on that host); each run takes an empty GPU.
#   smoke: 2 steps of the first run of HOST only, into OUT/smoke-<name>.
HOST=$1; SMOKE=$2
P=${PYTHON:-/root/autodl-tmp/venv-tt1-v2/bin/python}
OUT=/root/autodl-tmp/abc18; mkdir -p $OUT
SEEDS=${SEEDS:-/root/autodl-tmp/tt1-work/tree}
VAC="--modes 8 8 --ns 51 --save-every 10 --coil-point-rows"
QA_B="--beta 0.025 --bootstrap $VAC --seed-run $SEEDS/seeds/qa4-beta-free --vacuum-iota-floor 0.31 --iota-ceiling 0.495"
QH_B="--beta 0.025 --bootstrap $VAC --design-step-scale 1.0 --max-iterations 20000"
QI_B="--beta 0.025 --bootstrap $VAC --seed-run $SEEDS/prod/seed-dkx29-qi6-beta-A8 --design-step-scale 1.0"
declare -A LAB=([A]="--labelling start" [C]="--labelling equal_arc --label-weight 1e-3" [NL]="--labelling start --label-weight 0")
jobs=()
# C and NL claim GPUs first, then A
for m in C NL; do
  jobs+=("lhk14-19|qh-beta-$m|qh4-beta-A8-merc|$QH_B|$m" "lhk14-19|qa-beta-$m|qa4-beta-merc|$QA_B|$m" "lhk14-19|qi-beta-$m|qi6-beta-A8|$QI_B|$m")
done
jobs+=("lhk8-13|qa-beta-A|qa4-beta-merc|$QA_B|A" "lhk8-13|qi-beta-A|qi6-beta-A8|$QI_B|A")
for c in qa:qa3 qh:qh qi:qi; do jobs+=("lhk0-7|${c%%:*}-vac-C|${c##*:}|$VAC|C" "lhk0-7|${c%%:*}-vac-NL|${c##*:}|$VAC|NL" "lhk8-13|${c%%:*}-vac-A|${c##*:}|$VAC|A"); done
jobs+=("lhk8-13|qh-beta-A|qh4-beta-A8-merc|$QH_B|A")
cd "$(dirname "$0")/.."
free=($(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 100 {print $1}'))
i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r host name case args m <<< "$job"
  [ "$host" = "$HOST" ] || continue
  [ -n "$ONLY" ] && [[ "$name" != $ONLY ]] && continue
  [ -n "$SMOKE" ] && { name="smoke-$name"; steps=2; } || steps=2000
  [ -e "$OUT/$name" ] && { echo "skip $name: $OUT/$name exists"; continue; }
  G=${free[$i]}; i=$((i + 1))
  [ -z "$G" ] && { echo "no empty GPU left for $name"; continue; }
  setsid nohup env PYTHONPATH=$PWD COIL_CASE=$case JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 CUDA_VISIBLE_DEVICES=$G $P -u \
    examples/optimization/single_stage_free_boundary_optimization_three_term.py $args ${LAB[$m]} --steps $steps \
    --output $OUT/$name > $OUT/$name.log 2>&1 < /dev/null &
  echo "launched $name ($m, $case) on $G"
  [ -n "$SMOKE" ] && break
done
