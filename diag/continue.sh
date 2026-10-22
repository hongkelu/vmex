#!/bin/bash
# Continue 300-step runs to 2000 steps from their step-300 checkpoint, each on its own GPU once it finishes.
# usage: continue.sh JOBS_FILE   (lines: NAME|OUTDIR|GPU_UUID|PYTHON|DRIVER_CWD|PROBE|W|ENV|ARGS)
#   ARGS: the run's driver arguments without --steps, --output, --seed-run and --restart.
# A run that stopped before step 300 converged (SLSQP finished): it is left alone and recorded.
JOBS=$1
while true; do
  pending=0
  while IFS='|' read -r name out gpu py cwd probe w env args; do
    [ -z "$name" ] || [[ "$name" == \#* ]] && continue
    [ -e "$out/$name.cont" ] && continue
    pending=1
    pgrep -f "run $out/$name --" > /dev/null && continue
    run="$out/$name/run"
    if [ ! -e "$run/wout.step300.nc" ]; then echo "$(date) $name ended before step 300: converged or failed" > "$out/$name.cont"; continue; fi
    [ "$(nvidia-smi --id=$gpu --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ] || continue
    r="$out/$name-r300"; mkdir -p "$r"
    cp "$run/input.run" "$r/" && cp "$run/coils.step300.json" "$r/coils.json" && cp "$run/wout.step300.nc" "$r/wout.nc"
    (cd "$cwd" && setsid nohup env $env CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=true \
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 $py -u "$probe" "$w" run "$out/$name-cont" $args \
       --steps 1700 --restart "$r" --output "$out/$name-cont/run" > "$out/$name-cont.log" 2>&1 < /dev/null &)
    echo "$(date) $name continued on $gpu" | tee "$out/$name.cont"
  done < "$JOBS"
  [ $pending = 0 ] && break
  sleep 120
done
