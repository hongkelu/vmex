#!/bin/bash
# Historical: the launcher as used on lhk0-7 (paths there). The portable equivalent is <repo>/campaign/run.sh.
# launch.sh <gpu uuid> <arm: free|fixed> <case> [extra args]: one production run, every step saved
set -e
cd /root/autodl-tmp/vmex-qa-cases
u=$1; arm=$2; case=$3; shift 3
if [ "$arm" = free ]; then D=examples/optimization/single_stage_free_boundary_optimization_three_term.py; else D=examples/optimization/single_stage_optimization_coil_constraints.py; fi
out=prod/$arm-$case${TAG:-}
COIL_CASE=$case CUDA_VISIBLE_DEVICES=$u XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH=$PWD \
  nohup /root/autodl-tmp/venv-vmex/bin/python $D --beta 0.025 --bootstrap --modes 8 8 --ns 51 --steps 300 --save-every 1 --output $out "$@" > $out.log 2>&1 < /dev/null &
echo "$arm-$case on $u pid $!"
