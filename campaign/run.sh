#!/bin/bash
# run.sh <commit> <F|X|fixed_jaccheck_tmp> <COIL_CASE> <run options...> --output prod/<name>
#
# One run as in RUNS.md: the commit's code tree, the frozen venv, one GPU (CUDA_VISIBLE_DEVICES, default 0) and the
# campaign's environment. F = free-boundary three-term driver, X = fixed-boundary driver. For a run that built its
# own seed, add --seed-run <repo>/campaign/reference/runs/<run> to start from its exact original seed.
# Example: ./run.sh 91729930 F qa4-beta-merc --steps 1000 --seed-run seeds/qa4-beta-free \
#              --vacuum-iota-floor 0.31 --iota-ceiling 0.495 --output prod/free-qa4-beta-merc-vac0.31
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
commit=$1; arm=$2; case=$3; shift 3
tree="$here/code/vmex-${commit:0:8}"
[ -d "$tree" ] || { echo "no code tree $tree: run setup.sh"; exit 1; }
case $arm in
  F) driver=examples/optimization/single_stage_free_boundary_optimization_three_term.py ;;
  X) driver=examples/optimization/single_stage_optimization_coil_constraints.py ;;
  fixed_jaccheck_tmp) cp "$here/checks/fixed_jaccheck.py" "$tree/examples/optimization/fixed_jaccheck_tmp.py"
     driver=examples/optimization/fixed_jaccheck_tmp.py ;;
  *) echo "arm must be F, X or fixed_jaccheck_tmp"; exit 1 ;;
esac
cd "$tree"
export COIL_CASE=$case CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
       XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH="$tree"
exec "$here/venv/bin/python" "$driver" --beta 0.025 --bootstrap --modes 8 8 --ns 51 --save-every 1 "$@"
