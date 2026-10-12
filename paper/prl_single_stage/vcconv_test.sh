#!/bin/bash
# virtual-casing grid ladder: bash vcconv_test.sh <set> <vc> [modes]
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh; export JAX_PLATFORMS=cuda,cpu
n=$1; vc=$2; m=${3:-12}; out=runs/vcconv_test/m${m}vc${vc}/$n; mkdir -p $(dirname $out)
$PY eval_coilsets_vacuum.py coils/$n.json $out --modes $m $m --vc-grid $vc --bftol 1e-8 --bjftol 1e-6 --bnfev 400 > $out.log 2>&1
