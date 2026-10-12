#!/bin/bash
# radial-resolution convergence test at a tight boundary tolerance: bash nsconv_test.sh <set> <ns>
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh; export JAX_PLATFORMS=cuda,cpu
n=$1; ns=$2; out=runs/nsconv_test/ns${ns}/$n; mkdir -p $(dirname $out)
$PY eval_coilsets_vacuum.py coils/$n.json $out --modes 12 12 --vc-grid 64 --ns $ns --bftol 1e-8 --bjftol 1e-6 --bnfev 400 > $out.log 2>&1
