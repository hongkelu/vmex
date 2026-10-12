#!/bin/bash
# MHD force-residual ladder: bash ftolconv_test.sh <set> <ftol_final>
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh; export JAX_PLATFORMS=cuda,cpu
n=$1; ft=$2; out=runs/ftolconv_test/ft${ft}/$n; mkdir -p $(dirname $out)
$PY eval_coilsets_vacuum.py coils/$n.json $out --modes 12 12 --vc-grid 64 --ftol-final $ft --bftol 1e-8 --bjftol 1e-6 --bnfev 400 > $out.log 2>&1
