#!/bin/bash
# boundary-solve convergence test: bash bconv_test.sh <set> <ftol> <jftol> <nfev>
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh; export JAX_PLATFORMS=cuda,cpu
n=$1; ft=$2; jf=$3; nf=$4; out=runs/bconv_test/f${ft}/$n; mkdir -p $(dirname $out)
$PY eval_coilsets_vacuum.py coils/$n.json $out --modes 12 12 --vc-grid 64 --bftol $ft --bjftol $jf --bnfev $nf > $out.log 2>&1
