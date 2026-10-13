#!/bin/bash
# Round-2 convergence ladders on the two new allocations (<= 2 overlap jobs per node, GPU 0):
#   bash conv_round2.sh <jobA> <jobB>
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
A=$1; B=$2
run() { # job node set modes vc ns tag
  J=$1; r=$2; n=$3; m=$4; vc=$5; ns=$6; out=runs/conv2/m${m}_vc${vc}_ns${ns}/$n; mkdir -p $(dirname $out)
  setsid nohup srun --jobid=$J --overlap -N1 -n1 -c4 --gpus-per-task=1 --mem=0 -r $r bash -c "source env.sh; export JAX_PLATFORMS=cuda,cpu; \$PY eval_coilsets_vacuum.py coils/$n.json $out --modes $m $m --vc-grid $vc --ns $ns --ftol-final 1e-15 --niter-final 200000 --bftol 1e-8 --bjftol 1e-6 --bnfev 400" > $out.log 2>&1 < /dev/null &
}
# A: NS ladder at VC64 (99, 151) for Gil 30m F14 and hurfront_5ed3aa3c; VC96 NS50 for both
run $A 0 gilpar_L30_F14 12 64 99;   run $A 0 gilpar_L30_F14 12 64 151
run $A 1 hurfront_5ed3aa3c 12 64 99; run $A 1 hurfront_5ed3aa3c 12 64 151
run $A 2 gilpar_L30_F14 12 96 50;   run $A 3 hurfront_5ed3aa3c 12 96 50
# B: VC ladder (96, 128) at NS 99, plus NS50 VC128, and ours set
run $B 0 hurfront_5ed3aa3c 12 128 50; run $B 0 hurfront_5ed3aa3c 12 96 99
run $B 1 gilpar_L30_F14 12 128 50;    run $B 1 gilpar_L30_F14 12 96 99
run $B 2 hurfront_5ed3aa3c 12 64 50;  run $B 3 gilpar_L30_F14 12 64 50
echo "launched round-2 ladders on $A and $B"
