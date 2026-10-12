#!/bin/bash
# one chunk of converged metrics on a compute node: bash ffm_chunk.sh <chunk_index> <n_chunks>
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper; source env.sh; export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
k=$1; n=$2; args=""; i=0
for nm in $(cat runs/verified_cloud_names.txt) gilpar_L25_F9 gilpar_L25_F10 gilpar_L25_F10.5 gilpar_L25_F12 gilpar_L30_F9 gilpar_L30_F9.5 gilpar_L30_F10 gilpar_L30_F11 gilpar_L30_F12 gilpar_L30_F14; do
  [ $((i % n)) -eq $k ] && [ -f coils/$nm.json ] && args="$args $nm=coils/$nm.json"; i=$((i + 1)); done
$PY force_front_metrics.py runs/ffm1024_chunk$k.json $args > runs/ffm1024_chunk$k.log 2>&1
