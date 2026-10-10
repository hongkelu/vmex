#!/bin/bash
# Score one of our runs with Gil et al.'s pipeline:  bash score_run_gil.sh <run_dir> [step]
# 1. latest coils.stepN.json -> SIMSOPT JSON at reactor scale (paper/export_simsopt_coils.py)
# 2. their B.n statistics + their QFM surface (gil_postprocess.py, simsopt-env)
# 3. vmex fixed-boundary solve of the QFM surface + their 51-surface QS (qfm_qs_vmex.py), cross-checked with
#    SIMSOPT's own QS code (qs_simsopt_wout.py)
# 4. their engineering metrics incl. force, on the run's own LCFS (gil_coil_metrics.py)
# CPU except step 3 (GPU if available).  Run inside an allocation as a step, or on a compute node.
set -u
RUN=$1
cd /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper
export MPICH_GPU_SUPPORT_ENABLED=0 MPLCONFIGDIR=/tmp/mpl-$USER OMP_NUM_THREADS=16
SIM=/global/homes/h/hongkelu/.conda/envs/simsopt-env/bin/python
UW=/global/homes/h/hongkelu/.conda/envs/uwplasma-env/bin/python
if [ "${2:-}" != "" ]; then STEP=$2; else
  STEP=$(ls $RUN/coils.step*.json 2>/dev/null | sed -E 's/.*step([0-9]+)\.json/\1/' | sort -n | tail -1); fi
[ -z "$STEP" ] && { echo "no saved coils in $RUN"; exit 1; }
OUT=$RUN/gil_score_step$STEP
mkdir -p $OUT
echo "scoring $RUN step $STEP -> $OUT"
JAX_PLATFORMS=cpu $UW export_simsopt_coils.py $RUN/coils.step$STEP.json $OUT/biot_savart.json || exit 1
$SIM gil_postprocess.py $OUT/biot_savart.json $OUT --no-vmec --surface-wout $RUN/wout.step$STEP.nc || exit 1
source /pscratch/sd/h/hongkelu/freeboundary-single-stage/paper/env.sh
$PY qfm_qs_vmex.py $OUT/qfm_surface.json $OUT || exit 1
$SIM qs_simsopt_wout.py $OUT/wout_qfm.nc $OUT/qs_simsopt.json
JAX_PLATFORMS=cpu $UW gil_coil_metrics.py $OUT/biot_savart.json $OUT/gil_coil_metrics.json --surface-wout $RUN/wout.step$STEP.nc
echo "done $OUT"
