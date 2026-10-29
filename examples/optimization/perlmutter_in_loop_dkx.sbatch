#!/bin/bash
# In-loop DKX bootstrap rows on Perlmutter: 8 nodes x 4 A100 = 32 GPUs, one process per GPU.
#   rank 0      : the single-stage driver (VMEX three-term free boundary + SLSQP) on its own GPU
#   ranks 1..31 : DKX workers (python -m vmex.core.kinetic_bootstrap RENDEZVOUS), one surface each
# The workers meet the driver through a rendezvous file in the run directory (host, port, a per-job key; written
# owner-only), then exchange each surface's Boozer spectrum (a few kB) and its <j.B> value and gradient over TCP.
#
# NOT SUBMITTED. Before the first submission (needs the user's OK):
#   - the env has dkx 2.11.0; the rows need dkx >= 2.12 (KineticBootstrapMismatch._newton, #385) and the branch's
#     vmex (tt1-three-terms-campaign); the coil API needs ESSOS coil-distance-values (uwplasma/ESSOS#190, 33b16ae);
#   - copy the restart directory (input.run, coils.json, wout.nc, dkx_rows.json) to $RUN_IN.
#
#   sbatch perlmutter_dkx_rows.sbatch            # RUN_IN, STEPS, NSURF overridable: sbatch --export=ALL,STEPS=20 ...
#SBATCH -A m4656_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -J dkx-rows
#SBATCH -o %x-%j.out

set -euo pipefail
ROOT=/pscratch/sd/h/hongkelu/freeboundary-single-stage
source $ROOT/paper/env.sh
TREE=${TREE:-$ROOT/vmex-tt1-campaign}         # git worktree of hongkelu/vmex tt1-three-terms-campaign
ESSOS=${ESSOS:-$ROOT/ESSOS-coil-design}  # uwplasma/ESSOS coil-distance-values
RUN_IN=${RUN_IN:-$ROOT/dkx-rows/in/qi8gil-A-r03}  # a pass or round directory of the fixed-current rounds
STEPS=${STEPS:-50}
NSURF=${NSURF:-$((SLURM_NTASKS - 1))}         # one surface per worker GPU
OUT=$ROOT/dkx-rows/runs/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p $OUT
export RENDEZVOUS=$OUT/dkx_rendezvous.json
export COIL_CASE=qi6-beta-A8 PYTHONPATH=$TREE:$ESSOS
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 XLA_FLAGS=--xla_gpu_autotune_level=0
# the A100 40 GB nodes: g25 (the fine correction) takes 14.4 GiB, g15 1.7 GiB per surface with its gradient

cat > $OUT/rank.sh <<EOF
#!/bin/bash
if [ "\$SLURM_PROCID" = 0 ]; then
  exec $PY -u $TREE/examples/optimization/single_stage_free_boundary_optimization_three_term.py \\
    --restart $RUN_IN --beta 0.025 --modes 8 8 --ns 51 --save-every 5 --coil-point-rows --design-step-scale 1.0 \\
    --coil-limits 2.468 16.5 33.3 --labelling anchored --label-weight 1e-2 --bootstrap --profiles stellaris \\
    --in-loop-dkx --dkx-surfaces $NSURF --dkx-row-workers $((SLURM_NTASKS - 1)) --dkx-rendezvous $RENDEZVOUS \\
    --dkx-hold 1 --dkx-correction-grid 25 37 96 4 6 16 --steps $STEPS --output $OUT/run > $OUT/driver.log 2>&1
else
  exec $PY -u -m vmex.core.kinetic_bootstrap $RENDEZVOUS > $OUT/worker\$SLURM_PROCID.log 2>&1
fi
EOF
chmod +x $OUT/rank.sh
# --gpus-per-task=1 gives every rank its own GPU (CUDA_VISIBLE_DEVICES); none is shared
srun --gpu-bind=single:1 $OUT/rank.sh
