# source this before any vmex run on a compute node
export PY=$HOME/.conda/envs/uwplasma-env/bin/python
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMEX_COMPILATION_CACHE=1
export VMEX_COMPILATION_CACHE_DIR=/pscratch/sd/h/hongkelu/freeboundary-single-stage/jax_cache
export JAX_COMPILATION_CACHE_DIR=$VMEX_COMPILATION_CACHE_DIR
export MPLCONFIGDIR=/pscratch/sd/h/hongkelu/freeboundary-single-stage/.mplconfig
