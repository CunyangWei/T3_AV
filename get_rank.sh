#!/bin/bash
# select_gpu_device wrapper script
export LOCAL_RANK=${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor/rank_${RANK}"
exec $*
