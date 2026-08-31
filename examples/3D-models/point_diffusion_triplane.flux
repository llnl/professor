#!/bin/bash
#flux: -N 1
#flux: -n 4
#flux: -t 12h
#flux: -q pbatch
#flux: --output=point_diffusion_triplane.out
#flux: --error=point_diffusion_triplane.err


# For ROCM systems (e.g. Tuolumne), load configuration options from this file
# This line should be removed/modified for other systems
source setup_rocm_6_3_1.sh

# Override these variables to match your system configuration
PROF_TRAINER=prof-trainer
MODEL_ROOT=/p/lustre5/${USER}/models
DATA_ROOT=/p/lustre5/${USER}/data
RUN_DIR=${MODEL_ROOT}/point_diffusion_triplane
DATASET_DIR=${DATA_ROOT}/point_diffusion
VIS_CONFIG=${RUN_DIR}/point_diffusion.yaml

# Model configuration
GENERATOR_TYPE=3D-triplane
UPSCALE_TYPE=nearest
LOSS_TARGET=l1
BATCH_SIZE=128
LEARNING_RATE=1e-5
MAX_FEATURES=512
MIN_FEATURES=64

mkdir -p "${RUN_DIR}"

# Launch one training rank per GPU.
flux run -N 1 -n 4 -x -o fastload=on \
    -o mpibind=omp_proc_bind,omp_places "${PROF_TRAINER}" \
    --lr "${LEARNING_RATE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs 200 \
    --generator-type "${GENERATOR_TYPE}" \
    --upscale-type "${UPSCALE_TYPE}" \
    --max_feature "${MAX_FEATURES}" \
    --min_feature "${MIN_FEATURES}" \
    --loss_target "${LOSS_TARGET}" \
    --n_checkpoint 10 \
    --dataloader_workers 4 \
    --dataset_path "${DATASET_DIR}" \
    --keys concentration \
    --x_kernel 4 \
    --y_kernel 4 \
    --z_kernel 4 \
    --act-fun ReLU \
    --dataset_type 0 \
    --n_sims 1000000 \
    --run_directory "${RUN_DIR}" \
    --vis-config "${VIS_CONFIG}"
