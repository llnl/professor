#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -J prof-trainer
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -A llnl_ai_g
#SBATCH --exclusive

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /lustre/vescratch1/$USER/mlvenv

EXE=/lustre/scratch1/$USER/mlvenv/bin/prof-trainer
PFOLDER=/lustre/scratch1/$USER/models/analytical_data
DATADIR=/lustre/scratch1/$USER/data/analytical_data

mkdir -p $PFOLDER
cd $PFOLDER

export LOSS_TARGET=l1
export BATCH_SIZE=296
export LR=1e-5

export MAX_FEATURE=1024
export MIN_FEATURE=128

# this uses 4 gpus on one Node!
srun -N 1 -n 4 python -u $EXE \
    --lr $LR \
    --batch_size ${BATCH_SIZE} \
    --num_epochs 100 \
    --max_feature ${MAX_FEATURE} \
    --min_feature ${MIN_FEATURE} \
    --loss_target ${LOSS_TARGET} \
    --n_checkpoint 20 \
    --dataset_path ${DATADIR} \
    --keys 'function' \
    --y_kernel 4 \
    --x_kernel 4 \
    --dataset_type 0  \
    --n_sims 1000000
