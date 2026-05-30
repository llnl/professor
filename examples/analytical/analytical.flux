#!/bin/bash
#flux: -N 1
#flux: -n 4
#flux: -t 1h
#flux: -q pbatch

set -e

ROCM_VER=6.3.1

module load rocm/$ROCM_VER
module load "rocmcc/$ROCM_VER-magic"

MPICH_VER=8.1.32
module load cray-mpich/$MPICH_VER

# ######################
# ### STAR AMD ROCM  ###
# ######################

# TODO: use flux python module to set MASTER_ADDR from within the script
# get the hostname of the first node in the flux allocation
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
echo "first host: $firsthost"

# echo "$(flux getattr hostlist | /bin/hostlist -n 1) slots=8
# $(flux getattr hostlist | /bin/hostlist -n 2) slots=8" > hostlist

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any unused port number
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456
echo "$MASTER_ADDR"

# the AWS-OFI-RCCL plugin lets RCCL use libfabric instead of TCP sockets
# settings below taken from:
#   https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl#running-rccl-perf-tests
export LD_LIBRARY_PATH=/usr/workspace/prof/lib/aws-ofi-rccl/src:$LD_LIBRARY_PATH
export FI_CXI_ATS=0

export LD_PRELOAD="/opt/rocm-${ROCM_VER}/lib/libMIOpen.so"

# Point to node-local storage to cache MIOpen performance DB files and pre-compiled kernels
# These otherwise default to user home directories on NFS like ~/.config/miopen/ and ~/.cache/miopen
#   https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html
export MIOPEN_USER_DB_PATH=$TMPDIR"/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# this value is set to 3 above
# higher NET_GDR values enable GPU Direct RDMA in more situations
#   https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level-formerly-nccl-ib-gdr-level
export NCCL_NET_GDR_LEVEL=4

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level
export NCCL_P2P_LEVEL=4

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-min-nchannels
export NCCL_MIN_NRINGS=4

# to enable PCIe point-to-point between GPUs on a node
#  https://github.com/ROCmSoftwarePlatform/rccl#enabling-peer-to-peer-transport
export HSA_FORCE_FINE_GRAIN_PCIE=1

# ######################
# ### END AMD ROCM   ###
# ######################

EXE=/usr/workspace/prof/bin/prof-trainer
PFOLDER=/p/lustre5/$USER/models/analytical_data
DATADIR=/p/lustre5/$USER/data/analytical_data
VIS_CONFIG=/p/lustre5/$USER/models/analytical_data/analytical_visualization.yaml

mkdir -p $PFOLDER
cd $PFOLDER


export LOSS_TARGET=l1
export BATCH_SIZE=296
export LR=1e-5

export MAX_FEATURE=1024
export MIN_FEATURE=128

# this uses 4 gpus per Node!
flux run -N 1 -n 4 -x -o fastload=on -o mpibind=omp_proc_bind,omp_places $EXE \
    --lr $LR \
    --batch_size ${BATCH_SIZE} \
    --num_epochs 25 \
    --max_feature ${MAX_FEATURE} \
    --min_feature ${MIN_FEATURE} \
    --loss_target ${LOSS_TARGET} \
    --n_checkpoint 2000 \
    --dataset_path ${DATADIR} \
    --keys 'function' \
    --y_kernel 4 \
    --x_kernel 4 \
    --dataset_type 0  \
    --n_sims 1000000 \
    --vis-config  ${VIS_CONFIG}
