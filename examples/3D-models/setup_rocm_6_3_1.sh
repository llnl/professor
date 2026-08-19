#!/bin/bash

set -e

ROCM_VER=6.3.1
module load rocm/$ROCM_VER
module load "rocmcc/$ROCM_VER-magic"

MPICH_VER=8.1.32
module load cray-mpich/$MPICH_VER

# ######################
# ### STAR AMD ROCM  ###
# ######################

# Get the hostname of the first node in the Flux allocation.
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
echo "first host: $firsthost"

# Set MASTER_ADDR to the first compute node and MASTER_PORT to an unused port.
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456
echo "$MASTER_ADDR"

# the AWS-OFI-RCCL plugin lets RCCL use libfabric instead of TCP sockets
# settings below taken from:
#   https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl#running-rccl-perf-tests
AWS_OFI_RCCL_ROOT=${AWS_OFI_RCCL_ROOT:-/usr/workspace/prof/lib/aws-ofi-rccl/src}
export LD_LIBRARY_PATH=${AWS_OFI_RCCL_ROOT}:${LD_LIBRARY_PATH:-}
export FI_CXI_ATS=0

export LD_PRELOAD="/opt/rocm-${ROCM_VER}/lib/libMIOpen.so"

# Point to node-local storage to cache MIOpen performance DB files and pre-compiled kernels
# These otherwise default to user home directories on NFS like ~/.config/miopen/ and ~/.cache/miopen
#   https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html
export MIOPEN_USER_DB_PATH=${TMPDIR:?TMPDIR must be set}/my-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf -- "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Higher NET_GDR values enable GPU Direct RDMA in more situations.
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
