#!/bin/bash
#flux: --job-name=build_electron_tuolumne
#flux: --output='electron_build_tuolumne.out'
#flux: --error='electron_build_tuolumne.err'
#flux: -N 1
#flux: -n 32
#flux: -t 45m
#flux: -B geophys
#flux: -q pdebug

set -Eeuo pipefail
trap 'status=$?; printf "Build failed with status %s at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

module load rocm/7.2.1
module load gcc/13.3.1-magic

USER_NAME=${USER:-$(id -un)}
export ELECTRON_BUILD_ROOT="${ELECTRON_BUILD_ROOT:-/p/lustre5/${USER_NAME}/professor_electron}"
export COMPILER_ID=gcc-13.3.1-magic
export TORCH_VARIANT=rocm7.2
export TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2
export TORCH_VERSION=2.13.0
export TORCHVISION_VERSION=0.28.0
export PYTHON_VERSION=3.14.7
export NODE_VERSION=24.21.0
export TRIANGLE_VERSION=v20250106
export PROFESSOR_REF=${PROFESSOR_REF:-feature/sherman/electronApp}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/build_electron_app.sh"
