#!/bin/bash
##### These lines are for Slurm
#SBATCH -N 1
#SBATCH -J electron_build
#SBATCH -t 01:00:00
#SBATCH -A geophys
#SBATCH -p pbatch
#SBATCH --output electron_build_dane.out
#SBATCH --error electron_build_dane.err

set -Eeuo pipefail
trap 'status=$?; printf "Build failed with status %s at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

module load gcc/13.3.1-magic

export ELECTRON_BUILD_ROOT=/p/lustre2/sherman/professor_electron
export COMPILER_ID=gcc-13.3.1-magic
export TORCH_VERSION=2.13.0
export TORCHVISION_VERSION=0.28.0
export PYTHON_VERSION=3.14.7
export NODE_VERSION=24.21.0
export TRIANGLE_VERSION=v20250106
export PROFESSOR_REF=feature/sherman/electronApp

/usr/workspace/sherman/MADA/professor/electron/build_scripts/build_electron_app.sh
