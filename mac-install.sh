#!/bin/bash

set -e

installdir=$(pwd)/anaconda

# check for x86 or apple silicone
case $(uname -m) in
    x86_64)
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        sh Miniconda3-latest-MacOSX-x86_64.sh -b -f -p $installdir
        ;;
    arm64)
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        sh Miniconda3-latest-MacOSX-arm64.sh -b -f -p $installdir
        ;;
esac

# activate conda environment
source $installdir/bin/activate

# create an opence environment in conda (Python-3.11)
conda create -y -n professor python=3.11
conda activate professor

# install pytorch and dependencies
conda install -y pytorch::pytorch torchvision torchaudio -c pytorch
conda install -y mpi4py h5py matplotlib scipy ipython pyqt

# install professor
python -m pip install -e .[dev] --no-build-isolation
python -m pip install -e plugins/analytical
