#!/bin/bash

set -e

ROCMVERSION=6.3.1
module load rocm/$ROCMVERSION
module load cray-python/3.11.7

python3 -m venv /usr/workspace/${USER}/.mlvenv_professor_dev --system-site-packages
source /usr/workspace/${USER}/.mlvenv_professor_dev/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade "packaging>=24.2"
python -m pip install --upgrade setuptools lit ipython wheel flake8 Flake8-pyproject pandas
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3

