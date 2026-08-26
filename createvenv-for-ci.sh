#!/bin/bash

set -e

ROCMVERSION=7.2.1
module load rocm/$ROCMVERSION
module load cray-python/3.12.12

python3 -m venv /usr/workspace/${USER}/.mlvenv_professor_dev --system-site-packages
source /usr/workspace/${USER}/.mlvenv_professor_dev/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade "packaging>=24.2"
python -m pip install --upgrade setuptools lit ipython wheel flake8 Flake8-pyproject pandas
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

