#!/bin/bash

set -e

ROCMVERSION=7.2.1
module load rocm/$ROCMVERSION
module load cray-python/3.12.12

python3 -m venv mlvenv --system-site-packages
source mlvenv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools lit ipython wheel pandas numpy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
# python -m pip install torch==2025.9.23+rocm642 torchvision==2025.9.23+rocm642 --extra-index-url https://wci-repo.llnl.gov/repository/pypi-group/simple

# -e is for editable installs
python -m pip install -e .[dev]
python -m pip install -e plugins/analytical
