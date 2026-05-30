#!/bin/bash

set -e

module load python/3.11.5

python3 -m venv .mlvenv-rzhound --system-site-packages
source .mlvenv-rzhound/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools lit ipython wheel
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install -e .[dev]
