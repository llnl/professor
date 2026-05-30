#!/bin/bash

set -e

# before you can run this script, you must run the following in a new bash terminal
# module load python/3.12-anaconda-2024.10
# conda init

module load python/3.12-anaconda-2024.10

# needed for bash conda activate commands
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
#conda create -p /lustre/vescratch1/$USER/mlvenv python=3.12 -y
conda activate /lustre/vescratch1/$USER/mlvenv
conda install cuda==12.6.3 cuda-toolkit -c nvidia -y
# install nightly pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

python -m pip install .[dev]
python -m pip install plugins/analytical

#ROCMVERSION=6.2.4
#module load rocm/$ROCMVERSION
#module load cray-python/3.11.7
# i expect cuda 11 to have bad performance on gh200
# pytorch binaries don't exisit for cuda 12.5
#module load cuda/11.8
#module load python/3.12-anaconda-2024.10
#module load cray-python/3.11.7
#module load python/3.10-anaconda-2023.03
#python3 -m venv mlvenv --system-site-packages
#source mlvenv/bin/activate

#python -m pip install --upgrade pip
#python -m pip install --upgrade setuptools lit ipython wheel
#python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu118


#python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126



#python -m pip install torch --index-url https://download.pytorch.org/whl/cu118





#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118


#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118



#python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# -e is for editable installs
#python -m pip install -e .[dev]
#python -m pip install -e plugins/analytical
