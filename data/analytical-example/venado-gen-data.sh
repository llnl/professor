#!/bin/bash

set -e

module load python/3.12-anaconda-2024.10

# needed for bash conda activate commands
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /lustre/vescratch1/$USER/mlvenv

datafolder=/lustre/vescratch1/$USER/data/analytical_data
mkdir -p $datafolder

cp fouriermodes_study_mpi.py $datafolder
cd $datafolder

export MPLBACKEND=agg
srun -n 143 python -u fouriermodes_study_mpi.py -n 18
ls -1 *.h5 > filelist.txt
