#!/bin/bash

set -e

module load rocm/6.2.4
module load python/3.13.2
# source /usr/workspace/${USER}/.mlvenv_professor_dev/bin/activate

datafolder=/p/lustre5/${USER}/data/analytical_data
mkdir -p $datafolder

cp fouriermodes_study_mpi.py $datafolder
cd $datafolder

export MPLBACKEND=agg
flux run -n 96 python -u fouriermodes_study_mpi.py -n 18
ls -1 *.h5 > filelist.txt
