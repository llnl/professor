#!/bin/bash

set -e

module load cuda/12.2.2

python=/usr/workspace/prof/opence1.7.2/bin/python

datafolder=/p/gpfs1/${USER}/data/analytical_data
mkdir -p $datafolder

cp fouriermodes_study_mpi.py $datafolder
cd $datafolder

export MPLBACKEND=agg
jsrun -r 40 $python -u fouriermodes_study_mpi.py -n 18
ls -1 *.h5 > filelist.txt
