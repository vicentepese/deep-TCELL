#!/bin/bash 

#SBATCH --job-name=OPT
#SBATCH -p gpu
#SBATCH --time=03:00:00
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --array=0-20

# Submit
source env/bin/activate
python3 train_roberta_PMBEC.py --jobid $SLURM_JOBID
