#!/bin/bash
#SBATCH --job-name=EDA
#SBATCH -G 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --output=EDA.out
#SBATCH --error=EDA.err
#SBATCH --nodes=1

#SBATCH -t 10:00:00

source env/bin/activate
python3 eda.py