#!/bin/bash
#SBATCH --job-name=OPT
#SBATCH -p owners
#SBATCH --time=03:00:00
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -C GPU_MEM:32GB

source env/bin/activate
python3 train_roberta_multilabel.py