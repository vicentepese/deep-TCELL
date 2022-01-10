#!/bin/bash
#SBATCH --job-name=TRAIN_ROBERTA
#SBATCH -p gpu
#SBATCH --time=03:00:00
#SBATCH -G 1
#SBATCH -N 1

source env/bin/activate
python3 train_roberta_multilabel.py