#!/bin/bash 
#SBATCH --job-name=OPT_BATCH_SIZE
#SBATCH --output=OPT_BATCH_SIZE_${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=OPT_BATCH_SIZE_${SLURM_ARRAY_TASK_ID}.err
#SBATCH -p gpu
#SBATCH --time=08:00:00
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB
#SBATCH -C GPU_BRD:TESLA
#SBATCH --array=0-4

# Optimize batch size 
BATCH_SIZE=(8 16 24 32)

# Submit
source env/bin/activate
python3 train_roberta_multilabel.py --batch_size ${BATCH_SIZE[$SLURM_ARRAY_TASK_ID]}