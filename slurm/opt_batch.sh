#!/bin/bash 
#SBATCH --job-name=OPT_ALLEGRO
#SBATCH --output=OPT_ALLEGRO.out
#SBATCH --error=OPT_ALLEGRO.err
#SBATCH -p gpu
#SBATCH --time=05:00:00
#SBATCH -G 5

# Submit
source env/bin/activate
python3 allegro_opt.py