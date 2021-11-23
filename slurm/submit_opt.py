import os
import pdb
import sys
import tempfile

JOBS = [
    ('TCELL_grid_1', 'python3 train_roberta_multilabel.py --opt True --batch_size 8 --learning_rate 1e-5 --dropout 0.1'),
	('TCELL_grid_2', 'python3 train_roberta_multilabel.py --opt True --batch_size 12 --learning_rate 1e-5 --dropout 0.1'),
	('TCELL_grid_3', 'python3 train_roberta_multilabel.py --opt True --batch_size 16 --learning_rate 1e-5 --dropout 0.1'),
	('TCELL_grid_4', 'python3 train_roberta_multilabel.py --opt True --batch_size 24 --learning_rate 1e-5 --dropout 0.1'),
	('TCELL_grid_4', 'python3 train_roberta_multilabel.py --opt True --batch_size 32 --learning_rate 1e-5 --dropout 0.1'),
	('TCELL_grid_5', 'python3 train_roberta_multilabel.py --opt True --batch_size 64 --learning_rate 1e-5 --dropout 0.1')
    ]


def submit_job(jobname, experiment):

    content = '''#!/bin/bash
#SBATCH --job-name={0}
#SBATCH -p gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH --output=/home/users/vipese/Projects/deep-TCELL/logs/{0}.out
#SBATCH --error=/home/users/vipese/Projects/deep-TCELL/logs/{0}.err
##################################################

source /home/users/vipese/Projects/deep-TCELL/env/bin/activate

{1}
'''
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system('sbatch {}'.format(j.name))


if __name__ == '__main__':

    for job in JOBS:
        submit_job(job[0], job[1])

    print('All jobs have been submitted!')
