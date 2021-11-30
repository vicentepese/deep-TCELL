#!/bin/bash
srun -p gpu -N 2 --gpus-per-node 1 --cpus-per-gpu 1 --time 03:00:00 clearml-agent daemon --queue default