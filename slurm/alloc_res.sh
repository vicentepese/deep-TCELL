#!/bin/bash 
salloc -p gpu -N 1 --gpus-per-node 1 --cpus-per-gpu 1 --time 03:00:00
