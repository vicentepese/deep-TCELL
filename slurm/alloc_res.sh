#!/bin/bash 
salloc -p gpu -c 1 -G 1 -C GPU_BRD:TESLA -t 08:00:00
