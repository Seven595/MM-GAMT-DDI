#!/bin/bash
#BSUB -q gpu
#BSUB -gpgpu 1
#BSUB -app default
#BSUB -n 16
#BSUB -e error.%J
#BSUB -o output.%J
#BSUB -W 48:00
#BSUB -J ddi570covv4
module load cuda-11.4
module load anaconda3
source activate rdkit
python dataPrepare-570-cov-v4.py


