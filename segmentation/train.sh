#!/bin/bash -l

#SBATCH --partition=
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=
#SBATCH --cpus-per-task=

CONFIG=$1
WORK_DIR=$2

python -u train.py "${CONFIG}" --work-dir="${WORK_DIR}" --launcher="slurm"

