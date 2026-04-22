#!/bin/bash -l

#SBATCH --partition=
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=
#SBATCH --cpus-per-task=

CONFIG=$1
WORK_DIR=$2

python test.py "${CONFIG}" --work-dir="${WORK_DIR}" --launcher="slurm"

