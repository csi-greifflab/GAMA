#!/bin/bash

#SBATCH --account=nn9603k
#SBATCH --job-name=ig_array

#SBATCH --array=0-16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=accel
#SBATCH --mem-per-cpu=1G

#  5min
#SBATCH --time=0-00:05:00

## Set safer defaults for bash
set -o errexit
set -o nounset

module --quiet purge  # Clear any inherited modules
module load conda!!!!!!!!!!!!!!!!!!!
conda activate ig

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python3 singleGPU.py ${SLURM_ARRAY_TASK_ID}