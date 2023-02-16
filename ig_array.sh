#!/bin/bash

#SBATCH --account=nn9603k
#SBATCH --job-name=ig_array

#SBATCH --array=2-34
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=accel
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1

##  7d
#SBATCH --time=7-00:00:00

## Set safer defaults for bash
set -o errexit
set -o nounset


## module --quiet purge  # Clear any inherited modules
## module load Anaconda3/2020.11
## module list
## conda activate ig

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
## python3 testbla.py ${SLURM_ARRAY_TASK_ID} ./test
python3 singleGPU.py ${SLURM_ARRAY_TASK_ID} ./ig_mass