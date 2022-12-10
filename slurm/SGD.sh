#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=0-9%5
#SBATCH --time=0-11:0:00               # Time: D-H:M:S
#SBATCH --account=def-keli              # Account: def-keli/rrg-keli
#SBATCH --mem=120G                       # Memory in total
#SBATCH --nodes=1                       # Number of nodes requested.
#SBATCH --cpus-per-task=12              # Number of cores per task.
#SBATCH --gres=gpu:v100l:1

#SBATCH --job-name=sgd
#SBATCH --output=job_results/_%x_%a.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tristanengst@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=1
# Print some info for context.
pwd
hostname
date

echo "Starting job number $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
module load python/3.10
source ~/virtual_envs/py310MSAM/bin/activate

# Python will buffer output of your script unless you set this.
export PYTHONUNBUFFERED=1
export MKL_SERVICE_FORCE_INTEL=1

cd ~/projects/def-keli/tme3/NNSAM
wandb agent --count 1 tristanengst/NNSAM/p8ser26u

# Print completion time.
date