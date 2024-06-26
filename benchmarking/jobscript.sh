#!/bin/bash -x
#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j
#SBATCH --time=04:00:00
#SBATCH --partition=dc-ipu

# srun singularity run ~/tf2.sif -- python3 multi_ipu.py
srun singularity run ~/tf2.sif -- ./multi_testrun.sh
