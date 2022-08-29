#!/bin/sh
#SBATCH -Ahpsadm
#SBATCH -pdc-ipu
#SBATCH -N1
#SBATCH --time=20:00:00

# srun hostname |sort

# srun singularity run ~/tf2tonic.sif -- ./runfile_nmnist.sh
# srun singularity run ~/tf2.sif -- ./runfile_randman.sh
srun singularity run ~/tf2.sif -- python3 keras_train_util_ipu.py
