#!/bin/sh
#SBATCH -Aexalab
#SBATCH -pdc-ipu
#SBATCH -N1
#SBATCH --time=20:00:00

# srun hostname |sort


# srun singularity run ~/tf2tonic.sif -- ./runfile_nmnist_multi_ipu_benchmark.sh
# srun singularity run ~/tf2tonic.sif -- ./runfile_nmnist.sh
srun singularity run ~/tf2tonic.sif -- ./runfile_nmnist_multi_layer_benchmark.sh
# srun singularity run ~/tf2.sif -- ./runfile_randman.sh
# srun singularity run ~/tf2.sif -- python3 keras_train_util_ipu.py
# srun singularity run ~/tf2tonic.sif -- python3 nmnist_util.py
