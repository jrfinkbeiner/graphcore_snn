# module load SciPy-bundle/2021.10
# module load numba/0.55.1
# module load tqdm/4.62.3
# module load h5py/3.5.0-serial

export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic"

python3 train_nmnist.py --use_ipu=1 --impl_method=dense --profile_run 1
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/nmnist/test"}' python3 train_nmnist.py --use_ipu=1 --impl_method=dense --profile_run 1