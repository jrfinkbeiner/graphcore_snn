export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

python3 train_nmnist.py --use_ipu=1 --impl_method=dense --profile_run 1
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/nmnist/test"}' python3 train_nmnist.py --use_ipu=1 --impl_method=dense --profile_run 1