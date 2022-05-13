export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=1 --transpose_weights=1
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/nmnist/nmnist_large_constInit5_multiRow_epoch0"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1

# # for VARIABLE in 1 2 4 8 16 32 64 128 256
# for VARIABLE in 32 64
# do
# 	python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=$VARIABLE
# done