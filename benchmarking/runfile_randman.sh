export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/randman"

python3 train_randman.py --use_ipu=1 --impl_method=sparse_layer --profile_run 0
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/randman/sparse_layer_1024_1024_1024_1024_1024_1024_10_t50_bs48_nb5_prodRun_sp32_sgd"}' python3 train_randman.py --use_ipu=1 --impl_method=sparse_layer --profile_run 1
