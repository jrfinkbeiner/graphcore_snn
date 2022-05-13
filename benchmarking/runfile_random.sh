# python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/final_test_random_large_2states_baseline"}' python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0
