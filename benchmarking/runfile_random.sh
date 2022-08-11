# echo $LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/source/custom_dyn_dense_sparse_matmul/batched/standard
# echo $LD_LIBRARY_PATH
# export CUSTOM_CODELET_BASE_PATH=asd/fgh/in_runfile
# export LD_LIBRARY_PATH=/p/home/jusers/finkbeiner1/jureca/phd/pgi15_projects/graphcore_snn/build/lib:$LD_LIBRARY_PATH
python3 train_random.py --use_ipu=1 --impl_method=sparse_ops --profile_run=1 --transpose_weights=0
# # POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_test_random_large_2states_manyReduce"}' python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0
# # POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_random_2states_transpose_twoRow"}' python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=1

# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_layer/sparse_layer_02_selectSpikes_leftSideCounterInBreak"}' python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=1
# # POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_ops/sparse_layer_baseline_large"}' python3 train_random.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0
