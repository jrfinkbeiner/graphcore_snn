export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=16 --transpose_weights=0 --batchsize=192
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_nmnist_2states_notranspose_baseline"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0 --batchsize=48 --sparse_multiplier=16
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_nmnist_2states_transpose_MultiRowsSIMD_stateUpdBase"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=1 --batchsize=48 --sparse_multiplier=16

# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_gcMeeting/nmnist_sparse_layer_transpose_spmul16_bs048_seqlen10"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_ops --profile_run=1 --transpose_weights=1 --batchsize=48 --sparse_multiplier=16
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_vec/vec_nmnist_2states_var5Init_notranspose_baseline"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0 --batchsize=48 --sparse_multiplier=16

# # for VARIABLE in 1 2 4 8 16 32 64 128 256

# for BATCHSIZE in 6 48 96
for BATCHSIZE in 48
do
    # for LEARNING_RATE in 0.3 0.1 0.03 0.01 0.003
    for LEARNING_RATE in 30 10 3
    do
        # for VARIABLE in 1 2 4 8 16 32 64 128 256
        # python3 train_nmnist.py --use_ipu=1 --impl_method=dense --profile_run=0 --batchsize=$BATCHSIZE --lr=$LEARNING_RATE
        for VARIABLE in 8 16 32
        do
            python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=$VARIABLE --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1
        done
    done
done
