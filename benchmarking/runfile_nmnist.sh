export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic"

# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=32 --batchsize=48 --lr=0.0003 --transpose_weights=1
# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=4 --batchsize=48 --lr=0.001 --transpose_weights=1


# TF_POPLAR_FLAGS='--executable_cache_path=./cachedir_poplar' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.9 --num_ipus=1
# TF_POPLAR_FLAGS='--executable_cache_path=./cachedir_poplar' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.95 --num_ipus=1

# exit

# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=16 --batchsize=48 --transpose_weights=1 --second_thresh=0.9 --num_ipus=1 --lr=0.01 
# exit
# # python3 train_nmnist.py --use_ipu=0 --impl_method=dense --profile_run=0 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=0
# # python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1
# # python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_neal/nmnist_sparse_layer_transpose_1472_neal_benchmark_hidLayers3_numIPUs1_secondThresh0.9_evenAllocInpSpikesGrad_0.8"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.9 --num_ipus=1
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_neal/nmnist_sparse_layer_transpose_1472_neal_benchmark_numIPUs1_secondThresh0.95"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.95 --num_ipus=1

# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_neal/nmnist_sparse_layer_transpose_1472_neal_benchmark_numIPUs2_secondThresh0.9"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.9 --num_ipus=2
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_neal/nmnist_sparse_layer_transpose_1472_neal_benchmark_numIPUs2_secondThresh0.95"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --sparse_multiplier=16 --batchsize=48 --lr=0.001 --transpose_weights=1 --second_thresh=0.95 --num_ipus=2



# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=16 --transpose_weights=0 --batchsize=192
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_nmnist_2states_notranspose_baseline"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0 --batchsize=48 --sparse_multiplier=16
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/vec_nmnist_2states_transpose_MultiRowsSIMD_stateUpdBase"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=1 --batchsize=48 --sparse_multiplier=16

# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_gcMeeting/nmnist_sparse_layer_transpose_spmul16_bs048_seqlen10"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_ops --profile_run=1 --transpose_weights=1 --batchsize=48 --sparse_multiplier=16
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_vec/vec_nmnist_2states_var5Init_notranspose_baseline"}' python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --transpose_weights=0 --batchsize=48 --sparse_multiplier=16

# # for VARIABLE in 1 2 4 8 16 32 64 128 256

# python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=16 --batchsize=48 --transpose_weights=1 --second_thresh=0.9 --num_ipus=1 --lr=0.01 

for LEARNING_RATE in 0.01 0.001 0.0001
do
    for SPARSE_MUL in 16 64
    do
 
    	for SECOND_THRESH in 0.8 0.9 -100.0 0.5 0.95 0.0 1.0
	    do
    # python3 train_nmnist.py --use_ipu=0 --impl_method=dense --profile_run=0 --batchsize=$BATCHSIZE --lr=$LEARNING_RATE
    # for VARIABLE in 4 8 16 32 64 128
    # do

            python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=$SPARSE_MUL --batchsize=48 --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH
    
	    done
    done
done


exit

for VARIABLE in 16 64 8 
do

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 48 # 6 96
do
    # for LEARNING_RATE in 0.1 0.03 0.01 0.003 0.001 0.0003
    # for LEARNING_RATE in 0.03 0.01 0.003 0.001
    for LEARNING_RATE in 0.003 0.001 0.0003 0.0001
    # for LEARNING_RATE in 0.0001
    do
        # # for VARIABLE in 1 2 4 8 16 32 64 128 256
        # python3 train_nmnist.py --use_ipu=0 --impl_method=dense --profile_run=0 --batchsize=$BATCHSIZE --lr=$LEARNING_RATE
        # for VARIABLE in 4 8 16 32 64 128
        # do

            python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=$VARIABLE --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1
        done
    done
done
