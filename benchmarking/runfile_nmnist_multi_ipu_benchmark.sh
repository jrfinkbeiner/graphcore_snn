

for BATCHSIZE in 48 # 6 96
do
    for LEARNING_RATE in 0.001
    do
        for VARIABLE in 16
        do
            for NUM_IPUS in 8 16
            # for NUM_IPUS in 2 3 4 5
            do
                python3 train_nmnist.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --sparse_multiplier=$VARIABLE --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --num_ipus=$NUM_IPUS
                # CUDA_VISIBLE_DEVICES="3" python3 benchmark_multi_layer_nmnist.py --use_ipu=0 --impl_method=dense --profile_run=0 --sparse_multiplier=$VARIABLE --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --num_ipus=$NUM_IPUS
            done
            
        done
    done
done
