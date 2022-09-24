export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 48 # 6 96
do
    for LEARNING_RATE in 0.001
    do
        for MAX_ACTIVITY in 0.01 0.05 0.1
        do
            for SECOND_THRESH in 0.9 0.95 # 0.98
            do
                for NUM_HIDDEN_LAYERS in 2 3 4 5
                do
                    python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_layer --weight_mul=2.0 --sparse_size_inp=96 --dataset_name=NMNIST
                done 
            done
        done
    done
done
