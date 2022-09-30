export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 48 # 6 96
do
    for LEARNING_RATE in 0.001
    do
        for MAX_ACTIVITY in 0.01
        do
            for SECOND_THRESH in 0.9
            do
                for NUM_HIDDEN_LAYERS in 2 3 4 5
                do
		            CUDA_VISIBLE_DEVICES="2" python3 benchmarking_script.py --use_ipu=0 --impl_method=dense --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_layer --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD
                done 
            done
        done
    done
done
