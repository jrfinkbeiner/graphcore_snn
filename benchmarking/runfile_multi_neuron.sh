export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
# for BATCHSIZE in 12 24 # 6 96
for BATCHSIZE in 24 # 6 96
do
    for LEARNING_RATE in 0.003
    do
        for MAX_ACTIVITY in 0.05 0.01 0.025 0.1
        do
            for SECOND_THRESH in 0.9
            do
                for NUM_HIDDEN_LAYERS_BASE in 4 3 2
		do
		for NUM_NEURONS_PER_TILE in 16 # 2 4 8 # 16 # 32
                do
	            NUM_HIDDEN_LAYERS=$((NUM_NEURONS_PER_TILE/2*NUM_HIDDEN_LAYERS_BASE))
		    echo $NUM_HIDDEN_LAYERS
                    python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_neuron --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD --num_neurons_per_tile=$NUM_NEURONS_PER_TILE --ipu_id=8
                done 
		done
	    done
        done
    done
done
