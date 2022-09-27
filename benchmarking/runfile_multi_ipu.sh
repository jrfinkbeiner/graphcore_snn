export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

BASE_ID=1

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 24 # 6 96
do
    for LEARNING_RATE in 0.003
    do
        for MAX_ACTIVITY in 0.05
        do
            for SECOND_THRESH in 0.9
            do
		for NUM_HIDDEN_LAYERS_BASE in 3
		do
		for NUM_NEURONS_PER_TILE in 2 4 8 16
		do
		# for NUM_IPUs in 2 4 8
		for NUM_IPUs in 16
		do

		if [[ $NUM_IPUs -eq 2 ]]
		then
			IPU_ID=$((16 + 4*BASE_ID))
		elif [[ $NUM_IPUs -eq 4 ]]
		then
			IPU_ID=$((16 + 8 + 2*BASE_ID))
		elif [[ $NUM_IPUs -eq 8 ]]
		then
			IPU_ID=$((16 + 8 + 4 + BASE_ID))
		elif [[ $NUM_IPUs -eq 16 ]]
		then
			IPU_ID=30
		else
			IPU_ID=-1
		fi
			NUM_HIDDEN_LAYERS=$((NUM_NEURONS_PER_TILE/2*NUM_HIDDEN_LAYERS_BASE))
            # python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_neuron --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD --num_neurons_per_tile=$NUM_NEURONS_PER_TILE --ipu_id=$IPU_ID --num_ipus=$NUM_IPUs
            python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_neuron --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD --num_neurons_per_tile=$NUM_NEURONS_PER_TILE --ipu_id=$IPU_ID --num_ipus=$NUM_IPUs			
		done
	    	done 
		done
	    done
        done
    done
done
