export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

BASE_ID=1
MAX_NUM_IPUS=4

# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 24 # 6 96
# for BATCHSIZE in 48 96 192 384 # 6 96
do
    for LEARNING_RATE in 0.00003
    do
	for MAX_ACTIVITY in 0.05 # 0.025 # 0.01 0.1
        # for MAX_ACTIVITY in 0.05
        do
            for SECOND_THRESH in 0.9
            do
		for NUM_HIDDEN_LAYERS_BASE in 2
		do
		for NUM_NEURONS_PER_TILE in 2 4 8 # 16
		do
		for NUM_IPUs in 1
		do
			IPU_ID=-1
	            NUM_HIDDEN_LAYERS=$((NUM_NEURONS_PER_TILE/2*NUM_HIDDEN_LAYERS_BASE))
		    echo $IPU_ID
                    # PVTI_OPTIONS='{"enable":"true", "directory":"system_analyser_reports/nmnist_numIpus1_numHidLayers2"}' python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_neuron --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD --num_neurons_per_tile=$NUM_NEURONS_PER_TILE --ipu_id=$IPU_ID --num_ipus=$NUM_IPUs
                    POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports_stepsPerEpoch/shd_numIpus1_numHidLayers2", "autoReport.executionProfileProgramRunCount":"4"}' python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=1 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_neuron --weight_mul=2.0 --sparse_size_inp=48 --dataset_name=SHD --num_neurons_per_tile=$NUM_NEURONS_PER_TILE --ipu_id=$IPU_ID --num_ipus=$NUM_IPUs
		exit	
	    done
	    	done 
		done
	    done
        done
    done
done
