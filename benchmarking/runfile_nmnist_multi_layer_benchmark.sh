export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/tonic_fork"

# # # for BATCHSIZE in 6 48 96
# # for BATCHSIZE in 48 # 6 96
# for BATCHSIZE in 48 # 6 96
# do
#     for LEARNING_RATE in 0.001
#     do
#         for SECOND_THRESH in 0.9 # 0.95 # 0.98
#         do
#             for MAX_ACTIVITY in 0.005 0.01 0.02 0.025 0.05 0.1 0.2 0.25 0.5 
#             # for MAX_ACTIVITY in 0.05
#             do
#                 for NUM_HIDDEN_LAYERS in 2 3 4 5
#                 do  
# 		     # for DATASET_NAME in NMNIST SHD
# 		     # do
# 		          python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_layer --weight_mul=2.0 --sparse_size_inp=32 --dataset_name=NMNIST --ipu_id=8
# 	     	     # done
#                 done 
#             done
#         done
#     done
# done


# # for BATCHSIZE in 6 48 96
# for BATCHSIZE in 48 # 6 96
for BATCHSIZE in 48 # 6 96
do
    for LEARNING_RATE in 0.001
    do
        for SECOND_THRESH in 0.9 # 0.95 # 0.98
        do
            for MAX_ACTIVITY in 0.01
            # for MAX_ACTIVITY in 0.05
            do
                for NUM_HIDDEN_LAYERS in 2 3 4 5
                # for NUM_NEURONS_PER_TILE in 2 4 8
                do
	            # NUM_HIDDEN_LAYERS=$((NUM_NEURONS_PER_TILE/2*3))
		    echo $NUM_HIDDEN_LAYERS
                    python3 benchmarking_script.py --use_ipu=1 --impl_method=sparse_layer --profile_run=0 --max_activity=$MAX_ACTIVITY --batchsize=$BATCHSIZE --lr=$LEARNING_RATE --transpose_weights=1 --second_thresh=$SECOND_THRESH --num_hidden_layers=$NUM_HIDDEN_LAYERS --bench_mode=multi_layer --weight_mul=2.0 --sparse_size_inp=32 --dataset_name=NMNIST --ipu_id=0
                done 
	    done
        done
    done
done
