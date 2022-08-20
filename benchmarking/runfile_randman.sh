export PYTHONPATH="${PYTHONPATH}:/p/home/jusers/finkbeiner1/jureca/util/randman"

# python3 train_randman.py --use_ipu=1 --impl_method=dense --profile_run=1 --sparse_multiplier=16
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports/randman/sparse_layer_1024_1024_1024_1024_1024_1024_10_t50_bs48_nb5_prodRun_sp32_sgd"}' python3 train_randman.py --use_ipu=1 --impl_method=sparse_layer --profile_run 1


# python3 train_randman.py --use_ipu=1 --impl_method=dense --profile_run=0 --lr=0.05 --sparse_multiplier=16
# for SPARSE_MUL in 1 2 4 8 16 32 64 128 256
# do
#     python3 train_randman.py --use_ipu=1 --profile_run=0 --lr=0.05 --impl_method=sparse_layer --sparse_multiplier=$SPARSE_MUL
# done

# for LEARNING_RATE in 3.0 1.0 0.3 0.1 0.03 0.01 0.003
# for LEARNING_RATE in 0.1 0.03 0.01 0.003
for LEARNING_RATE in 1.0 0.3
do
    python3 train_randman.py --use_ipu=1 --profile_run=0 --lr=$LEARNING_RATE --impl_method=dense
    # for SPARSE_MUL in 4 8 16
    for SPARSE_MUL in 16
    do
        python3 train_randman.py --use_ipu=1 --profile_run=0 --lr=$LEARNING_RATE --impl_method=sparse_ops --sparse_multiplier=$SPARSE_MUL --transpose_weights=0
    #     python3 train_randman.py --use_ipu=1 --profile_run=0 --lr=$LEARNING_RATE --impl_method=sparse_layer --sparse_multiplier=$SPARSE_MUL --transpose_weights=1
    done
done
