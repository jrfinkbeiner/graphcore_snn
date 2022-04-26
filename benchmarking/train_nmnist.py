import os
import sys
import functools as ft
import numpy as np

from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, sparse2dense
from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, sparse2dense_ipu
# from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, sparse2dense, create_dataset_dense
# from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, create_dataset_sparse, sparse2dense_ipu


import tensorflow as tf
import tensorflow.keras as keras 
# from keras.callbacks import CSVLogger

from nmnist_util import get_nmnist_dataset

    # def reg_func(spikes_list):
    #     loss_reg = 0.0
    #     for spikes in spikes_list:
    #         loss_reg = loss_reg + activity_reg_lower(spikes) + activity_reg_upper(spikes)
    #     return loss_reg


def get_single_layer_reg_function(lambda_lower, nu_lower, lambda_upper, nu_upper):

    def activity_reg_lower(spikes):
        return lambda_lower * tf.math.reduce_mean( tf.nn.relu(nu_lower - spikes)**2 )

    def activity_reg_upper(spikes):
        return lambda_upper * tf.nn.relu(tf.math.reduce_mean(spikes - nu_upper))**2

    def reg_func(y_true, spikes):
        sum_spikes_per_neuron = tf.reduce_mean(spikes, axis=1) # given shape (batch, seq_len, neurons)
        loss_reg = activity_reg_lower(sum_spikes_per_neuron) + activity_reg_upper(sum_spikes_per_neuron)
        return loss_reg

    return reg_func

reg_func_single_layer = get_single_layer_reg_function(lambda_lower=100, nu_lower=1e-3, lambda_upper=100, nu_upper=15)




# def mse_softmax_loss_fn(y_target, y_pred):
#     print(y_target.shape)
#     print(y_pred.shape)
#     sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
#     print(sum_spikes.shape)
#     softmax_pred = tf.nn.softmax(sum_spikes, axis=1)
#     one_hot_target = tf.one_hot(y_target, softmax_pred.shape[-1], axis=-1, dtype=softmax_pred.dtype)
#     return tf.math.reduce_sum((softmax_pred-one_hot_target)**2)/y_target.shape[-1]
#     # return tf.math.reduce_max(softmax_pred)
#     # return tf.math.reduce_sum(sum_spikes)
#     # return tf.math.reduce_max((softmax_pred-one_hot_target)**2)/y_pred.shape[-1]

# # def mse_softmax_loss_with_reg_fn(y_target, y_pred):

# #     loss_class = mse_softmax_loss_fn(y_target, y_pred[-1])
# #     loss_reg = reg_func(y_target)
    
# #     return loss_class + 0.1 * loss_reg

# @tf.function(experimental_compile=True)
# def calc_accuracy(y_true, y_pred):
#     sum_spikes = tf.reduce_sum(y_pred, axis=1)
#     preds = tf.math.argmax(sum_spikes, axis=1)
#     # preds = tf.ones_like(y_true)
#     y_true = tf.cast(y_true, preds.dtype)
#     identical = tf.cast(preds==y_true, tf.float32)
#     return tf.math.reduce_mean(identical)

# @tf.function(experimental_compile=True)
# def calc_activity(y_true, y_pred):
#     sum_spikes = tf.reduce_mean(y_pred)*y_pred.shape[1]
#     return sum_spikes

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)


def get_sum_and_sparse_categorical_crossentropy_sparse_out(out_dim, transpose=False):
    def loss_fn(y_true, y_pred):
        y_pred_dense = sparse2dense_ipu(y_pred, out_dim)
        if transpose:
            y_pred_dense = tf.transpose(y_pred_dense, perm=[1, 0, 2])
        return sum_and_sparse_categorical_crossentropy(y_true, y_pred_dense)
    return loss_fn


def main(args):

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    ROOT_PATH_DATA = "/p/scratch/chpsadm/finkbeiner1/datasets"

    PROFILE_RUN = bool(args.profile_run)
    USE_IPU = bool(args.use_ipu)
    IMPL_METHOD = args.impl_method
    if USE_IPU:
        assert IMPL_METHOD is not None, "If `USE_IPU=True` the variable `IMPL_METHOD` must be set."
        assert IMPL_METHOD in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{IMPL_METHOD}'."
    SPARSE_METHOD = ("sparse" in IMPL_METHOD) and USE_IPU

    NUM_CLASSES = 10
    if PROFILE_RUN:
        NUM_EPOCHS = 10
        SEQ_LEN = 11
    else:
        NUM_EPOCHS = 10
        SEQ_LEN = 100 # 300


    IMAGE_DIMS = (34,34,2)

    DENSE_SIZES = [np.prod(IMAGE_DIMS), 1024, 512, 128, NUM_CLASSES]
    SPARSE_SIZES = [32, 48, 32, 16, 8]

    BATCHSIZE = 48
    if PROFILE_RUN:
        # NUM_SAMPLES_TRAIN = BATCHSIZE*4
        NUM_SAMPLES_TRAIN = BATCHSIZE*16
    else:
        NUM_SAMPLES_TRAIN = 54210

    print("DENSE_SIZES: ", DENSE_SIZES)
    print("SPARSE_SIZES: ", SPARSE_SIZES)
    print("BATCHSIZE: ", BATCHSIZE)
    print("NUM_SAMPLES_TRAIN: ", NUM_SAMPLES_TRAIN)
    print("SEQ_LEN: ", SEQ_LEN)

    rng = np.random.default_rng(42)

    BATCHSIZE_PER_STEP = BATCHSIZE
    STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE)
    TRAIN_STEPS_PER_EXECUTION = STEPS_PER_EPOCH

    DECAY_CONSTANT = 0.9
    THRESHOLD = 1.0

    LOG_FILE = None # f"log_sparse_bs{BATCHSIZE}.csv"

    # if SPARSE_METHOD:
    #     sys.exit()
    #     dataloader_train = get_nmnist_dataset(sparse_sizes[0], int(steps_per_epoch*batchsize), batchsize_per_step, image_dims, seq_len, sparse=True)
    # else:
    INP_DIM = SPARSE_SIZES[0] if SPARSE_METHOD else DENSE_SIZES[0]
    dataloader_train = get_nmnist_dataset(ROOT_PATH_DATA, SPARSE_METHOD, SEQ_LEN, INP_DIM, BATCHSIZE, sparse_size=SPARSE_SIZES[0], dims=IMAGE_DIMS)

    NUM_LAYERS = len(DENSE_SIZES)-1

    # # loss_fns = [reg_func_single_layer for _ in range(num_layers-1)]
    # # loss_fns.append(mse_softmax_loss_fn)
    # # loss_fns = {f"rnn_{i}" if i>0 else "rnn" : reg_func_single_layer for i in range(num_layers-1)}
    # loss_fns = {f"rnn_{i}" if i>0 else "rnn" : lambda x, y: 0.0 for i in range(num_layers-1)}
    # loss_fns[f"rnn_{num_layers-1}"] = mse_softmax_loss_fn

    # # metrics = [calc_activity for _ in range(num_layers-1)]
    # # metrics.append(calc_accuracy)
    # metrics = {f"rnn_{i}" if i>0 else "rnn": calc_activity for i in range(num_layers-1)}
    # metrics[f"rnn_{num_layers-1}"] = calc_accuracy

    if LOG_FILE is not None:
        csv_logger = keras.callbacks.CSVLogger(LOG_FILE, append=True, separator=';')
        callbacks = [csv_logger]
    else:
        callbacks = None

    print(dataloader_train)
    print(next(iter(dataloader_train)))

    # sys.exit()

    # train_ipu(
    #     num_epochs, 
    #     train_steps_per_execution, 
    #     batchsize_per_step,
    #     dataset_sparse,
    #     seq_len, 
    #     dense_sizes, 
    #     sparse_sizes, 
    #     decay_constant, 
    #     threshold,
    #     mse_softmax_loss_fn,
    #     metrics=calc_accuracy,
    #     steps_per_epoch=steps_per_epoch,
    #     callbacks=callbacks,
    # )



    if USE_IPU:

        method_to_loss_fn = {
            "dense": sum_and_sparse_categorical_crossentropy,
            "sparse_ops": get_sum_and_sparse_categorical_crossentropy_sparse_out(DENSE_SIZES[-1], transpose=False),
            "sparse_layer": get_sum_and_sparse_categorical_crossentropy_sparse_out(DENSE_SIZES[-1], transpose=True),
        }

        train_ipu(
            IMPL_METHOD,
            NUM_EPOCHS, 
            TRAIN_STEPS_PER_EXECUTION, 
            BATCHSIZE_PER_STEP,
            dataloader_train,
            SEQ_LEN, 
            DENSE_SIZES, 
            SPARSE_SIZES, 
            DECAY_CONSTANT, 
            THRESHOLD,
            method_to_loss_fn[IMPL_METHOD],
            metrics=None, #[calc_accuracy],
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            return_all=False
        )
    else:
        train_gpu(
            NUM_EPOCHS, 
            TRAIN_STEPS_PER_EXECUTION, 
            BATCHSIZE_PER_STEP,
            dataloader_train,
            SEQ_LEN, 
            DENSE_SIZES, 
            DECAY_CONSTANT, 
            THRESHOLD,
            sum_and_sparse_categorical_crossentropy,
            metrics=None, #calc_accuracy,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            return_all=False
        )


    # dataset_dense = get_nmnist_dataset_dense(batchsize, batchsize*steps_per_epoch)
    # train_gpu(
    #     num_epochs, 
    #     train_steps_per_execution, 
    #     batchsize,
    #     dataset_dense,
    #     seq_len, 
    #     dense_sizes, 
    #     decay_constant, 
    #     threshold,
    #     loss_fns,
    #     metrics=metrics,
    #     steps_per_epoch=steps_per_epoch,
    #     callbacks=callbacks,
    #     return_all=True,
    # )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Randman training optionally using the IPU and sparse implementations.")
    parser.add_argument('--use_ipu', type=int, default=1, help="Whether to use the IPU (default is `1` therefore `True`).")
    parser.add_argument('--impl_method', type=str, default=1, help="Implementation method to use, one of ['dense', 'sparse_ops', 'sparse_layer']."
                                                                    "Only used for `use_ipu=1`")
    parser.add_argument('--profile_run', type=int, default=0, help="Whether this is a profiling run (default is `0` therefore `Flase`), "
                                                                    "which uses shorter squence length, less data and only one epoch.")

    args = parser.parse_args()
    main(args)
