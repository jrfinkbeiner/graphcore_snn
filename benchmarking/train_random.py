import os
import sys
import functools as ft
import numpy as np

from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, gen_sparse_spikes, sparse2dense, create_dataset_dense
from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, create_dataset_sparse, sparse2dense_ipu

import tensorflow as tf
import tensorflow.keras as keras 


def get_dataloaders(rng, seq_len, num_inp_neurons, num_classes, num_total_samples, num_train_samples, batchsize, sparse, sparse_size):
    if sparse:
        assert sparse_size is not None, "If `sparse=True` the variable `sparse_size` must be set."

    # data, labels = make_spike_raster_dataset(rng, nb_classes=num_classes, nb_units=num_inp_neurons, nb_steps=seq_len, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=num_samples_per_class, alpha=2.0, shuffle=True)
    sparse_spike_ids, num_sparse_spikes = gen_sparse_spikes(rng, seq_len, num_total_samples, num_inp_neurons, sparse_size)
    sparse_spike_ids = sparse_spike_ids.transpose((1,0,2)) 
    num_sparse_spikes = num_sparse_spikes.transpose((1,0,2))
    labels = rng.integers(0, num_classes, size=(num_total_samples,)).astype(np.int32)
    
    labels_train,  labels_test  = labels[:num_train_samples], labels[num_train_samples:]

    if sparse:
        sparse_spike_ids_train, sparse_spike_ids_test = sparse_spike_ids[:num_train_samples], sparse_spike_ids[num_train_samples:]
        num_sparse_spikes_train, num_sparse_spikes_test = num_sparse_spikes[:num_train_samples], num_sparse_spikes[num_train_samples:]
        
        dataloader_train = create_dataset_sparse(sparse_spike_ids_train, num_sparse_spikes_train, labels_train, batchsize, shuffle=True) 
        dataloader_test = create_dataset_sparse(sparse_spike_ids_test, num_sparse_spikes_test, labels_test, batchsize, shuffle=False)
    else:
        dense_spike_ids = sparse2dense(sparse_spike_ids, num_sparse_spikes, num_inp_neurons)
        dense_spike_ids_train, dense_spike_ids_test = dense_spike_ids[:num_train_samples], dense_spike_ids[num_train_samples:]

        dataloader_train = create_dataset_dense(dense_spike_ids_train, labels_train, batchsize, shuffle=True) 
        dataloader_test = create_dataset_dense(dense_spike_ids_test, labels_test, batchsize, shuffle=False)

    return dataloader_train, dataloader_test

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)

# @tf.function(experimental_compile=True)
def calc_activity(y_true, y_pred):
    sum_spikes = tf.reduce_mean(y_pred)*y_pred.shape[1]
    return sum_spikes


# @tf.function(experimental_compile=True)
def calc_sparse_categorical_accuracy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1)
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, sum_spikes)


def get_sparse_func(func, out_dim, transpose=False):
    def sparse_fn(y_true, y_pred):
        y_pred_dense = sparse2dense_ipu(y_pred, out_dim)
        if transpose:
            y_pred_dense = tf.transpose(y_pred_dense, perm=[1, 0, 2])
        return func(y_true, y_pred_dense)
    sparse_fn.__name__ = func.__name__ + "_sparsified"
    return sparse_fn

def filter_layer_output(func, layer_id):
    def filter_fn(y_true, y_pred):
        print(func.__name__)
        print(y_pred)
        return func(y_true, y_pred[layer_id])
    filter_fn.__name__ = func.__name__ + f"_lay{layer_id}"
    return filter_fn

def main(args):

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    PROFILE_RUN = bool(args.profile_run)
    USE_IPU = bool(args.use_ipu)
    IMPL_METHOD = args.impl_method
    CALC_ACTIVITY = False
    TRANSPOSE_WEIGHTS = bool(args.transpose_weights)

    if USE_IPU:
        assert IMPL_METHOD is not None, "If `USE_IPU=True` the variable `IMPL_METHOD` must be set."
        assert IMPL_METHOD in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{IMPL_METHOD}'."
    SPARSE_METHOD = ("sparse" in IMPL_METHOD) and USE_IPU

    NUM_CLASSES = 8
    if PROFILE_RUN:
        NUM_EPOCHS = 1
        SEQ_LEN = 10
    else:
        NUM_EPOCHS = 5
        SEQ_LEN = 100

    # DENSE_SIZES = [2*34*34, 1024, 512, 128, NUM_CLASSES]
    # SPARSE_SIZES = [32, 48, 32, 16, 8]

    DENSE_SIZES = [2*34*34, 1024, 1024, 512, 128, NUM_CLASSES]
    SPARSE_SIZES = [32, 32, 32, 16, 8, 4]

    # DENSE_SIZES = [1024, 1024, 1024, NUM_CLASSES]
    # SPARSE_SIZES = [32, 32, 32, 8]

    # DENSE_SIZES = [8, 8 , NUM_CLASSES]
    # SPARSE_SIZES = [4, 4, 4]



    BATCHSIZE = 48
    if PROFILE_RUN:
        NUM_SAMPLES = BATCHSIZE*5
        NUM_SAMPLES_TRAIN = BATCHSIZE*4
    else:
        NUM_SAMPLES = BATCHSIZE*101
        NUM_SAMPLES_TRAIN = BATCHSIZE*100

    print("DENSE_SIZES: ", DENSE_SIZES)
    print("SPARSE_SIZES: ", SPARSE_SIZES)
    print("BATCHSIZE: ", BATCHSIZE)
    print("NUM_SAMPLES_TRAIN: ", NUM_SAMPLES_TRAIN)
    print("SEQ_LEN: ", SEQ_LEN)
    print()
    print("PROFILE_RUN: ", PROFILE_RUN)
    print("USE_IPU: ", USE_IPU)
    print("IMPL_METHOD: ", IMPL_METHOD)
    print("TRANSPOSE_WEIGHTS: ", TRANSPOSE_WEIGHTS)

    rng = np.random.default_rng(42)

    BATCHSIZE_PER_STEP = BATCHSIZE
    STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE)
    TRAIN_STEPS_PER_EXECUTION = STEPS_PER_EPOCH

    DECAY_CONSTANT = 0.9
    THRESHOLD = 1.0

    LOG_FILE = None # f"log_sparse_bs{BATCHSIZE}.csv"

    dataloader_train, dataloader_test = get_dataloaders(rng, SEQ_LEN, DENSE_SIZES[0], NUM_CLASSES, 
                    NUM_SAMPLES, NUM_SAMPLES_TRAIN, BATCHSIZE_PER_STEP if USE_IPU else BATCHSIZE, 
                    sparse=SPARSE_METHOD, sparse_size=SPARSE_SIZES[0])

    NUM_LAYERS = len(DENSE_SIZES)-1
    if LOG_FILE is not None:
        csv_logger = keras.callbacks.CSVLogger(LOG_FILE, append=True, separator=';')
        callbacks = [csv_logger]
    else:
        callbacks = None


    if USE_IPU:

        method_to_loss_fn = {
            "dense": sum_and_sparse_categorical_crossentropy,
            "sparse_ops": get_sparse_func(sum_and_sparse_categorical_crossentropy, DENSE_SIZES[-1], transpose=False),
            "sparse_layer": get_sparse_func(sum_and_sparse_categorical_crossentropy, DENSE_SIZES[-1], transpose=True),
        }

        method_to_metr_fn_to_last = {
            "dense": calc_sparse_categorical_accuracy,
            "sparse_ops": get_sparse_func(calc_sparse_categorical_accuracy, DENSE_SIZES[-1], transpose=False),
            "sparse_layer": get_sparse_func(calc_sparse_categorical_accuracy, DENSE_SIZES[-1], transpose=True),
        }
        method_to_calc_activity = {
            "dense": calc_activity,
            "sparse_ops": ft.partial(get_sparse_func, calc_activity, transpose=False),
            "sparse_layer": ft.partial(get_sparse_func, calc_activity, transpose=True),
        }


        loss_fn = method_to_loss_fn[IMPL_METHOD] if not CALC_ACTIVITY else filter_layer_output(method_to_loss_fn[IMPL_METHOD], NUM_LAYERS-1)

        if CALC_ACTIVITY:
            metrics = [filter_layer_output(method_to_metr_fn_to_last[IMPL_METHOD], NUM_LAYERS-1)]
            for i in range(NUM_LAYERS):
                met = method_to_calc_activity[IMPL_METHOD]
                if SPARSE_METHOD:
                    met = met(DENSE_SIZES[i+1])
                metrics.append(filter_layer_output(met, i))
        else:
            metrics = [method_to_metr_fn_to_last[IMPL_METHOD]]

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
            loss_fn,
            metrics=metrics,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            return_all=True if CALC_ACTIVITY else False,
            transpose_weights=TRANSPOSE_WEIGHTS,
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
            metrics=[calc_sparse_categorical_accuracy],
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            return_all=False
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Randman training optionally using the IPU and sparse implementations.")
    parser.add_argument('--use_ipu', type=int, default=1, help="Whether to use the IPU (default is `1` therefore `True`).")
    parser.add_argument('--impl_method', type=str, default=1, help="Implementation method to use, one of ['dense', 'sparse_ops', 'sparse_layer']."
                                                                    "Only used for `use_ipu=1`")
    parser.add_argument('--profile_run', type=int, default=0, help="Whether this is a profiling run (default is `0` therefore `Flase`), "
                                                                    "which uses shorter squence length, less data and only one epoch.")
    parser.add_argument('--transpose_weights', type=int, default=0, help="Whether to use the transposed weight matrix to better make use of vectorization."
                                                                        " For now only used with `impl_method=sparse_layer`. Default is 0 (False).")

    args = parser.parse_args()
    main(args)
