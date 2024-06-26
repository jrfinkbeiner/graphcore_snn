import os
import sys
import functools as ft
import numpy as np

from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, sparse2dense, create_dataset_dense
from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, create_dataset_sparse, sparse2dense_ipu

import tensorflow as tf
import tensorflow.keras as keras 

from randman import make_spike_raster_dataset, convert_raster_to_sparse_spikes


def get_dataloaders(rng, seq_len, num_inp_neurons, num_classes, num_samples_per_class, num_train_samples, batchsize, sparse=True, sparse_size=None):
    if sparse:
        assert sparse_size is not None, "If `sparse=True` the variable `sparse_size` must be set."

    # data, labels = make_spike_raster_dataset(rng, nb_classes=num_classes, nb_units=num_inp_neurons, nb_steps=seq_len, step_frac=1.0, dim_manifold=2, nb_spikes=1, nb_samples=num_samples_per_class, alpha=2.0, shuffle=True)
    data, labels = make_spike_raster_dataset(rng, nb_classes=num_classes, nb_units=num_inp_neurons, nb_steps=seq_len, step_frac=1.0, dim_manifold=1, nb_spikes=1, nb_samples=num_samples_per_class, alpha=2.0, shuffle=True)

    data_train, labels_train = data[:num_train_samples], labels[:num_train_samples]
    data_test,  labels_test  = data[num_train_samples:], labels[num_train_samples:]

    if sparse:
        data_train = convert_raster_to_sparse_spikes(rng, data_train, sparse_size)
        data_test = convert_raster_to_sparse_spikes(rng, data_test, sparse_size)

        dataloader_train = create_dataset_sparse(data_train[0], data_train[1], labels_train, batchsize, shuffle=True) 
        dataloader_test = create_dataset_sparse(data_test[0], data_test[1], labels_test, batchsize, shuffle=False)
    else:
        dataloader_train = create_dataset_dense(data_train, labels_train, batchsize, shuffle=True) 
        dataloader_test = create_dataset_dense(data_test, labels_test, batchsize, shuffle=False)

    return dataloader_train, dataloader_test

    # print(f"{data_sparse[0]=}")
    # print(f"{data_sparse[1]=}")
    # print(f"{np.max(data_sparse[1])=}")
    # data_dense_reconstructed = sparse2dense(data_sparse[0], data_sparse[1], dense_size, values=None, sparse_dim=-1)
    # print(np.all(data_dense==data_dense_reconstructed))


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
#     sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
#     softmax_pred = tf.nn.softmax(sum_spikes, axis=1)
#     one_hot_target = tf.one_hot(y_target, softmax_pred.shape[-1], axis=-1, dtype=softmax_pred.dtype)
#     return tf.math.reduce_sum((softmax_pred-one_hot_target)**2)/y_target.shape[-1]
#     # return tf.math.reduce_max(softmax_pred)
#     # return tf.math.reduce_sum(sum_spikes)
#     # return tf.math.reduce_max((softmax_pred-one_hot_target)**2)/y_pred.shape[-1]

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)


# def get_sum_and_sparse_categorical_crossentropy_sparse_out(out_dim, transpose=False):
#     def loss_fn(y_true, y_pred):
#         y_pred_dense = sparse2dense_ipu(y_pred, out_dim)
#         if transpose:
#             y_pred_dense = tf.transpose(y_pred_dense, perm=[1, 0, 2])
#         return sum_and_sparse_categorical_crossentropy(y_true, y_pred_dense)
#     return loss_fn

# def mse_softmax_loss_with_reg_fn(y_target, y_pred):

#     loss_class = mse_softmax_loss_fn(y_target, y_pred[-1])
#     loss_reg = reg_func(y_target)
    
#     return loss_class + 0.1 * loss_reg

# @tf.function(experimental_compile=True)
# def calc_accuracy(y_true, y_pred):
#     sum_spikes = tf.reduce_sum(y_pred, axis=1)
#     preds = tf.math.argmax(sum_spikes, axis=1)
#     # preds = tf.ones_like(y_true)
#     y_true = tf.cast(y_true, preds.dtype)
#     identical = tf.cast(preds==y_true, tf.float32)
#     return tf.math.reduce_mean(identical)

# @tf.function(experimental_compile=True)
def calc_activity(y_true, y_pred):
    sum_spikes = tf.reduce_mean(y_pred)*y_pred.shape[1]
    return sum_spikes


# @tf.function(experimental_compile=True)
def calc_sparse_categorical_accuracy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1)
    # preds = tf.math.argmax(sum_spikes, axis=1)
    # # preds = tf.ones_like(y_true)
    # y_true = tf.cast(y_true, preds.dtype)
    # identical = tf.cast(preds==y_true, tf.float32)
    # return tf.math.reduce_mean(identical)
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
        return func(y_true, y_pred[layer_id])
    filter_fn.__name__ = func.__name__ + f"_lay{layer_id}"
    return filter_fn

def main(args):

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    PROFILE_RUN = bool(args.profile_run)
    USE_IPU = bool(args.use_ipu)
    IMPL_METHOD = args.impl_method
    SPARSE_MULTIPLIER = args.sparse_multiplier
    LEARNING_RATE = args.lr
    CALC_ACTIVITY = True
    if USE_IPU:
        assert IMPL_METHOD is not None, "If `USE_IPU=True` the variable `IMPL_METHOD` must be set."
        assert IMPL_METHOD in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{IMPL_METHOD}'."
    SPARSE_METHOD = ("sparse" in IMPL_METHOD) and USE_IPU

    NUM_CLASSES = 8
    if PROFILE_RUN:
        NUM_EPOCHS = 1
        SEQ_LEN = 10
    else:
        NUM_EPOCHS = 50
        SEQ_LEN = 100

    DENSE_SIZES = [128, 512, 128, NUM_CLASSES]
    SPARSE_SIZES_BASE = [1, 2, 2, 1]
    SPARSE_SIZES = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE, DENSE_SIZES)]
    # SPARSE_SIZES = DENSE_SIZES # [20, 100, 10]
    # SPARSE_SIZES= DENSE_SIZES
    # DENSE_SIZES = [1024, 1024, 1024, 1024, 1024, 1024, NUM_CLASSES]
    # SPARSE_SIZES = [32,   32,   32,   32,   32,   32,   8]
    # SPARSE_SIZES = [64,   64,   64,   64,   64,   64,   8]
    # SPARSE_SIZES = [8,   64,   64,   8]
    
    IMAGE_DIMS = (34, 34, 2)
    DENSE_SIZES = [np.prod(IMAGE_DIMS), 1024, 1024, 512, 512, 128, NUM_CLASSES]
    SPARSE_SIZES_BASE = [4, 4, 4, 2, 2, 1, 1]
    SPARSE_SIZES = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE, DENSE_SIZES)]
    # SPARSE_SIZES = SPARSE_SIZES + [min(int(SPARSE_SIZES_BASE[-1]*SPARSE_MULTIPLIER), 8)]
    
    SPARSE_MULTIPLIER = 1
    NUM_CLASSES = 10
    DENSE_SIZES = [128, 512, 512, 512, 128, NUM_CLASSES]
    DENSE_SIZES = DENSE_SIZES[:1] + [int(0.5*d) for d in DENSE_SIZES[1:-1]] + DENSE_SIZES[-1:]
    SPARSE_SIZES_BASE = [64, 16, 16, 16, 8, 10]
    SPARSE_SIZES = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE, DENSE_SIZES)]

    BATCHSIZE = 48
    if PROFILE_RUN:
        NUM_SAMPLES_PER_CLASS = BATCHSIZE
        TRAIN_TEST_SPILT = 0.5
    else:
        NUM_SAMPLES_PER_CLASS = 1000
        TRAIN_TEST_SPILT = 0.8
    NUM_SAMPLES_TRAIN = int(NUM_CLASSES*NUM_SAMPLES_PER_CLASS*TRAIN_TEST_SPILT)

    print("IMPL_METHOD: ", IMPL_METHOD)
    print("DENSE_SIZES: ", DENSE_SIZES)
    print("SPARSE_SIZES: ", SPARSE_SIZES)
    print("BATCHSIZE: ", BATCHSIZE)
    print("NUM_SAMPLES_TRAIN: ", NUM_SAMPLES_TRAIN)
    print("SEQ_LEN: ", SEQ_LEN)
    print("SPARSE_MULTIPLIER: ", SPARSE_MULTIPLIER)
    print("LEARNING_RATE: ", LEARNING_RATE)

    rng = np.random.default_rng(42)

    BATCHSIZE_PER_STEP = BATCHSIZE
    STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE/8)
    TRAIN_STEPS_PER_EXECUTION = STEPS_PER_EPOCH

    DECAY_CONSTANT = 0.92
    THRESHOLD = 1.0

    LOG_FILE = f"improve_convergence_{int(DENSE_SIZES[1])}_large/{IMPL_METHOD}_randomIndOffset_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv"
    # LOG_FILE = f"improve_convergence_{int(DENSE_SIZES[1])}_large/{IMPL_METHOD}_randomIndOffset_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv"
    # LOG_FILE = f"convergence_sparsity_sweep_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_topK_sparse_multiplier_{SPARSE_MULTIPLIER}.csv"
    # LOG_FILE = f"convergence_learning_rate_sweep_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_topK_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv"
    # LOG_FILE = None

    dataloader_train, dataloader_test = get_dataloaders(rng, SEQ_LEN, DENSE_SIZES[0], NUM_CLASSES, 
                    NUM_SAMPLES_PER_CLASS, NUM_SAMPLES_TRAIN, BATCHSIZE_PER_STEP if USE_IPU else BATCHSIZE, 
                    sparse=SPARSE_METHOD, sparse_size=SPARSE_SIZES[0])

    NUM_LAYERS = len(DENSE_SIZES)-1
    # # loss_fns = [reg_func_single_layer for _ in range(num_layers-1)]
    # # loss_fns.append(mse_softmax_loss_fn)
    # # loss_fns = {f"rnn_{i}" if i>0 else "rnn" : reg_func_single_layer for i in range(num_layers-1)}
    # loss_fns = {f"rnn_{i}" if i>0 else "rnn" : lambda x, y: 0.0 for i in range(NUM_LAYERS-1)}
    # loss_fns[f"rnn_{NUM_LAYERS-1}"] = mse_softmax_loss_fn

    # # metrics = [calc_activity for _ in range(num_layers-1)]
    # # metrics.append(calc_accuracy)
    # metrics = {f"rnn_{i}" if i>0 else "rnn": calc_activity for i in range(NUM_LAYERS-1)}
    # metrics[f"rnn_{NUM_LAYERS-1}"] = calc_accuracy



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



    multi_grad_scale_fac = [
        # [dense/sparse for dense,sparse in zip(DENSE_SIZES[1:], SPARSE_SIZES[1:])],
        # [32.0, 24.0, 16.0, 8.0, 1.0],
        None
    ]
    LOG_FILES = [
        # f"improve_convergence_randman_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_randomIndOffset_calcGradScale_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv",
        # f"improve_convergence_randman_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_randomIndOffset_hardcodedGradScale_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv",
        # f"improve_convergence_randman_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_noRandomIndOffset_noneGradScale_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv"
        f"improve_convergence_randman_{int(DENSE_SIZES[1])}/{IMPL_METHOD}_sparseMul{SPARSE_MULTIPLIER}_lr{LEARNING_RATE:.0e}.csv"
    ]



    # NOTE grad_scale_facs = [24.0, 16.0, 12.0, 6.0, 1.0] worked well for:
    # SPARSE_MULTIPLIER = 1
    # NUM_CLASSES = 4
    # DENSE_SIZES = [64, 128, 128, 128, 64, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [64, 16, 16, 16, 8, 4]

    # if SPARSE_METHOD:
    #     # grad_scale_facs = [dense/sparse for dense,sparse in zip(DENSE_SIZES[1:], SPARSE_SIZES[1:])]
    #     grad_scale_facs = [32.0, 24.0, 16.0, 8.0, 1.0]
    # else:
    #     grad_scale_facs = None
    
    for LOG_FILE,grad_scale_facs in zip(LOG_FILES, multi_grad_scale_fac):
            
        if LOG_FILE is not None:
            csv_logger = keras.callbacks.CSVLogger(LOG_FILE, append=True, separator=';')
            callbacks = [csv_logger]
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        else:
            callbacks = None


        # grad_scale_facs = multi_grad_scale_fac[1]
        print("\ngrad_scale_facs")
        print(grad_scale_facs)

        if USE_IPU:
            train_ipu(
                IMPL_METHOD,
                NUM_EPOCHS, 
                TRAIN_STEPS_PER_EXECUTION, 
                BATCHSIZE_PER_STEP,
                dataloader_train.repeat(),
                SEQ_LEN, 
                DENSE_SIZES, 
                SPARSE_SIZES, 
                DECAY_CONSTANT, 
                THRESHOLD,
                loss_fn,
                metrics=metrics, #[calc_accuracy],
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=callbacks,
                return_all=True if CALC_ACTIVITY else False,
                transpose_weights=bool(args.transpose_weights),
                learning_rate=LEARNING_RATE,
                seed=44,
                grad_scale_facs=grad_scale_facs,
            )
        else:
            train_gpu(
                NUM_EPOCHS, 
                TRAIN_STEPS_PER_EXECUTION, 
                BATCHSIZE_PER_STEP,
                dataloader_train.repeat(),
                SEQ_LEN, 
                DENSE_SIZES, 
                DECAY_CONSTANT, 
                THRESHOLD,
                loss_fn,
                metrics=metrics, #calc_accuracy,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=callbacks,
                return_all=True if CALC_ACTIVITY else False,
                learning_rate=LEARNING_RATE,
                seed=44,
            )
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
    parser.add_argument('--transpose_weights', type=int, default=0, help="Whether to use the transposed weight matrix to better make use of vectorization."
                                                                        " For now only used with `impl_method=sparse_layer`. Default is 0 (False).")
    parser.add_argument('--sparse_multiplier', type=int, default=8, help="Factor to multiply sparse sizes with, default is 16.")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for optimizer, default `1e-2`.")

    args = parser.parse_args()
    main(args)
