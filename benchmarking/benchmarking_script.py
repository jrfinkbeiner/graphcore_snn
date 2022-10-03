from genericpath import isfile
import os
import sys
import functools as ft
import numpy as np

from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, sparse2dense, create_dataset_dense, TimingCallback

from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, sparse2dense_ipu, create_dataset_sparse
from multi_ipu import train_mutli_ipu_benchmarking, create_dataset_sparse_multi_ipu

# from keras_train_util import train_gpu, simple_loss_fn_dense, train_gpu, sparse2dense, create_dataset_dense
# from keras_train_util_ipu import train_ipu, simple_loss_fn_sparse, create_dataset_sparse, sparse2dense_ipu


import tensorflow as tf
import tensorflow.keras as keras 
# from keras.callbacks import CSVLogger

from nmnist_util import get_nmnist_dataset, create_nmnist_gener, get_nmnist_keras_dataset, load_dataset_to_tensor_dict

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




# # def mse_softmax_loss_fn(y_target, y_pred):
# #     print(y_target.shape)
# #     print(y_pred.shape)
# #     sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
# #     print(sum_spikes.shape)
# #     softmax_pred = tf.nn.softmax(sum_spikes, axis=1)
# #     one_hot_target = tf.one_hot(y_target, softmax_pred.shape[-1], axis=-1, dtype=softmax_pred.dtype)
# #     return tf.math.reduce_sum((softmax_pred-one_hot_target)**2)/y_target.shape[-1]
# #     # return tf.math.reduce_max(softmax_pred)
# #     # return tf.math.reduce_sum(sum_spikes)
# #     # return tf.math.reduce_max((softmax_pred-one_hot_target)**2)/y_pred.shape[-1]

# # # def mse_softmax_loss_with_reg_fn(y_target, y_pred):

# # #     loss_class = mse_softmax_loss_fn(y_target, y_pred[-1])
# # #     loss_reg = reg_func(y_target)
    
# # #     return loss_class + 0.1 * loss_reg

# # @tf.function(experimental_compile=True)
# # def calc_accuracy(y_true, y_pred):
# #     sum_spikes = tf.reduce_sum(y_pred, axis=1)
# #     preds = tf.math.argmax(sum_spikes, axis=1)
# #     # preds = tf.ones_like(y_true)
# #     y_true = tf.cast(y_true, preds.dtype)
# #     identical = tf.cast(preds==y_true, tf.float32)
# #     return tf.math.reduce_mean(identical)

# # @tf.function(experimental_compile=True)
# # def calc_activity(y_true, y_pred):
# #     sum_spikes = tf.reduce_mean(y_pred)*y_pred.shape[1]
# #     return sum_spikes

# def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
#     sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
#     return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)


def get_sum_and_sparse_categorical_crossentropy_sparse_out(out_dim, transpose=False):
    def loss_fn(y_true, y_pred):
        y_pred_dense = sparse2dense_ipu(y_pred, out_dim)
        if transpose:
            y_pred_dense = tf.transpose(y_pred_dense, perm=[1, 0, 2])
        return sum_and_sparse_categorical_crossentropy(y_true, y_pred_dense)
    return loss_fn

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)

def calc_activity(y_true, y_pred):
    sum_spikes = tf.reduce_mean(y_pred) # *y_pred.shape[1]
    return sum_spikes

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
        return func(y_true, y_pred[layer_id])
    filter_fn.__name__ = func.__name__ + f"_lay{layer_id}"
    return filter_fn

# def create_dataset_sparse(inp_spike_ids, num_inp_spikes, labels, batchsize, shuffle=True):
#     dataset = tf.data.Dataset.from_tensor_slices({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes, "targets": labels})
#     num_samples = labels.shape[0]
#     if shuffle:
#         dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
#     # dataset = dataset.repeat()
#     # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(batchsize, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     # dataset = dataset.prefetch(4)
#     return dataset




def main(args, ROOT_PATH_DATA):

    PROFILE_RUN = bool(args.profile_run)
    USE_IPU = bool(args.use_ipu)
    IMPL_METHOD = args.impl_method
    SPARSE_MULTIPLIER = args.sparse_multiplier
    CALC_ACTIVITY = False
    MULTIPROCESSING = True
    TRANSPOSE_WEIGHTS = bool(args.transpose_weights)
    BATCHSIZE = args.batchsize
    LEARNING_RATE = args.lr
    NUM_IPUS = args.num_ipus
    SECOND_THRESHOLD = args.second_thresh
    WEIGHT_MUL = args.weight_mul
    DATASET_NAME = args.dataset_name
    BENCH_MODE = args.bench_mode
    MAX_ACTIVITY = args.max_activity
    NUM_HIDDEN_LAYERS = args.num_hidden_layers
    SPARSE_SIZE_INP = args.sparse_size_inp
    IPU_ID = args.ipu_id if args.ipu_id >= 0 else None 
    NUM_NEURONS_PER_TILE = args.num_neurons_per_tile

    if USE_IPU:
        assert IMPL_METHOD is not None, "If `USE_IPU=True` the variable `IMPL_METHOD` must be set."
        assert IMPL_METHOD in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{IMPL_METHOD}'."
    SPARSE_METHOD = ("sparse" in IMPL_METHOD) and USE_IPU


    DATASET_NUM_CLASSES = {
        "NMNIST": 10,
        "DVSGesture": 11,
        "SHD": 20,
    }
    NUM_CLASSES = DATASET_NUM_CLASSES[DATASET_NAME]
    if PROFILE_RUN:
        NUM_EPOCHS = 1
        SEQ_LEN = 100
    else:
        NUM_EPOCHS = 21
        if BENCH_MODE != "multi_neuron":
            SEQ_LEN_DATASET = {
                "NMNIST": 100,
                "DVSGesture": 50,
                "SHD": 100,
            }
            SEQ_LEN = SEQ_LEN_DATASET[DATASET_NAME] # 300 for DVSGesture up to 97.5% acc
        else:
            SEQ_LEN = 10


    DATASET_TO_IMAGE_DIMS = {
        # "NMNIST": (34, 34, 2),
        "NMNIST": (24, 24, 2),
        # "DVSGesture": (64, 64, 2),
        "DVSGesture": (48, 48, 2),
        "SHD": (700, 1, 1),
    }
    IMAGE_DIMS = DATASET_TO_IMAGE_DIMS[DATASET_NAME]


    NEURON_TO_SPLIT = 1471*NUM_NEURONS_PER_TILE
    HIDDEN_LAYER_DENSE_SIZES = [int(NUM_NEURONS_PER_TILE*(NEURON_TO_SPLIT / NUM_HIDDEN_LAYERS // NUM_NEURONS_PER_TILE + (((NEURON_TO_SPLIT % int(NUM_HIDDEN_LAYERS*NUM_NEURONS_PER_TILE))) > NUM_NEURONS_PER_TILE*ilay))) for ilay in range(NUM_HIDDEN_LAYERS)]
    HIDDEN_LAYER_SPARSE_SIZES = [int((MAX_ACTIVITY*hid_dense_size//2)*2) for hid_dense_size in HIDDEN_LAYER_DENSE_SIZES]

    NEURON_TO_SPLIT = 1471*NUM_NEURONS_PER_TILE - ((NUM_CLASSES // NUM_NEURONS_PER_TILE + ((NUM_CLASSES % NUM_NEURONS_PER_TILE) > 0))* NUM_NEURONS_PER_TILE)
    HIDDEN_LAYER_DENSE_SIZES_LAST = [int(NUM_NEURONS_PER_TILE*(NEURON_TO_SPLIT / NUM_HIDDEN_LAYERS // NUM_NEURONS_PER_TILE + (((NEURON_TO_SPLIT % int(NUM_HIDDEN_LAYERS*NUM_NEURONS_PER_TILE))) > NUM_NEURONS_PER_TILE*ilay))) for ilay in range(NUM_HIDDEN_LAYERS)]
    HIDDEN_LAYER_SPARSE_SIZES_LAST = [int((MAX_ACTIVITY*hid_dense_size//2)*2) for hid_dense_size in HIDDEN_LAYER_DENSE_SIZES]


    # # benchmarking presentation
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), *HIDDEN_LAYER_DENSE_SIZES, NUM_CLASSES]
    # SPARSE_SIZES = [32, *HIDDEN_LAYER_SPARSE_SIZES, int(max((MAX_ACTIVITY*NUM_CLASSES//2)*2, 2))]


    DENSE_SIZES = [np.prod(IMAGE_DIMS), *HIDDEN_LAYER_DENSE_SIZES*(NUM_IPUS-1), *HIDDEN_LAYER_DENSE_SIZES_LAST, NUM_CLASSES]
    if BENCH_MODE in ["multi_layer", "multi_neuron"]:
        SPARSE_SIZES = [SPARSE_SIZE_INP, *HIDDEN_LAYER_SPARSE_SIZES*(NUM_IPUS-1), *HIDDEN_LAYER_SPARSE_SIZES_LAST, int(max((MAX_ACTIVITY*NUM_CLASSES//2)*2, 2))]
    else:
        SPARSE_SIZES_BASE = [64, 4, *[4]*(2*(NUM_IPUS-1)), 4, 10]
        SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]

    # if DATASET_NAME=="NMNIST":
    #     # benchmarking presentation
    #     # DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, *[1472]*(2*(NUM_IPUS-1)), 1076+384, NUM_CLASSES]
    #     # SPARSE_SIZES_BASE = [64, 4, *[4]*(2*(NUM_IPUS-1)), 4, 10]

    #     DENSE_SIZES = [np.prod(IMAGE_DIMS), *HIDDEN_LAYER_DENSE_SIZES*(NUM_IPUS-1), *HIDDEN_LAYER_DENSE_SIZES_LAST, NUM_CLASSES]
    #     if BENCH_MODE == "multi_neuron":
    #         SPARSE_SIZES = [SPARSE_SIZE_INP, *HIDDEN_LAYER_SPARSE_SIZES*(NUM_IPUS-1), *HIDDEN_LAYER_SPARSE_SIZES_LAST, int(max((MAX_ACTIVITY*NUM_CLASSES//2)*2, 2))]
    #     else:
    #         SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # elif DATASET_NAME=="DVSGesture":
    #     # benchmarking presentation
    #     DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, *[1472]*(2*(NUM_IPUS-1)), 1076, 384, NUM_CLASSES]
    #     # DENSE_SIZES = [np.prod(IMAGE_DIMS), 980, 980, *[1472]*(3*(NUM_IPUS-1)), 980, NUM_CLASSES]
    #     SPARSE_SIZES_BASE = [SPARSE_SIZE_INP, 4, *[4]*(2*(NUM_IPUS-1)), 4, 4, 10]
    #     SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # elif DATASET_NAME=="SHD":
    #     # benchmarking presentation
    #     DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, *[1472]*(2*(NUM_IPUS-1)), 1076+384, NUM_CLASSES]
    #     SPARSE_SIZES_BASE = [SPARSE_SIZE_INP, 4, *[4]*(2*(NUM_IPUS-1)), 4, 10]
    #     SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # else:
    #     raise ValueError(f"Unknown dataset name, got '{DATASET_NAME}'")

    # # benchmarking presentation
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, 1470, 1468, 1076+384, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [64, 4, 4, 4, 4, 10]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]

    # # benchmarking presentation
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, 1470, 1470, 1470, 1470, 1076+384, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [64, 4, 4, 4, 4, 4, 4, 10]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]


    # # benchmarking presentation
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), 512, 128, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [32, 4, 2, 1, 10]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]

    # BATCHSIZE = 48
    if PROFILE_RUN:
        NUM_SAMPLES_TRAIN_DATASET = {
            "NMNIST": BATCHSIZE*16,
            "DVSGesture": BATCHSIZE*16,
            "SHD": BATCHSIZE*16,
        }
        # NUM_SAMPLES_TRAIN = BATCHSIZE*4
        NUM_SAMPLES_TRAIN = NUM_SAMPLES_TRAIN_DATASET[DATASET_NAME]
    else:
        NUM_SAMPLES_TRAIN_DATASET = {
            # "NMNIST": BATCHSIZE*26*2,
            "NMNIST": 9984,
            "DVSGesture": 1077,
            "SHD": 8156,
        }
        NUM_SAMPLES_TRAIN = NUM_SAMPLES_TRAIN_DATASET[DATASET_NAME]

    MAX_SAMPLES = {
        "NMNIST": 60000,
        "DVSGesture": 1077,
        "SHD": 8156,
    }
    assert NUM_SAMPLES_TRAIN <= MAX_SAMPLES[DATASET_NAME]

    print("#################################################################################################")
    print("SPARSE_MULTIPLIER: ", SPARSE_MULTIPLIER)
    print("DENSE_SIZES: ", DENSE_SIZES)
    print("SPARSE_SIZES: ", SPARSE_SIZES)
    print("BATCHSIZE: ", BATCHSIZE)
    print("NUM_SAMPLES_TRAIN: ", NUM_SAMPLES_TRAIN)
    print("SEQ_LEN: ", SEQ_LEN)
    print()
    print("DATASET_NAME: ", DATASET_NAME)
    print("PROFILE_RUN: ", PROFILE_RUN)
    print("USE_IPU: ", USE_IPU)
    print("IMPL_METHOD: ", IMPL_METHOD)
    print("TRANSPOSE_WEIGHTS: ", TRANSPOSE_WEIGHTS)
    print("LEARNING_RATE: ", LEARNING_RATE)
    print("NUM_IPUS: ", NUM_IPUS)
    print("SECOND_THRESHOLD: ", SECOND_THRESHOLD) 
    print("WEIGHT_MUL: ", WEIGHT_MUL) 
    print("BENCH_MODE: ", BENCH_MODE) 
    print("NUM_HIDDEN_LAYERS: ", NUM_HIDDEN_LAYERS) 
    print("NUM_NEURONS_PER_TILE: ", NUM_NEURONS_PER_TILE) 
    print("MAX_ACTIVITY: ", MAX_ACTIVITY)

    USE_MULTI_IPU = True # NUM_IPUS > 1 # TODO change back!
    if USE_MULTI_IPU:
        print("WARNING: behaviour might be different than expected due to variable overwrite in multi-ipu version.")

    rng = np.random.default_rng(42)

    BATCHSIZE_PER_STEP = BATCHSIZE
    # STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE/4) # TODO change back !
    STEPS_PER_EPOCH = 1000 if BENCH_MODE == "multi_neuron" else 200 # TODO change back !
    TRAIN_STEPS_PER_EXECUTION = int(STEPS_PER_EPOCH // 2)

    print(STEPS_PER_EPOCH)

    DECAY_CONSTANT = 0.95
    THRESHOLD = 1.0 
    THRESHOLD_FISRT_AND_SECOND = [THRESHOLD, [*[SECOND_THRESHOLD]*(len(SPARSE_SIZES)-1)]]
    # THRESHOLD_FISRT_AND_SECOND = [THRMAX_ACTIVITYRESHOLD]*(len(SPARSE_SIZES)-2), -100]]
    
    # BASE_FOLDER = f"final_bench_results_multi_ipu/{DATASET_NAME}_{BENCH_MODE}/"
    # BASE_FOLDER = f"final_bench_results_fixedAct/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"
    # if IPU_ID is None:
    #     BASE_FOLDER = f"final_bench_results_fixedAct/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"
    #     # BASE_FOLDER = f"final_bench_results_multi_ipu/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"
    # else:
    #     BASE_FOLDER = f"final_bench_results/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"
    BASE_FOLDER = f"final_bench_results/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"

    # BASE_FOLDER = f"failure_analysis/{DATASET_NAME}_{BENCH_MODE}_numSuperBatches2/"
    REL_FOLER_NAME = f"{DATASET_NAME}_{BENCH_MODE}_{IMPL_METHOD}_weightMul{WEIGHT_MUL}/"
    SPECIFIC_NAME = f"{DATASET_NAME}_{BENCH_MODE}_{IMPL_METHOD}_weightMul{WEIGHT_MUL}_numIPUs{NUM_IPUS}_sparseMul{SPARSE_MULTIPLIER}_numHid{NUM_HIDDEN_LAYERS}_hidLayerSize{HIDDEN_LAYER_DENSE_SIZES[0]}_maxAct{MAX_ACTIVITY}_sparseSizeInp{SPARSE_SIZE_INP}_numNeuonsPT{NUM_NEURONS_PER_TILE}_secondThresh{SECOND_THRESHOLD}_decayConst{DECAY_CONSTANT}_lr{LEARNING_RATE:.0e}_batchize{BATCHSIZE}_stepsPerEpoch{STEPS_PER_EPOCH}"
    LOG_FILE = None # BASE_FOLDER + REL_FOLER_NAME + "log_" + SPECIFIC_NAME + ".csv"
    TIMING_FILE = BASE_FOLDER + REL_FOLER_NAME + "timing_" + SPECIFIC_NAME

    REL_FOLER_NAME_2 = f"{DATASET_NAME}_{BENCH_MODE}_{IMPL_METHOD}_weightMul{WEIGHT_MUL}/"
    SPECIFIC_NAME_2 = f"{DATASET_NAME}_{BENCH_MODE}_{IMPL_METHOD}_weightMul{WEIGHT_MUL}_numIPUs{NUM_IPUS}_sparseMul{SPARSE_MULTIPLIER}_numHid{NUM_HIDDEN_LAYERS}_hidLayerSize{HIDDEN_LAYER_DENSE_SIZES[0]}_maxAct{MAX_ACTIVITY}_sparseSizeInp{SPARSE_SIZE_INP}_numNeuonsPT{NUM_NEURONS_PER_TILE}_secondThresh{SECOND_THRESHOLD}_decayConst{DECAY_CONSTANT}_lr{3e-3:.0e}_batchize{BATCHSIZE}_stepsPerEpoch{STEPS_PER_EPOCH}"
    LOG_FILE_2 = None # BASE_FOLDER + REL_FOLER_NAME + "log_" + SPECIFIC_NAME + ".csv"
    TIMING_FILE_2 = BASE_FOLDER + REL_FOLER_NAME_2 + "timing_" + SPECIFIC_NAME_2


    if os.path.isfile(TIMING_FILE+".npz") or os.path.isfile(TIMING_FILE_2+".npz"):
        print("\ntiming file already exists")
        sys.exit()
    # print("\ntiming file doesn't already exists")
    # sys.exit()
    INP_DIM = SPARSE_SIZES[0] if SPARSE_METHOD else DENSE_SIZES[0]
    ITER_BATCHISIZE_MULTI_PROC = {
        "NMNIST": 10000,
        "DVSGesture": 10000,
        "SHD": 10000,
    }

    data = load_dataset_to_tensor_dict(DATASET_NAME, ROOT_PATH_DATA, SPARSE_METHOD, SEQ_LEN, INP_DIM, num_samples=NUM_SAMPLES_TRAIN, iter_batchsize=min(ITER_BATCHISIZE_MULTI_PROC[DATASET_NAME], NUM_SAMPLES_TRAIN))
    if SPARSE_METHOD:
        if USE_MULTI_IPU:
            dataloader_train = create_dataset_sparse_multi_ipu(data["inp_spike_ids"], data["num_inp_spikes"], data["targets"], BATCHSIZE, shuffle=True)   
        else:
            dataloader_train = create_dataset_sparse(data["inp_spike_ids"], data["num_inp_spikes"], data["targets"], BATCHSIZE, shuffle=True) 
        
    else:
        dataloader_train = create_dataset_dense(
            tf.convert_to_tensor(data["inp_spikes"], dtype=data["inp_spikes"].dtype), 
            tf.convert_to_tensor(data["targets"], dtype=data["targets"].dtype), 
            BATCHSIZE, 
            shuffle=True) 

    callbacks = [TimingCallback(TIMING_FILE)]
    if LOG_FILE is not None:
        os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE)), exist_ok=True)
        csv_logger = keras.callbacks.CSVLogger(LOG_FILE, append=True, separator=';')
        callbacks.append(csv_logger)

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

    NUM_LAYERS = len(DENSE_SIZES)-1
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

    if USE_IPU:
        if USE_MULTI_IPU:
            method_to_loss_fn = {
                "dense": sum_and_sparse_categorical_crossentropy,
                "sparse_ops": get_sum_and_sparse_categorical_crossentropy_sparse_out(DENSE_SIZES[-1], transpose=False),
                "sparse_layer": get_sum_and_sparse_categorical_crossentropy_sparse_out(DENSE_SIZES[-1], transpose=True),
            }
            OPT = tf.keras.optimizers.SGD if BENCH_MODE=="multi_neuron" else tf.keras.optimizers.Adam
            train_mutli_ipu_benchmarking(
                IMPL_METHOD,
                NUM_EPOCHS, 
                TRAIN_STEPS_PER_EXECUTION, 
                BATCHSIZE_PER_STEP,
                dataloader_train.repeat(),
                SEQ_LEN, 
                DENSE_SIZES, 
                SPARSE_SIZES, 
                DECAY_CONSTANT, 
                THRESHOLD_FISRT_AND_SECOND,
                sum_and_sparse_categorical_crossentropy,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=callbacks,
                return_all=False,
                transpose_weights=TRANSPOSE_WEIGHTS,
                learning_rate=LEARNING_RATE,
                num_ipus=NUM_IPUS,
                weight_mul=WEIGHT_MUL,
                ipu_id=IPU_ID,
                opt=OPT,
            )
        else:
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
                THRESHOLD_FISRT_AND_SECOND,
                loss_fn,
                metrics=metrics,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=callbacks,
                return_all=True if CALC_ACTIVITY else False,
                transpose_weights=TRANSPOSE_WEIGHTS,
                learning_rate=LEARNING_RATE,
                weight_mul=WEIGHT_MUL,
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
            # THRESHOLD,
            THRESHOLD_FISRT_AND_SECOND,
            loss_fn,
            metrics=metrics, #calc_accuracy,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            return_all=True if CALC_ACTIVITY else False,
            learning_rate=LEARNING_RATE,
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
    parser.add_argument('--impl_method', type=str, default="dense", help="Implementation method to use, one of ['dense', 'sparse_ops', 'sparse_layer']."
                                                                    "Only used for `use_ipu=1`")
    parser.add_argument('--profile_run', type=int, default=0, help="Whether this is a profiling run (default is `0` therefore `Flase`), "
                                                                    "which uses shorter squence length, less data and only one epoch.")
    parser.add_argument('--sparse_multiplier', type=int, default=16, help="Factor to multiply sparse sizes with, default is 16.")
    parser.add_argument('--transpose_weights', type=int, default=0, help="Whether to use the transposed weight matrix to better make use of vectorization."
                                                                        " For now only used with `impl_method=sparse_layer`. Default is 0 (False).")
    parser.add_argument('--batchsize', type=int, default=48, help="batchsize to use for training, default is 48.")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for optimizer, default `1e-2`.")
    parser.add_argument('--second_thresh', type=float, default=0.9, help="Second threshold, default `0.9`.")
    parser.add_argument('--num_ipus', type=int, default=1, help="Number of IPUs to use, default `1`.")
    parser.add_argument('--weight_mul', type=float, default=1.0, help="Weight multiplier in weight init, default `1`.")
    parser.add_argument('--dataset_name', type=str, default="NMNIST", help="dataset name, in ['NMNIST' (default), 'DVSGesture', 'SHD'].")
    parser.add_argument('--bench_mode', type=str, default="base", help="Benchmarking mode, in ['base' (default), 'multi_layer', 'multi_ipu', 'multi_neuron'].")
    parser.add_argument('--max_activity', type=float, default=0.05, help="Max activity in case of 'multi_layer'-mode.")
    parser.add_argument('--num_hidden_layers', type=int, default=2, help="Number of IPUs to use, default `1`.")
    parser.add_argument('--sparse_size_inp', type=int, default=64, help="sparse size for input.")
    parser.add_argument('--ipu_id', type=int, default=-1, help="IPU ID to use.")
    parser.add_argument('--num_neurons_per_tile', type=int, default=2, help="The maximal number of neurons per Tile.")

    args = parser.parse_args()

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    # ROOT_PATH_DATA = "/p/scratch/chpsadm/finkbeiner1/datasets"
    # ROOT_PATH_DATA = "/Data/pgi-15/datasets"
    # ROOT_PATH_DATA = "/p/scratch/icei-hbp-2022-0011/common/datasets/"
    ROOT_PATH_DATA = "/localdata/datasets/"

    main(args, ROOT_PATH_DATA)
