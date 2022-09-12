import os
import warnings
import sys
import functools as ft
from typing import Union, NamedTuple, List

import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python import ipu

class SparseBinaryVec(NamedTuple):
    ids: Union[tf.Tensor, tf.TensorShape]
    num_nzelements: Union[tf.Tensor, tf.TensorShape, int]

class TileMapping(NamedTuple):
    start_tile: int
    end_tile: int

def determine_neuron_tileMappings(dense_sizes, sparse_sizes, num_ipus=1, min_neurons_per_tile=1):
    tileMapping =  determine_neuron_tileMappings_multiIPU(dense_sizes, sparse_sizes, num_ipus, min_neurons_per_tile)
    return tileMapping


def tile_mapping_const_number_states_per_tile(num_neurons, neurons_per_tile, TILES_PER_IPU, TILE_OFFSET):

    num_ipus = len(TILE_OFFSET)
    USABLE_TILES_PER_IPU = [int(TILES_PER_IPU - tile_offs) for tile_offs in TILE_OFFSET]

    layerwise_max_tiles = np.ceil(num_neurons / neurons_per_tile)
    cumsum_num_neurons = np.cumsum(num_neurons)
    tile_mapping_possible = True

    cumsum_tiles = 0
    ipu_id = 0
    tileMappings = []
    for ilay,max_tiles_ilay in enumerate(layerwise_max_tiles):
        print(ilay, max_tiles_ilay)
        if max_tiles_ilay > USABLE_TILES_PER_IPU[ipu_id]:
            tile_mapping_possible = False
            break
        new_cumsum_tiles = cumsum_tiles + max_tiles_ilay
        # check whether additonal layer fits on current IPU, otherwise start mapping on next IPU
        if new_cumsum_tiles > USABLE_TILES_PER_IPU[ipu_id]:
            cumsum_tiles = 0
            new_cumsum_tiles = max_tiles_ilay
            ipu_id += 1
        
        if ipu_id >= num_ipus:
            tile_mapping_possible = False
            break
        
        start_tile = int(cumsum_tiles + TILE_OFFSET[ipu_id] + ipu_id * TILES_PER_IPU)
        end_tile = int(start_tile + max_tiles_ilay)
        tileMappings.append(TileMapping(start_tile, end_tile))
        cumsum_tiles = new_cumsum_tiles

    if tile_mapping_possible:
        return tileMappings
    else:
        return None


def determine_neuron_tileMappings_multiIPU(dense_sizes, sparse_sizes, num_ipus, min_neurons_per_tile):
    
    TILE_OFFSET = [1] + [0]*(num_ipus-1)
    TILES_PER_IPU = 1472 # hardcoded fpr IPUv2 MK2000

    num_neurons = np.asarray(dense_sizes[1:], dtype=np.int64)
    
    neurons_per_tile = min_neurons_per_tile
    tile_mapping_found = False
    while not tile_mapping_found:
        print(f"\nneurons_per_tile={neurons_per_tile}")
        print(f"num_neurons={num_neurons}")
        tile_mapping = tile_mapping_const_number_states_per_tile(num_neurons, neurons_per_tile, TILES_PER_IPU, TILE_OFFSET)
        neurons_per_tile += min_neurons_per_tile
        print(tile_mapping)
        if tile_mapping is not None:
            tile_mapping_found = True
        print(f"tile_mapping_found={tile_mapping_found}")

    return tile_mapping
    
def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse)).astype(np.float32)
    for ibatch in range(batchsize):
        for iseq in range(seq_len):
            sparse_spike_ids[iseq, ibatch, :] = rng.choice(size_dense, size_sparse, replace=False)
    num_sparse_spikes = rng.choice(size_sparse, (seq_len, batchsize, 1), replace=True).astype(np.float32)
    return sparse_spike_ids, num_sparse_spikes


def sparse2dense_ipu(spikes: SparseBinaryVec, dense_size: int):
    spike_ids, num_spikes = spikes.ids, spikes.num_nzelements
    assert len(spike_ids.shape) == 3, f"`spike_ids` must be tensor of rank 3, got {len(spike_ids.shape)}."
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], dense_size])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "sparse2dense", "libcustom_op.so")
    gp_path = os.path.join(base_path, "sparse2dense", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately
                                              inputs_with_gradients=[0], # TODO is this working ?
                                              attributes=f"{dense_size}",
                                            )[0]


def custom_multi_lif_layer_sparse(sparse_sizes, weights, init_state, sparse_inp_spikes, decay_constants, thresholds, tileMappings):

    inp_spike_ids = [t.ids for t in sparse_inp_spikes] 
    num_inp_spikes = [t.num_nzelements for t in sparse_inp_spikes]

    # TODO check that batch and seq_lens are consitent for all states, inp_spike_ids, num_inp_spikes
    assert isinstance(weights, (list, tuple))
    num_layers = len(weights)

    seq_and_batch_size = num_inp_spikes[0].shape[:2]
    seq_len = num_inp_spikes[0].shape[0]
    batch_size = num_inp_spikes[0].shape[1]

    output_types = [
        *[inp_spike_ids[ilay].dtype for ilay in range(num_layers)],
        # *[num_inp_spikes[ilay].dtype for ilay in range(num_layers)],  # TODO uncomment
        *[tf.float32 for ilay in range(num_layers)],
        *[init_state[ilay].dtype for ilay in range(num_layers)],
    ]
    output_shapes = [
        *[tf.TensorShape([*seq_and_batch_size, sparse_sizes[ilay]]) for ilay in range(1, num_layers+1)], 
        *[tf.TensorShape([*seq_and_batch_size, 2]) for ilay in range(1, num_layers+1)],
        *[tf.TensorShape([seq_len, *init_state[ilay].shape]) for ilay in range(num_layers)],
    ]

    outputs = {
        "output_types": output_types,
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": output_shapes
    }

    base_path = os.path.realpath(os.path.dirname(__file__))

    lib_path = os.path.join(base_path, "custom_lif_multi_layer", "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_lif_multi_layer", "custom_codelet.gp")

    dense_sizes = [w.shape[0] for w in weights]
    dense_sizes = [weights[0].shape[1], *dense_sizes]

    weights = [tf.transpose(ws, perm=[1, 0]) for ws in weights]
    inputs = [*weights, *init_state, *inp_spike_ids, *num_inp_spikes, *decay_constants, *thresholds]

    sizes_str = "_".join([str(val) for val in [*dense_sizes, *sparse_sizes, batch_size]])
    start_tiles = [tileMapping.start_tile for tileMapping in tileMappings]
    end_tiles = [tileMapping.end_tile for tileMapping in tileMappings]
    tileMapping_str = "_".join([str(val) for val in [*start_tiles, *end_tiles]])
    attributes_str = "_".join([sizes_str, tileMapping_str])

    inputs_with_gradients = list(range(num_layers))
    out = ipu.custom_ops.precompiled_user_op(inputs,
                                              lib_path,
                                              gp_path,
                                              name="custom_multi_lif_layer_sparse", # TF operation name
                                              op_name="Build",
                                              outs=outputs,
                                              inputs_with_gradients=inputs_with_gradients,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=attributes_str,
                                              gradient_attributes=attributes_str,
                                            )
    return out

class KerasMultiLIFLayerBase(keras.layers.Layer):
    def __init__(self, dense_shapes, decay_constant, threshold, transpose_weights=False, seed=None):
        super().__init__()
        assert len(dense_shapes) > 1, "`dense_shapes` must be of at least length 2, generating a network with no hidden layers."
        self.num_layers = len(dense_shapes)-1
        self.dense_shapes = dense_shapes
        self.decay_constant_value = decay_constant
        self.threshold_value = threshold
        self.version_multi_thresh = isinstance(self.threshold_value, (list, tuple)) and (len(self.threshold_value) > 1)
        self.transpose_weights = transpose_weights
        self.seed = seed
        if self.version_multi_thresh:
            self.current_second_threshs = self.threshold_value[1] if isinstance(self.threshold_value[1], (tuple, list)) \
                                                else [self.threshold_value[1] for i in range(self.num_layers)]
        else: 
            self.current_second_threshs = None

    def build(self, input_shape):

        def custom_init(in_feat, out_feat, dtype):
            limit = (6/(in_feat))**0.5 * 1.5
            shape = (out_feat, in_feat)
            return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

        if self.seed is not None:
            tf.random.set_seed(self.seed+2)

        self.ws = [tf.Variable(
            initial_value=custom_init(in_feat=self.dense_shapes[ilay], out_feat=self.dense_shapes[ilay+1], dtype=tf.float32),
            trainable=True,
            name=f"weights_{ilay}",
        ) for ilay in range(self.num_layers)]

        self.decay_constants = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.decay_constant_value), tf.float32),
            trainable=False,
            name=f"decay_cosntants_{ilay}",
        ) for ilay in range(1, self.num_layers+1)]
        
        if self.version_multi_thresh is True:
            self.thresholds = self.create_thresh_tensor([self.threshold_value[0]]*self.num_layers, self.current_second_threshs)
        else:
            self.thresholds = [tf.Variable(
                initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.threshold_value), tf.float32),
                trainable=False,
                name=f"thresholds_{ilay}",
            ) for ilay in range(1, self.num_layers+1)]

    def create_thresh_tensor(self, threshs_1, threshs_2):
        thresholds = [tf.Variable(
                initial_value=tf.stack([
                    tf.cast(tf.fill((self.dense_shapes[ilay+1],), thresh), tf.float32) for thresh in thrs
                ], axis = 0),
                trainable=False,
                name=f"thresholds_{ilay+1}",
            ) for ilay,thrs in enumerate(zip(threshs_1, threshs_2))
        ]
        return thresholds

    def call(self):
        raise NotImplementedError



class KerasMultiLIFLayerSparse(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, seed=None, num_ipus=1):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        assert len(dense_shapes) == len(sparse_shapes), "`dense_shapes` and `sparse_shapes` must have the same nmber of elements."
        self.sparse_shapes = sparse_shapes
        self.neuron_tileMappings = determine_neuron_tileMappings(dense_shapes, sparse_shapes, num_ipus, min_neurons_per_tile=2)

    def call(self, inp_spikes, init_states):
        out = custom_multi_lif_layer_sparse(self.sparse_shapes, self.ws, init_states, inp_spikes, self.decay_constants, self.thresholds, self.neuron_tileMappings)
        return out

def model_fn_sparse_layer(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, seed=None, num_ipus=1):
    num_layers = len(dense_shapes)-1

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype ) for ilay in range(1, num_layers)]]
    num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype ) for ilay in range(1, num_layers)]]
    inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

    init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32, name=f"init_state_{i}") for i in range(num_layers)]
    out = KerasMultiLIFLayerSparse(
            dense_shapes, sparse_shapes, decay_constant, threshold, seed, num_ipus
        )(inp_spikes, init_states)
    out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]

    out = SparseBinaryVec(out_spike_ids[-1], num_out_spikes[-1])

    dense_spikes = sparse2dense_ipu(out, dense_shapes[-1])
    output = tf.transpose(dense_spikes, perm=[1, 0, 2])

    return (inp_spike_ids, num_inp_spikes), output


class KerasSparseIdentity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        last_dim = input_shape.ids[-1]

        self.eye = tf.Variable(
            initial_value=tf.eye(last_dim),
            trainable=True,
            name=f"identity_eye",
        )

    def call(self, x):
        return type(x)(x.ids @ self.eye, x.num_nzelements)


def model_fn_sparse_layer_multi_ipu(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, seed=None, num_ipus=1):
    num_layers = len(dense_shapes)-1
    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.float32, name="num_inp_spikes")

    assert num_ipus > 1

    with ipu.keras.PipelineStage(0):
        spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2], name="inp_spike_ids_0_transp"), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype, name=f"intial_spike_ids_{ilay}") for ilay in range(1, num_layers)]]
        num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2], name="nup_inp_spikes_0_transp"), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype, name=f"intial_num_spikes_{ilay}") for ilay in range(1, num_layers)]]
        inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

        init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32, name=f"init_state_{i}") for i in range(num_layers)]
    
        out = KerasMultiLIFLayerSparse(
                dense_shapes, sparse_shapes, decay_constant, threshold, seed, num_ipus
            )(inp_spikes, init_states)
        out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]

        out = SparseBinaryVec(out_spike_ids[-1], num_out_spikes[-1])

    for iipu in range(1,num_ipus-1):
        with ipu.keras.PipelineStage(iipu):
            out = KerasSparseIdentity()(out)

    with ipu.keras.PipelineStage(num_ipus-1):
        dense_spikes = sparse2dense_ipu(out, dense_shapes[-1])
        output = tf.transpose(dense_spikes, perm=[1, 0, 2])
    
    return (inp_spike_ids, num_inp_spikes), output


def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)

def calc_sparse_categorical_accuracy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1)
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, sum_spikes)

def create_dataset_sparse(inp_spike_ids, num_inp_spikes, labels, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes, "targets": labels})
    num_samples = labels.shape[0]
    if shuffle:
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_ipu(
        method,
        num_epochs,
        train_steps_per_execution,
        batch_size,
        dataset,
        seq_len, 
        dense_sizes, 
        sparse_sizes, 
        decay_constant, 
        threshold,
        loss_fn,
        steps_per_epoch=None,
        learning_rate=1e-2,
        num_ipus=1
    ):
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    assert method in ["sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{method}'."


    if num_ipus > 1:
        method_to_model_fn = {
            "sparse_layer": ft.partial(model_fn_sparse_layer_multi_ipu, sparse_sizes, num_ipus=num_ipus),
        }
    else:
        method_to_model_fn = {
            "sparse_layer": ft.partial(model_fn_sparse_layer, sparse_sizes),
        }

    with strategy.scope():
        # init model
        inputs, outputs = method_to_model_fn[method](seq_len, dense_sizes, decay_constant, threshold, batch_size)
        model = keras.Model(inputs, outputs)

        if num_ipus > 1:
            device_mapping = [ ipu.pipelining_ops._ALL_DEVICES, *[num_ipus-1]*(num_ipus-1)]
            model.set_pipelining_options(
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Sequential,
                gradient_accumulation_steps_per_replica=1,
                device_mapping=device_mapping,
                offload_weight_update_variables=False,
            )

        # Set the infeed and outfeed options.
        model.set_infeed_queue_options(prefetch_depth=2)
        model.set_outfeed_queue_options(buffer_depth=2)

        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optim, loss_fn,
                    steps_per_execution=train_steps_per_execution,
        )
        # model.summary()

        model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, workers=batch_size)



def create_dataset_sparse(inp_spike_ids, num_inp_spikes, labels, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes}, labels))
    num_samples = labels.shape[0]
    if shuffle:
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_dataloader(rng, seq_len, num_inp_neurons, num_classes, num_total_samples, batchsize, sparse_size):
    sparse_spike_ids, num_sparse_spikes = gen_sparse_spikes(rng, seq_len, num_total_samples, num_inp_neurons, sparse_size)
    sparse_spike_ids = sparse_spike_ids.transpose((1,0,2)) 
    num_sparse_spikes = num_sparse_spikes.transpose((1,0,2))
    labels = rng.integers(0, num_classes, size=(num_total_samples,)).astype(np.int32)
    dataloader_train = create_dataset_sparse(sparse_spike_ids, num_sparse_spikes, labels, batchsize, shuffle=True) 
    return dataloader_train

if __name__ == '__main__':

    rng = np.random.default_rng(42)

    NUM_IPUS = 2
    LEARNING_RATE = 1e-2
    IMPL_METHOD = "sparse_layer"
    BATCHSIZE = 48
    NUM_EPOCHS = 2
    NUM_SAMPLES = int(48 * 32)
    TRAIN_STEPS_PER_EXECUTION = int(NUM_SAMPLES / BATCHSIZE)
    STEPS_PER_EPOCH = int(NUM_SAMPLES / BATCHSIZE)
    SEQ_LEN = 100
    DECAY_CONSTANT = 0.95
    THRESHOLD = [1.0, 0.95]

    NUM_CLASSES = 10
    IMAGE_DIMS = (34,34,2)
    DENSE_SIZES = [np.prod(IMAGE_DIMS), 1470, *[1472]*(2*(NUM_IPUS-1)), 1076+384, NUM_CLASSES]
    SPARSE_SIZES = [64, 64, *[64]*(2*(NUM_IPUS-1)), 64, NUM_CLASSES]

    dataloader_train = get_dataloader(rng, SEQ_LEN, DENSE_SIZES[0], NUM_CLASSES, 
                    NUM_SAMPLES, BATCHSIZE, sparse_size=SPARSE_SIZES[0])

    train_ipu(
        IMPL_METHOD,
        NUM_EPOCHS, 
        TRAIN_STEPS_PER_EXECUTION, 
        BATCHSIZE,
        dataloader_train.repeat(),
        SEQ_LEN, 
        DENSE_SIZES, 
        SPARSE_SIZES, 
        DECAY_CONSTANT, 
        THRESHOLD,
        sum_and_sparse_categorical_crossentropy,
        steps_per_epoch=STEPS_PER_EPOCH,
        learning_rate=LEARNING_RATE, 
        num_ipus=NUM_IPUS
    )



