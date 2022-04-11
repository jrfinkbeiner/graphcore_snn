import os
import warnings
import sys
import math
from typing import Union, NamedTuple
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import functools as ft
import time

from tensorflow.python import ipu

# import tensorflow.keras as keras

# tf.disable_v2_behavior()
# from tensorflow.python.framework.tensor_shape import TensorShape

class SparseBinaryVec(NamedTuple):
    ids: Union[tf.Tensor, tf.TensorShape]
    num_nzelements: Union[tf.Tensor, tf.TensorShape, int]

def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse)).astype(np.float32)
    for ibatch in range(batchsize):
        for iseq in range(seq_len):
            sparse_spike_ids[iseq, ibatch, :] = rng.choice(size_dense, size_sparse, replace=False)
    num_sparse_spikes = rng.choice(size_sparse, (seq_len, batchsize, 1), replace=True).astype(np.int32)
    return sparse_spike_ids, num_sparse_spikes

def sparse2dense(spike_ids, num_spikes, dense_size, values=None, sparse_dim=-1):
    assert sparse_dim == -1
    assert len(spike_ids.shape) == 3
    if values is not None:
        assert values.shape == spike_ids.shape
    sparse_shape = spike_ids.shape
    dense_shape = list(spike_ids.shape)
    dense_shape[sparse_dim] = dense_size
    dense_shape = [dense_shape[0], dense_shape[1], dense_shape[2]]

    dense_tensor = np.zeros((dense_shape[0], dense_shape[1], dense_shape[2]), dtype=np.float32)
    for iseq in range(spike_ids.shape[0]):
        for ibatch in range(spike_ids.shape[1]):
            if num_spikes is None:
                ids = spike_ids[iseq, ibatch, :].astype(np.int32)
            else:
                ids = spike_ids[iseq, ibatch, :num_spikes[iseq, ibatch, 0]].astype(np.int32)

            if values is None:
                dense_tensor[iseq, ibatch, ids] = 1
            else:
                if num_spikes is None:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :]
                else:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :num_spikes[iseq, ibatch, 0]]
    return dense_tensor

def sparse2dense_ipu(spikes: SparseBinaryVec, dense_size: int):
    spike_ids, num_spikes = spikes.ids, spikes.num_nzelements
    assert len(spike_ids.shape) == 3, f"`spike_ids` must be tensor of rank 3, got {len(spike_ids.shape)}."
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], dense_size])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "sparse2dense", "libcustom_op.so")
    gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "sparse2dense", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{dense_size}",
                                            )[0]


def dyn_dense_binary_sparse_matmul_op(matrix: tf.Tensor, sparse_vec: SparseBinaryVec):
    sparse_ids, num_nzelements = sparse_vec.ids, sparse_vec.num_nzelements

    outputs = {
        "output_types": [matrix.dtype],
        "output_shapes": [tf.TensorShape([*sparse_ids.shape[:-1], matrix.shape[0]])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "libcustom_op.so")
    gp_path  = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "custom_codelet.gp")
   
    out = ipu.custom_ops.precompiled_user_op([matrix, sparse_ids, num_nzelements],
                                              lib_path,
                                              gp_path,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              outs=outputs)[0]
    return out


def compute_sparse_spikes(state: tf.Tensor, thresholds: tf.Tensor, sparse_size: int, start_tile: int, end_tile: int):

    batch_size = state.shape[0]
    outputs = {
        "output_types": [state.dtype, tf.int32],
        "output_shapes": [tf.TensorShape((batch_size, sparse_size)), tf.TensorShape((batch_size, 1))],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "libcustom_op.so")
    gp_path  = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "custom_codelet.gp")
    
    attributes = "_".join([str(val) for val in [sparse_size, start_tile, end_tile]])
    print(attributes)
    out = ipu.custom_ops.precompiled_user_op([state, thresholds],
                                              lib_path,
                                              gp_path,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=attributes,
                                              gradient_attributes=attributes,
                                              outs=outputs)
    spike_ids, num_spikes = out[0], out[1]
    return SparseBinaryVec(spike_ids, num_spikes)


# TODO could also implement a self-recurrent version
def custom_multi_lif_layer_sparse(sparse_sizes, weights, init_state, sparse_inp_spikes, decay_constants, thresholds):

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
        *[num_inp_spikes[ilay].dtype for ilay in range(num_layers)],
        *[init_state[ilay].dtype for ilay in range(num_layers)],
    ]
    output_shapes = [
        *[tf.TensorShape([*seq_and_batch_size, sparse_sizes[ilay]]) for ilay in range(1, num_layers+1)], 
        *[tf.TensorShape([*seq_and_batch_size, 1]) for ilay in range(1, num_layers+1)],
        *[tf.TensorShape([seq_len, *init_state[ilay].shape]) for ilay in range(num_layers)],
    ]

    outputs = {
        "output_types": output_types,
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": output_shapes
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "libcustom_op.so")
    # gp_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "custom_codelet.gp")
    lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize", "libcustom_op.so")
    gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize", "custom_codelet.gp")

    inputs = [*weights, *init_state, *inp_spike_ids, *num_inp_spikes, *decay_constants, *thresholds]
    # sparse_sizes_str = "_".join([str(val) for val in sparse_sizes])
    dense_sizes = [w.shape[0] for w in weights]
    dense_sizes = [weights[0].shape[1], *dense_sizes]
    sizes_str = "_".join([str(val) for val in [*dense_sizes, *sparse_sizes, batch_size]])
    return ipu.custom_ops.precompiled_user_op(inputs,
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=sizes_str,
                                            )

class KerasMultiLIFLayerBase(keras.layers.Layer):
    def __init__(self, dense_shapes, decay_constant, threshold, seed=None):
        super().__init__()
        assert len(dense_shapes) > 1, "`dense_shapes` must be of at least length 2, generating a network with no hidden layers."
        self.num_layers = len(dense_shapes)-1
        self.dense_shapes = dense_shapes
        self.decay_constant_value = decay_constant
        self.threshold_value = threshold
        self.seed = seed

    def build(self, input_shape):
        w_init = tf.random_normal_initializer(0.0, 5.0, self.seed)
        self.ws = [tf.Variable(
            initial_value=w_init(shape=(self.dense_shapes[ilay+1], self.dense_shapes[ilay]), dtype=tf.float32),
            trainable=True,
            name="weights",
        ) for ilay in range(self.num_layers)]
        self.decay_constants = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.decay_constant_value, "decay_cosntants"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]
        self.thresholds = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.threshold_value, "thresholds"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]

    def call(self):
        raise NotImplementedError


class KerasMultiLIFLayerSparse(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        assert len(dense_shapes) == len(sparse_shapes), "`dense_shapes` and `sparse_shapes` must have the same nmber of elements."
        self.sparse_shapes = sparse_shapes

    def call(self, inp_spikes, init_states):
        return custom_multi_lif_layer_sparse(self.sparse_shapes, self.ws, init_states, inp_spikes, self.decay_constants, self.thresholds)


@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * 1/(beta*tf.math.abs(x)+1)**2
  return spikes, grad

def pure_tf_lif_step_sparse(weights, state, inp_, decay_constants, thresholds, sparse_dim):
    syn_inp = dyn_dense_binary_sparse_matmul_op(weights, inp_)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 1))
    new_state = state * decay_constants + (1 - decay_constants) * syn_inp
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    # spikes_out = heaviside_with_super_spike_surrogate(new_state-thresholds) # TODO this has to be a custom_op as well, select spikes_op
    # warnings.warn("Implement proper spike selection and its gradient.", UserWarning)

    # top_k_neurons = tf.cast(tf.math.top_k(spikes_out, k=sparse_dim, sorted=True).indices, tf.float32)
    # # TODO multiply with sum here
    # num_spikes = tf.expand_dims(tf.cast(tf.stop_gradient(tf.math.reduce_sum(spikes_out, axis=-1)), dtype=tf.int32), axis=-1)
    # spikes_out = SparseBinaryVec(top_k_neurons, num_spikes)

    start_tile = 1
    end_tile = int(new_state.shape[0]+1)

    spikes_out = compute_sparse_spikes(new_state, thresholds, sparse_dim, start_tile, end_tile)

    return spikes_out, new_state

class KerasMultiLIFLayerSparseCell(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        self.sparse_shapes = sparse_shapes
        state_size = [tf.TensorShape((dim,)) for dim in dense_shapes[1:]]
        for sparse_dim in sparse_shapes[1:]:
            state_size.extend((tf.TensorShape((sparse_dim,)), tf.TensorShape((1,))))
        print(state_size)
        self.state_size = state_size
        self.output_size = [ SparseBinaryVec(tf.TensorShape((sparse_dim,)), tf.TensorShape((1,))) for sparse_dim in sparse_shapes[1:]]
        # self.output_size = dense_shapes[-1]

    def call(self, inp_spikes, state):
        print(f"{state=}")
        neuron_states = state[:self.num_layers]
        outs = [SparseBinaryVec(ids,num_nzelements) for ids,num_nzelements in zip(state[self.num_layers::2], state[self.num_layers+1::2])]
        all_out_spikes = []
        all_neuron_states = []

        for ilay in range(self.num_layers):
            inp_ = inp_spikes if ilay==0 else outs[ilay-1]
            spikes_out, neuron_stat = pure_tf_lif_step_sparse(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay], self.sparse_shapes[ilay+1])
            all_neuron_states.append(neuron_stat)
            all_out_spikes.extend(spikes_out)
            
        state_new = [*all_neuron_states, *all_out_spikes]
        structured_out_spikes = [SparseBinaryVec(ids,num_nzelements) for ids,num_nzelements in zip(all_out_spikes[::2], all_out_spikes[1::2])]
        return structured_out_spikes, state_new

    def get_initial_state(self, inputs, batch_size, dtype):
        # print(self.dense_shapes)
        # return [tf.zeros((batch_size, dim)) for dim in self.dense_shapes[1:]]
        # print(self.state_size[0].as_list())
        # return [tf.zeros((batch_size, *stat_shape.as_list())) for stat_shape in self.state_size]
        init_state = [tf.zeros((batch_size, dim), dtype=tf.float32) for dim in self.dense_shapes[1:]]
        for sparse_dim in self.sparse_shapes[1:]:
            init_state.extend((tf.zeros((batch_size, sparse_dim), dtype=tf.float32), tf.zeros((batch_size, 1), dtype=tf.int32)))
        return init_state

def KerasMultiLIFLayerSparseOps(dense_shapes, sparse_shapes, decay_constant, threshold, seed=None, **kwargs):
    return tf.keras.layers.RNN(KerasMultiLIFLayerSparseCell(dense_shapes, sparse_shapes, decay_constant, threshold, seed), **kwargs)


def pure_tf_lif_step_dense(weights, state, inp_, decay_constants, thresholds):
    syn_inp = tf.matmul(inp_, weights, transpose_b=True)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 1))
    new_state = state * decay_constants + (1 - decay_constants) * syn_inp
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    spikes_out = heaviside_with_super_spike_surrogate(new_state-thresholds)
    return spikes_out, new_state

class KerasMultiLIFLayerDenseCell(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, decay_constant, threshold, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        state_shapes = dense_shapes[1:]*2
        self.state_size = [tf.TensorShape((dim,)) for dim in state_shapes]
        self.output_size = [tf.TensorShape((shape_,)) for shape_ in dense_shapes[1:]]
        # self.output_size = dense_shapes[-1]

    def call(self, inp_spikes, state):
        neuron_states = state[:self.num_layers]
        outs = state[self.num_layers:]
        all_out_spikes = []
        all_neuron_states = []

        for ilay in range(self.num_layers):
            inp_ = inp_spikes if ilay==0 else outs[ilay-1]
            spikes_out, neuron_stat = pure_tf_lif_step_dense(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay])
            all_neuron_states.append(neuron_stat)
            all_out_spikes.append(spikes_out)
            
        state_new = [*all_neuron_states, *all_out_spikes]
        return all_out_spikes, state_new

def KerasMultiLIFLayerDense(dense_shapes, decay_constant, threshold, seed=None, **kwargs):
    return tf.keras.layers.RNN(KerasMultiLIFLayerDenseCell(dense_shapes, decay_constant, threshold, seed), **kwargs)


def model_fn_sparse_layer(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, return_all=False, seed=None):
    if return_all:
        warnings.warn("All layers outputs will be returned. But note that only gradient propagation through the last layers outputs is implemented."
                    " Adding loss terms to other layers outputs will be ignored and will result in a wrong gradient.", UserWarning)

    num_layers = len(dense_shapes)-1

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype ) for ilay in range(1, num_layers)]]
    num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype ) for ilay in range(1, num_layers)]]
    inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

    init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32) for i in range(num_layers)]
    out = KerasMultiLIFLayerSparse(
            dense_shapes, sparse_shapes, decay_constant, threshold, seed
        )(inp_spikes, init_states)
    out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]

    # if return_all:
    #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    # else:
    #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
    #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
    #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])

    if return_all:
        out = [SparseBinaryVec(ids, num_nzelements) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    else:
        out = SparseBinaryVec(out_spike_ids[-1], num_out_spikes[-1])
    return (inp_spike_ids, num_inp_spikes), out


def model_fn_sparse_ops(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, return_all=False, seed=None):

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    inp_spikes = SparseBinaryVec(inp_spike_ids, num_inp_spikes)

    out_spikes = KerasMultiLIFLayerSparseOps(
            dense_shapes, sparse_shapes, decay_constant, threshold, seed, return_sequences=True
        )(inp_spikes)
    # if return_all:
    #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    # else:
    #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
    #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
    #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])

    if return_all:
        out = out_spikes
    else:
        out = out_spikes[-1]
    return (inp_spike_ids, num_inp_spikes), out


def model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, seed=None, return_all=False):
    inp_spikes = keras.Input(shape=(seq_len, dense_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32)
    out_spikes = KerasMultiLIFLayerDense(
            dense_shapes, decay_constant, threshold, seed, return_sequences=True
    )(inp_spikes)
    if return_all:
        out = out_spikes
    else:
        out = out_spikes[-1]
    return inp_spikes, out

def simple_loss_fn_dense(y_target, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def simple_loss_fn_sparse(y_target, y_pred: SparseBinaryVec):
    print(f"{y_pred=}")
    dense_spikes = tf.transpose(sparse2dense_ipu(y_pred, y_target.shape[-1]), perm=[1, 0, 2])
    sum_spikes = tf.reduce_sum(dense_spikes, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def create_dataset_sparse(inp_spike_ids, num_inp_spikes, targets, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes}, targets))
    # dataset = tf.data.Dataset.from_tensor_slices((inp_spike_ids, targets))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if shuffle:
        dataset = dataset.shuffle(targets.shape[0])
    dataset = dataset.prefetch(batchsize)
    return dataset

def create_dataset_dense(inp_spikes, targets, batchsize, shuffle=True):
    # dataset = tf.data.Dataset.from_tensor_slices(({"input_1": inp_spike_ids, "input_2": num_inp_spikes}, targets))
    dataset = tf.data.Dataset.from_tensor_slices((inp_spikes, targets))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if shuffle:
        dataset = dataset.shuffle(targets.shape[0])
    dataset = dataset.prefetch(batchsize)
    return dataset


def train_ipu( # TODO finish implementation
        num_epochs,
        train_steps_per_execution,
        batchsize_per_step,
        dataset,
        seq_len, 
        dense_shapes, 
        sparse_shapes, 
        decay_constant, 
        threshold,
        loss_fn,
        metrics=None,
        steps_per_epoch=None,
        callbacks=None,
        mode=None,
        return_all=False,
    ):
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    if mode is None:
        mode = "sparse_layer"

    assert mode in ["dense", "sparse_ops", "sparse_layer"], f"`mode` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{mode}'."

    model_to_model_fn = {
        "dense": model_fn_dense, 
        "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_shapes), 
        "sparse_layer": ft.partial(model_fn_sparse_layer, sparse_shapes),
    }

    with strategy.scope():
        # init model
        model = keras.Model(*model_fn_sparse_layer(seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, return_all=return_all))

        # model.set_pipelining_options(
        #     gradient_accumulation_steps_per_replica=2*num_ipus #train_steps_per_execution
        # )

        # Compile our model with Stochastic Gradient Descent as an optimizer
        # and Categorical Cross Entropy as a loss.
        # optim = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=False, name="SGD")
        optim = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9, nesterov=False, name="SGD")

        # model.add_loss()
        model.compile(optim, loss_fn,
                    metrics=metrics,
                    steps_per_execution=train_steps_per_execution,
        )

        model.summary()

        print('\nTraining')
        model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, workers=batchsize_per_step, callbacks=callbacks)
        # model.fit([inp_spike_ids, num_inp_spikes, init_states], targets, epochs=num_epochs, batch_size=batchsize, shuffle=False)


def train_gpu(
        num_epochs, 
        train_steps_per_execution,
        batchsize,
        dataset,
        seq_len, 
        dense_shapes, 
        decay_constant, 
        threshold,
        loss_fn,
        metrics=None,
        steps_per_epoch=None,
        callbacks=None,
        return_all=False,
    ):

    # init model
    model = keras.Model(*model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize, return_all=return_all))

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    optim = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9, nesterov=False, name="SGD")
    model.compile(optim, loss_fn,
                # metrics=["accuracy"],
                metrics=metrics,
                # steps_per_execution=train_steps_per_execution,
                # jit_compile=True
    )

    model.summary()

    print('\nTraining')
    model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, workers=batchsize, callbacks=callbacks) #, use_multiprocessing=True) #, 
                # validation_steps=1, validation_batch_size=10*batchsize)



@tf.function(experimental_compile=True)  # Make it fast.
def value_and_grad_on_batch(model, x, y, sparse=False, out_batch_first=True):
    with tf.GradientTape() as tape:
        out = model(x)
        if sparse:
            out = sparse2dense_ipu(out, y.shape[-1])
        if not out_batch_first:
            out = tf.transpose(out , perm=[1, 0, 2])
        loss = simple_loss_fn_dense(y, out)
        gradients = tape.gradient(loss, model.trainable_weights)
    return out, gradients

def check_values(a, b, name, *,rtol=1e-4, **kwargs):
    sucess = False
    same_shape = (len(a.shape) == len(b.shape)) and all([x == y for x,y in zip(a.shape, b.shape)])
    if same_shape:
        check_allclose = np.allclose(a, b, rtol=rtol, **kwargs)
        if check_allclose:
            sucess = True
    if sucess:
        print(f"\u001b[32m{name}: Success! Reults for tf and custom ipu implementation are identical.\u001b[0m")
    else:
        print(f"\u001b[31m{name}: Wrong results! Reults for tf and custom ipu implementation are not identical.\u001b[0m")
    return sucess

def test_sparse_vs_dense():

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_sequences = 2
    batchsize = num_sequences
    batchsize_per_step = batchsize
    seq_len = 200
    # dense_sizes = [102, 801, 799]
    dense_sizes = [4, 4]
    sparse_sizes = [3, 3] #dense_sizes
    decay_constant = 0.9
    threshold = 1.0
    num_layers = len(dense_sizes)-1
    model_seed = rng.integers(999999)

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(batchsize / batchsize_per_step)

    targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])
    dataset_sparse = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)
    inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
    dataset_dense = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    data_sparse = iter(dataset_sparse).next()
    data_dense = iter(dataset_dense).next()

    with strategy.scope():
        model_dense_ipu = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, threshold, batchsize, seed=model_seed, return_all=False))
        out_ipu_dense, grad_ipu_dense =  strategy.run(value_and_grad_on_batch, args=[model_dense_ipu, *data_dense, False])

    with strategy.scope():
        model_sparse_layer = keras.Model(*model_fn_sparse_layer(sparse_sizes, seq_len, dense_sizes, decay_constant, threshold, batchsize_per_step, seed=model_seed, return_all=False))
        out_sparse_layer, grad_sparse_layer =  strategy.run(value_and_grad_on_batch, args=[model_sparse_layer, *data_sparse, True, False])

    with strategy.scope():
        model_sparse_ops = keras.Model(*model_fn_sparse_ops(sparse_sizes, seq_len, dense_sizes, decay_constant, threshold, batchsize_per_step, seed=model_seed, return_all=False))
        out_sparse_ops, grad_sparse_ops =  strategy.run(value_and_grad_on_batch, args=[model_sparse_ops, *data_sparse, True])

    model_dense = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, threshold, batchsize, seed=model_seed, return_all=False))
    out_dense, grad_dense = value_and_grad_on_batch(model_dense, *data_dense, False)
    
 
    # print("\nforward")
    # print(out_dense.shape)
    # print(out_sparse.shape)
    # print(out_sparse)
    # print(out_dense)
    # print("\ngrad")
    # print(len(grad_sparse))
    # print(len(grad_dense))
    # print(grad_sparse)
    # print(grad_dense)

    # import matplotlib.pyplot as plt
    # for i in range(num_layers):
    #     quotient = (grad_sparse[i]/grad_dense[i]).numpy()
    #     print()
    #     print(quotient)
    #     plt.figure()
    #     plt.hist(quotient.flatten(), bins=int(np.sqrt(0.5*quotient.size)))
    #     plt.title(f"layer {i}")

    #     print(np.nanmin(quotient), np.nanmax(quotient))
    # plt.show()

    print("------------------ in --------------------------")
    print(data_sparse)
    print(data_dense)
    print("------------------ out -------------------------")
    print(out_sparse_ops)
    print(f"{out_sparse_ops.numpy().sum()=}")
    print()
    print(f"{out_sparse_layer.numpy().sum()=}")
    print()
    print(f"{out_dense.numpy().sum()=}")


    for i in range(num_layers):
        print(f"------------------- {i} ----------------------")
        print(f"{grad_dense[i]=}")
        print(f"{grad_sparse_ops[i]=}")
        print(f"{grad_sparse_layer[i]=}")
        print(tf.math.abs(grad_sparse_ops[i]-grad_sparse_layer[i]) > 1e-6)


    print()
    check_values(out_ipu_dense, out_dense, f"dense - out_spikes", rtol=1e-4, atol=1e-6)
    for i in range(num_layers):
        check_values(grad_ipu_dense[i], grad_dense[i], f"dense - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        # check_values(model_dense_ipu.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)
    print()
    check_values(out_sparse_layer, out_dense, f"sparse layer - out_spikes", rtol=1e-4, atol=1e-6)
    for i in range(num_layers):
        check_values(grad_sparse_layer[i], grad_dense[i], f"sparse layer - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        # check_values(model_sparse_layer.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)
    print()
    check_values(out_sparse_ops, out_dense, f"sparse ops - out_spikes", rtol=1e-4, atol=1e-6)
    for i in range(num_layers):
        check_values(grad_sparse_ops[i], grad_dense[i], f"sparse ops - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        # check_values(model_sparse_ops.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)

    print()
    check_values(out_sparse_ops, out_sparse_layer, f"sparse layer vs sparse ops - out_spikes", rtol=1e-4, atol=1e-6)
    for i in range(num_layers):
        check_values(grad_sparse_ops[i], grad_sparse_layer[i], f"sparse layer vs sparse ops - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        # check_values(model_sparse_ops.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)



def main():

    sparse_comp = True
    if sparse_comp:
        os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_epochs = 10
    num_sequences = 64
    batchsize = 2
    batchsize_per_step = 2
    seq_len = 12
    dense_sizes = [612, 512, 412]
    sparse_sizes = [42, 32, 22]
    decay_constant = 0.9
    threshold = 1.0

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(num_sequences / batchsize) # TODO or batchsize_per_step ?

    targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])

    if sparse_comp:
        dataset = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize)
        train_ipu(
            num_epochs, 
            train_steps_per_execution, 
            batchsize_per_step,
            dataset,
            seq_len, 
            dense_sizes, 
            sparse_sizes, 
            decay_constant, 
            threshold,
            simple_loss_fn
        )
    else:
        inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
        dataset = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize)
        train_gpu(
            num_epochs,  
            batchsize,
            dataset,
            seq_len, 
            dense_sizes, 
            decay_constant, 
            threshold,
            simple_loss_fn
        )


if __name__ == '__main__':
    test_sparse_vs_dense()
    # main()



