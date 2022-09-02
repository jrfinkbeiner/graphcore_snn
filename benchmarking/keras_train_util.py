import os
import warnings
import sys
import math
from typing import Union, NamedTuple
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import functools as ft


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


def get_shape(in_features: int, out_features: int, transpose: bool):
    # return (in_features, out_features) if transpose else (out_features, in_features)
    return (out_features, in_features)

class KerasMultiLIFLayerBase(keras.layers.Layer):
    def __init__(self, dense_shapes, decay_constant, threshold, transpose_weights=False, seed=None):
        super().__init__()
        assert len(dense_shapes) > 1, "`dense_shapes` must be of at least length 2, generating a network with no hidden layers."
        self.num_layers = len(dense_shapes)-1
        self.dense_shapes = dense_shapes
        self.decay_constant_value = decay_constant
        self.threshold_value = threshold
        self.transpose_weights = transpose_weights
        self.seed = seed

    def build(self, input_shape):

        def custom_init(in_feat, out_feat, dtype):
            # limit = (6/(in_feat + out_feat))**0.5
            limit = (6/(in_feat))**0.5
            shape = get_shape(in_feat, out_feat, self.transpose_weights)
            print(f"limit={limit}")
            # return tf.random.uniform(shape, minval=0, maxval=limit, dtype=dtype)
            return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

        # w_init = tf.random_normal_initializer(0.0, 10.0, self.seed)
        w_init = tf.random_normal_initializer(0.0, 0.1, self.seed)

        if self.seed is not None:
            tf.random.set_seed(self.seed+2)

        self.ws = [tf.Variable(
            # initial_value=w_init(shape=get_shape(self.dense_shapes[ilay], self.dense_shapes[ilay+1], self.transpose_weights), dtype=tf.float32),
            initial_value=custom_init(in_feat=self.dense_shapes[ilay], out_feat=self.dense_shapes[ilay+1], dtype=tf.float32),
            # initial_value=0.1*tf.ones(shape=get_shape(self.dense_shapes[ilay], self.dense_shapes[ilay+1], self.transpose_weights), dtype=tf.float32),
            trainable=True,
            name=f"weights_{ilay}",
        ) for ilay in range(self.num_layers)]

        self.decay_constants = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.decay_constant_value, f"decay_cosntants_{ilay}"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]
        self.thresholds = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.threshold_value, f"thresholds_{ilay}"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]

    def call(self):
        raise NotImplementedError


@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * (1/(beta*tf.math.abs(x)+1)**2)
  return spikes, grad

def pure_tf_lif_step_dense(weights, state, inp_, decay_constants, thresholds):
    syn_inp = tf.matmul(inp_, weights, transpose_b=True)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 1))
    new_state = state * decay_constants + (1 - decay_constants) * 10 * syn_inp # hard coded factor 20 in IPU code
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    spikes_out = heaviside_with_super_spike_surrogate(new_state-thresholds)
    return spikes_out, new_state

@tf.custom_gradient
def heaviside_with_super_spike_surrogate_secondthresh(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * (1/(beta*tf.math.abs(x)+1)**2) * tf.experimental.numpy.heaviside(x+0.05, 1)
  return spikes, grad

def pure_tf_lif_step_dense_secondthresh(weights, state, inp_, decay_constants, thresholds):
    syn_inp = tf.matmul(inp_, weights, transpose_b=True)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 1))
    new_state = state * decay_constants + (1 - decay_constants) * 10 * syn_inp # hard coded factor 20 in IPU code
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    spikes_out = heaviside_with_super_spike_surrogate_secondthresh(new_state-thresholds)
    return spikes_out, new_state


class KerasMultiLIFLayerDenseCell(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, decay_constant, threshold, transpose_weights=False, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, transpose_weights, seed)
        state_shapes = dense_shapes[1:]*2
        self.state_size = [tf.TensorShape((dim,)) for dim in state_shapes]
        self.output_size = [tf.TensorShape((shape_,)) for shape_ in dense_shapes[1:]]
        # self.output_size = dense_shapes[-1]

    def call(self, inp_spikes, state):
        neuron_states = state[:self.num_layers]
        outs = state[self.num_layers:]
        all_out_spikes = []
        all_neuron_states = []

        # for ilay in range(self.num_layers):
        #     inp_ = inp_spikes if ilay==0 else outs[ilay-1]
        #     spikes_out, neuron_stat = pure_tf_lif_step_dense(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay])
        #     all_neuron_states.append(neuron_stat)
        #     all_out_spikes.append(spikes_out)

        for ilay in range(self.num_layers-2):
            inp_ = inp_spikes if ilay==0 else outs[ilay-1]
            spikes_out, neuron_stat = pure_tf_lif_step_dense_secondthresh(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay])
            all_neuron_states.append(neuron_stat)
            all_out_spikes.append(spikes_out)
        for ilay in range(self.num_layers-2, self.num_layers):
            inp_ = outs[ilay-1]
            spikes_out, neuron_stat = pure_tf_lif_step_dense(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay])
            all_neuron_states.append(neuron_stat)
            all_out_spikes.append(spikes_out)
               
        state_new = [*all_neuron_states, *all_out_spikes]
        return all_out_spikes, state_new

def KerasMultiLIFLayerDense(dense_shapes, decay_constant, threshold, transpose_weights=False, seed=None, **kwargs):
    return tf.keras.layers.RNN(KerasMultiLIFLayerDenseCell(dense_shapes, decay_constant, threshold, transpose_weights, seed), **kwargs)

def model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, seed=None, return_all=False):
    inp_spikes = keras.Input(shape=(seq_len, dense_shapes[0]), batch_size=batchsize_per_step, name="inp_spikes", dtype=tf.float32)
    out_spikes = KerasMultiLIFLayerDense(
            dense_shapes, decay_constant, threshold, False, seed, return_sequences=True, time_major=False
    )(inp_spikes)
    if return_all:
        out = out_spikes
    else:
        out = out_spikes[-1]
    return inp_spikes, out

def simple_loss_fn_dense(y_target, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]


def create_dataset_dense(data, labels, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = tf.data.Dataset.from_tensor_slices({"inp_spikes": data, "targets": labels})
    num_samples = labels.shape[0]
    if shuffle:
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    # dataset = dataset.repeat()
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.prefetch(4)
    return dataset




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
        learning_rate=1e-2,
        seed=None,
    ):

    # init model
    inputs, outputs = model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize, return_all=return_all, seed=seed)
    targets = keras.Input((1,), name="targets")
    model = keras.Model([inputs, targets], outputs)

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    # optim = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9, nesterov=False, name="SGD")
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    
    model.add_loss(loss_fn(targets, outputs))
    if metrics is not None:
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        for metric in metrics:
            model.add_metric(metric(targets, outputs), metric.__name__)
    model.compile(optim,
                # metrics=["accuracy"],
                # metrics=metrics,
                steps_per_execution=train_steps_per_execution,
                jit_compile=True
    )

    model.summary()

    print('\nTraining')
    model.fit(dataset, batch_size=batchsize, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks) #, workers=batchsize, use_multiprocessing=True)
                # validation_steps=1, validation_batch_size=10*batchsize)


# def main():

#     sparse_comp = True
#     if sparse_comp:
#         os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

#     rng = np.random.default_rng(1)
#     num_epochs = 10
#     num_sequences = 64
#     batchsize = 2
#     batchsize_per_step = 2
#     seq_len = 12
#     dense_sizes = [612, 512, 412]
#     sparse_sizes = [42, 32, 22]
#     decay_constant = 0.9
#     threshold = 1.0

#     assert batchsize % batchsize_per_step == 0
#     train_steps_per_execution = int(num_sequences / batchsize) # TODO or batchsize_per_step ?

#     targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
#     inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])

#     inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
#     dataset = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize)
#     train_gpu(
#         num_epochs,  
#         batchsize,
#         dataset,
#         seq_len, 
#         dense_sizes, 
#         decay_constant, 
#         threshold,
#         simple_loss_fn_dense
#     )


# if __name__ == '__main__':
#     test_sparse_vs_dense()
#     # main()



