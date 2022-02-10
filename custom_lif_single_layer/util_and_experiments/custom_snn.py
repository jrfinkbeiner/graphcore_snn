import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

from tensorflow.python import ipu
tf.disable_v2_behavior()


def snn_init_weights_func(rng, u_thresh, in_firing_rate, in_features, out_features):
    mean = 0.0
    thresh_dist_fac = 1.
    postsyn_pot_kernal_variance = 0.5
    variance = (u_thresh / thresh_dist_fac)**2 / (in_features * in_firing_rate * postsyn_pot_kernal_variance)
    weights = rng.normal(size=(out_features, in_features)).astype(np.float32)
    weights = variance**0.5 * weights + mean
    return weights


def init_func_tf(num_neurons, batchsize, seq_len, size_sparse_in):
    num_layers = len(num_neurons) - 1
    weights = tuple(tf.placeholder(np.float32, [num_neurons[i+1], num_neurons[i]]) for i in range(num_layers))
    init_state = tuple(tf.placeholder(np.float32, [batchsize, nneurons]) for nneurons in num_neurons[1:])
    inp_spike_ids = tf.placeholder(np.float32, [seq_len, batchsize, size_sparse_in])
    num_inp_spikes = tf.placeholder(np.int32, [seq_len, batchsize, 1])
    decay_constants = tuple(tf.placeholder(np.float32, [nneurons]) for nneurons in num_neurons[1:])
    thresholds = tuple(tf.placeholder(np.float32, [nneurons]) for nneurons in num_neurons[1:])
    return weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds


def init_func_np(num_neurons, batchsize, decay_constant, threshold, init_weight_func):
    num_layers = len(num_neurons) - 1
    weights = tuple(init_weight_func(num_neurons[i], num_neurons[i+1]) for i in range(num_layers))
    init_state = tuple(np.zeros((batchsize, nneurons), dtype=np.float32) for nneurons in num_neurons[1:])
    decay_constants = tuple(decay_constant * np.ones((nneurons), dtype=np.float32) for nneurons in num_neurons[1:])
    thresholds = tuple(threshold * np.ones((nneurons), dtype=np.float32) for nneurons in num_neurons[1:])
    return weights, init_state, decay_constants, thresholds
    

def sparse2dense_ipu(spike_ids, num_spikes, dense_size: int):
    assert len(spike_ids.shape) == 3
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], dense_size])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "sparse2dense", "libcustom_op.so")
    gp_path = os.path.join(base_path, "..", "sparse2dense", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{int(dense_size)}",
                                            )


def custom_lif_layer(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds, size_sparse_out: int):

    batch_and_seq_size = num_inp_spikes.shape[:2]
    seq_len = num_inp_spikes.shape[0]

    outputs = {
        "output_types": [inp_spike_ids.dtype, inp_spike_ids.dtype, init_state.dtype],
        "output_shapes": [tf.TensorShape([*batch_and_seq_size, size_sparse_out]), tf.TensorShape([*batch_and_seq_size, 1]), tf.TensorShape([seq_len, *init_state.shape])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "custom_lif_layer", "libcustom_op.so")
    gp_path = os.path.join(base_path, "..", "custom_lif_layer", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{size_sparse_out}",
                                            )


def get_multi_layer_snn(sparse_out_sizes):

    def multi_layer_snn(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):
        with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
            spike_ids, num_spikes = inp_spike_ids, num_inp_spikes
            for ws, decay_consts, threshs, init_stat, sparse_out_size in zip(weights, decay_constants, thresholds, init_states, sparse_out_sizes):
                spike_ids, num_spikes, states = custom_lif_layer(ws, init_stat, spike_ids, num_spikes, decay_consts, threshs, sparse_out_size)
                num_spikes = tf.cast(num_spikes, tf.int32)
            return spike_ids, num_spikes

    return multi_layer_snn


def get_calc_loss_and_grad(snn_func, loss_fn, reg_fn=None):

    def calc_loss_and_grad(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds, targets_one_hot):
        pred_spike_ids, pred_num_spikes = snn_func(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds)
        pred_spikes_dense = sparse2dense_ipu(pred_spike_ids, pred_num_spikes, weights[-1].shape[0])[0]
        loss_task = loss_fn(pred_spikes_dense, targets_one_hot)
        loss_reg = reg_fn([pred_spikes_dense]) if reg_fn is not None else 0
        loss = loss_task + loss_reg
        grads = tf.gradients(loss, [*weights])
        # return (loss, (pred_spike_ids, pred_num_spikes, pred_spikes_dense)), grads
        return (loss_task, (pred_spike_ids, pred_num_spikes, pred_spikes_dense)), grads

    return calc_loss_and_grad


def get_update_func(loss_and_grad_func, optimizer):

    def update_func(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds, targets_one_hot, opt_state):
        (loss, aux) , grads = loss_and_grad_func(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds, targets_one_hot)
        opt_state, updates = optimizer(opt_state, grads)
        updated_weights = [ws+up for ws,up in zip(weights, updates)]
        return (loss, aux), updated_weights, opt_state

    return update_func


def get_accuracy_func(snn_func, normalized=True):

    def calc_accuracy(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds, target):
        pred_spike_ids, pred_num_spikes = snn_func(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds)
        pred_spikes_dense = sparse2dense_ipu(pred_spike_ids, pred_num_spikes, weights[-1].shape[0])[0]
        pred_sum_spikes = tf.math.reduce_sum(pred_spikes_dense, axis=0)

        pred = tf.math.argmax(pred_sum_spikes, axis=1)
        target = tf.cast(target, pred.dtype)
        comp = tf.cast(tf.math.equal(pred,target), tf.float32)
        res = tf.math.reduce_mean(comp) if normalized else tf.math.reduce_sum(comp)
        return res, pred, target, comp

    return calc_accuracy


def get_sgd(init_weights, learning_rate, alpha=None):

    def sgd(state, grads):
        return state, [-learning_rate*x for x in  grads]

    def sgd_with_mom(state, grads):
        update =  [alpha*st - learning_rate*gs for st,gs in zip(state, grads)]
        return update, update

    init_state = [np.zeros_like(x) for x in  init_weights]
    method = sgd if alpha is None else sgd_with_mom
    return init_state, method