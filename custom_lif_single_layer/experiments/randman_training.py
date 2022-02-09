import os
import functools as ft
import sys
import math
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import time
import randman

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
tf.disable_v2_behavior()

from custom_snn import sparse2dense_ipu, get_multi_layer_snn, init_func_tf, init_func_np, snn_init_weights_func


def convert_spike_times_to_sparse_raster(spike_times: np.ndarray, sparse_size: int, timestep: float = 1.0, max_time: Optional[float] = None, num_neurons: Optional[int] = None):
    """
    Convert spike times array to sparse spikes. 
    For now, all neurons must have same number of spike times.
    
    Args:
        spike_times: MoreArrays, spiketimes as array of shape batch_dim x spikes/neuron X 2
            training dim: (times, neuron_id)
    """
    if num_neurons is None:
        num_neurons = int(np.nanmax(spike_times[:,:,1]))+1
    if max_time is None:
        max_time = np.nanmax(spike_times[:,:,0])
    num_bins = int(max_time / timestep + 1)

    num_samples = spike_times.shape[0]
    spike_times_bins = (spike_times[:, :, 0] / timestep).astype(np.int32)
    spike_neuron_ids = spike_times[:, :, 1].astype(np.int32)
    spike_ids = np.empty((num_bins, num_samples, sparse_size), dtype=np.float32)
    num_spikes = np.empty((num_bins, num_samples, 1), dtype=np.int32)

    occurances = np.zeros(num_neurons+1)

    reshaped_int_spike_times = spike_times_bins.reshape((num_samples, -1, num_neurons))
    # reshaped_ids = spike_neuron_ids.reshape((num_samples, -1, num_neurons))

    num_spikes_per_neuron = reshaped_int_spike_times.shape[1]
    for i in range(num_spikes_per_neuron-1):
        for j in range(i+1, num_spikes_per_neuron):
            assert not np.any(reshaped_int_spike_times[:, i, :] == reshaped_int_spike_times[:, j, :]), "Time resolution too low. At least one neuron spikes twice in the same time-step."
    for isam in range(num_samples):
        for t in range(num_bins):
            arg_ids = np.argwhere(spike_times_bins[isam, :] == t).flatten()
            ids = spike_neuron_ids[isam, arg_ids]
            nspikes = len(ids)
            occurances[nspikes] += 1
            if nspikes > sparse_size:
                # print("more spikes than sparse")
                np.random.shuffle(ids) 
                nspikes = sparse_size
            spike_ids[t, isam, :nspikes] = ids[:nspikes]
            num_spikes[t, isam, 0] = nspikes
    return spike_ids, num_spikes, occurances


def check_spike_distribution(inp_num_spikes, occurence, inp_sparse_size, show = True):
    plt.figure()
    plt.hist(inp_num_spikes.flatten(), bins=inp_sparse_size)

    plt.figure()
    plt.yscale("log")
    plt.plot(occurence/(np.prod(inp_num_spikes.shape)), "x-")
    if show:
        plt.show()



def standardize(x,eps=1e-7):
    mi,_ = x.min(0)
    ma,_ = x.max(0)
    return (x-mi)/(ma-mi+eps)


def make_spiking_dataset(nb_classes=10, nb_units=100, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=1, 
                            nb_samples=1000, alpha=2.0, shuffle=True, classification=True, seed=None):
    """ Generates event-based generalized spiking randman classification/regression dataset. 
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work. 
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args: 
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)
    Returns: 
        A tuple of data,labels. The data is structured as numpy array 
        (sample x event x 2 ) where the last dimension contains 
        the relative [0,1] (time,unit) coordinates and labels.
    """

    data = []
    labels = []
    targets = []

    if seed is not None:
        np.random.seed(3)

    max_value = np.iinfo(np.int).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes,nb_spikes) )

    for k in range(nb_classes):
        x = np.random.rand(nb_samples,dim_manifold)
        submans = [ randman.Randman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k,i]) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(np.repeat(np.arange(nb_units).reshape(1,-1),nb_samples,axis=0))
            times.append(y.numpy())

        units = np.concatenate(units,axis=1)
        times = np.concatenate(times,axis=1)
        events = np.stack([times,units],axis=2)
        data.append(events)
        labels.append(k*np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=np.int)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:,:,0] *= nb_steps*step_frac
    # data = np.array(data, dtype=int)

    if classification:
        return data, labels
    else:
        return data, targets


def calc_loss(out_spikes_dense, targets_one_hot):
    sum_spikes = tf.math.reduce_sum(out_spikes_dense, axis=0)
    # log_softmax_pred = tf.nn.log_softmax(sum_spikes, axis=1)

    # norm_sum_spikes = tf.nn.softmax(sum_spikes, axis=1)
    # norm_sum_spikes = sum_spikes / tf.math.reduce_sum(sum_spikes, axis=1)[:, None]

    return tf.math.reduce_sum((sum_spikes-targets_one_hot)**2) / float(out_spikes_dense.shape[0])
    # return tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=sum_spikes, labels=targets_one_hot))


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

def activity_reg_lower(lambda_lower, nu_lower, spikes):
    return lambda_lower * tf.math.reduce_mean( tf.nn.relu(nu_lower - spikes)**2 )

def activity_reg_upper(lambda_upper, nu_upper, spikes):
    return lambda_upper * tf.nn.relu(tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean(spikes, axis=2), axis=0) - nu_upper))**2

def get_reg_function(lambda_lower, nu_lower, lambda_upper, nu_upper):

    activity_reg_lower_ = ft.partial(activity_reg_lower, lambda_lower, nu_lower)
    activity_reg_upper_ = ft.partial(activity_reg_upper, lambda_upper, nu_upper)

    def reg_func(spikes_pytree):
        g_lowers = [activity_reg_lower_(spikes) for spikes in spikes_pytree]
        g_uppers = [activity_reg_upper_(spikes) for spikes in spikes_pytree]
        g_sum_leaves = [x+y for x, y in zip(g_lowers, g_uppers)]
        num_layers = len(g_sum_leaves)
        loss_reg = tf.math.reduce_mean(tf.concatenate(g_sum_leaves)) if num_layers > 1 else g_sum_leaves[0]
        return loss_reg

    return reg_func



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


def main():
    rng = np.random.default_rng(1)
    num_epochs = 10
    num_classes = 2
    seq_len = 20
    batchsize = 32
    batchsize_val = 64

    num_spikes_per_inp_neuron = 1
    num_samples_per_class = 1024
    num_samples_per_class_val = 256

    num_samples = num_classes * num_samples_per_class
    num_samples_val = num_classes * num_samples_per_class_val
    num_batches = int(num_samples / batchsize)


    learning_rate = 1e-1
    momentum_alpha = 0.9
    sizes_dense = [256, 128, num_classes]
    sizes_sparse = [256, 128, num_classes]

    decay_constant = 0.95
    threshold = 1.0

    name_add = "_dense256_128"
    savefigs = False

    # lambda_lower=100
    # nu_lower=1e-3
    # lambda_upper=100
    # nu_upper=15

    assert len(sizes_sparse) == len(sizes_dense)
    assert all(s <= d for s,d in zip(sizes_sparse, sizes_dense))


    # TODO not stable for small value for `nb_units`
    data, labels = make_spiking_dataset(nb_classes=num_classes, nb_units=sizes_dense[0], nb_steps=seq_len, step_frac=1.0, dim_manifold=2, 
                            nb_spikes=num_spikes_per_inp_neuron, nb_samples=num_samples_per_class, alpha=2.0, shuffle=True, classification=True, seed=rng.integers(9999999999))
    inp_spike_ids, inp_num_spikes, occurence = convert_spike_times_to_sparse_raster(data, sizes_sparse[0])

    data_val, labels_val = make_spiking_dataset(nb_classes=num_classes, nb_units=sizes_dense[0], nb_steps=seq_len, step_frac=1.0, dim_manifold=2, 
                            nb_spikes=num_spikes_per_inp_neuron, nb_samples=num_samples_per_class_val, alpha=2.0, shuffle=True, classification=True, seed=rng.integers(9999999999))
    inp_spike_ids_val, inp_num_spikes_val, occurence_val = convert_spike_times_to_sparse_raster(data_val, sizes_sparse[0])
    labels_val = labels_val.astype(np.int32)
    
    # print(inp_spike_ids[:, 0])
    # print("\n-------------------------------------\n")
    # print(inp_num_spikes[:, 0].flatten())
    # print("\n-------------------------------------\n")
    # print(data[0])
    
    
    # check_spike_distribution(inp_num_spikes, occurence, sizes_sparse[0], show=True)
    # sys.exit()

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()


    with tf.device("cpu"):
        weights_pl, init_states_pl, inp_spike_ids_pl, num_inp_spikes_pl, decay_constants_pl, thresholds_pl = init_func_tf(sizes_dense, batchsize, seq_len, sizes_sparse[0])
        opt_state_pl = tuple(tf.placeholder(tf.float32, ws.shape) for ws in weights_pl)
        targets_one_hot_pl = tf.placeholder(tf.float32, [batchsize, num_classes])
        _, init_states_pl_val, inp_spike_ids_pl_val, num_inp_spikes_pl_val, _, _ = init_func_tf(sizes_dense, batchsize_val, seq_len, sizes_sparse[0])
        targets_pl_val = tf.placeholder(tf.int32, [batchsize_val])

    init_weight_func = ft.partial(snn_init_weights_func, rng, decay_constant, threshold)
    weights, init_states, decay_constants, thresholds = init_func_np(sizes_dense, batchsize, decay_constant, threshold, init_weight_func)
    _, init_states_val, _, _ = init_func_np(sizes_dense, num_samples_val, decay_constant, threshold, init_weight_func)


    snn_func = get_multi_layer_snn(sizes_sparse[1:])
    reg_fn = None #get_reg_function(lambda_lower, nu_lower, lambda_upper, nu_upper)
    opt_state, opt = get_sgd(weights, learning_rate, momentum_alpha)

    update_func = get_update_func(
        loss_and_grad_func=get_calc_loss_and_grad(snn_func, calc_loss, reg_fn),
        optimizer=opt,
    )
    normalized_acc = True
    acc_func = get_accuracy_func(snn_func, normalized_acc)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(update_func, [weights_pl, init_states_pl, inp_spike_ids_pl, num_inp_spikes_pl, decay_constants_pl, thresholds_pl, targets_one_hot_pl, opt_state_pl])
        xla_acc = ipu.ipu_compiler.compile(acc_func, [weights_pl, init_states_pl_val, inp_spike_ids_pl_val, num_inp_spikes_pl_val, decay_constants_pl, thresholds_pl, targets_pl_val])

    loss_vals = np.empty((num_epochs, num_batches))
    accuracies = np.empty((num_epochs))   
    times = np.empty((num_epochs, num_batches))
    num_spikes_arr = np.empty((num_epochs, num_batches))
    sample_idxs = np.arange(num_samples)
    with tf.Session() as sess:
        for epoch in range(num_epochs):
            rng.shuffle(sample_idxs)
            for ibatch in range(num_batches):

                batch_idxs = sample_idxs[ibatch*batchsize:(ibatch+1)*batchsize]
                inp_spike_ids_batch = inp_spike_ids[:, batch_idxs]
                inp_num_spikes_batch = inp_num_spikes[:, batch_idxs]
                target_batch = labels[batch_idxs]
                targets_one_hot_batch = tf.one_hot(target_batch, num_classes, axis=1).eval()

                t_start = time.time()
                (loss, aux), weights, opt_state = sess.run(xla_result, {
                    weights_pl:weights, 
                    init_states_pl:init_states, 
                    inp_spike_ids_pl:inp_spike_ids_batch, 
                    num_inp_spikes_pl:inp_num_spikes_batch, 
                    decay_constants_pl:decay_constants, 
                    thresholds_pl:thresholds, 
                    targets_one_hot_pl:targets_one_hot_batch,
                    opt_state_pl:opt_state
                })
                t_end = time.time()
                times[epoch, ibatch] = t_end-t_start

                pred_spike_ids, pred_num_spikes, pred_spikes_dense = aux

                # # print(weights[0][0])
                # print(pred_num_spikes.sum(axis=0)[:, 0])

                print(epoch, ibatch, loss, pred_num_spikes.sum())
                loss_vals[epoch,ibatch] = loss
                num_spikes_arr[epoch,ibatch] = pred_num_spikes.mean()




            # num_batches_val = math.ceil(num_samples_val / batchsize_val)
            # print(num_samples_val, batchsize_val, num_batches_val)
            # if num_batches_val > 1:
            #     start_idx = 0        
            #     res = 0.0
            #     for ibatch in range(num_batches_val):
            #         stop_idx = min(start_idx+batchsize_val, num_samples)
            #         res_ibatch = sess.run(xla_acc, {
            #             weights_pl:weights, 
            #             init_states_pl_val:[stat[start_idx:stop_idx] for stat in init_states_val], 
            #             inp_spike_ids_pl_val:inp_spike_ids_val[:, start_idx:stop_idx], 
            #             num_inp_spikes_pl_val:inp_num_spikes_val[:, start_idx:stop_idx], 
            #             decay_constants_pl:decay_constants, 
            #             thresholds_pl:thresholds, 
            #             targets_pl_val:labels_val[start_idx:stop_idx]
            #         })[0]
            #         if normalized_acc:
            #             size_ibatch = stop_idx - start_idx
            #             res += res_ibatch * size_ibatch
            #         else:
            #             res += res_ibatch
            #         start_idx = stop_idx
                
            #     if normalized_acc:
            #         res /= num_samples_val
            # else:
            #     res = sess.run(xla_acc, {
            #             weights_pl:weights, 
            #             init_states_pl_val:init_states_val, 
            #             inp_spike_ids_pl_val:inp_spike_ids_val, 
            #             num_inp_spikes_pl_val:inp_num_spikes_val, 
            #             decay_constants_pl:decay_constants, 
            #             thresholds_pl:thresholds, 
            #             targets_pl_val:labels_val
            #     })[0]

            # accuracies[epoch] = res
            # print(epoch, accuracies[epoch])


    print(loss_vals)
    print(accuracies)
    print(times)

    plt.figure()
    plt.plot(loss_vals.flatten())
    plt.ylabel("loss")
    if savefigs:
        plt.savefig(f"timings/randman_ipu_loss{name_add}.pdf")

    # plt.figure()
    # plt.plot(accuracies.flatten(), "x-")
    # plt.ylabel("accuracy")
    # if savefigs:
        # plt.savefig(f"timings/randman_ipu_acc{name_add}.pdf")

    plt.figure()
    plt.plot(times.flatten()[1:], "x-")
    plt.hlines(times.flatten()[1:].mean(), 0, times.size-1, "black")
    plt.ylabel("time")
    if savefigs:
        plt.savefig(f"timings/randman_ipu_time{name_add}.pdf")

    plt.figure()
    plt.plot(num_spikes_arr.flatten(), "x-")
    plt.ylabel("mean num spikes")
    if savefigs:
        plt.savefig(f"timings/randman_ipu_nspikes{name_add}.pdf")

    plt.show()

if __name__ == "__main__":

    main()
