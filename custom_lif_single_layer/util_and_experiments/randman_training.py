import os
import functools as ft
import sys
import math
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import time

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
tf.disable_v2_behavior()

from custom_snn import sparse2dense_ipu, get_multi_layer_snn, init_func_tf, init_func_np, snn_init_weights_func, get_sgd, get_update_func, get_calc_loss_and_grad, get_accuracy_func
from util_randman import make_spiking_dataset, convert_spike_times_to_sparse_raster, check_spike_distribution

def calc_loss(out_spikes_dense, targets_one_hot):
    sum_spikes = tf.math.reduce_sum(out_spikes_dense, axis=0)
    # log_softmax_pred = tf.nn.log_softmax(sum_spikes, axis=1)

    # norm_sum_spikes = tf.nn.softmax(sum_spikes, axis=1)
    # norm_sum_spikes = sum_spikes / tf.math.reduce_sum(sum_spikes, axis=1)[:, None]

    return tf.math.reduce_sum((sum_spikes-targets_one_hot)**2) / float(out_spikes_dense.shape[0])
    # return tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=sum_spikes, labels=targets_one_hot))

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

    init_weight_func = ft.partial(snn_init_weights_func, rng, threshold, num_spikes_per_inp_neuron/seq_len)
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
