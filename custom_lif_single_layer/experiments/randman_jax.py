import sys
import functools as ft
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map, tree_multimap
import matplotlib.pyplot as plt

from jax_snn import lif_layer_jax
from randman_training import make_spiking_dataset
from custom_snn import init_func_np, snn_init_weights_func

def create_one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[..., None] == jnp.arange(k), dtype)


def convert_spike_times_to_raster(spike_times: np.ndarray, timestep: float = 1.0, max_time: Optional[float] = None, num_neurons: Optional[int] = None, dtype=None):
    """
    Convert spike times array to spike raster array. 
    For now, all neurons need to have same number of spike times.
    
    Args:
        spike_times: MoreArrays, spiketimes as array of shape batch_dim x spikes/neuron X 2
            training dim: (times, neuron_id)
    """

    if dtype is None:
        dtype = np.int16
    # spike_times = spike_times.astype(np.uint16)
    if num_neurons is None:
        num_neurons = int(np.nanmax(spike_times[:,:,1]))+1
    if max_time is None:
        max_time = np.nanmax(spike_times[:,:,0])
    num_bins = int(max_time / timestep + 1)

    spike_raster = np.zeros((spike_times.shape[0], num_bins, num_neurons), dtype=dtype)
    batch_id = np.arange(spike_times.shape[0]).repeat(spike_times.shape[1])
    spike_times_flat = (spike_times[:, :, 0].flatten() / timestep).astype(dtype)
    neuron_ids = spike_times[:, :, 1].flatten().astype(dtype)
    np.add.at(spike_raster, (batch_id, spike_times_flat, neuron_ids), 1)
    return spike_raster


def multi_layer_lif(weights, init_states, inp_spikes, decay_constants, thresholds):
    spikes = inp_spikes
    for ws, decay_consts, threshs, init_stat in zip(weights, decay_constants, thresholds, init_states):
        states, spikes = lif_layer_jax(ws, init_stat, spikes, decay_consts, threshs)
        # states, spikes = vmap(lif_layer_jax, in_axes=(None, 0, 1, None, None), out_axes=(1, 1))(ws, init_stat, spikes, decay_consts, threshs)
    return states, spikes

multi_layer_lif_vmap = vmap(multi_layer_lif, in_axes=(None, 0, 1, None, None), out_axes=(1, 1))


def calc_loss(weights, init_states, inp_spikes, decay_constants, thresholds, targets_one_hot):
    spikes = inp_spikes
    
    print(tree_map(lambda x: x.shape, init_states))
    print(tree_map(lambda x: x.shape, inp_spikes))

    out_states, out_spikes = multi_layer_lif_vmap(weights, init_states, inp_spikes, decay_constants, thresholds)
    # out_states, out_spikes = multi_layer_lif(weights, init_states, inp_spikes, decay_constants, thresholds)

    sum_spikes = jnp.sum(out_spikes, axis=0)
    # norm_sum_spikes = jax.nn.softmax(sum_spikes, axis=1)
    # norm_sum_spikes = sum_spikes / jnp.sum(sum_spikes, axis=1)[:, None]
   

    loss_task = jnp.sum((sum_spikes-targets_one_hot)**2) / float(out_spikes.shape[0])
    loss = loss_task
    
    return loss, (out_states, spikes)


def get_sgd(init_weights, learning_rate, alpha=None):

    def sgd(state, grads):
        return state, tree_map(lambda x: -learning_rate*x, grads)

    def sgd_with_mom(state, grads):
        update = tree_multimap(lambda st,gs: alpha*st - learning_rate*gs, state, grads)
        return update, update

    init_state = tree_map(jnp.zeros_like, init_weights)

    method = sgd if alpha is None else sgd_with_mom
    # method = lambda st, ws: tree_multimap(method, st, ws)

    return init_state, method



def get_update_fn(opt):

    @jit
    def update(weights, init_states, inp_spikes, decay_constants, thresholds, targets_one_hot, opt_state):

        (loss, aux) , grads = value_and_grad(calc_loss, has_aux=True)(weights, init_states, inp_spikes, decay_constants, thresholds, targets_one_hot)
        
        opt_state, update = opt(opt_state, grads)
        new_weights = tree_multimap(lambda ws,up: ws+up, weights, update)

        return (loss, aux), new_weights, opt_state
    return update


@ft.partial(jit, static_argnums=6)
def calc_accuracy(weights, init_states, inp_spikes, decay_constants, thresholds, target, normalized=True):
    _, spikes_out = multi_layer_lif_vmap(weights, init_states, inp_spikes, decay_constants, thresholds)
    pred_sum_spikes = jnp.sum(spikes_out, axis=0)
    pred = jnp.argmax(pred_sum_spikes, axis=1)

    if normalized:
        return (pred==target).mean()
    else:
        return (pred==target).sum()



def main():
    rng = np.random.default_rng(1)
    num_epochs = 40
    num_classes = 4
    seq_len = 20
    batchsize = 32
    # batchsize = 8*10
    num_spikes_per_inp_neuron = 1
    num_samples_per_class = 1024
    num_samples_per_class_val = 64

    num_samples = num_classes * num_samples_per_class
    num_samples_val = num_classes * num_samples_per_class_val
    num_batches = int(num_samples / batchsize)


    learning_rate = 1e-1
    momentum_alpha = 0.9
    sizes_dense = [256, 128, num_classes]
    # sizes_dense = [64, 32, num_classes]

    decay_constant = 0.95
    threshold = 1.0

    lambda_lower=100
    nu_lower=1e-3
    lambda_upper=100
    nu_upper=15

    # TODO not stable for small value for `nb_units`
    data, labels = make_spiking_dataset(nb_classes=num_classes, nb_units=sizes_dense[0], nb_steps=seq_len, step_frac=1.0, dim_manifold=2, 
                            nb_spikes=num_spikes_per_inp_neuron, nb_samples=num_samples_per_class, alpha=2.0, shuffle=True, classification=True, seed=rng.integers(9999999999))
    inp_spikes = convert_spike_times_to_raster(data).transpose(1, 0, 2)
    print(inp_spikes.shape)
    # sys.exit()

    data_val, labels_val = make_spiking_dataset(nb_classes=num_classes, nb_units=sizes_dense[0], nb_steps=seq_len, step_frac=1.0, dim_manifold=2, 
                            nb_spikes=num_spikes_per_inp_neuron, nb_samples=num_samples_per_class_val, alpha=2.0, shuffle=True, classification=True, seed=rng.integers(9999999999))
    inp_spikes_val = convert_spike_times_to_raster(data_val).transpose(1, 0, 2)


    


    init_weight_func = ft.partial(snn_init_weights_func, rng, decay_constant, threshold)
    weights, init_states, decay_constants, thresholds = init_func_np(sizes_dense, batchsize, decay_constant, threshold, init_weight_func)
    _, init_states_val, _, _ = init_func_np(sizes_dense, num_samples_val, decay_constant, threshold, init_weight_func)

    opt_state, opt = get_sgd(weights, learning_rate, alpha=momentum_alpha)
    update_fn = get_update_fn(opt)


    loss_vals = np.empty((num_epochs, num_batches))
    accuracies = np.empty((num_epochs))
    sample_idxs = np.arange(num_samples)
    for epoch in range(num_epochs):
        rng.shuffle(sample_idxs)
        for ibatch in range(num_batches):

            batch_idxs = sample_idxs[ibatch*batchsize:(ibatch+1)*batchsize]
            inp_spikes_batch = inp_spikes[:, batch_idxs]
            target_batch = labels[batch_idxs]
            targets_one_hot_batch = create_one_hot(target_batch, num_classes)
            # targets_one_hot.eval(session=tf.compat.v1.Session())    

            # print("Hi")
            # print(inp_spike_ids_batch.shape)
            # print(inp_num_spikes_batch.shape)
            # print(targets_one_hot_batch.shape)

            (loss, aux), weights, opt_state = update_fn(weights, init_states, inp_spikes_batch, decay_constants, thresholds, targets_one_hot_batch, opt_state)
            loss_vals[epoch, ibatch] = loss
    
        accuracies[epoch] = calc_accuracy(weights, init_states_val, inp_spikes_val, decay_constants, thresholds, labels_val)
    
    print("\nlosses")
    print(loss_vals)
    print("\naccuracies")
    print(accuracies)


    plt.figure()
    plt.plot(loss_vals.flatten())
    plt.ylabel("loss")

    plt.figure()
    plt.plot(accuracies.flatten(), "x-")
    plt.ylabel("accuracy")

    plt.show()



if __name__ == "__main__":
    main()
