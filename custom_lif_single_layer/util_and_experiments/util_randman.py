from typing import Optional
import numpy as np
import randman

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
    import matplotlib.pyplot as plt
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