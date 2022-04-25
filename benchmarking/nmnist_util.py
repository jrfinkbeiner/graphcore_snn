import functools as ft
import bisect
import numpy as np
import tensorflow as tf


import tonic
import tonic.transforms as transforms
# from tfneuromorphic import nmnist

def find_first(a, tgt):
    '''
    returns the first element of tgt that is larger than a
    '''
    return bisect.bisect_left(a, tgt)


def get_tmad_slice(times, addrs, start_time, seq_len, ds_tm=1, ds_ad=1):
    '''
    Slices dataset to seq_len, return timestamp -- address array (e.g. tm, ad0, ad1, ad2 ...)
    '''
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time+seq_len)+idx_beg
        return np.column_stack([times[idx_beg:idx_end]//ds_tm, addrs[idx_beg:idx_end]//ds_ad])
    except IndexError:
        raise IndexError("Empty batch found")

def events_to_sparse_tensors(events,
                     deltat=1000,
                     seq_len=500,
                     ds_w=1,
                     ds_h=1,
                     sparse_size=128):

    times = events["t"]
    addrs = np.stack((events["p"], events["x"], events["y"]), axis=1) # TODO which order ?
    # # addrs = events[:, ["x", "y", "p"]]


    # print("events")
    # print(events)
    # print(times)
    # print(events["x"])
    # print(type(events))
    # # print(events.names)
    # print(events.shape)
    # print(addrs.shape)
    # import sys
    # sys.exit()

    n_dims = addrs.shape[1]
    t_start = times[0]
    ts = range(t_start, t_start + seq_len * deltat, deltat)
    data = np.zeros([seq_len, sparse_size, n_dims], dtype='int16')
    idx_start = 0
    idx_end = 0
    diff=0
    num_events  = np.zeros([seq_len], dtype='int16')
    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end > idx_start:
            ee = addrs[idx_start:idx_end]
            #pol, x, y = ee[:, 0], (ee[:, 1] // ds_w).astype('int16'), (ee[:, 2] // ds_h).astype('int16')

            l = len(ee)
            if l>sparse_size:
                diff += len(ee)-sparse_size
                choose = np.arange(l)
                np.random.shuffle(choose)

                choose = choose[:sparse_size]
                data[i,:sparse_size,:] = ee[choose,:]
                num_events[i] = sparse_size
            else:
                data[i,:l] = ee
                num_events[i] = l

        idx_start = idx_end
    return data, num_events

# def generate_tonic_nmnist_dataset():
#     import tonic
#     import tonic.transforms as transforms

#     sensor_size = tonic.datasets.NMNIST.sensor_size

#     transform_train = transforms.Compose([
#         # transforms.Crop(target_size=(28,28)),
#         # transforms.Denoise(filter_time=10000),
#         # transforms.TimeJitter(std=10),
#         # transforms.SpatialJitter(
#         #     variance_x=0.3, # TODO originally 2
#         #     variance_y=0.3, # TODO originally 2
#         #     clip_outliers=True
#         # ),
#         transforms.ToFrame(sensor_size, time_window=1000.0),
#         # transforms.ToFrame(n_time_bins=1000),
#     ])
#     transform_test = transforms.Compose([
#         # transforms.Denoise(filter_time=10000),
#         transforms.ToFrame(sensor_size, time_window=1000.0),
#     ])

#     dataset_train = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
#                                     train=True,
#                                     # transform=transform_train,
#                                     first_saccade_only=True)
#     dataset_test = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
#                                     train=False,
#                                     transform=transform_test,
#                                     first_saccade_only=True)
#     return dataset_train, dataset_test


def create_nmnist_gener(root, sparse, seq_len=300, sparse_size=None, num_samples=None, dataset='train', shuffle=None):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a generator function with yields data, num_events, target
    target: integer
    data: flattened float32 array of dimension seq_len x prod(sersor_size) containing flattened event addresses
    '''
    assert dataset in ['train','val','test']
    
    if sparse:
        assert sparse_size is not None, "For `sparse=True`, `sparse_size` must be given, got `None`."

    if shuffle is None:
        shuffle = True if dataset == 'train' else False
    
    if dataset == 'val':
        raise NotImplementedError()
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    if sparse:
        transform_train = transforms.Compose([
            # transforms.ToFrame(sensor_size, time_window=1000.0),
            ft.partial(events_to_sparse_tensors, deltat=1000,
                                        seq_len=seq_len,
                                        sparse_size=sparse_size)
        ])
    else:
        transform_train = transforms.Compose([
            # transforms.ToFrame(sensor_size, time_window=1000.0),
            transforms.ToFrame(sensor_size, n_time_bins=seq_len),
        ])
        # transform_test = transforms.Compose([
        #     # transforms.Denoise(filter_time=10000),
        #     # transforms.ToFrame(sensor_size, time_window=1000.0),
        #     transforms.ToFrame(sensor_size, n_time_bins=seq_len),
        # ])


    dataset = tonic.datasets.NMNIST(save_to=root,
                                train=dataset == 'train',
                                transform=transform_train,
                                first_saccade_only=False) # TODO decide for first saccade... has to match sparse implementation...

    if num_samples is None:
        num_samples = len(dataset)
    
    # idx_samples = np.arange(num_samples) 
    # idx_samples = np.arange(num_samples) 
    idx_samples = np.random.choice(len(dataset), num_samples, replace=False)
    
    def gen_dense():    
        if shuffle: np.random.shuffle(idx_samples)
        for i in idx_samples:
            data, label = dataset[i]
            data_flat = data.reshape(seq_len, -1).astype(np.float32)
            # yield data_flat, label
            yield {"inp_spikes": data_flat, "targets": label}
            # yield data_flat, label

    def gen_sparse():    
        if shuffle: np.random.shuffle(idx_samples)
        for i in idx_samples:
            data, label = dataset[i]
            yield {"inp_spike_ids": data[0].astype(np.float32), "num_inp_spikes": data[1].astype(np.int32), "targets": label}

    gen = gen_sparse if sparse else gen_dense

    
    # def get_dense(idx):
    #     idx = int(idx)
    #     data, label = dataset[idx]
    #     data_flat = data.reshape(seq_len, -1).astype(np.float32)
    #     return {"inp_spikes": data_flat, "targets": label}

    # def get_sparse(idx):
    #     print(idx)
    #     # idx = int(idx)
    #     # idx = tf.get_static_value(idx, partial=False)
    #     idx = idx.numpy()
    #     print(idx)
    #     data, label = dataset[idx]

    #     ds = np.array([np.prod(sensor_size[i:]) for i in range(1,len(sensor_size))] + [1], dtype=np.int16)
    #     spike_ids_flat = tf.math.reduce_sum([spike_ids[:,:,i]*d for i,d in enumerate(ds)],axis=0)

    #     return {"inp_spike_ids": spike_ids_flat.astype(np.float32), "num_inp_spikes": data[1].astype(np.int32), "targets": label}

    # gen = get_sparse if sparse else get_dense


    return gen, num_samples


# def cast_data_tf_dense(data, label):
#     return tf.cast(data, tf.float32), tf.cast(label, tf.float32)

def get_nmnist_dataset(root, sparse, seq_len, inp_dim, batchsize, sparse_size=None, num_samples=None, dims=None):
    # from snnax.utils.data import SequenceLoader
    gen_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples)
    # get_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples)
    
    # dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
    #                                                                         tf.TensorSpec(shape=(), dtype=tf.int32))))

    if dims is None:
        dims = (34,34,2)
    flatten_fn = ft.partial(flatten_data_tf, dims=dims)
    if sparse:
        dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spike_ids": tf.TensorSpec(shape=(seq_len, inp_dim, len(dims)), dtype=tf.float32),
                                                                                "num_inp_spikes": tf.TensorSpec(shape=(seq_len, ), dtype=tf.int32),
                                                                                "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})
    else:
        dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spikes": tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
                                                                                "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})

    # idx = np.arange(num_samples)
    # # dataset = tf.data.Dataset.from_tensor_slices(idx)
    # dataset = tf.data.Dataset.range(num_samples) # .as_numpy_iterator()

    if sparse:
        # TODO perform transform here instead of inside tonic dataset?, makes use of `num_parallel_calls`
        dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # # dataset = dataset.map(cast_data_tf_dense, num_parallel_calls=batchsize)
    # dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False) # TODO uncomment
    # # dataset = dataset.map(get_train, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.repeat()
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def flatten_data_tf(data, dims):
    '''
    Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
    dims: tuple whose elements specify the size of each dimension
    '''
    spike_ids = data["inp_spike_ids"]
    print(spike_ids)

    ds = np.array([np.prod(dims[i:]) for i in range(1,len(dims))] + [1], dtype=np.int16)
    spike_ids_flat = tf.math.reduce_sum([spike_ids[:,:,i]*d for i,d in enumerate(ds)],axis=0)
    data["inp_spike_ids"] = spike_ids_flat
    return data

# def cast_data_tf(input_data, label):
#     data, num_events = input_data
# # def cast_data_tf(data, num_events, label):
#     return (tf.cast(data, tf.float32), tf.expand_dims(tf.cast(num_events, tf.int32), axis=-1)), tf.cast(label, tf.int32) # cast label only because of IPUModel

# def get_nmnist_dataset(hdf5_filepath, sparse_size, num_samples, batchsize, dims=None, seq_len=300, sparse=True):

#     assert sparse, "Currently only the sparse nmnist dataset generator is implemented"

#     if dims is None:
#         dims = (2,34,34)
#     dtype = tf.int16

#     # gen_train = nmnist.create_gener('/Data/pgi-15/datasets/nmnist/n_mnist.hdf5', dataset='train', sparse_size = sparse_size, num_samples=num_samples, seq_len=seq_len)
#     gen_train = nmnist.create_gener(hdf5_filepath, dataset='train', sparse_size = sparse_size, num_samples=num_samples, seq_len=seq_len)
#     # it = gen_train()
#     # data, num_events, label = next(it)
#     # data_flat, _, _ = nmnist.flatten_data(data, num_events, label, dims = dims)
#     #  = nmnist.sparse_vector_to_dense(data, num_events, dims = dims)
    
#     dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, sparse_size, len(dims)), dtype=dtype),
#                                                                            tf.TensorSpec(shape=(seq_len), dtype=dtype)),
#                                                                           tf.TensorSpec(shape=(), dtype=dtype)))

#     flatten_fn = ft.partial(flatten_data_tf, dims=dims)
#     print(dataset)

#     # dataset = dataset.shuffle()
#     # dataset = dataset.prefetch(16) # TODO why 16 ?
#     # dataset = dataset.map(nmnist.flatten_data_tf)
#     dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.map(cast_data_tf, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.repeat()
#     dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
#     dataset = dataset.batch(batchsize, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset