import functools as ft
from typing import Optional, Callable
import bisect
import numpy as np
import tensorflow as tf

from multiprocessing import Pool, cpu_count
from multi_proc_helper import set_global_dataset, get_dataset_item

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


from dataclasses import dataclass
@dataclass(frozen=True)
class TimeSlice:
    seq_len: int
    def __call__(self, sequence):
        # print(sequence.shape)
        slice = np.zeros((self.seq_len, *sequence.shape[1:]))
        seq_to_use = min(self.seq_len, sequence.shape[0])
        slice[:seq_to_use] = sequence[:seq_to_use]
        return slice

def create_nmnist_dataset(root, sparse, seq_len=300, sparse_size=None, dataset='train', apply_flatten=False, delta_t=1000):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a `tonic.datasets.NMNIST` instance
    '''
    assert dataset in ['train','val','test']
    
    if sparse:
        assert sparse_size is not None, "For `sparse=True`, `sparse_size` must be given, got `None`."

    if dataset == 'val':
        raise NotImplementedError()
    
    sensor_size = tonic.datasets.NMNIST.sensor_size

    if sparse:
        transforms_list = [
            transforms.Denoise(filter_time=10000),
            ft.partial(events_to_sparse_tensors, deltat=delta_t,
                            seq_len=seq_len,
                            sparse_size=sparse_size),
        ]
        if apply_flatten:
            def flatten_fn(dims, data):
                return flatten_spike_ids(dims, data[0]), data[1]
            transforms_list.append(ft.partial(flatten_fn, sensor_size))

        transform_train = transforms.Compose(transforms_list)
    else:
        transform_train = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size, time_window=delta_t),
            TimeSlice(seq_len),
            # transforms.ToFrame(sensor_size, n_time_bins=seq_len),
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
    return dataset


def create_dense_batch(ids, dataset, batch_size):
    batched_data = []
    batched_labels = np.empty(batch_size, dtype=np.int32)
    for i,idx in enumerate(ids):
        data, label = dataset[idx]
        data_flat = data.reshape(data.shape[0], -1).astype(np.float32)
        # data_flat = data
        # label = 0
        # data_flat = np.empty((seq_len, np.prod(dataset.sensor_size))) #*data["x"][0]
        batched_data.append(data_flat)
        batched_labels[i] = label
    batched_data = np.stack(batched_data)
    return {"inp_spikes": batched_data, "targets": batched_labels}        

def create_sparse_batch(ids, dataset, batch_size, pool=None): #, seq_len):
    batched_inp_spike_ids = []
    batched_num_inp_spikes = []
    # batched_num_inp_spikes = np.empty((batch_size, seq_len, 1), dtype=np.int32)
    
    if pool is None:
        batched_labels = np.empty(batch_size, dtype=np.int32)
        for i,idx in enumerate(ids):
            data, label = dataset[idx]
            batched_inp_spike_ids.append(data[0])
            batched_num_inp_spikes.append(data[1])
            # batched_num_inp_spikes[i] = data[1]
            batched_labels[i] = label
    else:
        data = pool.map(get_dataset_item, ids)
        batched_labels = []
        for (inp_spike_ids,num_inp_spikes),label in data:
            batched_inp_spike_ids.append(inp_spike_ids)
            batched_num_inp_spikes.append(num_inp_spikes)
            batched_labels.append(label)
        batched_labels = np.asarray(batched_labels, dtype=np.int32)

    batched_inp_spike_ids = np.stack(batched_inp_spike_ids).astype(np.float32)
    batched_num_inp_spikes = np.stack(batched_num_inp_spikes)
    return {"inp_spike_ids": batched_inp_spike_ids, "num_inp_spikes": batched_num_inp_spikes, "targets": batched_labels}


def create_nmnist_gener(root, sparse, num_epochs=1, seq_len=300, sparse_size=None, num_samples=None, dataset='train', shuffle=None, batchsize=None, use_multiprocessing=False):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a generator function with yields data, num_events, target
    target: integer
    data: flattened float32 array of dimension seq_len x prod(sersor_size) containing flattened event addresses
    '''

    dataset = create_nmnist_dataset(root, sparse, seq_len=seq_len, sparse_size=sparse_size, dataset=dataset, apply_flatten=True if use_multiprocessing else False)

    if shuffle is None:
        shuffle = True if dataset == 'train' else False

    if num_samples is None:
        num_samples = len(dataset)
    if batchsize is not None:
        num_batches = int((num_samples//batchsize) * num_epochs)

    if use_multiprocessing:
        assert batchsize is not None

    # idx_samples = np.arange(num_samples) 
    # idx_samples = np.arange(num_samples) 
    print()
    print(len(dataset))
    print(num_samples)
    if shuffle:
        idx_samples_base = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        idx_samples_base = np.arange(num_samples)

    idx_samples = np.empty(num_samples*num_epochs, dtype=np.int64)
    for iepoch in range(num_epochs):
        if shuffle:
            np.random.shuffle(idx_samples_base)
        idx_samples[iepoch*num_samples:(iepoch+1)*num_samples] = idx_samples_base
    print(idx_samples)

    def gen_dense_batched():
        for ibatch in range(num_batches):
            inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
            ret_data = create_dense_batch(inds, dataset, batchsize)
            yield ret_data

    def gen_dense():
        for i in idx_samples:
            data, label = dataset[i]
            data_flat = data.reshape(data.shape[0], -1).astype(np.float32)
            yield {"inp_spikes": data_flat, "targets": label}

    def gen_sparse_batched():
        for ibatch in range(num_batches):
            inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
            ret_data = create_sparse_batch(inds, dataset, batchsize)
            yield ret_data
            # batched_inp_spike_ids = []
            # batched_num_inp_spikes = np.empty((batchsize, seq_len, 1), dtype=np.int32)
            # batched_labels = np.empty(batchsize, dtype=np.int32)
            # for i,idx in enumerate(idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]):
            #     data, label = dataset[idx]
            #     batched_inp_spike_ids.append(data[0])
            #     batched_num_inp_spikes[i] = data[1]
            #     batched_labels[i] = label
            # batched_inp_spike_ids = np.stack(batched_inp_spike_ids).astype(np.float32)
            # batched_num_inp_spikes = np.stack(batched_inp_spike_ids).astype(np.float32)
            # yield {"inp_spike_ids": batched_inp_spike_ids, "num_inp_spikes": batched_num_inp_spikes, "targets": label}

    def gen_sparse_batched_multiproc():
        # use at most one process per cpu or one process per sample
        with Pool(min(cpu_count(), batchsize), initializer=set_global_dataset, initargs=(dataset,)) as p:
            for ibatch in range(num_batches):
                inds = idx_samples[ibatch*batchsize:(ibatch+1)*batchsize]
                ret_data = create_sparse_batch(inds, dataset, batchsize, p)
                yield ret_data

    def gen_sparse():    
        for i in idx_samples:
            data, label = dataset[i]
            yield {"inp_spike_ids": data[0].astype(np.float32), "num_inp_spikes": data[1].astype(np.int32), "targets": label}

    if batchsize is None:
        gen = gen_sparse if sparse else gen_dense 
    else:
        if use_multiprocessing:
            # TODO implement multiprocessing !!!
            # gen = gen_sparse_batched_multiproc if sparse else gen_dense_batched_multiproc
            gen = gen_sparse_batched_multiproc if sparse else gen_dense_batched
        else:
            gen = gen_sparse_batched if sparse else gen_dense_batched

    return gen, num_samples


class KerasNMNIST(tf.keras.utils.Sequence):
    _sparse: bool = False
    _gen_batch: Callable

    def __init__(self, 
            dataset, 
            batch_size: int, 
            sparse: bool,
            shuffle: Optional[bool] = False, 
            rng: Optional[np.random.Generator] = None, 
            # processing_func: Optional[Callable] = None
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.processing_func = processing_func
        self.sparse = sparse
        self.shuffle = shuffle
        if shuffle:
            assert isinstance(rng, np.random.Generator), f"If `shuffle=True`, `rng` has to be an instance of `numpy.random.Generator`, got '{rng}'."
        self.rng = rng
        self._indices = np.arange(len(self.dataset))

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        self._sparse = value
        self._gen_batch = self._gen_sparse_batch if value else self._gen_dense_batch

    def _gen_dense_batch(self, ids):
        return create_dense_batch(ids, self.dataset, self.batch_size)

    def _gen_sparse_batch(self, ids):
        return create_sparse_batch(ids, self.dataset, self.batch_size)

    def __len__(self):
        return int(len(self.dataset) // self.batch_size)

    def __getitem__(self, idx):
        inds = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_batch = self._gen_batch(inds)
        return data_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self._indices)

def get_nmnist_keras_dataset(rng, root, sparse, batchsize, seq_len, sparse_size: Optional[int] = None):
    if sparse:
        assert sparse_size is not None

    tonic_dataset = create_nmnist_dataset(root, sparse, seq_len=seq_len, sparse_size=sparse_size, dataset='train')
    keras_dataset = KerasNMNIST(
            dataset=tonic_dataset, 
            batch_size=batchsize, 
            sparse=sparse,
            shuffle=True, 
            rng=rng, 
    )
    return keras_dataset


def get_nmnist_dataset(root, sparse, num_epochs, seq_len, inp_dim, batchsize, num_samples=None, dims=None, multiprocessing=False):
    if multiprocessing:
        gen_train, num_samples = create_nmnist_gener(root, sparse, num_epochs, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples, batchsize=batchsize, multiprocessing=multiprocessing)
    else:
        gen_train, num_samples = create_nmnist_gener(root, sparse, num_epochs, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples, multiprocessing=multiprocessing)
    # get_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples)

    # dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
    #                                                                         tf.TensorSpec(shape=(), dtype=tf.int32))))

    if dims is None:
        dims = (34,34,2)
    flatten_fn = ft.partial(flatten_data_tf, dims=dims)
    if multiprocessing:
        if sparse:
            dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spike_ids": tf.TensorSpec(shape=(batchsize, seq_len, inp_dim), dtype=tf.float32),
                                                                                    "num_inp_spikes": tf.TensorSpec(shape=(batchsize, seq_len, ), dtype=tf.int32),
                                                                                    "targets": tf.TensorSpec(shape=(batchsize, ), dtype=tf.int32)})
        else:
            dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spikes": tf.TensorSpec(shape=(batchsize, seq_len, inp_dim), dtype=tf.float32),
                                                                                    "targets": tf.TensorSpec(shape=(batchsize, ), dtype=tf.int32)})
    else:
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

    # dataset = dataset.map(lambda *args: args, num_parallel_calls=tf.data.AUTOTUNE)
    if not multiprocessing:
        if sparse:
            # TODO perform transform here instead of inside tonic dataset?, makes use of `num_parallel_calls`
            dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset




# def flatten_spike_ids(dims, spike_ids):
#     '''
#     Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
#     dims: tuple whose elements specify the size of each dimension
#     '''
#     ds = np.array([np.prod(dims[i:]) for i in range(1,len(dims))] + [1], dtype=np.int16)
#     spike_ids_flat = tf.math.reduce_sum([spike_ids[:,:,i]*d for i,d in enumerate(ds)],axis=0)
#     return spike_ids_flat

def flatten_spike_ids(sensor_dim, ids):
    dimx, dimy, dimp = sensor_dim
    return  ids[...,1] + dimy * (ids[...,2] + dimx * ids[...,0])


def flatten_data_tf(data, dims):
    '''
    Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
    dims: tuple whose elements specify the size of each dimension
    '''
    data["inp_spike_ids"] = flatten_spike_ids(dims, data["inp_spike_ids"])
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


def load_dataset_to_tensor_dict(root, sparse, seq_len, inp_dim, num_samples=None, iter_batchsize=None, shuffle=True):

    if iter_batchsize is None:
        iter_batchsize = 1000
    gen, num_samples = create_nmnist_gener(root, sparse, 1, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples, batchsize=iter_batchsize, shuffle=shuffle, use_multiprocessing=True)

    assert num_samples % iter_batchsize == 0, "`num_samples` must be divisible by `iter_batchsize`"

    if sparse:

        # TODO apply flatten here or in create_nmnist_gener !!!

        inp_spike_ids = np.empty((num_samples, seq_len, inp_dim), dtype=np.float32)
        num_inp_spikes = np.empty((num_samples, seq_len, 1), dtype=np.int32)
        labels = np.empty((num_samples,), dtype=np.int32)
        for i,data in enumerate(gen()):
            print(i)
            inp_spike_ids[i*iter_batchsize:(i+1)*iter_batchsize] = data["inp_spike_ids"]
            num_inp_spikes[i*iter_batchsize:(i+1)*iter_batchsize] = np.expand_dims(data["num_inp_spikes"], axis=-1)
            labels[i*iter_batchsize:(i+1)*iter_batchsize] = data["targets"]
        ret_val = {
            "inp_spike_ids": inp_spike_ids,
            "num_inp_spikes": num_inp_spikes,
            "targets": labels,
        }
    else:
        inp_spikes = np.empty((num_samples, seq_len, inp_dim), dtype=np.float32)
        labels = np.empty((num_samples,), dtype=np.int32)
        for i,data in enumerate(gen()):
            print(i)
            inp_spikes[i*iter_batchsize:(i+1)*iter_batchsize] = data["inp_spikes"]
            labels[i*iter_batchsize:(i+1)*iter_batchsize] = data["targets"]
        ret_val = {
            "inp_spikes": inp_spikes,
            "targets": labels,
        }
    return ret_val



if __name__ == "__main__":
    import sys
    gens = {}
    data = {}
    for use_sparse in [True, False]:
        sparse_str = "sparse" if use_sparse else "dense"
        gen, num_samples = create_nmnist_gener(
            root="/Data/pgi-15/datasets", 
            sparse=use_sparse, 
            num_epochs=1, 
            seq_len=100, 
            sparse_size=128, 
            num_samples=None, 
            dataset='train', 
            shuffle=False, 
            batchsize=1, 
            use_multiprocessing=True
        )
        gens[sparse_str] = gen
        data_next = next(gen())
        data[sparse_str] = data_next


        # use_multiprocessing = False
        # dataset = create_nmnist_dataset("/Data/pgi-15/datasets", use_sparse, seq_len=100, sparse_size=128, dataset="train", apply_flatten=True if use_multiprocessing else False)
        # data[sparse_str] = dataset[0]

    print()
    print(data["dense"]["inp_spikes"].shape)
    print(data["dense"]["inp_spikes"].sum(axis=2).astype(np.int32))
    print()
    print(data["sparse"]["num_inp_spikes"].shape)
    print(data["sparse"]["num_inp_spikes"])
    print()
    print(data["dense"]["inp_spikes"].shape)
    print(data["dense"]["inp_spikes"])
    # sys.exit()

    num_inp_spikes_dense = data["dense"]["inp_spikes"].sum(axis=2).astype(np.int32)[0]
    num_inp_spikes_sparse = data["sparse"]["num_inp_spikes"][0]
    inp_spikes_ids_dense = np.argwhere(data["dense"]["inp_spikes"][0] > 0)
    inp_spikes_ids_sparse = data["sparse"]["inp_spike_ids"][0]


    sensor_size = tonic.datasets.NMNIST.sensor_size
    def flatten_fn_general(dims, data):
        return flatten_spike_ids(dims, data[0]), data[1]
    flatten_fn = ft.partial(flatten_fn_general, sensor_size)
    print(sensor_size)


    # import sys
    # sys.exit()
    # for i in range(len(num_inp_spikes_sparse)-1):
    #     print(num_inp_spikes_dense[i], num_inp_spikes_sparse[i+1])
    
    def flatten_spike_id(sensor_dims, id_):
        # return id_[0]*sensor_dims[0]*sensor_dims[1] + id_[1]*sensor_dims[0] + id_[2]
        return  id_[1] + sensor_dims[1] * (id_[2] + id_[0] * sensor_dims[0])
        # return id_[0] +  id_[2] * sensor_dims[2] + sensor_dims[1]*sensor_dims[2] * id_[1]

    # def flatten_spike_ids_vec(sensor_dims, ids):
    #     return  ids[...,1] + sensor_dims[1] * (ids[...,2] + ids[...,0] * sensor_dims[0])

    def flatten_spike_ids_vec(dimx, dimy, dimp, ids):
        return  ids[...,1] + dimy * (ids[...,2] + dimx * ids[...,0])


    arr = np.arange(4*4*2).reshape((4,4,2))
    print(arr)
    print()
    print(arr[:,:,0])    
    print()
    print(arr[:,0,0])    

    # import sys
    # sys.exit()




    print()
    print(np.all(num_inp_spikes_dense[:-1] == num_inp_spikes_sparse[1:]))
    print()
    print(inp_spikes_ids_dense.shape)
    print(inp_spikes_ids_dense[:10])
    print()
    print(inp_spikes_ids_sparse.shape)
    print(inp_spikes_ids_sparse[:6, :3])
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[0, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[1, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[2, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[3, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[4, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[5, 0]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[5, 1]))
    # print(flatten_spike_id(sensor_size, inp_spikes_ids_sparse[5, 2]))
    # print()
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[0:0+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[1:1+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[2:2+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[3:3+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[4:4+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[5:5+1, 0:0+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[5:5+1, 1:1+1].reshape((1,1,3))))
    # print(flatten_spike_ids(sensor_size, inp_spikes_ids_sparse[5:5+1, 2:2+1].reshape((1,1,3))))
    # print()
    # print(flatten_spike_ids_vec(*sensor_size, inp_spikes_ids_sparse[0:6, 0:3]))
    # print()
    # print(inp_spikes_ids_sparse[inp_spikes_ids_sparse > 0].flatten())