import os
import sys
import functools as ft
import time
import numpy as np

import tensorflow as tf

import tonic
import tonic.transforms as transforms
from tfneuromorphic import nmnist


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)


def generate_tonic_nmnist_dataset():
    import tonic
    import tonic.transforms as transforms

    sensor_size = tonic.datasets.NMNIST.sensor_size

    transform_train = transforms.Compose([
        # transforms.Crop(target_size=(28,28)),
        # transforms.Denoise(filter_time=10000),
        # transforms.TimeJitter(std=10),
        # transforms.SpatialJitter(
        #     variance_x=0.3, # TODO originally 2
        #     variance_y=0.3, # TODO originally 2
        #     clip_outliers=True
        # ),
        transforms.ToFrame(sensor_size, time_window=1000.0),
        # transforms.ToFrame(n_time_bins=1000),
    ])
    transform_test = transforms.Compose([
        # transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size, time_window=1000.0),
    ])

    dataset_train = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
                                    train=True,
                                    # transform=transform_train,
                                    first_saccade_only=True)
    dataset_test = tonic.datasets.NMNIST(save_to='/Data/pgi-15/datasets',
                                    train=False,
                                    transform=transform_test,
                                    first_saccade_only=True)
    return dataset_train, dataset_test


def create_gener_dense(root, seq_len=300, num_samples=None, dataset='train', shuffle=None):
    '''
    root: root directory of tonic datasets
    seq_len: maximum sequence length
    dataset: 'train', 'val', or 'test
    
    returns a generator function with yields data, num_events, target
    target: integer
    data: flattened float32 array of dimension seq_len x prod(sersor_size) containing flattened event addresses
    '''
    assert dataset in ['train','val','test']
    
    if shuffle is None:
        shuffle = True if dataset == 'train' else False
    
    if dataset == 'val':
        raise NotImplementedError()
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform_train = transforms.Compose([
        # transforms.ToFrame(sensor_size, time_window=1000.0),
        transforms.ToFrame(sensor_size, n_time_bins=seq_len),
    ])
    transform_test = transforms.Compose([
        # transforms.Denoise(filter_time=10000),
        # transforms.ToFrame(sensor_size, time_window=1000.0),
        transforms.ToFrame(sensor_size, n_time_bins=seq_len),
    ])

    dataset = tonic.datasets.NMNIST(save_to=root,
                                train=True,
                                transform=transform_train,
                                first_saccade_only=False)

    if num_samples is None:
        num_samples = len(dataset)
    
    # idx_samples = np.arange(num_samples) 
    # idx_samples = np.arange(num_samples) 
    idx_samples = np.random.choice(len(dataset), num_samples, replace=False)
    if shuffle: np.random.shuffle(idx_samples)
    def gen():    
        for i in idx_samples:
            data, label = dataset[i]
            data_flat = data.reshape(seq_len, -1).astype(np.float32)
            yield data_flat, label
            # yield data_flat, label
    return gen, num_samples


    dataset_train, dataset_test = generate_tonic_nmnist_dataset()

# def cast_data_tf_dense(data, label):
#     return tf.cast(data, tf.float32), tf.cast(label, tf.float32)

def get_nmnist_dataset_dense(batchsize, num_samples=None):
    # from snnax.utils.data import SequenceLoader
    seq_len = 300
    gen_train, num_samples = create_gener_dense("/Data/pgi-15/datasets", seq_len=seq_len, num_samples=num_samples)
    
    dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, 2*34*34), dtype=tf.float32),
                                                                            tf.TensorSpec(shape=(), dtype=tf.int32))))
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    # # dataset = dataset.repeat()
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def flatten_data_tf(input_data, label, dims = (2,34,34)):
    '''
    Flattens N-dimensional data to a 1 dimensional integer, where N = len(dims).
    dims: tuple whose elements specify the size of each dimension
    '''
    data, num_events = input_data
# def flatten_data_tf(data, num_events, label, dims = (2,34,34)):
    ds = np.array([np.prod(dims[i:]) for i in range(1,len(dims))] + [1], dtype='int16')
    flat_data = tf.math.reduce_sum([data[:,:,i]*d for i,d in enumerate(ds)],axis=0)

    return (flat_data, num_events), label

def cast_data_tf(input_data, label):
    data, num_events = input_data
# def cast_data_tf(data, num_events, label):
    return (tf.cast(data, tf.float32), tf.expand_dims(tf.cast(num_events, tf.int32), axis=-1)), tf.cast(label, tf.float32)

def get_nmnist_dataset(sparse_size, num_samples, batchsize, dims=None, seq_len=300, sparse=True):

    assert sparse, "Currently only the sparse nmnist dataset generator is implemented"

    if dims is None:
        dims = (2,34,34)
    dtype = tf.int16

    gen_train = nmnist.create_gener('/Data/pgi-15/datasets/nmnist/n_mnist.hdf5', dataset='train', sparse_size = sparse_size, seq_len=300)
    # it = gen_train()
    # data, num_events, label = next(it)
    # data_flat, _, _ = nmnist.flatten_data(data, num_events, label, dims = dims)
    #  = nmnist.sparse_vector_to_dense(data, num_events, dims = dims)
    
    dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, sparse_size, len(dims)), dtype=dtype),
                                                                           tf.TensorSpec(shape=(seq_len), dtype=dtype)),
                                                                          tf.TensorSpec(shape=(), dtype=dtype)))

    flatten_fn = ft.partial(flatten_data_tf, dims=dims)
    print(dataset)

    # dataset = dataset.shuffle()
    # dataset = dataset.prefetch(16) # TODO why 16 ?
    # dataset = dataset.map(nmnist.flatten_data_tf)
    dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(cast_data_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    dataset = dataset.repeat()
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset


def benchmark_on_ipu(dataset):
    import tensorflow as tf
    from tensorflow.python import ipu
    import json

    benchmark_op = ipu.dataset_benchmark.dataset_benchmark(dataset, 10, 512)

    with tf.Session() as sess:
        json_string = sess.run(benchmark_op)
        json_object = json.loads(json_string[0])




def main():

    batchsize = 48
    dataset_dense = get_nmnist_dataset_dense(batchsize, batchsize*100)

    benchmark(dataset_dense)


if __name__ == "__main__":
    main()