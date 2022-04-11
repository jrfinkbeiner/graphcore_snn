import os
import sys
import functools as ft
import numpy as np

from keras_train_util import train_ipu, simple_loss_fn, train_gpu
import tensorflow as tf
import tensorflow.keras as keras 
# from keras.callbacks import CSVLogger
import tonic
import tonic.transforms as transforms
from tfneuromorphic import nmnist

    # def reg_func(spikes_list):
    #     loss_reg = 0.0
    #     for spikes in spikes_list:
    #         loss_reg = loss_reg + activity_reg_lower(spikes) + activity_reg_upper(spikes)
    #     return loss_reg


def get_single_layer_reg_function(lambda_lower, nu_lower, lambda_upper, nu_upper):

    def activity_reg_lower(spikes):
        return lambda_lower * tf.math.reduce_mean( tf.nn.relu(nu_lower - spikes)**2 )

    def activity_reg_upper(spikes):
        return lambda_upper * tf.nn.relu(tf.math.reduce_mean(spikes - nu_upper))**2

    def reg_func(y_true, spikes):
        sum_spikes_per_neuron = tf.reduce_mean(spikes, axis=1) # given shape (batch, seq_len, neurons)
        loss_reg = activity_reg_lower(sum_spikes_per_neuron) + activity_reg_upper(sum_spikes_per_neuron)
        return loss_reg

    return reg_func

reg_func_single_layer = get_single_layer_reg_function(lambda_lower=100, nu_lower=1e-3, lambda_upper=100, nu_upper=15)


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


# def cast_data_tf_dense(data, label):
#     return tf.cast(data, tf.float32), tf.cast(label, tf.float32)

def get_nmnist_dataset_dense(batchsize, num_samples=None):
    # from snnax.utils.data import SequenceLoader
    seq_len = 300
    gen_train, num_samples = create_gener_dense("/Data/pgi-15/datasets", seq_len=seq_len, num_samples=num_samples)
    
    dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, 2*34*34), dtype=tf.float32),
                                                                            tf.TensorSpec(shape=(), dtype=tf.int32))))
    # dataset = dataset.map(cast_data_tf_dense, num_parallel_calls=batchsize)
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    dataset = dataset.repeat()
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
    return (tf.cast(data, tf.float32), tf.expand_dims(tf.cast(num_events, tf.int32), axis=-1)), tf.cast(label, tf.int32) # cast label only because of IPUModel

def get_nmnist_dataset(sparse_size, num_samples, batchsize, dims=None, seq_len=300, sparse=True):

    assert sparse, "Currently only the sparse nmnist dataset generator is implemented"

    if dims is None:
        dims = (2,34,34)
    dtype = tf.int16

    gen_train = nmnist.create_gener('/Data/pgi-15/datasets/nmnist/n_mnist.hdf5', dataset='train', sparse_size = sparse_size, num_samples=num_samples, seq_len=seq_len)
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
    dataset = dataset.repeat()
    dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def mse_softmax_loss_fn(y_target, y_pred):
    print(y_target.shape)
    print(y_pred.shape)
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    print(sum_spikes.shape)
    softmax_pred = tf.nn.softmax(sum_spikes, axis=1)
    one_hot_target = tf.one_hot(y_target, softmax_pred.shape[-1], axis=-1, dtype=softmax_pred.dtype)
    return tf.math.reduce_sum((softmax_pred-one_hot_target)**2)/y_target.shape[-1]
    # return tf.math.reduce_max(softmax_pred)
    # return tf.math.reduce_sum(sum_spikes)
    # return tf.math.reduce_max((softmax_pred-one_hot_target)**2)/y_pred.shape[-1]

# def mse_softmax_loss_with_reg_fn(y_target, y_pred):

#     loss_class = mse_softmax_loss_fn(y_target, y_pred[-1])
#     loss_reg = reg_func(y_target)
    
#     return loss_class + 0.1 * loss_reg

@tf.function(experimental_compile=True)
def calc_accuracy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1)
    preds = tf.math.argmax(sum_spikes, axis=1)
    # preds = tf.ones_like(y_true)
    y_true = tf.cast(y_true, preds.dtype)
    identical = tf.cast(preds==y_true, tf.float32)
    return tf.math.reduce_mean(identical)

@tf.function(experimental_compile=True)
def calc_activity(y_true, y_pred):
    sum_spikes = tf.reduce_mean(y_pred)*y_pred.shape[1]
    return sum_spikes


def main():

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    image_dims = (2,34,34)
    seq_len = 100 # 300

    # dense_sizes = [np.prod(image_dims), 1024, 512, 128, 10] # reached 0.9731 (for steps_per_epoch=100, might be overfitting) 
    # dense_sizes = [np.prod(image_dims), 1024, 1024, 512, 128, 64, 10]
    dense_sizes = [np.prod(image_dims), 1024, 512, 128, 10]
    sparse_sizes = [32, 64*4, 32*4, 16*4, 8]
    # sparse_sizes = [32, 64, 32, 16, 8]

    print(dense_sizes)
    print(sparse_sizes)

    num_epochs = 50
    batchsize = 48
    batchsize_per_step = batchsize
    # steps_per_epoch = int(60000/batchsize)
    steps_per_epoch = int(54210/batchsize)
    train_steps_per_execution = steps_per_epoch #int(steps_per_epoch/10)

    decay_constant = 0.95
    threshold = 1.0

    # log_file = f"log_sparse_bs{batchsize}.csv"
    log_file = f"log_sparse_topK_bs{batchsize}_large_seqlen100.csv"


    num_layers = len(dense_sizes)-1
    # loss_fns = [reg_func_single_layer for _ in range(num_layers-1)]
    # loss_fns.append(mse_softmax_loss_fn)
    # loss_fns = {f"rnn_{i}" if i>0 else "rnn" : reg_func_single_layer for i in range(num_layers-1)}
    loss_fns = {f"rnn_{i}" if i>0 else "rnn" : lambda x, y: 0.0 for i in range(num_layers-1)}
    loss_fns[f"rnn_{num_layers-1}"] = mse_softmax_loss_fn

    # metrics = [calc_activity for _ in range(num_layers-1)]
    # metrics.append(calc_accuracy)
    metrics = {f"rnn_{i}" if i>0 else "rnn": calc_activity for i in range(num_layers-1)}
    metrics[f"rnn_{num_layers-1}"] = calc_accuracy

    if log_file is not None:
        csv_logger = keras.callbacks.CSVLogger(log_file, append=True, separator=';')
        callbacks = [csv_logger]
    else:
        callbacks = None


    dataset_sparse = get_nmnist_dataset(sparse_sizes[0], int(steps_per_epoch*batchsize), batchsize_per_step, image_dims, seq_len, sparse=True)
    train_ipu(
        num_epochs, 
        train_steps_per_execution, 
        batchsize_per_step,
        dataset_sparse,
        seq_len, 
        dense_sizes, 
        sparse_sizes, 
        decay_constant, 
        threshold,
        mse_softmax_loss_fn,
        metrics=calc_accuracy,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )


    # dataset_dense = get_nmnist_dataset_dense(batchsize, batchsize*steps_per_epoch)
    # train_gpu(
    #     num_epochs, 
    #     train_steps_per_execution, 
    #     batchsize,
    #     dataset_dense,
    #     seq_len, 
    #     dense_sizes, 
    #     decay_constant, 
    #     threshold,
    #     loss_fns,
    #     metrics=metrics,
    #     steps_per_epoch=steps_per_epoch,
    #     callbacks=callbacks,
    #     return_all=True,
    # )


if __name__ == "__main__":
    main()