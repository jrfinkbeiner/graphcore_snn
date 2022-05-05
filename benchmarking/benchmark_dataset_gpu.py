import time 
import functools as ft
import tensorflow as tf
import numpy as np
# import yappi

from nmnist_util import get_nmnist_dataset, create_nmnist_gener, flatten_data_tf
# from nmnist_util import create_nmnist_gener, flatten_data_tf





# def get_nmnist_dataset(root, sparse, seq_len, inp_dim, batchsize, num_samples=None, dims=None):
#     # from snnax.utils.data import SequenceLoader
#     gen_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=inp_dim, num_samples=num_samples)
#     # get_train, num_samples = create_nmnist_gener(root, sparse, seq_len=seq_len, sparse_size=sparse_size, num_samples=num_samples)
    
#     # dataset = tf.data.Dataset.from_generator(gen_train, output_signature=((tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
#     #                                                                         tf.TensorSpec(shape=(), dtype=tf.int32))))
#     # return gen_train

#     # if dims is None:
#     #     dims = (34,34,2)
#     # flatten_fn = ft.partial(flatten_data_tf, dims=dims)
#     if sparse:
#         dataset = tf.data.Dataset.from_generator(gen_train, output_signature={"inp_spike_ids": tf.TensorSpec(shape=(seq_len, inp_dim, len(dims)), dtype=tf.float32),
#                                                                                 "num_inp_spikes": tf.TensorSpec(shape=(seq_len, ), dtype=tf.int32),
#                                                                                 "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})
#     else:
#         dataset = tf.data.Dataset.from_generator(gen_train, output_signature={
#                                                                                 "inp_spikes": tf.TensorSpec(shape=(seq_len, inp_dim), dtype=tf.float32),
#                                                                                 # "inp_spikes": tf.TensorSpec(shape=(None, ), dtype=dtype),
#                                                                                 "targets": tf.TensorSpec(shape=(), dtype=tf.int32)})

#     def ret_dataset(*args):
#         return dataset



#     # if sparse:
#     #     # TODO perform transform here instead of inside tonic dataset?, makes use of `num_parallel_calls`
#     #     dataset = dataset.map(flatten_fn, num_parallel_calls=tf.data.AUTOTUNE)


#     dataset = (tf.data.Dataset.range(1)
#         .flat_map(ret_dataset)
#         # .interleave(  # Parallelize data reading
#         #     ret_dataset,
#         #     num_parallel_calls=tf.data.AUTOTUNE
#         # )
#         .batch(batchsize, drop_remainder=True)
#         .prefetch(tf.data.AUTOTUNE)
#     ,)[0]

#     # dataset = (dataset
#     #     # .interleave(  # Parallelize data reading
#     #     #     ret_dataset,
#     #     #     num_parallel_calls=tf.data.AUTOTUNE
#     #     # )
#     #     .batch(batchsize, drop_remainder=True)
#     #     .prefetch(tf.data.AUTOTUNE)
#     # ,)[0]


#     # dataset = (tf.data.Dataset.range(2)
#     #     .interleave(  # Parallelize data reading
#     #         ret_dataset,
#     #         num_parallel_calls=tf.data.AUTOTUNE
#     #     )
#     #     .batch(  # Vectorize your mapped function
#     #         batchsize,
#     #         drop_remainder=True)
#     #     # .map(  # Parallelize map transformation
#     #     #     time_consuming_map,
#     #     #     num_parallel_calls=tf.data.AUTOTUNE
#     #     # )
#     #     # .cache()  # Cache data
#     #     # .map(  # Reduce memory usage
#     #     #     memory_consuming_map,
#     #     #     num_parallel_calls=tf.data.AUTOTUNE
#     #     # )
#     #     .prefetch(tf.data.AUTOTUNE)
#     # ,)[0]

#     return dataset





# ROOT_PATH_DATA = "/Data/pgi-15/datasets/"
ROOT_PATH_DATA = "/p/scratch/chpsadm/finkbeiner1/datasets"
SPARSE_METHOD = True
MULTIPROCESSING = False

NUM_EPOCHS = 4
NUM_CLASSES = 10
SEQ_LEN = 100

IMAGE_DIMS = (34,34,2)
SPARSE_DIM = 32

BATCHSIZE = 48
NUM_SAMPLES_TRAIN = BATCHSIZE*16*8

BATCHSIZE_PER_STEP = BATCHSIZE
STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE)
TRAIN_STEPS_PER_EXECUTION = STEPS_PER_EPOCH

INP_DIM = SPARSE_DIM if SPARSE_METHOD else np.prod(IMAGE_DIMS)

# dataset = get_nmnist_dataset(ROOT_PATH_DATA, SPARSE_METHOD, SEQ_LEN, INP_DIM, BATCHSIZE, num_samples=NUM_SAMPLES_TRAIN, dims=IMAGE_DIMS)
dataset = get_nmnist_dataset(ROOT_PATH_DATA, SPARSE_METHOD, NUM_EPOCHS, SEQ_LEN, INP_DIM, BATCHSIZE, num_samples=NUM_SAMPLES_TRAIN, dims=IMAGE_DIMS, multiprocessing=MULTIPROCESSING)


def benchmark(dataset, num_epochs):
    # start_time = time.perf_counter()
    start_time = time.time()
    # for epoch_num in range(num_epochs):
    #     print(epoch_num)
    #     for i,sample in enumerate(dataset):
    #         print(i)
    #         # print()
    #         # print(sample)
    #         # Performing a training step
    #         # time.sleep(0.05)
    #         pass
    for i,sample in enumerate(dataset):
        pass
    print("Execution time:", time.time() - start_time)


def get_timing(dataset, num_epochs=5):
    # yappi.set_clock_type("cpu")
    yappi.set_clock_type("wall")
    yappi.clear_stats()
    yappi.start()
    for epoch_num in range(num_epochs):
        # for i,sample in enumerate(dataset):
        for sample in dataset():
            pass
    yappi.stop()
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()


print(f"NUM_SAMPLES_TRAIN = {NUM_SAMPLES_TRAIN}")
print(f"NUM_BATCHES = {NUM_SAMPLES_TRAIN//BATCHSIZE}")
benchmark(dataset, NUM_EPOCHS)
# get_timing(dataset)

# ipu.dataset_benchmark.dataset_benchmark(dataset, NUM_EPOCHS, TRAIN_STEPS_PER_EXECUTION, print_stats=True, apply_options=True, do_memcpy=True)