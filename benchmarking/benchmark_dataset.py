print("00000")

import tensorflow as tf
from tensorflow.python import ipu
from nmnist_util import get_nmnist_dataset # , create_nmnist_gener, get_nmnist_keras_dataset


ROOT_PATH_DATA = "/p/scratch/chpsadm/finkbeiner1/datasets"
SPARSE_METHOD = True

NUM_EPOCHS = 1
NUM_CLASSES = 10
SEQ_LEN = 10

IMAGE_DIMS = (34,34,2)

BATCHSIZE = 48
NUM_SAMPLES_TRAIN = BATCHSIZE*16

BATCHSIZE_PER_STEP = BATCHSIZE
STEPS_PER_EPOCH = int(NUM_SAMPLES_TRAIN/BATCHSIZE)
TRAIN_STEPS_PER_EXECUTION = STEPS_PER_EPOCH

INP_DIM = 32 if SPARSE_METHOD else np.prod(IMAGE_DIMS)

print("00100")
dataset = get_nmnist_dataset(ROOT_PATH_DATA, SPARSE_METHOD, NUM_EPOCHS, SEQ_LEN, INP_DIM, BATCHSIZE, num_samples=NUM_SAMPLES_TRAIN, dims=IMAGE_DIMS)

print("00200")
benchmark_op = ipu.dataset_benchmark.dataset_benchmark(dataset, NUM_EPOCHS, TRAIN_STEPS_PER_EXECUTION, print_stats=True, apply_options=True, do_memcpy=True)


import json

print("00300")
with tf.compat.v1.Session() as sess:
    json_string = sess.run(benchmark_op)
    json_object = json.loads(json_string[0])
print(json_object)