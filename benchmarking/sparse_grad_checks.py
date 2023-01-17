import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from keras_train_util import model_fn_dense, create_dataset_dense
from nmnist_util import load_dataset_to_tensor_dict

def simple_loss_fn_dense(y_target, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, SEQ_LEN, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)

@tf.function(experimental_compile=True)  # Make it fast.
def value_and_grad_on_batch(model, x, y):
    with tf.GradientTape() as tape:
        out = model(x)
        # loss = simple_loss_fn_dense(y, out[-1])
        loss = sum_and_sparse_categorical_crossentropy(y, out[-1])
        gradients = tape.gradient(loss, model.trainable_weights)
    return out, gradients

def cosine_similarity(a, b):
    return np.sum(a*b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_scale(a, b):
    return np.nanmean(a / b)


def check_values(a, b, name, *,rtol=1e-4, **kwargs):
    sucess = False
    same_shape = (len(a.shape) == len(b.shape)) and all([x == y for x,y in zip(a.shape, b.shape)])
    if same_shape:
        check_allclose = np.allclose(a, b, rtol=rtol, **kwargs)
        if check_allclose:
            sucess = True
    if sucess:
        print(f"\u001b[32m{name}: Success! Reults for tf and custom ipu implementation are identical.\u001b[0m")
    else:
        print(f"\u001b[31m{name}: Wrong results! Reults for tf and custom ipu implementation are not identical.\u001b[0m")
    return sucess

def test_sparse_vs_dense():

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    model_seed = rng.integers(999999)

    num_sequences = 24*16
    BATCHSIZE = num_sequences
    SEQ_LEN = 100

    DENSE_SIZES = [16, 7, 3, 2]

    # # benchmarking presentation
    NUM_CLASSES = 10
    IMAGE_DIMS = (34, 34, 2)
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), 1024, 1024, 512, 512, 128, NUM_CLASSES]
    DENSE_SIZES = [np.prod(IMAGE_DIMS), 1472, 1076, 384, NUM_CLASSES]



    num_layers = len(DENSE_SIZES)-1 



    # inp_spikes = tfp.distributions.Bernoulli(
    #     logits=None,
    #     # probs=np.full(shape=(1), fill_value=0.1),
    #     probs=0.03,
    #     dtype=tf.float32,
    #     validate_args=False,
    #     allow_nan_stats=True,
    #     name='Bernoulli'
    # ).sample((BATCHSIZE, SEQ_LEN, DENSE_SIZES[0]))
    # targets = rng.uniform(0.0, 1.0, size=(num_sequences, DENSE_SIZES[-1])).astype(np.float32)
    # dataset_dense = create_dataset_dense(inp_spikes, targets, BATCHSIZE, shuffle=False)
    # data_dense = iter(dataset_dense).next().values()

    ROOT_PATH_DATA = "/Data/pgi-15/datasets"
    data = load_dataset_to_tensor_dict(ROOT_PATH_DATA, False, SEQ_LEN, np.prod(IMAGE_DIMS), num_samples=BATCHSIZE, iter_batchsize=min(10000, BATCHSIZE))
    dataloader_train = create_dataset_dense(
        tf.convert_to_tensor(data["inp_spikes"], dtype=tf.float32), 
        tf.convert_to_tensor(data["targets"], dtype=tf.float32), 
        BATCHSIZE, 
        shuffle=True
    )
    data_dense_dict = iter(dataloader_train).next()
    data_dense = data_dense_dict.values()

    # print(data_dense)
    # sys.exit()

    decay_constant = 0.95
    threshold = 1.0
    second_thresh = 0.9
    first_and_second_threshold = [threshold, [*[second_thresh]*(num_layers-2), -100, -100]]


    print("\n################################# DENSE #######################################")
    model_dense = tf.keras.Model(*model_fn_dense(SEQ_LEN, DENSE_SIZES, decay_constant, threshold, BATCHSIZE, seed=model_seed, return_all=True))
    out_dense, grad_dense = value_and_grad_on_batch(model_dense, *data_dense)
    

    model_dense = tf.keras.Model(*model_fn_dense(SEQ_LEN, DENSE_SIZES, decay_constant, first_and_second_threshold, BATCHSIZE, seed=model_seed, return_all=True))
    out_dense, grad_dense_second_thresh = value_and_grad_on_batch(model_dense, *data_dense)



    print("\ngrad_dense")
    print(grad_dense)
    print("\ngrad_dense_second_thresh")
    print(grad_dense_second_thresh)

    print(f"\ninput activity: mean = {np.mean(data_dense_dict['inp_spikes'])}, std = {np.std(np.mean(data_dense_dict['inp_spikes'], axis=2))}, max = {np.max(np.mean(data_dense_dict['inp_spikes'], axis=2))}")
    print(f"\ninput num spikes: mean = {np.mean(np.sum(data_dense_dict['inp_spikes'], axis=2))}, std = {np.std(np.sum(data_dense_dict['inp_spikes'], axis=2))}, max = {np.max(np.sum(data_dense_dict['inp_spikes'], axis=2))}")
    for i in range(num_layers):
        print()
        check_values(grad_dense_second_thresh[i], grad_dense[i], f"{i}: sparse layer - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        print(f"{i}: activity = {np.mean(out_dense[i])}")
        print(f"{i}: cossine_similarity = \u001b[33m{cosine_similarity(grad_dense_second_thresh[i], grad_dense[i])}\u001b[0m")
        print(f"{i}: mean_scale = \u001b[36m{mean_scale(grad_dense_second_thresh[i], grad_dense[i])}\u001b[0m")


if __name__ == "__main__":
    test_sparse_vs_dense()