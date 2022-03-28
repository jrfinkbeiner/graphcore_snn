import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import functools as ft
import time

# import tensorflow.keras as keras
from tensorflow.python import ipu
# tf.disable_v2_behavior()
# from tensorflow.python.framework.tensor_shape import TensorShape

def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse)).astype(np.float32)
    for ibatch in range(batchsize):
        for iseq in range(seq_len):
            sparse_spike_ids[iseq, ibatch, :] = rng.choice(size_dense, size_sparse, replace=False)
    num_sparse_spikes = rng.choice(size_sparse, (seq_len, batchsize, 1), replace=True).astype(np.int32)
    return sparse_spike_ids, num_sparse_spikes

def sparse2dense(spike_ids, num_spikes, dense_size, values=None, sparse_dim=-1):
    assert sparse_dim == -1
    assert len(spike_ids.shape) == 3
    if values is not None:
        assert values.shape == spike_ids.shape
    sparse_shape = spike_ids.shape
    dense_shape = list(spike_ids.shape)
    dense_shape[sparse_dim] = dense_size
    dense_shape = [dense_shape[0], dense_shape[1], dense_shape[2]]

    dense_tensor = np.zeros((dense_shape[0], dense_shape[1], dense_shape[2]), dtype=np.float32)
    for iseq in range(spike_ids.shape[0]):
        for ibatch in range(spike_ids.shape[1]):
            if num_spikes is None:
                ids = spike_ids[iseq, ibatch, :].astype(np.int32)
            else:
                ids = spike_ids[iseq, ibatch, :num_spikes[iseq, ibatch, 0]].astype(np.int32)

            if values is None:
                dense_tensor[iseq, ibatch, ids] = 1
            else:
                if num_spikes is None:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :]
                else:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :num_spikes[iseq, ibatch, 0]]
    return dense_tensor

def sparse2dense_ipu(spike_ids, num_spikes, dense_size: int):
    assert len(spike_ids.shape) == 3, f"`spike_ids` must be tensor of rank 3, got {len(spike_ids.shape)}."
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], dense_size])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "sparse2dense", "libcustom_op.so")
    gp_path = os.path.join(base_path, "sparse2dense", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{dense_size}",
                                            )


# TODO could also implement a self-recurrent version
def custom_multi_lif_layer_sparse(sparse_sizes, weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):

    # TODO check that batch and seq_lens are consitent for all states, inp_spike_ids, num_inp_spikes
    assert isinstance(weights, (list, tuple))
    num_layers = len(weights)

    seq_and_batch_size = num_inp_spikes[0].shape[:2]
    seq_len = num_inp_spikes[0].shape[0]
    batch_size = num_inp_spikes[0].shape[1]

    output_types = [
        *[inp_spike_ids[ilay].dtype for ilay in range(num_layers)],
        *[num_inp_spikes[ilay].dtype for ilay in range(num_layers)],
        *[init_state[ilay].dtype for ilay in range(num_layers)],
    ]
    output_shapes = [
        *[tf.TensorShape([*seq_and_batch_size, sparse_sizes[ilay]]) for ilay in range(1, num_layers+1)], 
        *[tf.TensorShape([*seq_and_batch_size, 1]) for ilay in range(1, num_layers+1)],
        *[tf.TensorShape([seq_len, *init_state[ilay].shape]) for ilay in range(num_layers)],
    ]

    outputs = {
        "output_types": output_types,
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": output_shapes
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "libcustom_op.so")
    # gp_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "custom_codelet.gp")
    lib_path = os.path.join(base_path, "custom_lif_layer_vectorize", "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_lif_layer_vectorize", "custom_codelet.gp")

    inputs = [*weights, *init_state, *inp_spike_ids, *num_inp_spikes, *decay_constants, *thresholds]
    # sparse_sizes_str = "_".join([str(val) for val in sparse_sizes])
    dense_sizes = [w.shape[0] for w in weights]
    dense_sizes = [weights[0].shape[1], *dense_sizes]
    sizes_str = "_".join([str(val) for val in [*dense_sizes, *sparse_sizes, batch_size]])
    return ipu.custom_ops.precompiled_user_op(inputs,
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=sizes_str,
                                            )

class KerasMultiLIFLayerBase(keras.layers.Layer):
    def __init__(self, dense_shapes, decay_constant, threshold, seed=None):
        super().__init__()
        assert len(dense_shapes) > 1, "`dense_shapes` must be of at least length 2, generating a network with no hidden layers."
        self.num_layers = len(dense_shapes)-1
        self.dense_shapes = dense_shapes
        self.decay_constant_value = decay_constant
        self.threshold_value = threshold
        self.seed = seed

    def build(self, input_shape):
        w_init = tf.random_normal_initializer(0.0, 1.0, self.seed)
        self.ws = [tf.Variable(
            initial_value=w_init(shape=(self.dense_shapes[ilay+1], self.dense_shapes[ilay]), dtype=tf.float32),
            trainable=True,
            name="weights",
        ) for ilay in range(self.num_layers)]
        self.decay_constants = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.decay_constant_value, "decay_cosntants"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]
        self.thresholds = [tf.Variable(
            initial_value=tf.cast(tf.fill((self.dense_shapes[ilay],), self.threshold_value, "thresholds"), tf.float32),
            trainable=False,
        ) for ilay in range(1, self.num_layers+1)]

    def call(self):
        raise NotImplementedError


class KerasMultiLIFLayerSparse(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        assert len(dense_shapes) == len(sparse_shapes), "`dense_shapes` and `sparse_shapes` must have the same nmber of elements."
        self.sparse_shapes = sparse_shapes

    def call(self, inp_spike_ids, num_inp_spikes, init_states):
        return custom_multi_lif_layer_sparse(self.sparse_shapes, self.ws, init_states, inp_spike_ids, num_inp_spikes, self.decay_constants, self.thresholds)


@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * 1/(beta*tf.math.abs(x)+1)**2
  return spikes, grad

def pure_tf_lif_step(weights, state, inp_, decay_constants, thresholds):
    syn_inp = tf.matmul(inp_, weights, transpose_b=True)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 0))
    new_state = state * decay_constants + (1 - decay_constants) * syn_inp
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    spikes_out = heaviside_with_super_spike_surrogate(new_state-thresholds)
    return spikes_out, new_state

class KerasMultiLIFLayerDenseCell(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, decay_constant, threshold, seed=None):
        super().__init__(dense_shapes, decay_constant, threshold, seed)
        state_shapes = dense_shapes[1:]*2
        self.state_size = [tf.TensorShape((dim,)) for dim in state_shapes]
        self.output_size = [tf.TensorShape((shape_,)) for shape_ in dense_shapes[1:]]
        # self.output_size = dense_shapes[-1]

    def call(self, inp_spikes, state):
        neuron_states = state[:self.num_layers]
        outs = state[self.num_layers:]
        all_out_spikes = []
        all_neuron_states = []

        for ilay in range(self.num_layers):
            inp_ = inp_spikes if ilay==0 else outs[ilay-1]
            spikes_out, neuron_stat = pure_tf_lif_step(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], self.thresholds[ilay])
            all_neuron_states.append(neuron_stat)
            all_out_spikes.append(spikes_out)
            
        state_new = [*all_neuron_states, *all_out_spikes]
        return all_out_spikes, state_new

def KerasMultiLIFLayerDense(dense_shapes, decay_constant, threshold, seed=None, **kwargs):
    return tf.keras.layers.RNN(KerasMultiLIFLayerDenseCell(dense_shapes, decay_constant, threshold, seed), **kwargs)


def model_fn_sparse(seq_len, dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step, seed=None):
    num_layers = len(dense_shapes)-1

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32)
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32) 
    
    
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype ) for ilay in range(1, num_layers)]]
    num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype ) for ilay in range(1, num_layers)]]

    init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32) for i in range(num_layers)]
    out = KerasMultiLIFLayerSparse(
            dense_shapes, sparse_shapes, decay_constant, threshold, seed
        )(spike_ids, num_spikes, init_states)
    out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]    
    dense_out_spikes_last_layer = sparse2dense_ipu(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32), dense_shapes[-1])[0]
    return (inp_spike_ids, num_inp_spikes), tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])

def model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, seed=None):
    num_layers = len(dense_shapes)-1
    inp_spikes = keras.Input(shape=(seq_len, dense_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32)
    # inp_spikes_transp = [tf.transpose(inp_spikes, perm=[1, 0, 2]), *[tf.zeros((dense_shapes[ilay]), dtype=inp_spikes.dtype ) for ilay in range(1, num_layers)]]

    # init_out_spikes = [tf.zeros((dense_shapes[ilay]), dtype=inp_spikes.dtype ) for ilay in range(num_layers)]
    # init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32) for i in range(num_layers)]
    # comb_init_state = [*init_states, *init_out_spikes]

    out_spikes = KerasMultiLIFLayerDense(
            dense_shapes, decay_constant, threshold, seed, return_sequences=True
        )(inp_spikes) #, comb_init_state)
    # out_spikes, final_internal_states = out 
    return inp_spikes, out_spikes[-1]

def simple_loss_fn(y_target, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]


def create_dataset_sparse(inp_spike_ids, num_inp_spikes, targets, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"input_1": inp_spike_ids, "input_2": num_inp_spikes}, targets))
    # dataset = tf.data.Dataset.from_tensor_slices((inp_spike_ids, targets))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if shuffle:
        dataset = dataset.shuffle(targets.shape[0])
    dataset = dataset.prefetch(16) # TODO why 16 ?
    return dataset

def create_dataset_dense(inp_spikes, targets, batchsize, shuffle=True):
    # dataset = tf.data.Dataset.from_tensor_slices(({"input_1": inp_spike_ids, "input_2": num_inp_spikes}, targets))
    dataset = tf.data.Dataset.from_tensor_slices((inp_spikes, targets))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if shuffle:
        dataset = dataset.shuffle(targets.shape[0])
    dataset = dataset.prefetch(16) # TODO why 16 ?
    return dataset



def train_ipu(
        num_epochs, 
        train_steps_per_execution,
        batchsize_per_step,
        dataset,
        seq_len, 
        dense_shapes, 
        sparse_shapes, 
        decay_constant, 
        threshold,
        loss_fn,
    ):
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        # init model
        model = keras.Model(*model_fn_sparse(seq_len, dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step))
        # model.set_pipelining_options(
        #     gradient_accumulation_steps_per_replica=2*num_ipus #train_steps_per_execution
        # )

        # Compile our model with Stochastic Gradient Descent as an optimizer
        # and Categorical Cross Entropy as a loss.
        model.compile('sgd', loss_fn,
                    # metrics=["accuracy"],
                    steps_per_execution=train_steps_per_execution,
                    # run_eagerly=False,
        )

        print('\nTraining')
        model.fit(dataset, epochs=num_epochs)
        # model.fit([inp_spike_ids, num_inp_spikes, init_states], targets, epochs=num_epochs, batch_size=batchsize, shuffle=False)

        model.summary()


def train_gpu(
        num_epochs, 
        batchsize,
        dataset,
        seq_len, 
        dense_shapes, 
        decay_constant, 
        threshold,
        loss_fn,
    ):

    # init model
    model = keras.Model(*model_fn_dense(seq_len, dense_shapes, decay_constant, threshold, batchsize))

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', loss_fn,
                # metrics=["accuracy"],
    )

    print('\nTraining')
    model.fit(dataset, epochs=num_epochs)
    model.summary()


@tf.function(experimental_compile=True)  # Make it fast.
def value_and_grad_on_batch(model, x, y):
    with tf.GradientTape() as tape:
        out = model(x)
        loss = simple_loss_fn(y, out)
        gradients = tape.gradient(loss, model.trainable_weights)
    return out, gradients

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

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_sequences = 24
    batchsize = num_sequences
    batchsize_per_step = batchsize
    seq_len = 50
    dense_sizes = [102, 801, 799]
    sparse_sizes = dense_sizes
    decay_constant = 0.9
    threshold = 1.0
    num_layers = len(dense_sizes)-1
    model_seed = rng.integers(999999)

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(batchsize / batchsize_per_step)

    targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])
    dataset_sparse = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)
    inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
    dataset_dense = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    data_sparse = iter(dataset_sparse).next()
    with strategy.scope():
        model_sparse = keras.Model(*model_fn_sparse(seq_len, dense_sizes, sparse_sizes, decay_constant, threshold, batchsize_per_step, model_seed))
        out_sparse, grad_sparse =  strategy.run(value_and_grad_on_batch, args=[model_sparse, *data_sparse])


    model_dense = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, threshold, batchsize, model_seed))
    data_dense = iter(dataset_dense).next()
    out_dense, grad_dense = value_and_grad_on_batch(model_dense, *data_dense)
    
    # print("\nforward")
    # print(out_dense.shape)
    # print(out_sparse.shape)
    # print(out_sparse)
    # print(out_dense)
    # print("\ngrad")
    # print(len(grad_sparse))
    # print(len(grad_dense))
    # print(grad_sparse)
    # print(grad_dense)

    # import matplotlib.pyplot as plt
    # for i in range(num_layers):
    #     quotient = (grad_sparse[i]/grad_dense[i]).numpy()
    #     print()
    #     print(quotient)
    #     plt.figure()
    #     plt.hist(quotient.flatten(), bins=int(np.sqrt(0.5*quotient.size)))
    #     plt.title(f"layer {i}")

    #     print(np.nanmin(quotient), np.nanmax(quotient))
    # plt.show()


    check_values(out_sparse, out_dense, f"out_spikes", rtol=1e-4, atol=1e-6)
    for i in range(num_layers):
        check_values(grad_sparse[i], grad_dense[i], f"grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        # check_values(model_sparse.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)

def main():

    sparse_comp = True
    if sparse_comp:
        os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_epochs = 10
    num_sequences = 64
    batchsize = 2
    batchsize_per_step = 2
    seq_len = 12
    dense_sizes = [612, 512, 412]
    sparse_sizes = [42, 32, 22]
    decay_constant = 0.9
    threshold = 1.0

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(batchsize / batchsize_per_step)

    targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])

    if sparse_comp:
        dataset = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize)
        train_ipu(
            num_epochs, 
            train_steps_per_execution, 
            batchsize_per_step,
            dataset,
            seq_len, 
            dense_sizes, 
            sparse_sizes, 
            decay_constant, 
            threshold,
            simple_loss_fn
        )
    else:
        inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
        dataset = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize)
        train_gpu(
            num_epochs,  
            batchsize,
            dataset,
            seq_len, 
            dense_sizes, 
            decay_constant, 
            threshold,
            simple_loss_fn
        )


if __name__ == '__main__':
    test_sparse_vs_dense()
    # main()



