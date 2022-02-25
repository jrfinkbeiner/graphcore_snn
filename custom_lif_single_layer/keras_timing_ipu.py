import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import functools as ft
import time

# import tensorflow.keras as keras
from tensorflow.python import ipu
# tf.disable_v2_behavior()

def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse))
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
def custom_lif_layer(size_sparse_out, weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):

    batch_and_seq_size = num_inp_spikes.shape[:2]
    seq_len = num_inp_spikes.shape[0]
    outputs = {
        "output_types": [inp_spike_ids.dtype, inp_spike_ids.dtype, init_state.dtype],
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": [tf.TensorShape([*batch_and_seq_size, size_sparse_out]), tf.TensorShape([*batch_and_seq_size, 1]), tf.TensorShape([seq_len, *init_state.shape])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "libcustom_op.so")
    # gp_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "custom_codelet.gp")
    lib_path = os.path.join(base_path, "custom_lif_layer_repeat", "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_lif_layer_repeat", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{size_sparse_out}",
                                            )


class KerasLIFLayerSparse(keras.layers.Layer):
    def __init__(self, units, input_dim_dense, out_dim_sparse, decay_constant, threshold):
        super().__init__()
        self.units = units
        self.input_dim_dense = input_dim_dense
        self.out_dim_sparse = out_dim_sparse
        self.decay_constant_value = decay_constant
        self.threshold_value = threshold

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.units, self.input_dim_dense), dtype=tf.float32),
            trainable=True,
            name="weights",
        )
        self.decay_constants = tf.Variable(
            initial_value=tf.cast(tf.fill((self.units,), self.decay_constant_value, "decay_cosntants"), tf.float32),
            trainable=False,
        )
        self.thresholds = tf.Variable(
            initial_value=tf.cast(tf.fill((self.units,), self.threshold_value, "thresholds"), tf.float32),
            trainable=False,
        )

    def call(self, inp_spike_ids, num_inp_spikes, init_state):
        return custom_lif_layer(self.out_dim_sparse, self.w, init_state, inp_spike_ids, num_inp_spikes, self.decay_constants, self.thresholds)


def model_fn_ipu(seq_len, dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step):
    num_layers = len(dense_shapes)-1
    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32)
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32) 

    print(inp_spike_ids.shape)
    print(num_inp_spikes.shape)


    spike_ids = tf.transpose(inp_spike_ids, perm=[1, 0, 2])
    # num_spikes = tf.transpose(num_inp_spikes, perm=[1, 0, 2])
    num_spikes = tf.cast(tf.fill((seq_len, batchsize_per_step, 1), sparse_shapes[0]), dtype=tf.int32)

    print(spike_ids.shape)
    print(num_spikes.shape)
    for i in range(num_layers):

        init_state = tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32)
        spike_ids, num_spikes, states = KerasLIFLayerSparse(
                dense_shapes[i+1], dense_shapes[i], sparse_shapes[i+1], decay_constant, threshold
            )(spike_ids, num_spikes, init_state)
        num_spikes = tf.cast(num_spikes, tf.int32)

    print(spike_ids.shape)
    print(num_spikes.shape)

    dense_out_spikes = sparse2dense_ipu(spike_ids, num_spikes, dense_shapes[-1])[0]

    print(dense_out_spikes.shape)

    return (inp_spike_ids, num_inp_spikes), dense_out_spikes[-1]


# def model_fn_ipu_with_init_state(seq_len, dense_shapes, sparse_shapes, decay_constant, threshold):
#     num_layers = len(dense_shapes)-1
#     inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), dtype=tf.float32)
#     num_inp_spikes = keras.Input(shape=(seq_len, 1), dtype=tf.int32) 
#     init_states = [keras.Input(shape=(batchsize_per_step, dense_shapes[i]), dtype=tf.float32) for i in range(num_layers)]

#     print(inp_spike_ids.shape)
#     print(num_inp_spikes.shape)


#     spike_ids = tf.transpose(inp_spike_ids, perm=[1, 0, 2])
#     num_spikes = tf.transpose(num_inp_spikes, perm=[1, 0, 2])

#     print(spike_ids.shape)
#     print(num_spikes.shape)
#     for i in range(num_layers):
#         spike_ids, num_spikes, states = KerasLIFLayerSparse(
#                 dense_shapes[i+1], dense_shapes[i], sparse_shapes[i+1], decay_constant, threshold
#             )(spike_ids, num_spikes, init_states[i])
#         num_spikes = tf.cast(num_spikes, tf.int32)

#     print(spike_ids.shape)
#     print(num_spikes.shape)

#     dense_out_spikes = sparse2dense_ipu(spike_ids, num_spikes, dense_shapes[-1])[0]

#     return (inp_spike_ids, num_inp_spikes, init_states), dense_out_spikes[-1]


# def loss_fn(y_pred, y_target):
#     out_size = y_target.shape[-1]
#     print("000000")
#     spike_ids_pred = y_pred[0] 
#     num_spikes_pred = y_pred[1]
#     fin_states_pred = y_pred[2]
#     print("000100")
#     print(spike_ids_pred.shape)
#     print(num_spikes_pred.shape)
#     dense_out_spikes = sparse2dense_ipu(spike_ids_pred, num_spikes_pred, out_size)[0]
#     print("000200")
#     return tf.math.reduce_sum((dense_out_spikes[-1]-y_target)**2)/out_size


def loss_fn(y_pred, y_target):
    print(y_pred.shape)
    print(y_target.shape)
    return tf.math.reduce_sum((y_pred-y_target)**2)/y_target.shape[-1]

def create_dataset(inp_spike_ids, num_inp_spikes, targets, batchsize):
    print(inp_spike_ids.shape)
    print(num_inp_spikes.shape)
    print(targets.shape)
    print(batchsize)

    dataset = tf.data.Dataset.from_tensor_slices(({"input_1": inp_spike_ids, "input_2": num_inp_spikes}, targets))
    # dataset = tf.data.Dataset.from_tensor_slices((inp_spike_ids, targets))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.shuffle(targets.shape[0])
    # dataset = dataset.prefetch(16)
    return dataset.prefetch(16)


class SNNDataset(keras.utils.Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, inp_spike_ids, num_inp_spikes, init_states, targets, batch_size, rng):

        self.inp_spike_ids = inp_spike_ids        
        self.num_inp_spikes = num_inp_spikes
        self.init_states = init_states
        self.targets = targets
        self.batch_size = batch_size
        self.indices = np.arange(self.targets.shape[0])
        self.rng = rng

    def __len__(self):
        return math.ceil(self.targets.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        inps_batch = [
            self.inp_spike_ids[:, inds],
            self.num_inp_spikes[:, inds],
            [init_stat[inds] for init_stat in self.init_states],
        ]
        target_batch = self.targets[inds]
        return inps_batch, target_batch
    
    def on_epoch_end(self):
        self.rng.shuffle(self.indices)


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
        # inp_spike_ids,
        # num_inp_spikes,
        # init_states,
        # targets,
    ):


    print(train_steps_per_execution)
    
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    # inp_spike_ids = tf.convert_to_tensor(inp_spike_ids.transpose(1,0,2))
    # num_inp_spikes = tf.convert_to_tensor(num_inp_spikes.transpose(1,0,2))
    # init_states = [tf.convert_to_tensor(stat) for stat in init_states]
    # targets = tf.convert_to_tensor(targets)

    with strategy.scope():
        # init model
        model = keras.Model(*model_fn_ipu(seq_len, dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step))
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





def main():

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_epochs = 10
    num_sequences = 64
    batchsize = 2
    batchsize_per_step = 2
    seq_len = 12
    dense_sizes = [512, 512, 512]
    sparse_sizes = [8, 8, 8]
    decay_constant = 0.9
    threshold = 1.0

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(batchsize / batchsize_per_step)


    targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])
    # init_states = [np.zeros((num_sequences, nneurons), dtype=np.float32) for nneurons in dense_sizes[1:]]

    # dataset = SNNDataset(inp_spike_ids, num_inp_spikes, init_states, targets, batchsize, rng)
    dataset = create_dataset(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize)

    train_ipu(
        num_epochs, 
        train_steps_per_execution, 
        batchsize,
        dataset,
        seq_len, 
        dense_sizes, 
        sparse_sizes, 
        decay_constant, 
        threshold
    )


if __name__ == '__main__':
    main()



