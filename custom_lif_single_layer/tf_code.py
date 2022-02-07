import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

# import tensorflow.keras as keras
from tensorflow.python import ipu
tf.disable_v2_behavior()

from pure_tf_snn import pure_tf_lif_layer
from jax_snn import calc_result_and_gradient_jax


NUM_LAYERS = 4
BATCHSIZE = 32 # breaks at 15 for SEQ_LEN=1, breaks at 2 for SEQ_LEN>1, irrespective of seed (therefore of input and output)
SEQ_LEN = 35 # still good for 1000 with BATCHSIZE=1, 
SIZE_IN = 6
SIZE_SPARSE_IN = 6
SIZE_OUT = 7
SIZE_SPARSE_OUT = 7


# def model_fn_sequential_multilayer(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent):
#     input_layer = keras.Input(shape=(None, dim))
#     x = input_layer
#     for _ in range(num_layers):
#         x = tf.keras.layers.RNN(SNNBlock(dim, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent), return_sequences=True)(x)
#     return input_layer, x[:, -1, :] # only last item of sequence


def test_sparse2dense():
    inp_spikes = np.array([[[1, 4, 5], [0, 3, 6]]], dtype=np.float32)
    num_spikes = np.array([[[2], [3]]], dtype=np.int32)
    size_dense = 8
    correct_result = np.array([[[0., 1., 0., 0., 1., 0., 0., 0.],
                                [1., 0., 0., 1., 0., 0., 1., 0.]]], dtype=np.float32)

    dense_tensor = sparse2dense(inp_spikes, num_spikes, size_dense)
    assert np.all(dense_tensor == correct_result)

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



# # @tf.custom_gradient
# def sparse2dense_tf(spike_ids, num_spikes, dense_size, sparse_dim=-1):
#     assert sparse_dim == -1
#     assert len(spike_ids.shape) == 3

#     seq_dense_out_list = []
#     for iseq in range(spike_ids.shape[0]):
#         batched_dense_out_list = []
#         for ibatch in range(spike_ids.shape[1]):
#             nspikes = tf.cast(num_spikes[iseq, ibatch, 0], tf.int32)
#             # # # ids = np.asarray(spike_ids[iseq, ibatch, :nspikes], dtype=tf.int32)
#             ids = tf.cast(spike_ids[iseq, ibatch, :nspikes], tf.int32)
#             # # # ids = spike_ids[iseq, ibatch, :nspikes]
#             # # dense_tensor[iseq, ibatch, :] = tf.one_hot(ids, dense_size)

#             # # ids = spike_ids[iseq, ibatch, :num_spikes[iseq, ibatch, 0]].astype(np.int32)
#             # # dense_tensor[iseq, ibatch, ids] = 1
#             # # indices = tf.stack([rows, columns], axis=1)
#             # # for id in ids:
#             # dmi0 = tf.ones_like(ids)*iseq
#             # dim1 = tf.ones_like(ids)*ibatch
#             # indices = tf.stack([dmi0, dim1, ids], axis=1)
#             # dense_tensor = tf.tensor_scatter_nd_update(dense_tensor, indices, tf.ones_like(ids, dtype=tf.float32))

#             batched_dense_out_list.append(tf.math.reduce_sum(tf.one_hot(ids, dense_size, axis=0), axis=0))
#         seq_dense_out_list.append(tf.stack(batched_dense_out_list))
#     dense_out = tf.stack(seq_dense_out_list)
        
#     # def grad(upstream):
#     #     return np.zeros_like(dense_tensor), np.zeros_like(dense_tensor), np.zeros_like(dense_tensor)
#     return dense_out
  

def sparse2dense_ipu(spike_ids, num_spikes, dense_size: int):
    assert len(spike_ids.shape) == 3
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
def custom_lif_layer(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):

    batch_and_seq_size = num_inp_spikes.shape[:2]

    outputs = {
        "output_types": [inp_spike_ids.dtype, inp_spike_ids.dtype, init_state.dtype],
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": [tf.TensorShape([*batch_and_seq_size, SIZE_SPARSE_OUT]), tf.TensorShape([*batch_and_seq_size, 1]), tf.TensorShape([SEQ_LEN, *init_state.shape])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{SIZE_SPARSE_OUT}",
                                            )

def calc_result_and_gradient_ipu(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        spike_ids, num_spikes = inp_spike_ids, num_inp_spikes
        for ws, decay_consts, threshs, init_stat in zip(weights, decay_constants, thresholds, init_states):
            spike_ids, num_spikes, states = custom_lif_layer(ws, init_stat, spike_ids, num_spikes, decay_consts, threshs)
            num_spikes = tf.cast(num_spikes, tf.int32)
        dense_out_spikes = sparse2dense_ipu(spike_ids, num_spikes, weights[-1].shape[0])[0]
        sum = tf.math.reduce_sum((dense_out_spikes-1.0)**2)
        grads = tf.gradients(sum, [*weights, init_state[0], inp_spike_ids, spike_ids])
        return (spike_ids, num_spikes, states, dense_out_spikes) , grads


def calc_result_and_gradient_pure_tf(weights, init_states, inp_spikes, decay_constants, thresholds):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        spikes = inp_spikes
        for ws, decay_consts, threshs, init_stat in zip(weights, decay_constants, thresholds, init_states):
            spikes, states = pure_tf_lif_layer(ws, init_stat, spikes, decay_consts, threshs)
        sum = tf.math.reduce_sum((spikes-1.0)**2)
        grads = tf.gradients(sum, [*weights, init_state[0], inp_spikes])
        return (spikes, states), grads


# def calc_result_and_gradient_manuel(matrix: np.ndarray, sparse_vec: np.ndarray, num_elements: np.ndarray, loss_weights: np.ndarray):
#     num_elements = num_elements[0]
#     sparse_vec = sparse_vec.astype(np.int32)
#     dense_vec = np.zeros(matrix.shape[1])
#     dense_vec[sparse_vec[:num_elements]] = 1
#     result = np.sum(matrix[:, sparse_vec[:num_elements]], axis=1)
    
#     grads_weights = np.outer(loss_weights, dense_vec)
#     grads_inp = np.zeros(sparse_vec.shape, dtype=np.float32)
#     grads_inp[:num_elements] = np.matmul(matrix[:, sparse_vec[:num_elements]].transpose(1,0), loss_weights)
#     grads = [grads_weights, grads_inp, np.zeros(1)]
#     return result, grads


def check_fwd_results_self_consistent(out_spikes_ids_ipu, num_out_spikes_ipu, states_ipu):
    result_checks = np.zeros((SEQ_LEN, BATCHSIZE), dtype=bool)
    for iseq in range(SEQ_LEN):
        for ibatch in range(BATCHSIZE):
            above_thresh_ids = np.argwhere(states_ipu[iseq, ibatch] > threshold).flatten()
            num_out_spikes_iseq_ibatch = int(num_out_spikes_ipu[iseq, ibatch,0])
            if len(above_thresh_ids) == num_out_spikes_iseq_ibatch:
                if np.all(above_thresh_ids == np.sort(out_spikes_ids_ipu[iseq, ibatch, :num_out_spikes_iseq_ibatch])):
                    result_checks[iseq, ibatch] = True
    if np.all(result_checks):
        print("\u001b[32mIPU: Success! Spikes match state results.\u001b[0m")
        return True
    else:
        raise AssertionError("\u001b[31mWrong results! Spikes do not match state results.\u001b[0m")
        return False

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


if __name__ == '__main__':


    assert SIZE_IN >= SIZE_SPARSE_IN
    assert SIZE_OUT >= SIZE_SPARSE_OUT

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(5)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    # TODO adujst code below for arbitrary shapes
    # TODO this, however, must also include the sparse sizes
    # TODO especailly the hardcoded SIZE_SPARSE_OUT in the custom op
    shapes = [SIZE_IN]
    for i in range(NUM_LAYERS):
        shapes.append(SIZE_OUT)

    with tf.device("cpu"):
        weights = tuple(tf.placeholder(np.float32, [shapes[i+1], shapes[i]]) for i in range(NUM_LAYERS))
        init_state = tuple(tf.placeholder(np.float32, [BATCHSIZE, SIZE_OUT]) for i in range(NUM_LAYERS))
        inp_spike_ids = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN])
        num_inp_spikes = tf.placeholder(np.int32, [SEQ_LEN, BATCHSIZE, 1])
        inp_spikes = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_IN])
        decay_constants = tuple(tf.placeholder(np.float32, [SIZE_OUT]) for i in range(NUM_LAYERS))
        thresholds = tuple(tf.placeholder(np.float32, [SIZE_OUT]) for i in range(NUM_LAYERS))

        xla_result_puretf = calc_result_and_gradient_pure_tf(weights, init_state, inp_spikes, decay_constants, thresholds)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(calc_result_and_gradient_ipu, [weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds])
        

    with tf.Session() as sess:
        threshold = 1.0

        a = tuple(rng.normal(size=(shapes[i+1], shapes[i])).astype(np.float32) for i in range(NUM_LAYERS))
        # a = []
        # for i in range(NUM_LAYERS-1):
        #     a.append(rng.normal(size=(SIZE_OUT, SIZE_OUT)).astype(np.float32))
        # a = tuple(a)
        b = tuple(rng.normal(size=(BATCHSIZE, SIZE_OUT)).astype(np.float32) for i in range(NUM_LAYERS))
        c = np.empty((SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN))
        for ibatch in range(BATCHSIZE):
            for iseq in range(SEQ_LEN):
                c[iseq, ibatch, :] = rng.choice(SIZE_IN, SIZE_SPARSE_IN, replace=False)
        d = rng.choice(SIZE_SPARSE_IN, (SEQ_LEN, BATCHSIZE, 1), replace=True).astype(np.int32)
        es = tuple(0.95 * np.ones((SIZE_OUT), dtype=np.float32) for ilay in range(NUM_LAYERS))
        fs = tuple(threshold * np.ones((SIZE_OUT), dtype=np.float32) for ilay in range(NUM_LAYERS))
        g = sparse2dense(c, d, SIZE_IN)
        # g = np.ones(1, dtype=np.int32)

        assert np.max(d) <= SIZE_SPARSE_IN

        results_ipu, grads_ipu = sess.run(xla_result, feed_dict={weights: a, init_state: b, inp_spike_ids: c, num_inp_spikes: d, decay_constants: es, thresholds: fs})

        results_puretf, grads_puretf = sess.run(xla_result_puretf, feed_dict={weights: a, init_state: b, inp_spikes: g, decay_constants: es, thresholds: fs}) #, size_out_sparse: g})

    out_spikes_ids_ipu, num_out_spikes_ipu, states_ipu, out_spikes_ipu = results_ipu
    dLdweights_ipu = grads_ipu[:NUM_LAYERS]
    dLdInitState_ipu, dLdInpSpikes_ipu_sparse, dLdOutSpikes_ipu_sparse = grads_ipu[NUM_LAYERS:]
    dLdInpSpikes_ipu = sparse2dense(c, None, SIZE_IN, dLdInpSpikes_ipu_sparse)

    out_spikes_puretf, states_puretf = results_puretf
    dLdweights_puretf = grads_puretf[:NUM_LAYERS]
    dLdInitState_puretf, dLdInpSpikes_puretf = grads_puretf[NUM_LAYERS:]

    results_jax, grads_jax = calc_result_and_gradient_jax(weights=a, init_state=b, inp_spikes=g, decay_constants=es, thresholds=fs)
    out_spikes_jax, states_jax = results_jax
    dLdweights_jax, dLdInitState_jax, dLdInpSpikes_jax = grads_jax

    # print(dLdweights_ipu.shape)
    # print(dLdweights_jax.shape)
    # print(dLdweights_puretf.shape)

    # np.set_printoptions(suppress=True)
    # print("\n--------------------------------------------------------------------\n")
    # print(c)
    # print(d)
    # print("\n--------------------------------------------------------------------\n")
    # print(g)
    # print("\n--------------------------------------------------------------------\n")
    # print(out_spikes_ipu)
    # print(np.sum(out_spikes_ipu))
    # print("\n---------------------- dLdweights ----------------------------------\n")
    # print(dLdweights_ipu)
    # print("\n--------------------------------------------------------------------\n")
    # print(dLdweights_puretf)
    # print("\n--------------------------------------------------------------------\n")
    # print(np.isclose(dLdweights_ipu,dLdweights_puretf, rtol=1e-4))
    # print("\n--------------------------------------------------------------------\n")


    # # dLdInpSpikes_ipu = dLdInpSpikes_ipu * 0.05

    # print("\n---------------------- dLdInpSpikes --------------------------------\n")
    # print(dLdInpSpikes_ipu)
    # print("\n--------------------------------------------------------------------\n")
    # print(dLdInpSpikes_puretf)
    # print("\n--------------------------------------------------------------------\n")
    # print(np.isclose(dLdInpSpikes_ipu,dLdInpSpikes_puretf, rtol=1e-4))
    # print("\n--------------------------------------------------------------------\n")


    check_fwd_results_self_consistent(out_spikes_ids_ipu, num_out_spikes_ipu, states_ipu)
    print()
    check_values(states_ipu, states_puretf, "States - ipu-puretf", rtol=1e-4, atol=1e-6)
    check_values(states_jax, states_puretf, "States - jax-puretf", rtol=1e-4, atol=1e-6)
    check_values(out_spikes_ipu, out_spikes_puretf, "out_spikes - ipu-puretf")
    check_values(out_spikes_jax, out_spikes_puretf, "out_spikes - jax-puretf")
    print()
    for i in range(NUM_LAYERS):
        check_values(dLdweights_ipu[i], dLdweights_puretf[i], f"dLdweights[{i}] - ipu-puretf", rtol=1e-4, atol=1e-6)
        check_values(dLdweights_jax[i], dLdweights_puretf[i], f"dLdweights[{i}] - jax-puretf", rtol=1e-4, atol=1e-6)
        check_values(dLdweights_ipu[i], dLdweights_jax[i], f"dLdweights[{i}] - ipu-jax", rtol=1e-4, atol=1e-6)
        print()
    check_values(dLdInpSpikes_ipu, dLdInpSpikes_puretf, "dLdInpSpikes - ipu-puretf", rtol=1e-4, atol=1e-6)
    check_values(dLdInpSpikes_jax, dLdInpSpikes_puretf, "dLdInpSpikes - jax-puretf", rtol=1e-4, atol=1e-6)
    check_values(dLdInpSpikes_ipu, dLdInpSpikes_jax, "dLdInpSpikes - ipu-jax", rtol=1e-4, atol=1e-6)
