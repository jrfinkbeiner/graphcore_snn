import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft
import time

# import tensorflow.keras as keras
from tensorflow.python import ipu
tf.disable_v2_behavior()

from util_and_experiments.pure_tf_snn import pure_tf_lif_layer


NUM_LAYERS = 2
BATCHSIZE = 2
SEQ_LEN = 12
SIZE_IN = 128
SIZE_SPARSE_IN = 8
SIZE_OUT = 128
SIZE_SPARSE_OUT = 8


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
    lib_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{SIZE_SPARSE_OUT}",
                                            )

def calc_result_and_gradient_ipu(weights, init_states, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):
    spike_ids, num_spikes = inp_spike_ids, num_inp_spikes
    for ilay, (ws, decay_consts, threshs, init_stat) in enumerate(zip(weights, decay_constants, thresholds, init_states)):
        with tf.variable_scope(f"lif_layer_{ilay}", reuse=tf.AUTO_REUSE) as scope:
            spike_ids, num_spikes, states = custom_lif_layer(ws, init_stat, spike_ids, num_spikes, decay_consts, threshs)
            num_spikes = tf.cast(num_spikes, tf.int32)
    with tf.variable_scope(f"sparse2dense", reuse=tf.AUTO_REUSE) as scope:
        dense_out_spikes = sparse2dense_ipu(spike_ids, num_spikes, weights[-1].shape[0])[0]
    with tf.variable_scope(f"loss_calc", reuse=tf.AUTO_REUSE) as scope:
        sum_ = tf.math.reduce_sum((dense_out_spikes-1.0)**2)
        grads = tf.gradients(sum_, [*weights, init_states[0], inp_spike_ids, spike_ids])
        return (spike_ids, num_spikes, states, dense_out_spikes) , grads


def calc_result_and_gradient_pure_tf(weights, init_states, inp_spikes, decay_constants, thresholds):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        spikes = inp_spikes
        for ws, decay_consts, threshs, init_stat in zip(weights, decay_constants, thresholds, init_states):
            spikes, states = pure_tf_lif_layer(ws, init_stat, spikes, decay_consts, threshs)
        sum = tf.math.reduce_sum((spikes-1.0)**2)
        grads = tf.gradients(sum, [*weights, init_states[0], inp_spikes])
        return (spikes, states), grads


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
    # with ipu.scopes.ipu_scope("/device:IPU:0"):
        weights = tuple(tf.placeholder(np.float32, [shapes[i+1], shapes[i]]) for i in range(NUM_LAYERS))
        init_state = tuple(tf.placeholder(np.float32, [BATCHSIZE, SIZE_OUT]) for i in range(NUM_LAYERS))
        inp_spike_ids = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN])
        num_inp_spikes = tf.placeholder(np.int32, [SEQ_LEN, BATCHSIZE, 1])
        inp_spikes = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_IN])
        decay_constants = tuple(tf.placeholder(np.float32, [SIZE_OUT]) for i in range(NUM_LAYERS))
        thresholds = tuple(tf.placeholder(np.float32, [SIZE_OUT]) for i in range(NUM_LAYERS))

        # xla_result_puretf = calc_result_and_gradient_pure_tf(weights, init_state, inp_spikes, decay_constants, thresholds)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(calc_result_and_gradient_ipu, [weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds])
       

    times = []    


    with tf.Session() as sess:
        
        for i in range(1):

            threshold = 1.0

            # with ipu.scopes.ipu_scope("/device:IPU:0"):
            a = tuple(rng.normal(size=(shapes[i+1], shapes[i])).astype(np.float32) for i in range(NUM_LAYERS))
            b = tuple(rng.normal(size=(BATCHSIZE, SIZE_OUT)).astype(np.float32) for i in range(NUM_LAYERS))
            c = np.empty((SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN))
            for ibatch in range(BATCHSIZE):
                for iseq in range(SEQ_LEN):
                    c[iseq, ibatch, :] = rng.choice(SIZE_IN, SIZE_SPARSE_IN, replace=False)
            d = rng.choice(SIZE_SPARSE_IN, (SEQ_LEN, BATCHSIZE, 1), replace=True).astype(np.int32)
            es = tuple(0.95 * np.ones((SIZE_OUT), dtype=np.float32) for ilay in range(NUM_LAYERS))
            fs = tuple(threshold * np.ones((SIZE_OUT), dtype=np.float32) for ilay in range(NUM_LAYERS))
            # g = sparse2dense(c, d, SIZE_IN)

            # assert np.max(d) <= SIZE_SPARSE_IN


            time_start = time.time()
            results_ipu, grads_ipu = sess.run(xla_result, feed_dict={weights: a, init_state: b, inp_spike_ids: c, num_inp_spikes: d, decay_constants: es, thresholds: fs})
            # results_puretf, grads_puretf = sess.run(xla_result_puretf, feed_dict={weights: a, init_state: b, inp_spikes: g, decay_constants: es, thresholds: fs}) #, size_out_sparse: g})
            time_end = time.time()
            times.append(time_end-time_start)
            print(i, times[-1])


    # out_spikes_ids_ipu, num_out_spikes_ipu, states_ipu, out_spikes_ipu = results_ipu
    # dLdweights_ipu = grads_ipu[:NUM_LAYERS]
    # dLdInitState_ipu, dLdInpSpikes_ipu_sparse, dLdOutSpikes_ipu_sparse = grads_ipu[NUM_LAYERS:]
    # dLdInpSpikes_ipu = sparse2dense(c, None, SIZE_IN, dLdInpSpikes_ipu_sparse)

    # out_spikes_puretf, states_puretf = results_puretf
    # dLdweights_puretf = grads_puretf[:NUM_LAYERS]
    # dLdInitState_puretf, dLdInpSpikes_puretf = grads_puretf[NUM_LAYERS:]

    print()
    if len(times) > 1:
        print(np.mean(times[1:]))
    else:
        print(times[0])
