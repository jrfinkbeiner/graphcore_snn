import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

from tensorflow.python import ipu
tf.disable_v2_behavior()


BATCHSIZE = 10
SEQ_LEN = 11
SIZE_IN = 31
SIZE_SPARSE_IN = 7
SIZE_OUT = 63
SIZE_SPARSE_OUT = 13


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

def calc_result_and_gradient_ipu(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        result = custom_lif_layer(weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds)
        out_spikes_ids, num_out_spikes, states = result
        sum = tf.math.reduce_sum(out_spikes_ids)
        grads = tf.gradients(sum, [weights, inp_spike_ids])
        return result, grads

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
            correct_result = False
            # print()
            # print(iseq, ibatch)
            # print(above_thresh_ids)
            # print(out_spikes_ids_ipu[iseq, ibatch, :num_out_spikes_iseq_ibatch])

            if len(above_thresh_ids) == num_out_spikes_iseq_ibatch:
                if np.all(above_thresh_ids == out_spikes_ids_ipu[iseq, ibatch, :num_out_spikes_iseq_ibatch]):
                    result_checks[iseq, ibatch] = True

    # print(result_checks)
    # print(np.all(result_checks))
    if np.all(result_checks):
        print("Sucess! Spikes match state results.")
    else:
        raise AssertionError("Wrong results! Spikes do not match state results.")



if __name__ == '__main__':

    assert SIZE_IN >= SIZE_SPARSE_IN
    assert SIZE_OUT >= SIZE_SPARSE_OUT

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(2)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        weights = tf.placeholder(np.float32, [SIZE_OUT, SIZE_IN])
        init_state = tf.placeholder(np.float32, [BATCHSIZE, SIZE_OUT])
        inp_spike_ids = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN])
        num_inp_spikes = tf.placeholder(np.int32, [SEQ_LEN, BATCHSIZE, 1])
        decay_constants = tf.placeholder(np.float32, [SIZE_OUT])
        thresholds = tf.placeholder(np.float32, [SIZE_OUT])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(calc_result_and_gradient_ipu, [weights, init_state, inp_spike_ids, num_inp_spikes, decay_constants, thresholds])

    with tf.Session() as sess:
        threshold = 1.0

        a = rng.normal(size=(SIZE_OUT, SIZE_IN)).astype(np.float32)
        b = rng.normal(size=(BATCHSIZE, SIZE_OUT)).astype(np.float32)
        c = np.empty((SEQ_LEN, BATCHSIZE, SIZE_SPARSE_IN))
        for ibatch in range(BATCHSIZE):
            for iseq in range(SEQ_LEN):
                c[iseq, ibatch, :] = rng.choice(SIZE_IN, SIZE_SPARSE_IN, replace=False)
        d = rng.choice(SIZE_IN, (SEQ_LEN, BATCHSIZE, 1), replace=True).astype(np.int32)
        e = 0.95 * np.ones(SIZE_OUT, dtype=np.float32)
        f = threshold  * np.ones(SIZE_OUT, dtype=np.float32)

        results_ipu, grads_ipu = sess.run(xla_result, feed_dict={weights: a, init_state: b, inp_spike_ids: c, num_inp_spikes: d, decay_constants: e, thresholds: f})

    print(len(grads_ipu))
    print(grads_ipu[0].shape)

    out_spikes_ids_ipu, num_out_spikes_ipu, states_ipu = results_ipu
    check_fwd_results_self_consistent(*results_ipu)


