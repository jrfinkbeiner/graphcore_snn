import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

# import tensorflow.keras as keras
from tensorflow.python import ipu
tf.disable_v2_behavior()


def test_sparse2dense():
    inp_spikes = np.array([[[1, 4, 5], [0, 3, 6]]], dtype=np.float32)
    num_spikes = np.array([[[2], [3]]], dtype=np.int32)
    size_dense = 8
    correct_result = np.array([[[0., 1., 0., 0., 1., 0., 0., 0.],
                                [1., 0., 0., 1., 0., 0., 1., 0.]]], dtype=np.float32)

    dense_tensor = sparse2dense(inp_spikes, num_spikes, size_dense)
    assert np.all(dense_tensor == correct_result)

def sparse2dense(spike_ids, num_spikes, dense_size, sparse_dim=-1):
    assert sparse_dim == -1
    assert len(spike_ids.shape) == 3
    sparse_shape = spike_ids.shape
    dense_shape = list(spike_ids.shape)
    dense_shape[sparse_dim] = dense_size
    dense_shape = [dense_shape[0], dense_shape[1], dense_shape[2]]

    dense_tensor = np.zeros((dense_shape[0], dense_shape[1], dense_shape[2]), dtype=np.float32)
    for iseq in range(spike_ids.shape[0]):
        for ibatch in range(spike_ids.shape[1]):
            ids = spike_ids[iseq, ibatch, :num_spikes[iseq, ibatch, 0]].astype(np.int32)
            dense_tensor[iseq, ibatch, ids] = 1
    return dense_tensor

# def sparse2dense_ipu(spike_ids, num_spikes, dense_size: int):
#     assert len(spike_ids.shape) == 3
#     outputs = {
#         "output_types": [spike_ids.dtype],
#         "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], int(dense_size)])],
#     }

#     print(f"{int(dense_size)=}")

#     base_path = os.path.realpath(os.path.dirname(__file__))
#     lib_path = os.path.join(base_path, "libcustom_op.so")
#     gp_path = os.path.join(base_path, "custom_codelet.gp")

#     print(base_path)

#     return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
#                                               lib_path,
#                                               gp_path,
#                                               outs=outputs,
#                                               separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
#                                               attributes=f"{int(dense_size)}",
#                                             )


def sparse2dense_ipu(spike_ids, num_spikes, dense_size):
    assert len(spike_ids.shape) == 3
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:2], int(dense_size)])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=f"{int(dense_size)}",
                                            )


def calc_result_and_gradient_ipu(spike_ids, num_spikes, target):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        dense_out_spikes = sparse2dense_ipu(spike_ids, num_spikes, int(target.shape[-1]))[0]
        print(dense_out_spikes)
        sum = tf.math.reduce_sum((dense_out_spikes-target)**2)
        grads = tf.gradients(sum, [spike_ids, dense_out_spikes])
        # grads = tf.zeros_like(spike_ids)
        return dense_out_spikes , grads


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


    SEQ_LEN = 11
    BATCHSIZE = 5
    SIZE_DENSE = 6 
    SIZE_SPARSE = 4

    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(2)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        spike_ids = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_SPARSE])
        num_spikes = tf.placeholder(np.int32, [SEQ_LEN, BATCHSIZE, 4])
        dense_spikes_target = tf.placeholder(np.float32, [SEQ_LEN, BATCHSIZE, SIZE_DENSE])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(calc_result_and_gradient_ipu, [spike_ids, num_spikes, dense_spikes_target])
        

    with tf.Session() as sess:

        spike_ids_np = np.empty((SEQ_LEN, BATCHSIZE, SIZE_SPARSE), dtype=np.float32)
        for ibatch in range(BATCHSIZE):
            for iseq in range(SEQ_LEN):
                spike_ids_np[iseq, ibatch, :] = rng.choice(SIZE_DENSE, SIZE_SPARSE, replace=False)
        num_spikes_np = rng.choice(SIZE_SPARSE, (SEQ_LEN, BATCHSIZE, 4), replace=True).astype(np.int32)
        dense_target_np = rng.uniform(size=[SEQ_LEN, BATCHSIZE, SIZE_DENSE]).astype(dtype=np.float32)
        print(f"{dense_target_np.shape=}")
        assert np.max(num_spikes_np) <= SIZE_SPARSE

        print("Before")
        dense_spikes_ipu, grads_ipu = sess.run(xla_result, feed_dict={spike_ids: spike_ids_np, num_spikes: num_spikes_np, dense_spikes_target: dense_target_np})
        print("After")

    # print("Hello")
    # print("Hello")
    test_sparse2dense()
    dense_spikes_np = sparse2dense(spike_ids_np, num_spikes_np, SIZE_DENSE)

    print()

    dLdsparse_spikes_ids, dLddense_spikes = grads_ipu


    print("\n--------------------------------------------------------------------\n")
    print(dense_spikes_np)
    print("\n--------------------------------------------------------------------\n")
    print(dense_spikes_ipu)
    print("\n--------------------------------------------------------------------\n")

    # print(grads_ipu)

    
    manuel_grad = np.zeros(dLdsparse_spikes_ids.shape, dtype=np.float32)
    for iseq in range(dLdsparse_spikes_ids.shape[0]):
        for ibatch in range(dLdsparse_spikes_ids.shape[1]):
            manuel_grad[iseq, ibatch, :] = dLddense_spikes[iseq, ibatch, spike_ids_np[iseq, ibatch, :].astype(np.int32)]

    # manuel_grad = dLddense_spikes[spike_ids_np.astype(np.int32)].reshape(spike_ids_np.shape)


    print("\n--------------------------------------------------------------------\n")
    print(manuel_grad[0, :, :])
    print("\n--------------------------------------------------------------------\n")
    print(dLdsparse_spikes_ids[0, :, :])
    print("\n--------------------------------------------------------------------\n")

    check_values(dense_spikes_ipu, dense_spikes_np, "dense spikes")
    check_values(dLdsparse_spikes_ids, manuel_grad, "dLdsparse_spikes_ids", rtol=1e-4, atol=1e-6)
    

