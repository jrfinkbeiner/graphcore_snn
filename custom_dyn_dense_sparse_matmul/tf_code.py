import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

from tensorflow.python import ipu
tf.disable_v2_behavior()

def dyn_dense_binary_sparse_matmul_op(matrix, sparse_vec, num_elements):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [tf.TensorShape([matrix.shape[0]])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libcustom_op.so")
    gp_path = os.path.join(base_path, "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([matrix, sparse_vec, num_elements],
                                              lib_path,
                                              gp_path,
                                              outs=outputs)

def calc_result_and_gradient_ipu(matrix, sparse_vec, num_elements, loss_weights):
    with tf.variable_scope(f"some_name", reuse=tf.AUTO_REUSE) as scope:
        result = dyn_dense_binary_sparse_matmul_op(matrix, sparse_vec, num_elements)[0]
        sum = tf.math.reduce_sum(loss_weights*result)
        grads = tf.gradients(sum, [matrix, sparse_vec])
        return result, grads

def calc_result_and_gradient_manual(matrix: np.ndarray, sparse_vec: np.ndarray, num_elements: np.ndarray, loss_weights: np.ndarray):
    num_elements = num_elements[0]
    sparse_vec = sparse_vec.astype(np.int32)
    dense_vec = np.zeros(matrix.shape[1])
    dense_vec[sparse_vec[:num_elements]] = 1
    result = np.sum(matrix[:, sparse_vec[:num_elements]], axis=1)
    
    grads_weights = np.outer(loss_weights, dense_vec)
    grads_inp = np.zeros(sparse_vec.shape, dtype=np.float32)
    grads_inp[:num_elements] = np.matmul(matrix[:, sparse_vec[:num_elements]].transpose(1,0), loss_weights)
    grads = [grads_weights, grads_inp, np.zeros(1)]
    return result, grads



if __name__ == '__main__':
    
    SIZE_OUT = 128
    SIZE_IN = 128
    SIZE_SPARSE = 32
    NUM_NZELEMENTS = 12

    assert NUM_NZELEMENTS <= SIZE_SPARSE
    os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    np.random.seed(2)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        matrix = tf.placeholder(np.float32, [SIZE_OUT, SIZE_IN])
        sparse_vec = tf.placeholder(np.float32, [SIZE_SPARSE])
        num_elements = tf.placeholder(np.int32, [1])
        loss_weights = tf.placeholder(np.float32, [SIZE_OUT])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(calc_result_and_gradient_ipu, [matrix, sparse_vec, num_elements, loss_weights])

    with tf.Session() as sess:
        # a = np.array([
        #     [1, 2, 3],
        #     [4, 5, 6],
        #     [7, 8, 9],
        #     [10, 11, 12],
        # ], dtype=np.float32)
        # b = np.array([0,2,100], dtype=np.float32)
        # c = np.array([NUM_NZELEMENTS], dtype=np.int32)
        # d = np.array([0.5, 0.25, 0.125], dtype=np.float32)
        # # d = np.ones(SIZE_OUT).astype(np.float32)

        a = np.random.randn(SIZE_OUT, SIZE_IN).astype(np.float32)
        b = np.random.choice(SIZE_IN, SIZE_SPARSE, replace=False).astype(np.int32)
        c = np.array([NUM_NZELEMENTS], dtype=np.int32)
        d = np.random.randn(SIZE_OUT).astype(np.float32)

        result_ipu, grads_ipu = sess.run(xla_result, feed_dict={matrix: a, sparse_vec: b, num_elements: c, loss_weights: d})

    result_man, grads_man = calc_result_and_gradient_manual(a, b, c, d)

    # Show result from the IPU and compare to the manual/CPU implementation:
    print(f"\nresults identical: {np.allclose(result_ipu, result_man, rtol=1e-5)}")
    print("IPU:", result_ipu)
    print("MAN:", result_man)
    print(f"\ngradients weights dLdW identical: {np.allclose(grads_ipu[0], grads_man[0], rtol=1e-6)}")
    print("IPU:", grads_ipu[0])
    print("MAN:", grads_man[0])
    print(f"\ngradients inputs dLdx identical: {np.allclose(grads_ipu[1], grads_man[1], rtol=1e-5)}")
    print("IPU:", grads_ipu[1])
    print("MAN:", grads_man[1])
