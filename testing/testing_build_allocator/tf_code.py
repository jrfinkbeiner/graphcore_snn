import os
from typing import List
import tensorflow as tf
from tensorflow.python import ipu


def _add_op(xs: List[tf.Tensor]):

    outputs = {
        "output_types": [xs[0].dtype],
        "output_shapes": [tf.TensorShape(xs[0].shape)],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libcustom_op.so")

    return ipu.custom_ops.precompiled_user_op(xs,
                                              lib_path,
                                              outs=outputs,
                                            #   separate_gradients=False, # to calculate gradients separately
                                            #   inputs_with_gradients=[0], # TODO is this working ?
                                            #   attributes=f"{dense_size}",
                                            )[0]


@tf.function(experimental_compile=True)
def add_tensor_list_fn(xs):
    out = _add_op(xs)        
    return out

@tf.function(experimental_compile=True)
def add_tensor_list_fn_with_grad(xs):
    with tf.GradientTape() as tape:
        out = _add_op(xs)
        loss = tf.reduce_sum(out)
        gradients = tape.gradient(loss, xs)
    return out, gradients

def main():

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    data_sparse = iter(dataset_sparse).next()
    data_sparse = ((data_sparse["inp_spike_ids"], data_sparse["num_inp_spikes"]), data_sparse["targets"])
    data_dense = iter(dataset_dense).next().values()

    # with strategy.scope():
    #     model_dense_ipu = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, threshold, batchsize, seed=model_seed, return_all=False))
    #     out_ipu_dense, grad_ipu_dense =  strategy.run(value_and_grad_on_batch, args=[model_dense_ipu, *data_dense, False])

    with strategy.scope():
        model_sparse_layer = keras.Model(*model_fn_sparse_layer(sparse_sizes, seq_len, dense_sizes, decay_constant, threshold, batchsize_per_step, seed=model_seed, return_all=False, transpose_weights=True, num_ipus=num_ipus))
        
        
        
        
        out_sparse_layer, grad_sparse_layer =  strategy.run(value_and_grad_on_batch, args=[model_sparse_layer, *data_sparse, True, False])







if __name__ == "__main__":
    main()