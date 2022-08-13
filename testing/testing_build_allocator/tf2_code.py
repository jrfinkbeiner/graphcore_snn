import os
from typing import List

import numpy as np
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
                                              separate_gradients=False, # to calculate gradients separately
                                            #   inputs_with_gradients=[0], # TODO is this working ?
                                              attributes=f"some_attrib",
                                            )[0]

@tf.function(experimental_compile=True)
def add_tensor_list_fn(*xs):
    out = _add_op(xs)        
    return out

# grad op not implemented yet for custom op
# @tf.function(experimental_compile=True)
# def add_tensor_list_fn_with_grad(*xs):
#     with tf.GradientTape() as tape:
#         out = _add_op(xs)
#         loss = tf.reduce_sum(out)
#         gradients = tape.gradient(loss, xs)
#     return out, gradients

def main():
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = 1
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    NUM_XS = 10
    DIM = 8

    with strategy.scope():

        w_init = tf.random_normal_initializer(0.0, 2.0, 42)
        xs_tf = [tf.Variable(
            initial_value=np.ones(DIM, dtype=np.float32),
            # initial_value=w_init(shape=(DIM,), dtype=tf.float32),
            trainable=True,
            # validate_shape=True,
            # caching_device=None,
            name=f"x{i}",
        ) for i in range(NUM_XS)]
                
        out =  strategy.run(add_tensor_list_fn, args=xs_tf)
        # out, grad =  strategy.run(add_tensor_list_fn_with_grad, args=xs_tf)

    print(out)




if __name__ == "__main__":
    main()