# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import functools as ft
from typing import Union, NamedTuple, List
import warnings
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python import ipu


from keras_train_util_ipu import (
    KerasMultiLIFLayerSparse, 
    KerasMultiLIFLayerSparseOps, 
    # KerasSparseIdentity, 
    simple_loss_fn_sparse, 
    simple_loss_fn_dense, 
    dyn_dense_binary_sparse_matmul_op,
    sparse2dense_ipu,
    gen_sparse_spikes,
    # model_fn_sparse_layer,
)

class SparseBinaryVec(NamedTuple):
    ids: Union[tf.Tensor, tf.TensorShape]
    num_nzelements: Union[tf.Tensor, tf.TensorShape, int]

class TileMapping(NamedTuple):
    start_tile: int
    end_tile: int


def simple_loss_fn_dense_multi_ipu(pipeline_stage, y_target, y_pred):
    with keras.ipu.PipelineStage(pipeline_stage):
        sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
        loss = tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]
    return loss


class KerasSparseIdentity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        print(input_shape)
        last_dim = input_shape.ids[-1]

        self.eye = tf.Variable(
            initial_value=tf.eye(last_dim),
            trainable=True,
            name=f"identity_eye",
        )

    def call(self, x):
        print("\nIDENTITY")
        print(x)
        return type(x)(x.ids @ self.eye, x.num_nzelements)


# def sparse2dense_ipu(spike_ids, num_spikes, dense_size):
#     assert len(spike_ids.shape) == 3

#     spike_ids = tf.transpose(spike_ids, perm=[1,0,2])
#     num_spikes = tf.transpose(num_spikes, perm=[1,0,2])
    
#     outputs = {
#         "output_types": [spike_ids.dtype],
#         "output_shapes": [tf.TensorShape([*spike_ids.shape[:2], int(dense_size)])],
#     }
    
#     base_path = os.path.realpath(os.path.dirname(__file__))
#     lib_path = os.path.join(base_path,  "..", "custom_lif_multi_layer", "sparse2dense", "libcustom_op.so")
#     gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "sparse2dense", "custom_codelet.gp")

#     return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
#                                               lib_path,
#                                               gp_path,
#                                               outs=outputs,
#                                               separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
#                                               inputs_with_gradients=[0],
#                                               attributes=f"{int(dense_size)}",
#                                             )


# def dyn_dense_binary_sparse_matmul_op(matrix: tf.Tensor, sparse_vec: SparseBinaryVec, tileMapping: TileMapping):
#     sparse_ids, num_nzelements = sparse_vec.ids, sparse_vec.num_nzelements

#     outputs = {
#         "output_types": [matrix.dtype],
#         "output_shapes": [tf.TensorShape([*sparse_ids.shape[:-1], matrix.shape[0]])],
#     }

#     base_path = os.path.realpath(os.path.dirname(__file__))
#     # lib_path = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "libcustom_op.so")
#     # gp_path  = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "custom_codelet.gp")
#     # lib_path = os.path.join(base_path, "..", "test_build", "lib", "custom_dynamic_sparse", "custom_dyn_dense_sparse_matmul", "batched", "standard", "libcustom_op.so")
#     lib_path = os.path.join(base_path, "..", "build", "custom_ops", "libcustom_dyn_dense_sparse_matmul_standard.so")
#     # gp_path  = os.path.join(base_path, "..", "source", "custom_dyn_dense_sparse_matmul", "batched", "standard", "custom_codelet.gp")

#     attributes = "_".join([str(val) for val in [tileMapping.start_tile, tileMapping.end_tile]])

#     # print(matrix.dtype, sparse_ids.dtype, num_nzelements.dtype)

#     out = ipu.custom_ops.precompiled_user_op([matrix, sparse_ids, num_nzelements],
#                                               lib_path,
#                                             #   gp_path,
#                                               name="dyn_dense_binary_sparse_matmul_op", # TF operation name
#                                               op_name="Build",
#                                               attributes=attributes,
#                                               gradient_attributes=attributes, # TODO only because of not working automatic alloc
#                                               inputs_with_gradients=[0, 1], # TODO is this working ?
#                                               separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
#                                               outs=outputs)[0]
#     return out

class SparseDenseMatmulLayer(keras.layers.Layer):
    def __init__(self, dense_size):
        super().__init__()
        assert dense_size is not None, "`dense_size` must be a scalar."
        self.dense_size = dense_size
        self.tileMapping = TileMapping(1, dense_size+1)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer(0.0, 2.0, 42) 
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.dense_size, self.dense_size), dtype=tf.float32),
            trainable=True,
            name=f"weights",
        )
    def call(self, spike_ids, num_spikes):
        return dyn_dense_binary_sparse_matmul_op(self.w, SparseBinaryVec(spike_ids, num_spikes), self.tileMapping)




class Sparse2DenseLayer(keras.layers.Layer):
    def __init__(self, dense_size):
        super().__init__()
        assert dense_size is not None, "`dense_size` must be a scalar."
        self.dense_size = dense_size

    def call(self, sparse_spikes):
        return sparse2dense_ipu(sparse_spikes, self.dense_size)

# def model_fn_sparse2dense(dense_shape, batch_size):
#     spike_ids = keras.Input(shape=[12, 4], batch_size=batch_size, dtype=tf.float32)
#     num_spikes = keras.Input(shape=[12, 4], batch_size=batch_size, dtype=tf.int32)

#     with ipu.keras.PipelineStage(0):
#         sparse2dense_outputs = Sparse2DenseLayer(dense_shape)(SparseBinaryVec(spike_ids, num_spikes))
#         print(sparse2dense_outputs)
#         sparse2dense_outputs = tf.transpose(sparse2dense_outputs, perm=[1,0,2])

#     with ipu.keras.PipelineStage(1):
#         x = keras.layers.Reshape((12*6,))(sparse2dense_outputs)
#         x = keras.layers.Dense(12*6)(x)
#         output = keras.layers.Reshape((12, 6))(x)

#     return keras.Model(inputs=[spike_ids, num_spikes], outputs=output)

def model_fn_sparse2dense(dense_shape, batch_size):
    spike_ids = keras.Input(shape=[12, 4], batch_size=batch_size, dtype=tf.float32)
    num_spikes = keras.Input(shape=[12, 4], batch_size=batch_size, dtype=tf.int32)

    with keras.ipu.PipelineStage(0):
        sparse2dense_outputs = Sparse2DenseLayer(dense_shape)(SparseBinaryVec(spike_ids, num_spikes))
        print(sparse2dense_outputs)
        sparse2dense_outputs = tf.transpose(sparse2dense_outputs, perm=[1,0,2])

    with keras.ipu.PipelineStage(1):
        x = keras.layers.Reshape((12*6,))(sparse2dense_outputs)
        x = keras.layers.Dense(12*6)(x)
        output = keras.layers.Reshape((12, 6))(x)

    return keras.Model(inputs=[spike_ids, num_spikes], outputs=output)


def sparse2dense():
    # batch_size = 2
    # N = 320
    # vector_len = 12
    # data_shape = [N, 12]
    # output_shape = [N, 1]

    # rng = np.random.default_rng(1)

    # cfg = ipu.config.IPUConfig()
    # cfg.auto_select_ipus = 2
    # cfg.configure_ipu_system()

    # strategy = ipu.ipu_strategy.IPUStrategy()
    # with strategy.scope():
    #     model = model_fn(dense_shape=6, batch_size=batch_size)
    #     model.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
    #                                  device_mapping=[ipu.pipelining_ops._ALL_DEVICES] + [1])

    #     optimizer = tf.keras.optimizers.SGD()
    #     model.compile(
    #         optimizer, keras.losses.mean_absolute_error, steps_per_execution=4
    #     )

    #     input_1 = tf.random.normal([N, 12, 4])
    #     input_2 = rng.choice(4, (N, 12, 1), replace=True).astype(np.int32)
    #     output = tf.random.normal([N, 12, 6])
        
    #     dataset = tf.data.Dataset.from_tensor_slices(
    #         (
    #             {
    #                 "input_1": input_1,
    #                 "input_2": input_2,
    #             },
    #             output,
    #         )
    #     )
    #     dataset = dataset.batch(batch_size, drop_remainder=True)

    #     model.fit(dataset, epochs=10)


    batch_size = 2
    N = 320
    vector_len = 12
    data_shape = [N, 12]
    output_shape = [N, 1]

    rng = np.random.default_rng(1)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model = model_fn_sparse2dense(dense_shape=6, batch_size=batch_size)
        model.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                                     device_mapping=[ipu.pipelining_ops._ALL_DEVICES] + [1])

        optimizer = tf.keras.optimizers.SGD()
        model.compile(
            optimizer, keras.losses.mean_absolute_error, steps_per_execution=4
        )

        input_1 = tf.random.normal([N, 12, 4])
        input_2 = rng.choice(4, (N, 12, 1), replace=True).astype(np.int32)
        output = tf.random.normal([N, 12, 6])
        
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_1": input_1,
                    "input_2": input_2,
                },
                output,
            )
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)

        model.fit(dataset, epochs=10)

def model_fn_matmul(dense_size, sparse_size, out_size, batch_size):
    spike_ids = keras.Input(shape=[sparse_size, ], batch_size=batch_size, dtype=tf.float32)
    num_spikes = keras.Input(shape=[1, ], batch_size=batch_size, dtype=tf.int32)

    with keras.ipu.PipelineStage(0):
        x = SparseDenseMatmulLayer(dense_size)(spike_ids, num_spikes)

    with keras.ipu.PipelineStage(1):
        # x = keras.layers.Reshape((12*6,))(sparse2dense_outputs)
        output = keras.layers.Dense(out_size)(x)
        # output = keras.layers.Reshape((12, 6))(x)

    return keras.Model(inputs=[spike_ids, num_spikes], outputs=output)


def sparse_matmul():
    batch_size = 64
    num_samples = 16*batch_size
    dense_size = 32
    sparse_size = 16
    out_size = 8

    rng = np.random.default_rng(1)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model = model_fn_matmul(dense_size=dense_size, sparse_size=sparse_size, out_size=out_size, batch_size=batch_size)
        model.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                                     device_mapping=[ipu.pipelining_ops._ALL_DEVICES] + [1])

        optimizer = tf.keras.optimizers.SGD()
        model.compile(
            optimizer, keras.losses.mean_absolute_error, steps_per_execution=4
        )

        input_1, input_2 = gen_sparse_spikes(rng, 1, num_samples, dense_size, sparse_size)
        output = tf.random.normal([num_samples, out_size])
        
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_1": input_1[0],
                    "input_2": input_2[0],
                },
                output,
            )
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)

        model.fit(dataset, epochs=10)


# def simple_loss_fn_sparse(time_axis, y_target, y_pred: SparseBinaryVec):
#     dense_spikes = tf.transpose(sparse2dense_ipu(y_pred, y_target.shape[-1]), perm=[1, 0, 2])
#     sum_spikes = tf.reduce_sum(dense_spikes, axis=time_axis) # (batch, seq, neurons)

#     print(dense_spikes.shape)
#     print(sum_spikes.shape)
#     print(y_target.shape)

#     return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def model_fn_sparse_ops(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, transpose_weights=False, return_all=False, seed=None):

    if transpose_weights:
        raise ValueError("`transpose_weights` for sparse ops is not implemented yet.")

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.float32, name="num_inp_spikes")
    
    inp_spikes = SparseBinaryVec(inp_spike_ids, num_inp_spikes)


    with keras.ipu.PipelineStage(0):
        out_spikes = KerasMultiLIFLayerSparseOps(
                dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step, seed, return_sequences=True
            )(inp_spikes)
        # if return_all:
        #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
        # else:
        #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
        #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
        #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])


        # print(out_spikes)

        if return_all:
            out = out_spikes
        else:
            out = out_spikes[-1]
    
    with keras.ipu.PipelineStage(1):
        # print(out)
        # print(out.ids.shape)
        # print(out.num_nzelements.shape)

        dense_spikes = sparse2dense_ipu(out, dense_shapes[-1])
        output = dense_spikes
        # out = tf.transpose(dense_spikes, perm=[1, 0, 2])
        # out = KerasSparseIdentity()(out)

    return (inp_spike_ids, num_inp_spikes), output

def model_fn_sparse_layer(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, transpose_weights=False, return_all=False, seed=None, num_ipus=1, weight_mul=1.0):
    if return_all:
        warnings.warn("All layers outputs will be returned. But note that only gradient propagation through the last layers outputs is implemented."
                    " Adding loss terms to other layers outputs will be ignored and will result in a wrong gradient.", UserWarning)

    num_layers = len(dense_shapes)-1

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    # spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype ) for ilay in range(1, num_layers)]]
    # num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype ) for ilay in range(1, num_layers)]]
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.repeat(tf.expand_dims(tf.range(0, sparse_shapes[ilay], delta=1,  dtype=inp_spike_ids.dtype), axis=0), batchsize_per_step, axis=0) for ilay in range(1, num_layers)]]
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.repeat(tf.expand_dims(tf.range(dense_shapes[ilay]-5, dense_shapes[ilay]-5+sparse_shapes[ilay], delta=1,  dtype=inp_spike_ids.dtype), axis=0), batchsize_per_step, axis=0) for ilay in range(1, num_layers)]]
    # num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.cast(tf.fill((batchsize_per_step,1), sparse_shapes[ilay]), dtype=num_inp_spikes.dtype) for ilay in range(1, num_layers)]]
    inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

    init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32, name=f"init_state_{i}") for i in range(num_layers)]
    out = KerasMultiLIFLayerSparse(
            dense_shapes, sparse_shapes, decay_constant, threshold, transpose_weights, seed, num_ipus, weight_mul
        )(inp_spikes, init_states)
    out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]

    # if return_all:
    #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    # else:
    #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
    #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
    #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])

    if return_all:
        out = [SparseBinaryVec(ids, num_nzelements) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    else:
        out = SparseBinaryVec(out_spike_ids[-1], num_out_spikes[-1])

    out = KerasSparseIdentity()(out)
    dense_spikes = sparse2dense_ipu(out, dense_shapes[-1])
    output = tf.transpose(dense_spikes, perm=[1, 0, 2])

    return (inp_spike_ids, num_inp_spikes), output


class KerasSparseIdentity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        last_dim = input_shape.ids[-1]

        self.eye = tf.Variable(
            initial_value=tf.eye(last_dim),
            trainable=True,
            name=f"identity_eye",
        )

    def call(self, x):
        return type(x)(x.ids @ self.eye, x.num_nzelements)


def model_fn_sparse_layer_multi_ipu(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, transpose_weights=False, return_all=False, seed=None, num_ipus=1, weight_mul=1.0):
    if return_all:
        warnings.warn("All layers outputs will be returned. But note that only gradient propagation through the last layers outputs is implemented."
                    " Adding loss terms to other layers outputs will be ignored and will result in a wrong gradient.", UserWarning)

    num_layers = len(dense_shapes)-1
    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.float32, name="num_inp_spikes")

    assert num_ipus > 1

    with ipu.keras.PipelineStage(0):
        
        # spike_ids_0 = tf.transpose(inp_spike_ids, perm=[1, 0, 2], name="inp_spike_ids_0_transp")
        # num_spikes_0 = tf.transpose(inp_spike_ids, perm=[1, 0, 2], name="inp_spike_ids_0_transp")

        # spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2], name="inp_spike_ids_0_transp"), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype, name=f"intial_spike_ids_{ilay}") for ilay in range(1, num_layers)]]
        # num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2], name="nup_inp_spikes_0_transp"), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype, name=f"intial_num_spikes_{ilay}") for ilay in range(1, num_layers)]]
        spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.repeat(tf.expand_dims(tf.range(0, sparse_shapes[ilay], delta=1,  dtype=inp_spike_ids.dtype), axis=0), batchsize_per_step, axis=0) for ilay in range(1, num_layers)]]
        num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.cast(tf.fill((batchsize_per_step,1), sparse_shapes[ilay]), dtype=num_inp_spikes.dtype) for ilay in range(1, num_layers)]]
        inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

        init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32, name=f"init_state_{i}") for i in range(num_layers)]
    
        out = KerasMultiLIFLayerSparse(
                dense_shapes, sparse_shapes, decay_constant, threshold, transpose_weights, seed, num_ipus, weight_mul
            )(inp_spikes, init_states)
        out_spike_ids, num_out_spikes, states = out[:num_layers], out[num_layers:2*num_layers], out[2*num_layers:]

        # if return_all:
        #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
        # else:
        #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
        #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
        #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])

        if return_all:
            raise ValueError("curretly `return_all=True` is not supported")
            out = [SparseBinaryVec(ids, num_nzelements) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
        else:
            out = SparseBinaryVec(out_spike_ids[-1], num_out_spikes[-1])


    for iipu in range(1,num_ipus-1):
        with ipu.keras.PipelineStage(iipu):
            out = KerasSparseIdentity()(out)

    with ipu.keras.PipelineStage(num_ipus-1):
        # if num_ipus < 12:
        #     out = KerasSparseIdentity()(out)
        dense_spikes = sparse2dense_ipu(out, dense_shapes[-1])
        output = tf.transpose(dense_spikes, perm=[1, 0, 2])

    # for ipu_id in range(2, num_ipus):
    #     with ipu.keras.PipelineStage(ipu_id):
    #         output = tf.matmul(output, tf.eye(output.shape[-1]), transpose_b=False)
    
    return (inp_spike_ids, num_inp_spikes), output


def train_ipu(method, NUM_IPUS):

    rng = np.random.default_rng(1)

    batch_size = 64
    num_samples = 16*batch_size
    # dense_sizes = [32, 33, 34]
    # sparse_sizes = [15, 16, 17]
    # dense_sizes = [2942, 2942, 1470, 1470, 1000, 1000, 1000, 256, 8]
    # sparse_sizes = [32, 128, 64, 64, 48, 48, 48, 16, 8]
    # dense_sizes = [2942, 1472, 1468, 1002, 1000, 256, 8]
    # sparse_sizes = [32, 64, 64, 48, 48, 16, 8]
    
    # dense_sizes = [100, 1470, 1370, 8]
    # sparse_sizes = [32, 64, 64, 8]
    dense_sizes = [100, 1470, 1470, 1470, 8]
    # dense_sizes = [100, 70, 70, 70, 8]
    sparse_sizes = [32, 64, 64, 64, 8]

    dense_sizes = [2940, 1470, 1470, 1470, 1370, 8]
    sparse_sizes = [32, 64, 64, 64, 64, 8]


    dense_sizes = [2940, *[1470, 1470]*(NUM_IPUS-1), 1470, 1370, 8]
    sparse_sizes = [32, *[64, 64]*(NUM_IPUS-1), 64, 64, 8]

    # dense_sizes = [2940, 1472, 1470, 1474, 1370, 8]
    # sparse_sizes = [32, 64, 64, 64, 64, 8]


    # # # weird error:
    # # '''
    # # tensorflow.python.framework.errors_impl.InternalError: [Poplar][Compile engine] graph_cycle_error: tensorflow/compiler/plugin/poplar/driver/poplar_compiler.cc:1923 Field LIFStateUpdateInPlaceTwoNeuronSIMD.state writes to elements [0:2) of variable 'keras_multi_lif_layer_sparse/custom_multi_lif_layer_sparse/user-op.13:3/init_state' that are written to elsewhere via Field LIFStateUpdateInPlaceTwoNeuronSIMD.state in compute set 21
    # # This is an error because all vertices in a compute set can run concurrently and the order of accesses is not guaranteed. To fix this place these two vertices in different compute sets. [Op:__inference_pipeline_function_1240]
    # # '''
    # dense_sizes = [100, 256, 256, 8]
    # sparse_sizes = [32, 64, 64, 8]

    # dense_sizes = [100, 256, 256, 8]
    # sparse_sizes = [32, 64, 64, 8]
    # dense_sizes = [100, 128, 126, 8]
    # sparse_sizes = [32, 64, 64, 8]
    # dense_sizes = [100, 128, 8]
    # sparse_sizes = [32, 48, 8]
    # # dense_sizes =  list(range(32, 32+4))
    # # # sparse_sizes = list(range(15, 15+5))
    # # sparse_sizes = list(range(32, 32+4))


    seq_len = 12

    decay_constant = 0.9 
    threshold = 1.0 
    return_all = False
    # # loss_fn = ft.partial(simple_loss_fn_sparse, 1)
    # loss_fn = simple_loss_fn_sparse
    loss_fn = simple_loss_fn_dense
    # loss_fn = ft.partial(simple_loss_fn_dense_multi_ipu, 1)

    # NUM_IPUS = 1

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = NUM_IPUS
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    # if method is None:
    #     method = "sparse_layer"

    assert method in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{method}'."


    if NUM_IPUS > 1:
        method_to_model_fn = {
            # "dense": model_fn_dense, 
            "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_sizes, transpose_weights=False),
            "sparse_layer": ft.partial(model_fn_sparse_layer_multi_ipu, sparse_sizes, transpose_weights=True, num_ipus=NUM_IPUS),
        }
    else:
        method_to_model_fn = {
            # "dense": model_fn_dense, 
            "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_sizes, transpose_weights=False),
            "sparse_layer": ft.partial(model_fn_sparse_layer, sparse_sizes, transpose_weights=True),
        }

    input_1, input_2 = gen_sparse_spikes(rng, seq_len, num_samples, dense_sizes[0], sparse_sizes[0])
    output = tf.random.normal([num_samples, dense_sizes[-1]])
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "inp_spike_ids": input_1.transpose(1,0,2),
                "num_inp_spikes": input_2.transpose(1,0,2),
                # "targets": output
            },
            output
        )
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)


    with strategy.scope():
        # init model

        inputs, outputs = method_to_model_fn[method](seq_len, dense_sizes, decay_constant, threshold, batch_size, return_all=return_all)
        # targets = keras.Input((1,), name="targets")
        # model = keras.Model([inputs, targets], outputs)
        model = keras.Model(inputs, outputs)

        if NUM_IPUS > 1:
            device_mapping = [ ipu.pipelining_ops._ALL_DEVICES, *[NUM_IPUS-1]*(NUM_IPUS-1)]
            # print("\ndevice_mapping")
            # print(device_mapping)
            # sys.exit()
            model.set_pipelining_options(
                # pipeline_schedule=ipu.keras.pipeline.SequentialPipelineModel,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Sequential,
                gradient_accumulation_steps_per_replica=1,
                device_mapping=device_mapping,
                offload_weight_update_variables=False,
                # device_mapping=[0, 1]
            )
        # model.load_weights("./model_save_weights")
        # # model = tf.keras.models.load_model("./model_save", compile=False)

        # # Set the infeed and outfeed options.
        # model.set_infeed_queue_options(prefetch_depth=2)
        # model.set_outfeed_queue_options(buffer_depth=2)

        optim = tf.keras.optimizers.Adam(learning_rate=1e-2) # NOTE 1e-2 worked quite well
        # optim = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=0.9, nesterov=False, name="SGD")

        # model.add_loss(loss_fn(targets, outputs))
        # if metrics is not None:
        #     if not isinstance(metrics, (list, tuple)):
        #         metrics = [metrics]
        #     for metric in metrics:
        #         model.add_metric(metric(targets, outputs), metric.__name__)
        model.compile(optim, loss_fn,
                    steps_per_execution=4, # TODO change this
        )

        model.summary()
        # print(model.get_pipeline_stage_assignment())
        # print(model._pipeline_stage_assignment)
        # _pipeline_stage_assignment

        start_time = time.time()

        print('\nTraining')
        model.fit(dataset, epochs=4)


        print("\nFinal time: ", time.time()-start_time)


def create_dataset_sparse_multi_ipu(inp_spike_ids, num_inp_spikes, labels, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes}, labels))
    num_samples = labels.shape[0]
    if shuffle:
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    # dataset = dataset.repeat()
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.prefetch(4)
    return dataset

def train_mutli_ipu_benchmarking(
        method,
        num_epochs,
        train_steps_per_execution,
        batch_size,
        dataset,
        seq_len, 
        dense_sizes, 
        sparse_sizes, 
        decay_constant, 
        threshold,
        loss_fn,
        steps_per_epoch=None,
        callbacks=None,
        return_all=False,
        transpose_weights=False,
        learning_rate=1e-2,
        num_ipus=1,
        seed=None,
        weight_mul=1.0,
        ipu_id=None,
        opt=None,
        **optim_kwargs
    ):
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    if ipu_id is None:
        ipu_config.auto_select_ipus = num_ipus
    else:
        ipu_config.select_ipus = ipu_id
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    assert method in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{method}'."


    if num_ipus > 1:
        method_to_model_fn = {
            # "dense": model_fn_dense, 
            "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_sizes, transpose_weights=False),
            "sparse_layer": ft.partial(model_fn_sparse_layer_multi_ipu, sparse_sizes, transpose_weights=True, num_ipus=num_ipus, weight_mul=weight_mul),
        }
    else:
        method_to_model_fn = {
            # "dense": model_fn_dense, 
            "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_sizes, transpose_weights=False),
            "sparse_layer": ft.partial(model_fn_sparse_layer, sparse_sizes, transpose_weights=True, weight_mul=weight_mul),
        }

    with strategy.scope():
        # init model
        inputs, outputs = method_to_model_fn[method](seq_len, dense_sizes, decay_constant, threshold, batch_size, return_all=return_all)
        model = keras.Model(inputs, outputs)

        if num_ipus > 1:
            device_mapping = [ ipu.pipelining_ops._ALL_DEVICES, *[num_ipus-1]*(num_ipus-1)]
            model.set_pipelining_options(
                # pipeline_schedule=ipu.keras.pipeline.SequentialPipelineModel,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Sequential,
                gradient_accumulation_steps_per_replica=1,
                device_mapping=device_mapping,
                offload_weight_update_variables=False,
                # device_mapping=[0, 1]
            )

        # Set the infeed and outfeed options.
        model.set_infeed_queue_options(prefetch_depth=2)
        model.set_outfeed_queue_options(buffer_depth=2)

        if opt is not None:
            optim = opt(learning_rate=learning_rate, **optim_kwargs)
        else:
            optim = tf.keras.optimizers.Adam(learning_rate=learning_rate, **optim_kwargs)

        # optim = tf.keras.optimizers.Adam(learning_rate=learning_rate, **optim_kwargs) # NOTE 1e-2 worked quite well
        # optim = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=0.9, nesterov=False, name="SGD")

        model.compile(optim, loss_fn,
                    # metrics=metrics,
                    steps_per_execution=train_steps_per_execution,
        )
        model.summary()
        start_time = time.time()
        print('\nTraining')
        model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, workers=batch_size, callbacks=callbacks)
        print("\nFinal time: ", time.time()-start_time)





if __name__ == "__main__":
    # sparse2dense() # TODO broken
    # sparse_matmul()
    # train_ipu("sparse_ops")
    train_ipu("sparse_layer", NUM_IPUS=2)
    # main()
