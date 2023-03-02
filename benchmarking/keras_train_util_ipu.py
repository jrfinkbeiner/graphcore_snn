import os
import warnings
import sys
import functools as ft
from typing import Union, NamedTuple, List

import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_addons as tfa


from tensorflow.python import ipu

from keras_train_util import KerasMultiLIFLayerBase, model_fn_dense, create_dataset_dense

class SparseBinaryVec(NamedTuple):
    ids: Union[tf.Tensor, tf.TensorShape]
    num_nzelements: Union[tf.Tensor, tf.TensorShape, int]

class TileMapping(NamedTuple):
    start_tile: int
    end_tile: int

def determine_neuron_tileMappings(dense_sizes, sparse_sizes, num_ipus=1, min_neurons_per_tile=1):

    if num_ipus > 1:
        tileMapping =  determine_neuron_tileMappings_multiIPU(dense_sizes, sparse_sizes, num_ipus, min_neurons_per_tile)
    else:
        tileMapping =  determine_neuron_tileMappings_multiIPU(dense_sizes, sparse_sizes, num_ipus, min_neurons_per_tile)
        # tileMapping =  determine_neuron_tileMappings_singleIPU(dense_sizes, sparse_sizes, min_neurons_per_tile)
    return tileMapping


def tile_mapping_const_number_states_per_tile(num_neurons, neurons_per_tile, TILES_PER_IPU, TILE_OFFSET):

    num_ipus = len(TILE_OFFSET)
    USABLE_TILES_PER_IPU = [int(TILES_PER_IPU - tile_offs) for tile_offs in TILE_OFFSET]

    num_neurons_total = np.sum(num_neurons).astype(np.float64)

    layerwise_max_tiles = np.ceil(num_neurons / neurons_per_tile)
    cumsum_num_neurons = np.cumsum(num_neurons)
    cumsum_max_tiles = cumsum_num_neurons.astype(np.float64) / neurons_per_tile
    tile_mapping_possible = True

    cumsum_tiles = 0
    ipu_id = 0
    start_tiles = []
    end_tiles = []
    tileMappings = []
    for ilay,max_tiles_ilay in enumerate(layerwise_max_tiles):
        print(ilay, max_tiles_ilay)
        if max_tiles_ilay > USABLE_TILES_PER_IPU[ipu_id]:
            tile_mapping_possible = False
            break
        new_cumsum_tiles = cumsum_tiles + max_tiles_ilay
        # check whether additonal layer fits on current IPU, otherwise start mapping on next IPU
        if new_cumsum_tiles > USABLE_TILES_PER_IPU[ipu_id]:
            cumsum_tiles = 0
            new_cumsum_tiles = max_tiles_ilay
            ipu_id += 1
        
        if ipu_id >= num_ipus:
            tile_mapping_possible = False
            break
        
        start_tile = int(cumsum_tiles + TILE_OFFSET[ipu_id] + ipu_id * TILES_PER_IPU)
        end_tile = int(start_tile + max_tiles_ilay)
        tileMappings.append(TileMapping(start_tile, end_tile))
        cumsum_tiles = new_cumsum_tiles

    if tile_mapping_possible:
        return tileMappings
    else:
        return None


def determine_neuron_tileMappings_multiIPU(dense_sizes, sparse_sizes, num_ipus, min_neurons_per_tile):
    
    TILE_OFFSET = [1] + [0]*(num_ipus-1)
    TILES_PER_IPU = 1472 # hardcoded fpr IPUv2 MK2000
    USABLE_TILES_PER_IPU = [int(TILES_PER_IPU - tile_offs) for tile_offs in TILE_OFFSET]
    # USABLE_TILES_TOTAL = sum(USABLE_TILES_PER_IPU)

    num_neurons = np.asarray(dense_sizes[1:], dtype=np.int64)
    num_neurons_total = np.sum(num_neurons).astype(np.float64)

    # max_num_tiles_to_use = np.ceil(num_neurons_total / min_neurons_per_tile)
    # if max_num_tiles_to_use < USABLE_TILES_PER_IPU[0]:
    #     tileMapping = determine_neuron_tileMappings_singleIPU(dense_sizes, sparse_sizes, min_neurons_per_tile, tile_offset=TILE_OFFSET[0])
    #     return tileMapping
    
    neurons_per_tile = min_neurons_per_tile
    tile_mapping_found = False
    while not tile_mapping_found:
        print(f"\nneurons_per_tile={neurons_per_tile}")
        print(f"num_neurons={num_neurons}")
        tile_mapping = tile_mapping_const_number_states_per_tile(num_neurons, neurons_per_tile, TILES_PER_IPU, TILE_OFFSET)
        neurons_per_tile += min_neurons_per_tile
        print(tile_mapping)
        if tile_mapping is not None:
            tile_mapping_found = True
        print(f"tile_mapping_found={tile_mapping_found}")

    return tile_mapping
    
    # layerwise_max_tiles = np.ceil(num_neurons / min_neurons_per_tile)
    # cumsum_num_neurons = np.cumsum(dense_sizes[1:])
    # cumsum_max_tiles = cumsum_num_neurons.astype(np.float64) / min_neurons_per_tile
    # max_tile_mapping_possible = True

    # cumsum_tiles = 0
    # ipu_id = 0
    # start_tiles = []
    # end_tiles = []
    # tileMappings = []
    # for ilay,max_tiles_ilay in enumerate(layerwise_max_tiles):
    #     print(ilay, max_tiles_ilay)
    #     if max_tiles_ilay > USABLE_TILES_PER_IPU[ipu_id]:
    #         max_tile_mapping_possible = False
    #         break
    #     new_cumsum_tiles = cumsum_tiles + max_tiles_ilay
    #     # check whether additonal layer fits on current IPU, otherwise start mapping on next IPU
    #     if new_cumsum_tiles > USABLE_TILES_PER_IPU[ipu_id]:
    #         cumsum_tiles = 0
    #         new_cumsum_tiles = max_tiles_ilay
    #         ipu_id += 1
        
    #     if ipu_id >= num_ipus:
    #         max_tile_mapping_possible = False
    #         break
        
    #     start_tile = int(cumsum_tiles + TILE_OFFSET[ipu_id] + ipu_id * TILES_PER_IPU)
    #     end_tile = int(start_tile + max_tiles_ilay)
    #     tileMappings.append(TileMapping(start_tile, end_tile))
    #     cumsum_tiles = new_cumsum_tiles

    # if max_tile_mapping_possible:
    #     return tileMappings


    # raise NotImplementedError("multiIpu Mapping not fully impleneted yet. Tilemapping on one IPU was not desireable and mapping with min_neurons_per_tile not possible.")

    

def determine_neuron_tileMappings_singleIPU(dense_sizes, sparse_sizes, min_neurons_per_tile, tile_offset=1):

    neuron_tileMappings = []

    dense_sizes = np.asarray(dense_sizes, dtype=np.int64)
    sparse_sizes = np.asarray(sparse_sizes, dtype=np.int64)
    num_neurons_fptype = dense_sizes[1:].astype(np.float64)

    num_tiles = 1472 # hardcoded fpr IPUv2 MK2000
    # tile_offset = 1
    max_num_tiles_to_use = num_tiles-tile_offset # TODO substract batchsize because of mapped operations?
    num_layers = len(dense_sizes)-1

    weighted_num_neurons_total = int(np.sum(dense_sizes[1:]*sparse_sizes[:-1]))

    weight_factor = sparse_sizes[:-1] # weighting based on input spikes for now
    num_tiles_fptype = (num_neurons_fptype * weight_factor * max_num_tiles_to_use).astype(np.float64) / float(weighted_num_neurons_total)
    num_neurons_per_tile_ilay = np.ceil(num_neurons_fptype / num_tiles_fptype)
    num_neurons_per_tile_ilay = np.maximum(np.full_like(num_neurons_per_tile_ilay, min_neurons_per_tile), num_neurons_per_tile_ilay)
    num_tiles_ilay = np.ceil(num_neurons_fptype / num_neurons_per_tile_ilay).astype(np.int32)        
    num_neurons_per_tile_ilay = num_neurons_per_tile_ilay.astype(np.int32)

    end_tiles = np.cumsum(num_tiles_ilay)+tile_offset
    start_tiles = np.empty_like(end_tiles)
    start_tiles[1:] = end_tiles[:-1]
    start_tiles[0] = tile_offset

    tileMappings = [TileMapping(stt, endt) for stt, endt in zip(start_tiles, end_tiles)]
    return tileMappings

def determine_spikeGen_tileMappings(num_layers, batch_size):
    num_tiles = 1472 # hardcoded fpr IPUv2 MK2000

    start_tiles = num_tiles-np.arange(1,num_layers+1)*batch_size
    end_tiles = start_tiles+batch_size
    if start_tiles[0] < 0:
        raise RuntimeError(f"Spike generation tilemapping does not work. Only works for `num_layers * batch_size < num_tiles = 1472`, but got {num_layers * batch_size}.")

    tileMappings = [TileMapping(stt, endt) for stt, endt in zip(start_tiles, end_tiles)]

    print("\ndetermine_spikeGen_tileMappings")
    print(start_tiles)
    print(end_tiles)

    return tileMappings

def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse)).astype(np.float32)
    for ibatch in range(batchsize):
        for iseq in range(seq_len):
            sparse_spike_ids[iseq, ibatch, :] = rng.choice(size_dense, size_sparse, replace=False)
    num_sparse_spikes = rng.choice(size_sparse, (seq_len, batchsize, 1), replace=True).astype(np.float32)
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
                ids = spike_ids[iseq, ibatch, :int(num_spikes[iseq, ibatch, 0])].astype(np.int32)

            if values is None:
                dense_tensor[iseq, ibatch, ids] = 1
            else:
                if num_spikes is None:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :]
                else:
                    dense_tensor[iseq, ibatch, ids] = values[iseq, ibatch, :num_spikes[iseq, ibatch, 0]]
    return dense_tensor

def sparse2dense_ipu(spikes: SparseBinaryVec, dense_size: int):
    spike_ids, num_spikes = spikes.ids, spikes.num_nzelements
    assert len(spike_ids.shape) == 3, f"`spike_ids` must be tensor of rank 3, got {len(spike_ids.shape)}."
    outputs = {
        "output_types": [spike_ids.dtype],
        "output_shapes": [tf.TensorShape([*spike_ids.shape[:-1], dense_size])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "sparse2dense", "libcustom_op.so")
    gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "sparse2dense", "custom_codelet.gp")

    return ipu.custom_ops.precompiled_user_op([spike_ids, num_spikes],
                                              lib_path,
                                              gp_path,
                                              outs=outputs,
                                              separate_gradients=False, # to calculate gradients separately
                                              inputs_with_gradients=[0], # TODO is this working ?
                                              attributes=f"{dense_size}",
                                            )[0]


def dyn_dense_binary_sparse_matmul_op(matrix: tf.Tensor, sparse_vec: SparseBinaryVec, tileMapping: TileMapping):
    sparse_ids, num_nzelements = sparse_vec.ids, sparse_vec.num_nzelements

    outputs = {
        "output_types": [matrix.dtype],
        "output_shapes": [tf.TensorShape([*sparse_ids.shape[:-1], matrix.shape[0]])],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "libcustom_op.so")
    # gp_path  = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "custom_codelet.gp")
    # lib_path = os.path.join(base_path, "..", "test_build", "lib", "custom_dynamic_sparse", "custom_dyn_dense_sparse_matmul", "batched", "standard", "libcustom_op.so")
    lib_path = os.path.join(base_path, "..", "build", "custom_ops", "libcustom_dyn_dense_sparse_matmul_standard.so")
    # gp_path  = os.path.join(base_path, "..", "source", "custom_dyn_dense_sparse_matmul", "batched", "standard", "custom_codelet.gp")

    attributes = "_".join([str(val) for val in [tileMapping.start_tile, tileMapping.end_tile]])

    print(matrix.dtype, sparse_ids.dtype, num_nzelements.dtype)

    out = ipu.custom_ops.precompiled_user_op([matrix, sparse_ids, num_nzelements],
                                              lib_path,
                                            #   gp_path,
                                              name="dyn_dense_binary_sparse_matmul_op", # TF operation name
                                              op_name="Build",
                                              attributes=attributes,
                                              gradient_attributes=attributes, # TODO only because of not working automatic alloc
                                              inputs_with_gradients=[0, 1], # TODO is this working ?
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              outs=outputs)[0]
    print(out.dtype)
    return out

def dyn_dense_binary_sparse_matmul_op_parallel(matrix: List[tf.Tensor], sparse_vec: List[SparseBinaryVec], tileMappings: List[TileMapping], first_layer_grad_bool: bool):
    assert isinstance(matrix, (list, tuple))
    num_layers = len(matrix)
    assert len(sparse_vec) == num_layers
    assert len(tileMappings) == num_layers
        
    sparse_ids = [spvec.ids for spvec in sparse_vec] 
    num_nzelements = [spvec.num_nzelements for spvec in sparse_vec]

    outputs = {
        "output_types": [mat.dtype for mat in matrix],
        "output_shapes": [tf.TensorShape([*spids.shape[:-1], mat.shape[0]]) for spids, mat in zip(sparse_ids, matrix)],
    }

    # inp_types = [tensor.dtype for tensor in [*matrix, *sparse_ids, *num_nzelements]]
    # print("inp_types")
    # print(inp_types)
    # print("outputs['output_types']")
    # print(outputs["output_types"])
    # # sys.exit()
    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "libcustom_op.so")
    # gp_path  = os.path.join(base_path, "..", "custom_dyn_dense_sparse_matmul", "batched", "custom_codelet.gp")
    # lib_path = os.path.join(base_path, "..", "test_build", "lib", "custom_dynamic_sparse", "custom_dyn_dense_sparse_matmul", "batched", "standard", "libcustom_op.so")
    lib_path = os.path.join(base_path, "..", "build", "custom_ops", "libcustom_dyn_dense_sparse_matmul_standard_parallel.so")
    # gp_path  = os.path.join(base_path, "..", "source", "custom_dyn_dense_sparse_matmul", "batched", "standard", "custom_codelet.gp")

    start_tiles = [tileMapping.start_tile for tileMapping in tileMappings]
    end_tiles = [tileMapping.end_tile for tileMapping in tileMappings]
    attributes = "_".join([str(val) for val in [*start_tiles, *end_tiles]])

    print(f"attributes={attributes}")

    inputs_with_gradients = list(range(2*num_layers)) if first_layer_grad_bool else [*list(range(num_layers)), *list(range(num_layers+1, 2*num_layers))] 

    out = ipu.custom_ops.precompiled_user_op([*matrix, *sparse_ids, *num_nzelements],
                                              lib_path,
                                            #   gp_path,
                                              name="dyn_dense_binary_sparse_matmul_parallel_op", # TF operation name
                                              op_name="Build",
                                              attributes=attributes,
                                              gradient_attributes=attributes, # TODO only because automatic alloc is not working 
                                              inputs_with_gradients=inputs_with_gradients, # TODO is this working ?
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              outs=outputs)
    return out


def compute_sparse_spikes(state: tf.Tensor, thresholds: tf.Tensor, sparse_size: int, tileMapping: TileMapping):

    batch_size = state.shape[0]
    outputs = {
        "output_types": [state.dtype, tf.int32],
        "output_shapes": [tf.TensorShape((batch_size, sparse_size)), tf.TensorShape((batch_size, 1))],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "libcustom_op.so")
    # gp_path  = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "custom_codelet.gp")
    # lib_path = os.path.join(base_path, "..", "test_build", "lib", "custom_dynamic_sparse", "custom_select_spikes", "twoThresh", "libcustom_op.so")
    lib_path = os.path.join(base_path, "..", "build", "custom_ops", "libcustom_select_spikes_twoThresh.so")
    # lib_path = os.path.join(base_path, "..", "source", "custom_select_spikes", "twoThresh", "libcustom_op.so")

    attributes = "_".join([str(val) for val in [sparse_size, tileMapping.start_tile, tileMapping.end_tile]])
    out = ipu.custom_ops.precompiled_user_op([state, thresholds],
                                              lib_path,
                                            #   gp_path,
                                              name="compute_sparse_spikes_op", # TF operation name
                                              op_name="Build",
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=attributes,
                                              inputs_with_gradients=[0],
                                              gradient_attributes=attributes,
                                              outs=outputs)
    spike_ids, num_spikes = out[0], out[1]
    return SparseBinaryVec(spike_ids, num_spikes)

def compute_sparse_spikes_parallel(states: List[tf.Tensor], thresholds: List[tf.Tensor], sparse_sizes: List[int], tileMappings: List[TileMapping]):

    assert isinstance(states, (list, tuple))
    num_layers = len(states)
    assert len(thresholds) == num_layers
    assert len(sparse_sizes) == num_layers
    assert len(tileMappings) == num_layers

    batch_size = states[0].shape[0]
    outputs = {
        "output_types": [
            *[state.dtype for state in states], 
            # *[tf.int32 for _ in range(num_layers)] # TODO change when possible
            *[tf.float32 for _ in range(num_layers)]
        ],
        "output_shapes": [
            *[tf.TensorShape((batch_size, sparse_size)) for sparse_size in sparse_sizes],
            *[tf.TensorShape((batch_size, 1)) for i in range(num_layers)]
        ],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    # lib_path = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "libcustom_op.so")
    # gp_path  = os.path.join(base_path, "..", "custom_select_spikes", "twoThresh", "custom_codelet.gp")
    # lib_path = os.path.join(base_path, "..", "test_build", "lib", "custom_dynamic_sparse", "custom_select_spikes", "twoThresh", "libcustom_op.so")
    lib_path = os.path.join(base_path, "..", "build", "custom_ops", "libcustom_select_spikes_twoThresh_parallel.so")
    # lib_path = os.path.join(base_path, "..", "source", "custom_select_spikes", "twoThresh", "libcustom_op.so")


    start_tiles = [tileMapping.start_tile for tileMapping in tileMappings]
    end_tiles = [tileMapping.end_tile for tileMapping in tileMappings]
    attributes = "_".join([str(val) for val in [*sparse_sizes, *start_tiles, *end_tiles]])
    out = ipu.custom_ops.precompiled_user_op([*states, *thresholds],
                                              lib_path,
                                              name="compute_sparse_spikes_parallel_op", # TF operation name
                                              op_name="Build",
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=attributes,
                                              inputs_with_gradients=list(range(num_layers)),
                                              gradient_attributes=attributes,
                                              outs=outputs)

    sparse_spikes = [SparseBinaryVec(spike_ids, num_spikes) for spike_ids, num_spikes in zip(out[:num_layers], out[num_layers:])]
    return sparse_spikes


# TODO could also implement a self-recurrent version
def custom_multi_lif_layer_sparse(sparse_sizes, transpose_weights, weights, init_state, sparse_inp_spikes, decay_constants, thresholds, tileMappings):

    inp_spike_ids = [t.ids for t in sparse_inp_spikes] 
    num_inp_spikes = [t.num_nzelements for t in sparse_inp_spikes]

    # TODO check that batch and seq_lens are consitent for all states, inp_spike_ids, num_inp_spikes
    assert isinstance(weights, (list, tuple))
    num_layers = len(weights)

    seq_and_batch_size = num_inp_spikes[0].shape[:2]
    seq_len = num_inp_spikes[0].shape[0]
    batch_size = num_inp_spikes[0].shape[1]

    output_types = [
        *[inp_spike_ids[ilay].dtype for ilay in range(num_layers)],
        # *[num_inp_spikes[ilay].dtype for ilay in range(num_layers)],  # TODO uncomment
        *[tf.float32 for ilay in range(num_layers)],
        *[init_state[ilay].dtype for ilay in range(num_layers)],
    ]
    output_shapes = [
        *[tf.TensorShape([*seq_and_batch_size, sparse_sizes[ilay]]) for ilay in range(1, num_layers+1)], 
        *[tf.TensorShape([*seq_and_batch_size, 2]) for ilay in range(1, num_layers+1)],
        *[tf.TensorShape([seq_len, *init_state[ilay].shape]) for ilay in range(num_layers)],
    ]

    outputs = {
        "output_types": output_types,
        # "output_types": [inp_spike_ids.dtype, num_inp_spikes.dtype, init_state.dtype], # TODO uncomment when int isuue is fixed
        "output_shapes": output_shapes
    }

    base_path = os.path.realpath(os.path.dirname(__file__))

    if transpose_weights:
        lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize_transpose", "libcustom_op.so")
        gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize_transpose", "custom_codelet.gp")
    else:
        # lib_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "libcustom_op.so")
        # gp_path = os.path.join(base_path, "custom_lif_layer_loop_noCopy", "custom_codelet.gp")
        # lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_baseline", "libcustom_op.so")
        # gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_baseline", "custom_codelet.gp")
        # lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_dynamic", "libcustom_op.so")
        # gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_dynamic", "custom_codelet.gp")
        lib_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize", "libcustom_op.so")
        gp_path = os.path.join(base_path, "..", "custom_lif_multi_layer", "custom_lif_layer_vectorize", "custom_codelet.gp")



    # # sparse_sizes_str = "_".join([str(val) for val in sparse_sizes])
    dense_sizes = [w.shape[0] for w in weights]
    # if transpose_weights:
    #     dense_sizes = [*dense_sizes, weights[-1].shape[1]]
    # else:
    dense_sizes = [weights[0].shape[1], *dense_sizes]

    if transpose_weights:
        weights = [tf.transpose(ws, perm=[1, 0]) for ws in weights]

    inputs = [*weights, *init_state, *inp_spike_ids, *num_inp_spikes, *decay_constants, *thresholds]

    sizes_str = "_".join([str(val) for val in [*dense_sizes, *sparse_sizes, batch_size]])
    start_tiles = [tileMapping.start_tile for tileMapping in tileMappings]
    end_tiles = [tileMapping.end_tile for tileMapping in tileMappings]
    tileMapping_str = "_".join([str(val) for val in [*start_tiles, *end_tiles]])
    attributes_str = "_".join([sizes_str, tileMapping_str]) if transpose_weights else sizes_str
    print(start_tiles)
    print(end_tiles)
    print(attributes_str)
    # sys.exit()
    # TODO implement for all
    # inputs_with_gradients = [*list(range(num_layers)), *list(range(2*num_layers, 3*num_layers))] if transpose_weights else None
    inputs_with_gradients = list(range(num_layers)) if transpose_weights else None
    out = ipu.custom_ops.precompiled_user_op(inputs,
                                              lib_path,
                                              gp_path,
                                              name="custom_multi_lif_layer_sparse", # TF operation name
                                              op_name="Build",
                                              outs=outputs,
                                              inputs_with_gradients=inputs_with_gradients,
                                              separate_gradients=False, # to calculate gradients separately. Allows to only calculate weight gradient without implementing the others
                                              attributes=attributes_str,
                                              gradient_attributes=attributes_str,
                                            )
    return out

class KerasMultiLIFLayerSparse(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, transpose_weights=False, seed=None, num_ipus=1, weight_mul=1.0):
        super().__init__(dense_shapes, decay_constant, threshold, transpose_weights, seed, weight_mul)
        assert len(dense_shapes) == len(sparse_shapes), "`dense_shapes` and `sparse_shapes` must have the same nmber of elements."
        self.sparse_shapes = sparse_shapes
        self.neuron_tileMappings = determine_neuron_tileMappings(dense_shapes, sparse_shapes, num_ipus, min_neurons_per_tile=2 if transpose_weights else 1)
        print(self.neuron_tileMappings)
        # sys.exit()

    def adjust_multi_thresh(self):
        decay_val = 0.5
        layerwise_targets = [self.threshold_value[1] for i in range(self.num_layers)]
        self.current_second_threshs = [decay_val*curr_thr-(1-decay_val)*(curr_thr-taget_thr) for curr_thr, taget_thr in zip(self.current_second_threshs, layerwise_targets)]
        for ilay,curr_thresh in enumerate(self.current_second_threshs):
            self.thresholds[ilay][1, :] = curr_thresh

    def call(self, inp_spikes, init_states):
        out = custom_multi_lif_layer_sparse(self.sparse_shapes, self.transpose_weights, self.ws, init_states, inp_spikes, self.decay_constants, self.thresholds, self.neuron_tileMappings)

        # if self.version_multi_thresh:
        #     self.adjust_multi_thresh()

        return out

def pure_tf_lif_step_sparse(weights, state, inp_, decay_constants, thresholds, sparse_dim, neuron_tileMapping, spikeGen_tileMapping):
    syn_inp = dyn_dense_binary_sparse_matmul_op(weights, inp_, neuron_tileMapping)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 1))
    new_state = state * decay_constants + (1 - decay_constants) * 10 * syn_inp # TODO hyperparameter times 10 hardcoded

    spikes_out = compute_sparse_spikes(new_state, thresholds, sparse_dim, spikeGen_tileMapping)
    return spikes_out, new_state

def pure_tf_lif_step_sparse_parallel(weights, neuron_states, input_spikes, decay_constants, thresholds, sparse_dim, neuron_tileMapping, spikeGen_tileMapping, state_bins):
    
    num_layers = len(weights)
    # syn_inps = [dyn_dense_binary_sparse_matmul_op(weights[i], input_spikes[i], neuron_tileMapping[i]) for i in range(num_layers)]
    syn_inps = dyn_dense_binary_sparse_matmul_op_parallel(weights, input_spikes[:len(weights)], neuron_tileMapping, False)

    decay_constants_concat = tf.concat(decay_constants, axis=0)
    thresholds_concat = tf.concat(thresholds, axis=0)
    neuron_states_concat = tf.concat(neuron_states, axis=1)
    syn_inps_concat = tf.concat(syn_inps, axis=1)

    neuron_states_concat = neuron_states_concat - tf.stop_gradient(neuron_states_concat * tf.experimental.numpy.heaviside(neuron_states_concat-thresholds_concat, 1))
    new_states_concat = neuron_states_concat * decay_constants_concat + (1 - decay_constants_concat) * 10 * syn_inps_concat # TODO hyperparameter times 10 hardcoded
    new_states = [new_states_concat[:, state_bins[i]:state_bins[i+1]] for i in range(num_layers)]

    # spikes_out = [compute_sparse_spikes(new_states[i], thresholds[i], sparse_dim[i], spikeGen_tileMapping[i])  for i in range(num_layers)]
    spikes_out = compute_sparse_spikes_parallel(new_states, thresholds, sparse_dim, spikeGen_tileMapping)
    return spikes_out, new_states


class KerasMultiLIFLayerSparseCell(KerasMultiLIFLayerBase):
    def __init__(self, dense_shapes, sparse_shapes, decay_constant, threshold, batchsize, seed=None, parallel_execution=True):
        super().__init__(dense_shapes, decay_constant, threshold, False, seed)
        self.sparse_shapes = sparse_shapes
        state_size = [tf.TensorShape((dim,)) for dim in dense_shapes[1:]]
        for sparse_dim in sparse_shapes[1:]:
            state_size.extend((tf.TensorShape((sparse_dim,)), tf.TensorShape((1,))))
        out_spike_size = [ SparseBinaryVec(tf.TensorShape((sparse_dim,)), tf.TensorShape((1,))) for sparse_dim in sparse_shapes[1:]]
        self.state_size = state_size # + out_spike_size
        self.output_size = out_spike_size
        # self.output_size = dense_shapes[-1]

        self.neuron_tileMappings = determine_neuron_tileMappings(dense_shapes, sparse_shapes)
        self.spikeGen_tileMappings = determine_spikeGen_tileMappings(len(dense_shapes)-1, batchsize)

        state_bins = [0]
        end_id = 0
        for i in range(1, len(dense_shapes)):
            end_id += dense_shapes[i]
            state_bins.append(end_id)
        print("state_bins", state_bins)
        self.state_bins = state_bins
        self.parallel_execution = parallel_execution

    def call(self, inp_spikes, state):
        neuron_states = state[:self.num_layers]
        outs = [SparseBinaryVec(ids,num_nzelements) for ids,num_nzelements in zip(state[self.num_layers::2], state[self.num_layers+1::2])]
        # outs = state[self.num_layers:]
        

        if self.parallel_execution:
            inps = [inp_spikes]
            inps_add = [outs[i] for i in range(self.num_layers)]
            all_inps = inps + inps_add
            all_out_spikes, all_neuron_states = pure_tf_lif_step_sparse_parallel(self.ws, neuron_states, all_inps, self.decay_constants, 
                                                self.thresholds, self.sparse_shapes[1:], self.neuron_tileMappings, self.spikeGen_tileMappings, self.state_bins)
        else:
            all_out_spikes = []
            all_neuron_states = []

            for ilay in range(self.num_layers):
                inp_ = inp_spikes if ilay==0 else outs[ilay-1]
                spikes_out, neuron_stat = pure_tf_lif_step_sparse(self.ws[ilay], neuron_states[ilay], inp_, self.decay_constants[ilay], 
                                                    self.thresholds[ilay], self.sparse_shapes[ilay+1], self.neuron_tileMappings[ilay], self.spikeGen_tileMappings[ilay])
                all_neuron_states.append(neuron_stat)
                all_out_spikes.append(spikes_out)


        unstructured_out_spikes = [item for sublist in all_out_spikes for item in sublist]
        state_new = [*all_neuron_states, *unstructured_out_spikes]
        
        return all_out_spikes, state_new

    def get_initial_state(self, inputs, batch_size, dtype):
        # print(self.dense_shapes)
        # return [tf.zeros((batch_size, dim)) for dim in self.dense_shapes[1:]]
        # print(self.state_size[0].as_list())
        # return [tf.zeros((batch_size, *stat_shape.as_list())) for stat_shape in self.state_size]
        init_state = [tf.zeros((batch_size, dim), dtype=tf.float32) for dim in self.dense_shapes[1:]]
        for sparse_dim in self.sparse_shapes[1:]:
            init_state.extend((tf.zeros((batch_size, sparse_dim), dtype=tf.float32), tf.zeros((batch_size, 1), dtype=tf.float32)))
        return init_state

def KerasMultiLIFLayerSparseOps(dense_shapes, sparse_shapes, decay_constant, threshold, batchsize, seed=None, parallel_execution=True, **kwargs):
    return tf.keras.layers.RNN(KerasMultiLIFLayerSparseCell(dense_shapes, sparse_shapes, decay_constant, threshold, batchsize, seed, parallel_execution), **kwargs)


def model_fn_sparse_layer(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, transpose_weights=False, return_all=False, seed=None, num_ipus=1, weight_mul=1.0):
    if return_all:
        warnings.warn("All layers outputs will be returned. But note that only gradient propagation through the last layers outputs is implemented."
                    " Adding loss terms to other layers outputs will be ignored and will result in a wrong gradient.", UserWarning)

    num_layers = len(dense_shapes)-1

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    spike_ids = [tf.transpose(inp_spike_ids, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step, sparse_shapes[ilay]), dtype=inp_spike_ids.dtype ) for ilay in range(1, num_layers)]]
    num_spikes = [tf.transpose(num_inp_spikes, perm=[1, 0, 2]), *[tf.zeros((batchsize_per_step,1), dtype=num_inp_spikes.dtype ) for ilay in range(1, num_layers)]]
    inp_spikes = [SparseBinaryVec(ids,nz_elemts) for ids,nz_elemts in zip(spike_ids, num_spikes)]

    init_states = [tf.zeros((batchsize_per_step, dense_shapes[i+1]), dtype=tf.float32) for i in range(num_layers)]
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
    return (inp_spike_ids, num_inp_spikes), out

def model_fn_sparse_ops(sparse_shapes, seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, transpose_weights=False, return_all=False, seed=None, parallel_execution=True):

    if transpose_weights:
        raise ValueError("`transpose_weights` for sparse ops is not implemented yet.")

    inp_spike_ids = keras.Input(shape=(seq_len, sparse_shapes[0]), batch_size=batchsize_per_step, dtype=tf.float32, name="inp_spike_ids")
    num_inp_spikes = keras.Input(shape=(seq_len, 1), batch_size=batchsize_per_step, dtype=tf.int32, name="num_inp_spikes")
    
    inp_spikes = SparseBinaryVec(inp_spike_ids, num_inp_spikes)


    out_spikes = KerasMultiLIFLayerSparseOps(
            dense_shapes, sparse_shapes, decay_constant, threshold, batchsize_per_step, seed, parallel_execution, return_sequences=True
        )(inp_spikes)
    # if return_all:
    #     out = [tf.transpose(sparse2dense_ipu(SparseBinaryVec(ids, tf.cast(num_nzelements, tf.int32)), dense_shapes[-1]), perm=[1, 0, 2]) for ids,num_nzelements in zip(out_spike_ids, num_out_spikes)]
    # else:
    #     sparse_out_spikes_last_layer = SparseBinaryVec(out_spike_ids[-1], tf.cast(num_out_spikes[-1], tf.int32))
    #     dense_out_spikes_last_layer = sparse2dense_ipu(sparse_out_spikes_last_layer, dense_shapes[-1])
    #     out = tf.transpose(dense_out_spikes_last_layer, perm=[1, 0, 2])


    if return_all:
        out = out_spikes
    else:
        out = out_spikes[-1]

    return (inp_spike_ids, num_inp_spikes), out

def simple_loss_fn_dense(y_target, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def simple_loss_fn_sparse(y_target, y_pred: SparseBinaryVec):
    dense_spikes = tf.transpose(sparse2dense_ipu(y_pred, y_target.shape[-1]), perm=[1, 0, 2])
    sum_spikes = tf.reduce_sum(dense_spikes, axis=1) # (batch, seq, neurons)
    return tf.math.reduce_sum((sum_spikes-y_target)**2)/y_target.shape[-1]

def sum_and_sparse_categorical_crossentropy(y_true, y_pred):
    sum_spikes = tf.reduce_sum(y_pred, axis=1) # (batch, seq_len, neurons)
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, sum_spikes, from_logits=True)

def create_dataset_sparse(inp_spike_ids, num_inp_spikes, labels, batchsize, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices({"inp_spike_ids": inp_spike_ids, "num_inp_spikes": num_inp_spikes, "targets": labels})
    num_samples = labels.shape[0]
    if shuffle:
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)
    # dataset = dataset.repeat()
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.prefetch(4)
    return dataset

def gradient_transformers_scale(scale_facs):
    def grad_transf(args):
        return [(grad*scale_fac, var) for (grad,var),scale_fac in zip(args, scale_facs)]
    return grad_transf

def train_ipu(
        method,
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
        metrics=None,
        steps_per_epoch=None,
        callbacks=None,
        return_all=False,
        transpose_weights=False,
        learning_rate=1e-2,
        num_ipus=1,
        seed=None,
        grad_scale_facs=None,
        weight_mul=1.0,
    ):
    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    # if method is None:
    #     method = "sparse_layer"

    assert method in ["dense", "sparse_ops", "sparse_layer"], f"`method` must be one of 'dense', 'sparse_ops', 'sparse_layer' or None, got '{method}'."

    method_to_model_fn = {
        "dense": model_fn_dense, 
        "sparse_ops": ft.partial(model_fn_sparse_ops, sparse_shapes, transpose_weights=transpose_weights, parallel_execution=True), 
        "sparse_layer": ft.partial(model_fn_sparse_layer, sparse_shapes, transpose_weights=transpose_weights, num_ipus=num_ipus, weight_mul=weight_mul),
    }

    with strategy.scope():
        # init model
        inputs, outputs = method_to_model_fn[method](seq_len, dense_shapes, decay_constant, threshold, batchsize_per_step, return_all=return_all, seed=seed)
        targets = keras.Input((1,), name="targets")
        model = keras.Model([inputs, targets], outputs)
        # model.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
        #                        device_mapping=[ ipu.pipelining_ops._ALL_DEVICES, 1])
        # model.load_weights("./model_save_weights")
        # # model = tf.keras.models.load_model("./model_save", compile=False)

        # Set the infeed and outfeed options.
        model.set_infeed_queue_options(prefetch_depth=2)
        model.set_outfeed_queue_options(buffer_depth=2)

        # # Compile our model with Stochastic Gradient Descent as an optimizer
        # # optim = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=False, name="SGD")
        # # optim = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=False, name="SGD")
        # # optim = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9, nesterov=False, name="SGD")
        # optim = tf.keras.optimizers.Adam(learning_rate=1e-2) # NOTE 1e-2 worked quite well
        # # optim = tf.keras.optimizers.Adam(learning_rate=2.5e-2) # NOTE 1e-2 worked quite well
        # # optim = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.0, nesterov=False, name="SGD")

        optim_kwargs = {}
        if grad_scale_facs is not None:
            optim_kwargs["gradient_transformers"] = [gradient_transformers_scale(grad_scale_facs)]

        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate, **optim_kwargs) # NOTE 1e-2 worked quite well
        # optim = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=0.9, nesterov=False, name="SGD")

        model.add_loss(loss_fn(targets, outputs))
        if metrics is not None:
            if not isinstance(metrics, (list, tuple)):
                metrics = [metrics]
            for metric in metrics:
                model.add_metric(metric(targets, outputs), metric.__name__)
        model.compile(optim, # loss_fn,
                    # metrics=metrics,
                    steps_per_execution=train_steps_per_execution,
        )

        model.summary()

        print('\nTraining')
        model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, workers=batchsize_per_step, callbacks=callbacks)
        # model.fit([inp_spike_ids, num_inp_spikes, init_states], targets, epochs=num_epochs, batch_size=batchsize, shuffle=False)

        # # model.save("./model_save")
        # model.save_weights("./model_trained_weights", save_format="tf")


@tf.function(experimental_compile=True)  # Make it fast.
def value_and_grad_on_batch(dense_shapes, model, x, y, sparse=False, out_batch_first=True):
    with tf.GradientTape() as tape:
        out = model(x)
        if sparse:
            sparse_out = out
            # out = sparse2dense_ipu(out, y.shape[-1])
            out = [sparse2dense_ipu(out[ilay], dense_shapes[ilay]) for ilay in range(len(out))]
        if not out_batch_first:
            # out = tf.transpose(out , perm=[1, 0, 2])
            out = [tf.transpose(out[ilay] , perm=[1, 0, 2]) for ilay in range(len(out))]
        loss = simple_loss_fn_dense(y, out[-1])
        # loss = sum_and_sparse_categorical_crossentropy(y, out)
        gradients = tape.gradient(loss, model.trainable_weights)
    if sparse:
        return out, gradients, sparse_out
    else:
        return out, gradients

def cosine_similarity(a, b):
    return np.sum(a*b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_scale(a, b):
    return np.nanmean(a / b)


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

    # os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    rng = np.random.default_rng(1)
    num_sequences = 24
    batchsize = num_sequences
    batchsize_per_step = batchsize
    seq_len = 100
    # # dense_sizes = [102, 801, 799]
    # dense_sizes = [128, 256, 64]
    # # dense_sizes = [4, 4, 4]
    # # dense_sizes = [4*1024, 1024, 1024, 1024]
    # sparse_sizes = dense_sizes
    # # sparse_sizes = [2*1024, 1024, 1024, 1024]
    # # sparse_sizes = [3, 3] #dense_sizes

    # dense_sizes =  list(range(32, 32+5))
    # sparse_sizes = list(range(32, 32+5))

    # dense_sizes = [100, 256, 256, 8]
    # sparse_sizes = [32, 64, 64, 8]
    # dense_sizes = [100, 64, 64, 32,  4]
    dense_sizes = [100, 32,  4]
    # sparse_sizes = [32, 32, 32, 32, 8]
    sparse_sizes = dense_sizes
    
    # dense_sizes = [100, 256, 256, 256, 8]
    # sparse_sizes = [16, 32, 32, 32, 8]

    # dense_sizes = [int(34*34*2), 1470, 512, 128, 10]
    # sparse_sizes = [48, 48, 32, 16, 10]

    # sparse_sizes = dense_sizes
    # sparse_sizes = [32, 64, 64, 8]

    dense_sizes = [16, 7, 2]
    # sparse_sizes = [8, 4, 2 ,2]
    # dense_sizes = [16*6, 34*6, 18*6, 16]
    sparse_sizes = dense_sizes

    # SPARSE_MULTIPLIER = 1
    # NUM_CLASSES = 10
    # DENSE_SIZES = [512*2, 512, 512, 512, 128, NUM_CLASSES]
    # DENSE_SIZES = DENSE_SIZES[:1] + [int(0.5*d) for d in DENSE_SIZES[1:-1]] + DENSE_SIZES[-1:]
    # SPARSE_SIZES_BASE = [32, 32, 32, 32, 16, 10]
    # # SPARSE_SIZES = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE, DENSE_SIZES)]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # dense_sizes = DENSE_SIZES
    # sparse_sizes = SPARSE_SIZES

    # SPARSE_MULTIPLIER = 1
    # NUM_CLASSES = 10
    # DENSE_SIZES = [512, 512, 512, 128, NUM_CLASSES]
    # DENSE_SIZES = DENSE_SIZES[:1] + [int(0.5*d) for d in DENSE_SIZES[1:-1]] + DENSE_SIZES[-1:]
    # SPARSE_SIZES_BASE = [64, 32, 32, 16, 10]
    # # SPARSE_SIZES = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE, DENSE_SIZES)]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # dense_sizes = DENSE_SIZES
    # sparse_sizes = SPARSE_SIZES


    # SPARSE_MULTIPLIER = 32
    # IMAGE_DIMS = (34, 34)
    # NUM_CLASSES = 10
    # # benchmarking presentation
    # dense_sizes = [np.prod(IMAGE_DIMS), 1024, 1024, 512, 512, 128, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [4, 4, 4, 2, 2, 1, 1]
    # sparse_sizes = [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[:-1], dense_sizes[:-1])]
    # sparse_sizes = sparse_sizes + [min(int(SPARSE_SIZES_BASE[-1]*SPARSE_MULTIPLIER), 8)]


    # # benchmarking presentation
    # SPARSE_MULTIPLIER = 16
    # IMAGE_DIMS = (34,34,2)
    # DENSE_SIZES = [np.prod(IMAGE_DIMS), 1472, 1076+384, NUM_CLASSES]
    # SPARSE_SIZES_BASE = [64, 4, 3, 2, 10]
    # SPARSE_SIZES = SPARSE_SIZES_BASE[:1] + [min(dense, int(sparse*SPARSE_MULTIPLIER)) for sparse,dense in zip(SPARSE_SIZES_BASE[1:], DENSE_SIZES[1:])]
    # dense_sizes = DENSE_SIZES
    # sparse_sizes = SPARSE_SIZES

    decay_constant = 0.9
    threshold = 1.0
    second_thresh = 0.9
    num_layers = len(dense_sizes)-1 
    model_seed = rng.integers(999999)

    assert batchsize % batchsize_per_step == 0
    train_steps_per_execution = int(batchsize / batchsize_per_step)

    targets = rng.uniform(0.0, 1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
    # targets = rng.choice(10, size=(num_sequences,)).astype(np.int32)
    # print(targets)
    # sys.exit()
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])
    dataset_sparse = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)
    inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
    dataset_dense = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize, shuffle=False)

    num_ipus = 1

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    data_sparse = iter(dataset_sparse).next()
    data_sparse = ((data_sparse["inp_spike_ids"], data_sparse["num_inp_spikes"]), data_sparse["targets"])
    data_dense = iter(dataset_dense).next().values()

    # # with strategy.scope():
    # #     model_dense_ipu = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, threshold, batchsize, seed=model_seed, return_all=False))
    # #     out_ipu_dense, grad_ipu_dense =  strategy.run(value_and_grad_on_batch, args=[model_dense_ipu, *data_dense, False])

    first_and_second_threshold = [threshold, [*[second_thresh]*(len(sparse_sizes)-2), -100]]

    print("\n############################# SPARSE LAYER ###################################")
    with strategy.scope():
        model_sparse_layer = keras.Model(*model_fn_sparse_layer(sparse_sizes, seq_len, dense_sizes, decay_constant, first_and_second_threshold, batchsize_per_step, seed=model_seed, return_all=True, transpose_weights=True, num_ipus=num_ipus))
        out_sparse_layer, grad_sparse_layer, sparse_out_layer =  strategy.run(value_and_grad_on_batch, args=[dense_sizes[1:], model_sparse_layer, *data_sparse, True, False])
    #     out =  strategy.run(value_and_grad_on_batch, args=[dense_sizes[1:], model_sparse_layer, *data_sparse, True, False])
    # print(out)
    # print()
    # print(out[-1])
    # print()
    # print(out[-1].ids)
    # print()
    # print(out[-1].num_nzelements)

    print("success")

    # sys.exit()

    print("\n################################# DENSE #######################################")
    model_dense = keras.Model(*model_fn_dense(seq_len, dense_sizes, decay_constant, first_and_second_threshold, batchsize, seed=model_seed, return_all=True))
    out_dense, grad_dense = value_and_grad_on_batch(dense_sizes[1:], model_dense, *data_dense, False)
    

    # print("\nsparse_out_layer")
    # print(sparse_out_layer)
    # print("\ngrad_sparse_layer")
    # print(grad_sparse_layer)
    # print("\ngrad_dense")
    # print(grad_dense)


    # for i in range(num_layers):
    #     print()
    #     print(out_sparse_layer[i].shape, out_dense[i].shape)
    #     dense_size = out_dense[i].shape[-1]
    #     print(f"{i}: activity sparse = {np.mean(out_sparse_layer[i])}, dense =  {np.mean(out_dense[i])}, sparse_size/dense_size = {sparse_sizes[i+1]/dense_sizes[i+1]}")
    #     print(f"{i}: max activity sparse = {np.max(np.mean(out_sparse_layer[i], axis=2))}, max activity dense =  {np.max(np.mean(out_dense[i], axis=2))}")
    #     print(f"{i}: activity from num spikes: mean = {np.mean(sparse_out_layer[i].num_nzelements[...,0])/dense_size}, max = {np.max(sparse_out_layer[i].num_nzelements[...,0])/dense_size}")
    #     print(f"{i}: activity from num grad spikes: mean = {np.mean(sparse_out_layer[i].num_nzelements[...,1])/dense_size}, max = {np.max(sparse_out_layer[i].num_nzelements[...,1])/dense_size}")
    #     print(f"{i}: activity from only num grad spikes: mean = {np.mean(np.diff(sparse_out_layer[i].num_nzelements, axis=-1))/dense_size}, max = {np.max(np.diff(sparse_out_layer[i].num_nzelements, axis=-1))/dense_size}")
    #     print(f"{i}: num spikes: mean = {np.mean(sparse_out_layer[i].num_nzelements[...,0])}, max = {np.max(sparse_out_layer[i].num_nzelements[...,0])}")
    #     print(f"{i}: num grad spikes: mean = {np.mean(sparse_out_layer[i].num_nzelements[...,1])}, max = {np.max(sparse_out_layer[i].num_nzelements[...,1])}")
    #     print(f"{i}: only um grad spikes: mean = {np.mean(np.diff(sparse_out_layer[i].num_nzelements, axis=-1))}, max = {np.max(np.diff(sparse_out_layer[i].num_nzelements, axis=-1))}")


    #     check_values(out_sparse_layer[i], out_dense[i], f"{i}: sparse layer - out_spikes[{i}]", rtol=1e-4, atol=1e-6)
    #     check_values(grad_sparse_layer[i], grad_dense[i], f"{i}: sparse layer - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
    #     print(f"{i}: cossine_similarity = {cosine_similarity(grad_sparse_layer[i], grad_dense[i])}")
    #     print(f"{i}: mean_scale = {mean_scale(grad_sparse_layer[i], grad_dense[i])}")
    #     print(np.argwhere(out_sparse_layer[i] != out_dense[i]))

    # sys.exit()


    print("\n############################## SPARSE OPS ####################################")
    with strategy.scope():
        model_sparse_ops = keras.Model(*model_fn_sparse_ops(sparse_sizes, seq_len, dense_sizes, decay_constant, threshold, batchsize_per_step, seed=model_seed, return_all=True, parallel_execution=True))
        # model_sparse_ops.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
        #                        device_mapping=[ ipu.pipelining_ops._ALL_DEVICES, 1])
        out_sparse_ops, grad_sparse_ops, sparse_out_ops = strategy.run(value_and_grad_on_batch, args=[dense_sizes[1:], model_sparse_ops, *data_sparse, True])
        # out_sparse_layer, grad_sparse_layer =  strategy.run(value_and_grad_on_batch, args=[model_sparse_ops, *data_sparse, True])


    print("\ndata_sparse")
    print(data_sparse)
    # print("\ndata dense")
    # print(data_dense)
    print("\nforward")
    # print(out_ipu_dense.shape)
    # print(out_dense.shape)
    # print(out_ipu_dense)
    print(out_dense)
    print(out_sparse_layer)
    # print(out_sparse_ops)
    print("\ngrad")
    # print(len(grad_ipu_dense))
    print(len(grad_sparse_layer))
    print(len(grad_dense))
    print()
    # print(grad_ipu_dense)
    print("\ngrad_sparse_layer")
    print(grad_sparse_layer)
    print("\ngrad_sparse_ops")
    print(grad_sparse_ops)
    print("\ngrad_dense")
    print(grad_dense)

    # print("\nsparse_out_layer")
    # print(sparse_out_layer)
    # print("\nsparse_out_layer")
    # print(sparse_out_layer)

    # np.savetxt("grad_sparse_layer.txt", grad_sparse_layer[0])
    # np.savetxt("grad_sparse_ops.txt", grad_sparse_ops[0])

    # import matplotlib.pyplot as plt
    # for i in range(num_layers):
    #     quotient = (grad_sparse[i]/grad_dense[i]).numpy()
    #     print()
    #     print(quotient)
    #     plt.figure()
    #     plt.hist(quotient.flatten(), bins=int(np.sqrt(0.5*quotient.size)))
    #     plt.title(f"layer {i}")

    #     print(np.nanmin(quotient), np.nanmax(quotient))model
    # print(data_sparse)
    # print(data_dense)
    # print("------------------ out -------------------------")
    # print(out_sparse_ops)
    # print(f"{out_sparse_ops.numpy().sum()=}")
    # print()
    # print(f"{out_sparse_layer.numpy().sum()=}")
    # print()
    # print(f"{out_dense.numpy().sum()=}")

    print("\nMAKE SURE TO USE A WEIGHT INIT THAT USES THE SEED!!!\n")

    # print()
    # check_values(out_ipu_dense, out_dense, f"dense - out_spikes", rtol=1e-4, atol=1e-6)
    # for i in range(num_layers):
    #     check_values(grad_ipu_dense[i], grad_dense[i], f"dense - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
    #     # check_values(model_dense_ipu.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)
    print()
    for i in range(num_layers):
        print(out_sparse_layer[i].shape, out_dense[i].shape)
        print(f"{i}: activity sparse = {np.mean(out_sparse_layer[i])}, dense =  {np.mean(out_dense[i])}, sparse_size/dense_size = {sparse_sizes[i+1]/dense_sizes[i+1]}")
        print(f"{i}: max activity sparse = {np.max(np.mean(out_sparse_layer[i], axis=2))}, max activity dense =  {np.max(np.mean(out_dense[i], axis=2))}")
        check_values(out_sparse_layer[i], out_dense[i], f"{i}: sparse layer - out_spikes[{i}]", rtol=1e-4, atol=1e-6)
        check_values(grad_sparse_layer[i], grad_dense[i], f"{i}: sparse layer - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        print(f"{i}: cossine_similarity = {cosine_similarity(grad_sparse_layer[i], grad_dense[i])}")
        print(f"{i}: mean_scale = {mean_scale(grad_sparse_layer[i], grad_dense[i])}")
        print(np.argwhere(out_sparse_layer[i] != out_dense[i]))
        # check_values(model_sparse_layer.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)
    print()
    for i in range(num_layers):
        print(out_sparse_ops[i].shape, out_dense[i].shape)
        print(f"{i}: activity = {np.mean(out_sparse_ops[i])}, dense =  {np.mean(out_dense[i])}, sparse_size/dense_size = {sparse_sizes[i+1]/dense_sizes[i+1]}")
        print(f"{i}: max activity sparse = {np.max(np.mean(out_sparse_ops[i], axis=2))}, max activity dense =  {np.max(np.mean(out_dense[i], axis=2))}")
        check_values(out_sparse_ops[i], out_dense[i], f"{i}: sparse ops - out_spikes[{i}]", rtol=1e-4, atol=1e-6)
        check_values(grad_sparse_ops[i], grad_dense[i], f"{i}: sparse ops - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
        print(f"{i}: cossine_similarity = {cosine_similarity(grad_sparse_ops[i], grad_dense[i])}")
        print(f"{i}: mean_scale = {mean_scale(grad_sparse_ops[i], grad_dense[i])}")
        print(np.argwhere(out_sparse_ops[i] != out_dense[i]))
        # check_values(model_sparse_ops.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)

    # print()
    # check_values(out_ipu_dense, out_sparse_layer, f"sparse layer vs dense ipu - out_spikes", rtol=1e-4, atol=1e-6)
    # for i in range(num_layers):
    #     check_values(grad_ipu_dense[i], grad_sparse_layer[i], f"sparse layer vs dense ipu - grad_weights[{i}]", rtol=1e-4, atol=1e-6)
    #     # check_values(model_sparse_ops.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)

    print()

    for i in range(num_layers):
        check_values(out_sparse_ops[i], out_sparse_layer[i], f"{i}: sparse layer vs sparse ops - out_spikes[{i}]", rtol=1e-2, atol=1e-4)
        check_values(grad_sparse_ops[i], grad_sparse_layer[i], f"{i}: sparse layer vs sparse ops - grad_weights[{i}]", rtol=1e-2, atol=1e-4)
        print(f"{i}: cossine_similarity = {cosine_similarity(grad_sparse_ops[i], grad_sparse_layer[i])}")
        print(f"{i}: mean_scale = {mean_scale(grad_sparse_ops[i], grad_sparse_layer[i])}")
        print(np.argwhere(out_sparse_ops[i] != out_sparse_layer[i]))
        # check_values(model_sparse_ops.trainable_weights[i], model_dense.trainable_weights[i], f"weights[{i}]", rtol=1e-4, atol=1e-6)



# def main():

#     sparse_comp = True
#     if sparse_comp:
#         os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

#     rng = np.random.default_rng(1)
#     num_epochs = 10
#     num_sequences = 64
#     batchsize = 2
#     batchsize_per_step = 2
#     seq_len = 12
#     dense_sizes = [612, 512, 412]
#     sparse_sizes = [42, 32, 22]
#     decay_constant = 0.9
#     threshold = 1.0

#     assert batchsize % batchsize_per_step == 0
#     train_steps_per_execution = int(num_sequences / batchsize) # TODO or batchsize_per_step ?

#     targets = rng.uniform(1.0, size=(num_sequences, dense_sizes[-1])).astype(np.float32)
#     inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, num_sequences, dense_sizes[0], sparse_sizes[0])

#     if sparse_comp:
#         dataset = create_dataset_sparse(inp_spike_ids.transpose(1, 0, 2), num_inp_spikes.transpose(1, 0, 2), targets, batchsize)
#         train_ipu(
#             num_epochs, 
#             train_steps_per_execution, 
#             batchsize_per_step,
#             dataset,
#             seq_len, 
#             dense_sizes, 
#             sparse_sizes, 
#             decay_constant, 
#             threshold,
#             simple_loss_fn
#         )
#     else:
#         inp_spikes = sparse2dense(inp_spike_ids, num_inp_spikes, dense_sizes[0])
#         dataset = create_dataset_dense(inp_spikes.transpose(1, 0, 2), targets, batchsize)
#         train_gpu(
#             num_epochs,  
#             batchsize,
#             dataset,
#             seq_len, 
#             dense_sizes, 
#             decay_constant, 
#             threshold,
#             simple_loss_fn
#         )


if __name__ == '__main__':
    test_sparse_vs_dense()
    # main()



