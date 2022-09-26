import numpy as np 
from typing import NamedTuple

class TileMapping(NamedTuple):
    start_tile: int
    end_tile: int

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


def determine_neuron_tileMappings_multiIPU(dense_sizes, num_ipus, min_neurons_per_tile):
    
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



NUM_CLASSES = 10
NUM_NEURONS_PER_TILE = 16
NUM_HIDDEN_LAYERS_BASE = 3
NUM_HIDDEN_LAYERS = int(NUM_NEURONS_PER_TILE//2 * NUM_HIDDEN_LAYERS_BASE)


NEURON_TO_SPLIT = 1471*NUM_NEURONS_PER_TILE - ((NUM_CLASSES // NUM_NEURONS_PER_TILE + ((NUM_CLASSES % NUM_NEURONS_PER_TILE) > 0))* NUM_NEURONS_PER_TILE)
HIDDEN_LAYER_DENSE_SIZES_LAST = [int(NUM_NEURONS_PER_TILE*(NEURON_TO_SPLIT / NUM_HIDDEN_LAYERS // NUM_NEURONS_PER_TILE + (((NEURON_TO_SPLIT % int(NUM_HIDDEN_LAYERS*NUM_NEURONS_PER_TILE))) > NUM_NEURONS_PER_TILE*ilay))) for ilay in range(NUM_HIDDEN_LAYERS)]

print(HIDDEN_LAYER_DENSE_SIZES_LAST)


determine_neuron_tileMappings_multiIPU([None, *HIDDEN_LAYER_DENSE_SIZES_LAST], 1, 2)
