#include <iostream>
#include <algorithm>
#include <vector>
// #include <boost/optional.hpp>
#include <cmath> // ceil
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TargetType.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Zero.hpp>
#include <popops/Fill.hpp>
#include <popnn/NonLinearityDef.hpp> // TODO delete after sigmoid non-lin was replaced by custom non-lin
#include <popops/ElementWise.hpp>
#include <popops/Loop.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Reduce.hpp>
// #include <popops/Operation.hpp>
#include <popops/Cast.hpp>
#include <poprand/RandomGen.hpp>

#include <poplar/StringRef.hpp>

template<typename T>
void printVector(std::vector<T> vec) {
  std::cout << "{";
  for (auto val: vec) {
    std::cout << val << ", ";
  }
  std::cout << "}"<< std::endl;
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

std::vector<std::string> split_string(const std::string& s, char seperator)
{
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(seperator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );
        output.push_back(substring);
        prev_pos = ++pos;
    }
    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word
    return output;
}

std::vector<size_t> convert_vecOfStr_to_vecOfSizet(const std::string& s, char seperator) {
  std::vector<std::string> sparse_sizes_strs = split_string(s, seperator);
  auto num_layers = sparse_sizes_strs.size();
  std::vector<size_t> outputs;
  for (unsigned i=0; i<num_layers; ++i){
    size_t size_sparse_out;
    sscanf(sparse_sizes_strs[i].c_str(), "%zu", &size_sparse_out);
    outputs.push_back(size_sparse_out);
  }
  return outputs; 
}

void clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &dnai](const poplar::Tensor &t){return graph.clone(t, dnai, poplar::TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);});
}

std::vector<poplar::Tensor> clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  clone_tensor_vector(graph, src, dst, 0, dnai);
  return dst;
}

void clone_tensor_vector(poplar::Graph& graph, const poplar::Type &type, const std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &type, &dnai](const poplar::Tensor &t){return graph.clone(type, t, dnai);});
}

std::vector<poplar::Tensor> clone_tensor_vector(poplar::Graph& graph, const poplar::Type &type, const std::vector<poplar::Tensor> &src, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  // std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &type, &dnai](const poplar::Tensor &t){return graph.clone(type, t, dnai);});
  clone_tensor_vector(graph, type, src, dst, offset, dnai);
  return dst;
}

void cast_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, const std::vector<poplar::Tensor> &dst, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  for (unsigned i=0; i<src.size(); ++i){
    prog.add(popops::cast(graph, src[i], dst[i], dnai));
  }
}

std::vector<poplar::Tensor> cast_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, poplar::Type &dtype, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  // std::vector<poplar::Tensor> dst;
  // std::transform(src.begin(), src.end(), dst.begin(), [&graph, &dtype, &prog,  &dnai](const poplar::Tensor &t) -> poplar::Tensor {return popops::cast(graph, t, dtype, prog, dnai);});
  std::vector<poplar::Tensor> dst = clone_tensor_vector(graph, dtype, src, 0);
  cast_tensor_vector(graph, src, dst, prog, dnai);
  return dst;
}

void zero_tensor_vector(poplar::Graph& graph, std::vector<poplar::Tensor> &vec, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::for_each(vec.begin(), vec.end(), [&graph, &prog, &dnai](poplar::Tensor &t){popops::zero(graph, t, prog, dnai);});
}

void extend_tensor_vector(std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst){
  std::transform(src.begin(), src.end(), std::back_inserter(dst), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
}


struct BatchedSparseSpikes {
  poplar::Tensor spike_ids;
  poplar::Tensor num_spikes;

  // BatchedSparseSpikes(poplar::Tensor &spike_ids, poplar::Tensor &num_spikes)
  //   : spike_ids{spike_ids}
  //   , num_spikes{num_spikes} {};
};


struct LIFParams {
  popnn::rnn::RnnParams rnn;

  // number of neurons in this lif-layer
  size_t numNeurons;

  /// If true the LIF function returns the entire sequence of outputs,
  /// otherwise it returns just the final output.
  bool outputFullSequence = true;
  /// If this parameter is set to false then the GRU will skip the
  /// calculation of the gradients of the inputs.
  bool calcInputGradients = true;
  /// Activation function.
  popnn::NonLinearityType surrogateFnction = popnn::NonLinearityType::SIGMOID; // TODO implement surrogate derivative

  LIFParams();
  LIFParams(popnn::rnn::RnnParams rnn, size_t &numNeurons) 
    : rnn{rnn}
    , numNeurons{numNeurons}
    {};
};


std::vector<unsigned> get_tensor_tile_ids(poplar::Graph &graph, poplar::Tensor &t){
  auto tile_map = graph.getTileMapping(t);
  std::vector<size_t> tile_map_sizes;

  std::vector<unsigned> tile_ids;
  for (unsigned i=0; i<tile_map.size(); ++i){
    if (tile_map[i].size() > 0){
      tile_ids.push_back(i);
    }
  }
  return tile_ids;
}



std::tuple<unsigned, unsigned, bool> get_start_end_is_contigious(std::vector<size_t> &tile_allocation_sizes){
  unsigned start_tile;
  unsigned end_tile;
  unsigned num_switches{0};
  bool prev_size_gr_zero = false;
  for (unsigned i=0; i<tile_allocation_sizes.size(); ++i){
    bool this_size_gr_zero = (tile_allocation_sizes[i]>0);
    if (!prev_size_gr_zero && this_size_gr_zero){
      start_tile = i;
      prev_size_gr_zero = this_size_gr_zero;
      ++num_switches;
    } else if (prev_size_gr_zero && !this_size_gr_zero) {
      end_tile = i;
      prev_size_gr_zero = this_size_gr_zero;
      ++num_switches;
    }
  }
  if (tile_allocation_sizes.back()>0){
      end_tile = tile_allocation_sizes.size();
      ++num_switches;
  }
  bool is_contigous = (num_switches==2);
  return {start_tile, end_tile, is_contigous};
}

std::tuple<unsigned, unsigned, bool> get_start_end_is_contigious(poplar::Graph &graph, poplar::Tensor &t){
  auto tile_map = graph.getTileMapping(t);
  std::vector<size_t> tile_map_sizes;
  std::transform(tile_map.begin(), tile_map.end(), std::back_inserter(tile_map_sizes), [](std::vector<poplar::Interval> &vec) {return vec.size();});
  return get_start_end_is_contigious(tile_map_sizes);
}


std::vector<unsigned> get_ipu_ids_from_tensor_vec(poplar::Graph &graph, std::vector<poplar::Tensor> &tensor_vec) {
  std::vector<unsigned> start_tiles;
  std::vector<unsigned> ipu_ids_from_tensor_vec;
  unsigned num_tiles_per_ipu =  graph.getTarget().getTilesPerIPU();

  for (unsigned ilay=0; ilay<tensor_vec.size(); ++ilay){
    // auto tile_map = graph.getTileMapping(tensor_vec[ilay]);
    // std::vector<size_t> tile_map_sizes;
    // std::transform(tile_map.begin(), tile_map.end(), std::back_inserter(tile_map_sizes), [](std::vector<poplar::Interval> &vec) {return vec.size();});
    // auto [start_tile, end_tile, is_contiguous] = get_start_end_is_contigious(tile_map_sizes);
    auto [start_tile, end_tile, is_contiguous] = get_start_end_is_contigious(graph, tensor_vec[ilay]);
    
    unsigned ipu_id_start = start_tile / num_tiles_per_ipu;
    unsigned ipu_id_end = (end_tile-1) / num_tiles_per_ipu;

    if ((ipu_id_start!=ipu_id_end) || !is_contiguous){
      throw poputil::poplibs_error("Tensors have to be allocated on a single IPU and be contigous.");
    }   

    ipu_ids_from_tensor_vec.push_back(ipu_id_start);
  }
  return ipu_ids_from_tensor_vec;
}

unsigned get_tensor_ipu_id(poplar::Graph &graph, poplar::Tensor &t){
  auto tile_mapping = graph.getTileMapping(t);
  unsigned num_tiles = tile_mapping.size();
  unsigned num_tiles_per_ipu = graph.getTarget().getTilesPerIPU();
  unsigned ipu_id;
  for (unsigned i=0; i<num_tiles; ++i){
    if (tile_mapping[i].size() > 0){
      ipu_id = i / num_tiles_per_ipu;
      break;
    }
  }
  return ipu_id;
}

std::vector<unsigned> get_tensor_ipu_id(poplar::Graph &graph, std::vector<poplar::Tensor> &ts){
  std::vector<unsigned> ipu_ids;
  std::transform(ts.begin(), ts.end(), std::back_inserter(ipu_ids), [&graph](poplar::Tensor &t){return get_tensor_ipu_id(graph, t);});
  return ipu_ids;
}

std::vector<unsigned> get_relative_layer_id_on_ipu(const std::vector<unsigned> &layer_to_ipu_id){
  std::cout << "\nget_relative_layer_id_on_ipu" << std::endl;
  std::vector<unsigned> layer_id_per_ipu(layer_to_ipu_id.size());
  size_t num_layers{layer_to_ipu_id.size()};
  auto num_ipus_it = std::max_element(layer_to_ipu_id.begin(), layer_to_ipu_id.end());
  unsigned num_ipus = *num_ipus_it + 1;
  std::cout << "\nnum_ipus" << num_ipus << std::endl;
  std::vector<unsigned> num_layers_per_ipu(num_ipus, 0);
  for (unsigned i=0; i<num_layers; ++i){
    layer_id_per_ipu[i] = num_layers_per_ipu[layer_to_ipu_id[i]];
    num_layers_per_ipu[layer_to_ipu_id[i]] += 1;
  }
  return layer_id_per_ipu;
} 

std::vector<std::vector<unsigned>> get_layer_ids_per_ipu(poplar::Graph &graph, std::vector<poplar::Tensor> &ts){
  const std::vector<unsigned> layers_to_ipu_mapping(get_tensor_ipu_id(graph, ts));

  const unsigned num_ipus{graph.getTarget().getNumIPUs()};
  std::vector<std::vector<unsigned>> layer_ids_per_ipu(num_ipus);
  size_t num_layers{ts.size()};
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    unsigned ipu_id = layers_to_ipu_mapping[ilay];
    layer_ids_per_ipu[ipu_id].push_back(ilay);
  }
  return layer_ids_per_ipu;
} 


//---------------------------------------------------------- allocation ---------------------------------------------------------------------------------

poplar::Tensor alloc_linearly(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, unsigned offset, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  poputil::mapTensorLinearlyWithOffset(graph, allocTensor, offset);
  return allocTensor;
}

poplar::Tensor alloc_linearly(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, unsigned offset, unsigned minElementsPerTile, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  unsigned minGrainSize{1};
  poputil::mapTensorLinearlyWithOffset(graph, allocTensor, minElementsPerTile, minGrainSize, offset);
  return allocTensor;
}

poplar::Tensor alloc_neuronwise(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, std::vector<size_t> neuronMapping, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[neuronDim];
  for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
    graph.setTileMapping(allocTensor.slice(ineuron, ineuron+1, neuronDim), neuronMapping[ineuron]);
  }
  return allocTensor;
}

poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, std::vector<size_t> neuronMapping, const poplar::DebugNameAndId &dnai = {}) {

  std::vector<size_t> shape_slice;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_slice), [](size_t t) -> size_t {return t;});

  const size_t rank = shape.size();
  const size_t numNeurons = neuronMapping.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  std::vector<size_t> numNeuronsPerTile(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    numNeuronsPerTile[tile] = 0;
  }
  for (unsigned i = 0; i < numNeurons; ++i) {
    numNeuronsPerTile[neuronMapping[i]] += 1;
  }

  std::vector<poplar::Tensor> allocTensorVector;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
  
    const auto numNeuronsThisTile = numNeuronsPerTile[tile];
    // printVector(thisTileMap);
    if (numNeuronsThisTile == 0) {
      continue;
    }

    shape_slice[neuronDim] = numNeuronsThisTile;
    poplar::Tensor allocTensorSlice = graph.addVariable(type, shape_slice, dnai);
    graph.setTileMapping(allocTensorSlice, tile);
    allocTensorVector.push_back(allocTensorSlice);
  }

  poplar::Tensor allocTensor = poplar::concat(allocTensorVector, neuronDim);
  return allocTensor;
}

poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, poplar::Graph::TileToTensorMapping neuronTileMapping, const poplar::DebugNameAndId &dnai = {}) {

  std::vector<size_t> shape_slice;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_slice), [](size_t t) -> size_t {return t;});

  const size_t rank = shape.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  std::vector<poplar::Tensor> allocTensorVector;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    } else if (neuronTileMapping[tile].size() > 1) {
      throw poputil::poplibs_error("For `alloc_neuronwise_contiguous` the neurons on every tile have to be contiguos, meaning there can be only one neuron interval per tile in `neuronTileMapping`.");
    }
    
    const auto neuronRange = neuronTileMapping[tile][0];
    const auto numNeuronsThisTile = neuronRange.size();

    // This should never happen...
    if (numNeuronsThisTile == 0) {
      continue;
    }

    shape_slice[neuronDim] = numNeuronsThisTile;
    poplar::Tensor allocTensorSlice = graph.addVariable(type, shape_slice, dnai);
    graph.setTileMapping(allocTensorSlice, tile);
    allocTensorVector.push_back(allocTensorSlice);
  }
  poplar::Tensor allocTensor = poplar::concat(allocTensorVector, neuronDim);
  poplar::Tensor allocTensor_tileContigous = graph.clone(allocTensor, dnai, poplar::TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
  return allocTensor_tileContigous;
}


poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, size_t &start_tile, size_t &end_tile, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[neuronDim];
  size_t num_tiles_this_tensor = end_tile - start_tile;
  size_t num_neurons_per_tile = (numNeurons + num_tiles_this_tensor - 1) / num_tiles_this_tensor;
  size_t num_full_tiles = (numNeurons % num_neurons_per_tile == 0) ?  num_tiles_this_tensor-1 : num_tiles_this_tensor-1;

  size_t start_neuron = 0;
  size_t end_neuron = 0;
  for (unsigned itile = 0; itile < num_full_tiles; ++itile){
    end_neuron = start_neuron+num_neurons_per_tile;
    graph.setTileMapping(allocTensor.slice(start_neuron, end_neuron, neuronDim), itile+start_tile);
    start_neuron = end_neuron;
  }
  if (num_full_tiles != num_tiles_this_tensor) {
    graph.setTileMapping(allocTensor.slice(end_neuron, numNeurons, neuronDim), end_tile-1);
  }
  poplar::Tensor allocTensor_tileContigous = graph.clone(allocTensor, dnai, poplar::TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
  return allocTensor_tileContigous;
}


//---------------------------------------------- forward -----------------------------------------

void performBatchedLIFStateUpdateInPlace(poplar::Graph &graph, std::vector<poplar::Tensor> &weights, 
                            std::vector<poplar::Tensor> &state, std::vector<BatchedSparseSpikes> &inp_spikes, 
                            std::vector<poplar::Tensor> &decay_constants, std::vector<poplar::Tensor> &oneMinus_decay_constants, 
                            std::vector<poplar::Tensor> &thresholds,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "performBatchedLIFStateUpdateInPlace"});
  poplar::TargetType target_type = graph.getTarget().getTargetType();
  const auto numTiles = graph.getTarget().getNumTiles();
  size_t num_layers = weights.size();


  std::vector<poplar::Tensor> syn_input;
  if (target_type != poplar::TargetType::IPU) {
    syn_input = clone_tensor_vector(graph, state, {dnai, "syn_input"});
    zero_tensor_vector(graph, syn_input, prog, {dnai, "zero_syn_input"});
  }

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = weights[ilay].elementType();
    size_t batchsize = state[ilay].dim(0);

    auto neuronTileMapping = graph.getTileMapping(weights[ilay][0], true);

    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();
        // poplar::Tensor neuronWeights = weights[ilay].slice(neuronRange, 1).dimRoll(1, 0); // TODO does this create new tensors ?
        poplar::Tensor neuronStates = state[ilay].slice(neuronRange, 1);
        poplar::Tensor neuronDecay_constants = decay_constants[ilay].slice(neuronRange);
        poplar::Tensor neuronOneMinus_decay_constants = oneMinus_decay_constants[ilay].slice(neuronRange);
        poplar::Tensor neuronThresholds = thresholds[ilay][0].slice(neuronRange);

        poplar::Tensor neuronWeights = weights[ilay].slice(neuronRange, 1);
        poplar::Tensor neuronSyn_input;
        if (target_type != poplar::TargetType::IPU) {
          neuronSyn_input = syn_input[ilay].slice(neuronRange, 1);
        }
        

        const auto num_neurons = neuronWeights.dim(1);
        std::string vertexType;
        if (target_type == poplar::TargetType::IPU) {
          if ((num_neurons == 2) && (dtype == poplar::FLOAT)) {
            vertexType = "LIFStateUpdateInPlaceTwoNeuronSIMD";
          } else if (dtype == poplar::FLOAT) {
            // // if ((num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
            //   if (dtype == poplar::FLOAT) {
            vertexType = "LIFStateUpdateInPlaceMultiNeuronSIMD";
          }
          // could vectorize batches for multiple of 6
          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, vertexType,
                                      // {{"weights", weights[ilay][neuronId]},
                                      {{"weights", neuronWeights.flatten()},
                                      {"state", neuronStates[ibatch]},
                                      // {"syn_input", neuronSyn_input[ibatch]}, // TODO necessary for LIFStateUpdateInPlaceMultiNeuron
                                      {"inp_spikes_ids", inp_spikes[ilay].spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                      {"num_inp_spikes", inp_spikes[ilay].num_spikes[ibatch][0]},
                                      {"decay_constant", neuronDecay_constants},
                                      {"oneMinus_decay_constant", neuronOneMinus_decay_constants},
                                      {"threshold", neuronThresholds},
                                      {"num_neurons", neuronWeights.dim(1)}});
            // !!! TODO !!! totally bogus tile mapping, must be improved
            // should be based on weights mapping
            graph.setTileMapping(v, tile);
            // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
            graph.setPerfEstimate(v, 1);
          }
        } else {
          vertexType = poputil::templateVertex("LIFStateUpdateInPlaceMultiNeuron", dtype);

          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, vertexType,
                                      // {{"weights", weights[ilay][neuronId]},
                                      {{"weights", neuronWeights.flatten()},
                                      {"state", neuronStates[ibatch]},
                                      {"syn_input", neuronSyn_input[ibatch]}, // TODO necessary for LIFStateUpdateInPlaceMultiNeuron
                                      {"inp_spikes_ids", inp_spikes[ilay].spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                      {"num_inp_spikes", inp_spikes[ilay].num_spikes[ibatch][0]},
                                      {"decay_constant", neuronDecay_constants},
                                      {"oneMinus_decay_constant", neuronOneMinus_decay_constants},
                                      {"threshold", neuronThresholds},
                                      {"num_neurons", neuronWeights.dim(1)}});
          graph.setTileMapping(v, tile);
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
          }
        }
      }
    }
  }
  std::cout << "DONE" << std::endl;
  prog.add(poplar::program::Execute(cs));
}

unsigned determine_start_tile_spike_gen(const unsigned &layer_to_ipu_mapping, const unsigned &layer_id_per_ipu, const unsigned &batchsize, const unsigned &num_tiles_per_ipu){
  unsigned tile_offset = 1;
  return num_tiles_per_ipu * layer_to_ipu_mapping + batchsize * layer_id_per_ipu + tile_offset;
}



//---------------------------------------------- backward -----------------------------------------


void calcLIFWeightGrad_singleThread(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                        const std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  
  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});
  const size_t num_layers = dLdweights.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = dLdweights[ilay].elementType();
    size_t sparse_out_dim = fwdInpSpikes[ilay].spike_ids.dim(1);
    size_t batchsize = fwdInpSpikes[ilay].spike_ids.dim(0);

    auto neuronTileMapping = graph.getTileMapping(dLdweights[ilay][0], true);

    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();

        poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 1); // TODO does this create new tensors ?
        poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);

        // std::cout << "neuronDLdWeights.isContiguous(): " << neuronDLdWeights.isContiguous() << std::endl;
        // std::cout << "neuronDLdState.isContiguous(): " << neuronDLdState.isContiguous() << std::endl;



        const unsigned num_neurons = neuronDLdWeights.dim(1);
        std::string vertexType;
        poplar::TargetType target_type = graph.getTarget().getTargetType();
        if ((target_type == poplar::TargetType::IPU) && (num_neurons == 2) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradTwoNeuronSIMD";
          // vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else if ((target_type == poplar::TargetType::IPU) && (num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else {
          vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype);
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        }

        std::cout << "vertexType: " << vertexType << std::endl;

        auto v = graph.addVertex(cs, vertexType,
                                  {{"dLdState", neuronDLdState.flatten()},
                                  {"fwd_inp_spikes_ids", fwdInpSpikes[ilay].spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                                  {"fwd_num_inp_spikes", fwdInpSpikes[ilay].num_spikes.dimRoll(1, 0)[0]},
                                  {"sparse_out_dim", sparse_out_dim},
                                  {"batchsize", batchsize},
                                  {"num_neurons", num_neurons},
                                  // {"num_weights_per_neuron", neuronDLdWeights.dim(0)},
                                  {"dLdweights", neuronDLdWeights.flatten()}
                                  });
        // !!! TODO !!! totally bogus tile mapping, must be improved
        // should be based on state mapping
        graph.setTileMapping(v, tile); 
        // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        graph.setPerfEstimate(v, 1);
      }
    }
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFWeightGrad(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                        const std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  
  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});
  const size_t num_layers = dLdweights.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = dLdweights[ilay].elementType();
    size_t sparse_out_dim = fwdInpSpikes[ilay].spike_ids.dim(2);
    size_t batchsize = fwdInpSpikes[ilay].spike_ids.dim(1);
    size_t num_threads = dLdweights[ilay].dim(0);
    size_t batchsize_per_thread = batchsize / num_threads;

    std::cout << "ilay: " << ilay << std::endl;
    std::cout << "fwdInpSpikes[ilay].num_spikes.shapeToString(): " << fwdInpSpikes[ilay].num_spikes.shapeToString() << std::endl;

    if (batchsize < num_threads) {
      throw poputil::poplibs_error("For `calcLIFWeightGrad`: `batchsize` must be greater or equal to `num_threads`.");
    }
    
    // [0][0] because of replicated tensor due to multi thread approach
    auto neuronTileMapping = graph.getTileMapping(dLdweights[ilay][0][0], true);

    size_t occupied_tile_counter{0};
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();

        poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 2); // TODO does this create new tensors ?
        poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);
        poplar::Tensor fwd_inp_spikes_ids_tile = fwdInpSpikes[ilay].spike_ids[occupied_tile_counter];
        poplar::Tensor fwd_num_inp_spikes_tile = fwdInpSpikes[ilay].num_spikes[occupied_tile_counter];

        const unsigned num_neurons = neuronDLdWeights.dim(2);
        
        std::string vertexType;
        poplar::TargetType target_type = graph.getTarget().getTargetType();
        if ((target_type == poplar::TargetType::IPU) && (num_neurons == 2) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradTwoNeuronSIMD";
          // vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else if ((target_type == poplar::TargetType::IPU) && (num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else {
          vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype);
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        }

        for (unsigned thread_id = 0; thread_id < num_threads; ++thread_id) {

          size_t batch_id_start = thread_id*batchsize_per_thread;
          size_t batch_id_end = std::min((thread_id+1)*batchsize_per_thread, batchsize);

          poplar::Tensor neuronDLdWeightsThread = neuronDLdWeights[thread_id];
          poplar::Tensor neuronDLdStateThread = neuronDLdState.slice(batch_id_start, batch_id_end, 0);
          poplar::Tensor fwd_inp_spikes_ids_thread = fwd_inp_spikes_ids_tile.slice(batch_id_start, batch_id_end, 0);
          poplar::Tensor fwd_num_inp_spikes_thread = fwd_num_inp_spikes_tile.slice(batch_id_start, batch_id_end, 0); //.dimRoll(1, 0)[0];

          auto v = graph.addVertex(cs, vertexType,
                                    {{"dLdState", neuronDLdStateThread.flatten()},
                                    {"fwd_inp_spikes_ids", fwd_inp_spikes_ids_thread.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                                    {"fwd_num_inp_spikes", fwd_num_inp_spikes_thread.flatten()},
                                    {"num_thresholds", fwd_num_inp_spikes_thread.dim(1)},
                                    {"sparse_out_dim", sparse_out_dim},
                                    {"batchsize", neuronDLdStateThread.dim(0)},
                                    {"num_neurons", num_neurons},
                                    // {"num_weights_per_neuron", neuronDLdWeights.dim(0)},
                                    {"dLdweights", neuronDLdWeightsThread.flatten()}
                                    });
          // !!! TODO !!! totally bogus tile mapping, must be improved
          // should be based on state mapping
          graph.setTileMapping(v, tile); 
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
      ++occupied_tile_counter;
    }
  }
  prog.add(poplar::program::Execute(cs));
}


unsigned get_num_tiles_of_mapping(const poplar::Graph::TileToTensorMapping neuronTileMapping){
  unsigned num_tiles_total = neuronTileMapping.size();
  unsigned num_tiles_mapping{0};
  for (unsigned tile = 0; tile < num_tiles_total; ++tile) {
    if (!neuronTileMapping[tile].empty()) {
      num_tiles_mapping+=1;
    }
  }
  return num_tiles_mapping;
}


const std::vector<poplar::Tensor> performSharedUpdate(poplar::Graph &graph, const std::vector<poplar::Tensor> &oneMinus_decay_constants, 
                                                      std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, 
                                                      const poplar::DebugNameAndId &dnai = {}){

  std::vector<poplar::Tensor> intermediate_dLdState = clone_tensor_vector(graph, dLdState, {dnai, "intermediate_dLdState"});

  auto tile_map_dLdState = graph.getTileMapping(dLdState[0]);
  auto tile_map_iter_dLdState = graph.getTileMapping(intermediate_dLdState[0]);

  const auto num_lays{dLdState.size()};
  for (unsigned ilay=0; ilay<num_lays; ++ilay){
    const auto batchsize{dLdState[ilay].dim(0)};
    prog.add(poplar::program::Copy(dLdState[ilay], intermediate_dLdState[ilay]));
    popops::mulInPlace(graph, intermediate_dLdState[ilay], oneMinus_decay_constants[ilay].expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);
  }
  return intermediate_dLdState;
}



poplar::Tensor replicate_and_alloc_tensor(poplar::Graph &graph, poplar::Tensor &src, poplar::Graph::TileToTensorMapping tileMapping, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){

  size_t num_tiles_this_layer = std::accumulate(tileMapping.begin(), tileMapping.end(), 0, [](size_t a, std::vector<poplar::Interval>& tileMapping){return a + (tileMapping.size()>0);});
  std::vector<size_t> src_shape = src.shape();
  std::vector<size_t> repl_shape = {num_tiles_this_layer, };
  repl_shape.insert(repl_shape.end(), src_shape.begin(), src_shape.end());

  poplar::Tensor replicated_tensor = graph.addVariable(src.elementType(), repl_shape, {dnai, "addVariable_replicated_tensor"});

  // allocate tensor to tiles  
  unsigned allocated_num_rows{0};
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    const auto thisTileMap = tileMapping[tile];
    if (thisTileMap.size()>0){
      graph.setTileMapping(replicated_tensor[allocated_num_rows], tile);
      ++allocated_num_rows; 
    }
  }

  // copy elements from src to replicated tensor
  // poplar::Tensor broadcast_src = src.broadcast(num_tiles_this_layer, 0);
  poplar::Tensor broadcast_src = src.expand({0}).upsample(num_tiles_this_layer, 0, poplar::UpsampleMethod::REPEAT);
  prog.add(poplar::program::Copy(broadcast_src, replicated_tensor, false, {dnai, "copy_to_replicated_tensor"}));

  return replicated_tensor;
}