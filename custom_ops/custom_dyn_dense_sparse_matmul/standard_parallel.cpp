#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <popops/Cast.hpp>

#include <iostream>

#include "custom_dynamic_sparse/custom_dyn_dense_sparse_matmul/batched/standard/poplar_code.hpp"
#include "custom_dynamic_sparse/sparse_spikes.hpp"
#include "custom_dynamic_sparse/string_util.hpp"
#include "custom_dynamic_sparse/vector_util.hpp"


/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 5; // 5 neccessary for IPU model?
}

extern "C" void Build_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {
  is_elementwise = false;
  is_stateless = true;
}


poplar::Tensor alloc_linearly(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, unsigned offset, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  poputil::mapTensorLinearlyWithOffset(graph, allocTensor, offset);
  return allocTensor;
}

poplar::Tensor alloc_neuronwise(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, size_t start_tile, size_t end_tile, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[neuronDim];
  size_t num_tiles_this_tensor = end_tile - start_tile;
  size_t num_neurons_per_tile = (numNeurons + num_tiles_this_tensor - 1) / num_tiles_this_tensor;
  size_t num_full_tiles = (numNeurons % num_neurons_per_tile == 0) ?  num_tiles_this_tensor-1 : num_tiles_this_tensor-1;

  std::cout << "start_tile: " << start_tile << std::endl;
  std::cout << "end_tile: " << end_tile << std::endl;
  std::cout << "num_tiles_this_tensor: " << num_tiles_this_tensor << std::endl;
  std::cout << "num_neurons_per_tile: " << num_neurons_per_tile << std::endl;
  std::cout << "num_full_tiles: " << num_full_tiles << std::endl;

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
  return allocTensor;
}


extern "C" poplar::Tensor Build_allocator(
    poplar::Graph& graph,
    std::uint32_t operand,
    const std::vector<size_t>& shape,
    poplar::Type type,
    const std::string& attributes,
    const std::string& debug_prefix) {
  
  std::cout << "\nBuild_allocator\n" << std::endl;

  poplar::DebugNameAndId dnai{debug_prefix};

  // {*dense_sizes, *sparse_sizes, batchsize}
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  size_t start_tile = atrribute_sizes[0];
  size_t end_tile = atrribute_sizes[1];
  // size_t num_neurons = atrribute_sizes[2];
  // size_t batchsize = atrribute_sizes[4];

  size_t neuronDim = 0;
  std::string tensor_name;
  poplar::Tensor allocTensor;

  std::cout << "\noperand: " << operand << std::endl;
  printVector(atrribute_sizes);

  switch (operand) {
    case 0: tensor_name = "weights";
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, start_tile, end_tile, {dnai, tensor_name});
            break;
    case 1: tensor_name = "inp_spike_ids";
            allocTensor = alloc_linearly(graph, shape, type, 0, {dnai, tensor_name});
            break;
    case 2: tensor_name = "num_inp_spikes";
            allocTensor = alloc_linearly(graph, shape, type, 0, {dnai, tensor_name});
            break;
  }
  return allocTensor;
}


poplar::Tensor alloc_matrix_manually(poplar::Graph& graph, poplar::Tensor &matrix_bad_alloc, size_t start_tile, size_t end_tile, bool copy_values, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::cout << matrix_bad_alloc.shapeToString() <<std::endl;
  std::cout << start_tile << " " << end_tile << "\n" << std::endl;
  size_t neuronDim = 0;
  poplar::Tensor matrix = alloc_neuronwise(graph, matrix_bad_alloc.shape(), matrix_bad_alloc.elementType(), neuronDim, start_tile, end_tile, {dnai, "alloc matrix"});
  if (copy_values) {
    prog.add(poplar::program::Copy(matrix_bad_alloc, matrix, false, {dnai, "copy to matrix"}));
  }
  return matrix;
}

poplar::Tensor alloc_matrix_manually(poplar::Graph& graph, poplar::Tensor &matrix_bad_alloc, const std::string& attributes, bool copy_values, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::cout << "\nalloc manually" << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  size_t start_tile = atrribute_sizes[0];
  size_t end_tile = atrribute_sizes[1];
  return alloc_matrix_manually(graph, matrix_bad_alloc, start_tile, end_tile, copy_values, prog, dnai);
}

std::vector<poplar::Tensor> alloc_tensor_vector_manually(poplar::Graph& graph, std::vector<poplar::Tensor> &matrix_bad_alloc, const std::string& attributes, bool copy_values, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  size_t num_layers = matrix_bad_alloc.size();
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  const std::vector<size_t> start_tiles(atrribute_sizes.begin(), atrribute_sizes.begin()+num_layers);
  const std::vector<size_t> end_tiles(atrribute_sizes.begin()+num_layers, atrribute_sizes.begin()+2*num_layers);

  std::vector<poplar::Tensor> dst;
  for (unsigned ilay = 0; ilay < num_layers; ++ilay){
    dst.push_back(alloc_matrix_manually(graph, matrix_bad_alloc[ilay], start_tiles[ilay], end_tiles[ilay], copy_values, prog, dnai));
  }
  return dst;
}


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  if ((inputs.size() % 3) != 0) {
    throw poputil::poplibs_error("The number of input tensors must be a multiple of three. (Three for each layer, `weights`, `spike_ids` and `num_spikes`.)");
  }  

  size_t num_layers = inputs.size() / 3;

  std::vector<poplar::Tensor> matrix_bad_alloc(inputs.begin(),inputs.begin()+num_layers);
  std::vector<poplar::Tensor> spike_ids_fptype(inputs.begin()+num_layers,inputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> num_spikes_int(inputs.begin()+2*num_layers,inputs.begin()+3*num_layers);

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    if (matrix_bad_alloc[ilay].numElements() == 0) {
      return poplar::program::Sequence();
    }

    if (matrix_bad_alloc[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 2.");
    }

    if (spike_ids_fptype[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2.");
    }

    if (num_spikes_int[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[2]' must be tensor of rank 2.");
    }

    if (spike_ids_fptype[ilay].dim(1) > matrix_bad_alloc[ilay].dim(1)) {
      throw poputil::poplibs_error("Dimension 1 of 'inputs[1]' must be smaller or equal to 'inputs[0]' dimension 1.");
    }

  //   if (num_spikes_int[ilay].elementType() != poplar::INT) { // TODO uncomment
  //     throw poputil::poplibs_error("Input 'inputs[2]' must be of type 'int'.");
  //   }
  }

  std::cout << "\nNUM_TILES: " << graph.getTarget().getNumTiles() << std::endl;

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmulParallel"};
  auto prog = poplar::program::Sequence();

  std::vector<poplar::Tensor> matrix = alloc_tensor_vector_manually(graph, matrix_bad_alloc, attributes, true, prog, dnai);
  // TODO should later be a reinterpret cast
  std::vector<poplar::Tensor> spike_ids = cast_tensor_vector(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  // TODO should later be a reinterpret cast
  std::vector<poplar::Tensor> num_spikes = cast_tensor_vector(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});

  // auto neuronTileMapping = graph.getTileMapping(matrix.dimShuffle({1,0})[0], true);
  // std::cout << "SPARSE OPS"<< std::endl;
  // for (unsigned tile=0; tile<1472; ++tile){
  //   if (neuronTileMapping[tile].size() > 0){
  //     std::cout << tile << ": " << neuronTileMapping[tile].size() << std::endl;
  //   }
  // }

  std::vector<poplar::Tensor> result;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    size_t batchsize = spike_ids[ilay].dim(0);
    size_t numNeurons = matrix[ilay].dim(0);
    auto dtype = matrix[ilay].elementType();
    result.push_back(graph.addVariable(dtype, {batchsize, numNeurons}));
  }
  calcDynDenseSparseProd(graph, matrix, spike_ids, num_spikes, result, prog, {dnai, "calcDynDenseSparseProd"});

  extend_tensor_vector(result, outputs);
  return prog;
}

/// The gradient op requires its own metadata. Since it does not have any
/// internal state we can mark the op as stateless.
/// For stateless ops only one instance of the op is compiled even when
/// we ask for the gradient multiple times (e.g. we use tf.gradients() in
/// the python code).
extern "C"
void Build_grad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {

  is_stateless = true;
}


// TODO calculate gradients as separate funcitons !

/// Define the gradient op.
extern "C"
poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix) {


  std::cout << "\nEntering Build_grad\n" << std::endl;
  std::cout << "\nNUM_TILES: " << std::endl;
  std::cout << "\nNUM_TILES: " << graph.getTarget().getNumTiles() << std::endl;

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmulParallelGrads"};

  size_t num_layers = fwd_inputs.size() / 3;
  auto prog = poplar::program::Sequence();


  // auto dLdy = gradients[0];
  // auto weights = fwd_inputs[0];
  // auto spike_ids_fptype = fwd_inputs[1];
  // // TODO should later be a reinterpret cast
  // auto spike_ids = popops::cast(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  // auto num_spikes_int = fwd_inputs[2];
  // // TODO should later be a reinterpret cast
  // auto num_spikes = popops::cast(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});


  std::vector<poplar::Tensor> dLdy(gradients.begin(), gradients.begin()+num_layers);
  std::vector<poplar::Tensor> weights_bad_alloc(fwd_inputs.begin(), fwd_inputs.begin()+num_layers);
  // TODO remove and use line above when alloc works properly!
  std::vector<poplar::Tensor> weights = alloc_tensor_vector_manually(graph, weights_bad_alloc, attributes, true, prog, dnai);
  std::vector<poplar::Tensor> spike_ids_fptype(fwd_inputs.begin()+num_layers, fwd_inputs.begin()+2*num_layers);
  // TODO should later be a reinterpret cast
  std::vector<poplar::Tensor> spike_ids = cast_tensor_vector(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  std::vector<poplar::Tensor> num_spikes_int(fwd_inputs.begin()+2*num_layers, fwd_inputs.begin()+3*num_layers);
  // TODO should later be a reinterpret cast
  std::vector<poplar::Tensor> num_spikes = cast_tensor_vector(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});

  // if (input_grad_index == 0) { // gradient wrt weights
  //   poplar::Tensor dLdweights = graph.clone(weights, {dnai, "dLdweights"});
  //   popops::zero(graph, dLdweights, prog, dnai);
  //   calcWeightsGrad(graph, dLdweights, spike_ids, num_spikes, dLdy, prog, {dnai, "/calcWeightsGrad"});
  //   outputs.push_back(dLdweights);
  // } else if (input_grad_index == 1) { // gradient wrt spike_ids
  //   poplar::Tensor dLdx = calcInpSpikesGradRowWise(graph, weights, spike_ids, dLdy, prog, {});
  //   outputs.push_back(dLdx);
  // } else if (input_grad_index == 2) { // gradient wrt num_nzelements
  //   throw poputil::poplibs_error("The gradient wrt the number of input spikes is not defined and should not be necessary for any computation.");  
  // } else {
  //   throw poputil::poplibs_error("DynDenseBinarySparseMatmul only has 3 inputs.");
  // }

  std::vector<poplar::Tensor> dLdweights = clone_tensor_vector(graph, weights, {dnai, "dLdweights"});
  zero_tensor_vector(graph, dLdweights, prog, {dnai, "zero dLdweights"});
  calcWeightsGrad(graph, dLdweights, spike_ids, num_spikes, dLdy, prog, {dnai, "calcWeightsGrad"});

  std::cout << "dLdweights.size(): " << dLdweights.size() << std::endl;

  const bool first_layer_inp_grad = false; 
  std::vector<poplar::Tensor> dLdx = calcInpSpikesGradRowWise(graph, weights, spike_ids, dLdy, first_layer_inp_grad, prog, {dnai, "calcInpSpikesGradRowWise"});

  // poplar::Tensor dLdn = graph.clone(num_spikes_int, {dnai, "dLdnum_inp_spikes"}); 
  // outputs.push_back(dLdn);

  std::cout << "dLdweights.size(): " << dLdweights.size() << std::endl;
  std::cout << "dLdx.size(): " << dLdx.size() << std::endl;

  extend_tensor_vector(dLdweights, outputs);
  // // TODO does it expect some form of gradient if it is not calcuated for the first input ()
  // // ANSWER: No, it does not expect a tenosr to be returned for input tensors for which no gradient should be calculated
  // if (!first_layer_inp_grad) {
  //   poplar::Tensor none_tensor; 
  //   outputs.push_back(none_tensor);
  // }
  extend_tensor_vector(dLdx, outputs);

  return prog;
}