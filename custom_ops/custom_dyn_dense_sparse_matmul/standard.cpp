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


poplar::Tensor alloc_matrix_manually(poplar::Graph& graph, poplar::Tensor &matrix_bad_alloc, const std::string& attributes, bool copy_values, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::cout << "\nalloc manually" << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  size_t start_tile = atrribute_sizes[0];
  size_t end_tile = atrribute_sizes[1];
  std::cout << matrix_bad_alloc.shapeToString() <<std::endl;
  std::cout << start_tile << " " << end_tile << "\n" << std::endl;
  size_t neuronDim = 0;
  poplar::Tensor matrix = alloc_neuronwise(graph, matrix_bad_alloc.shape(), matrix_bad_alloc.elementType(), neuronDim, start_tile, end_tile, {dnai, "alloc matrix"});
  if (copy_values) {
    prog.add(poplar::program::Copy(matrix_bad_alloc, matrix, false, {dnai, "copy to matrix"}));
  }
  return matrix;
}

// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  auto matrix_bad_alloc = inputs[0];
  auto spike_ids_fptype = inputs[1];
  auto num_spikes_int = inputs[2];

  if (inputs.size() != 3) {
    throw poputil::poplibs_error("DynDenseBinarySparseProduct requires 3 inputs");
  }

  if (matrix_bad_alloc.numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (matrix_bad_alloc.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 2.");
  }

  if (spike_ids_fptype.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2.");
  }

  if (num_spikes_int.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be tensor of rank 2.");
  }

  if (spike_ids_fptype.dim(1) > matrix_bad_alloc.dim(1)) {
    throw poputil::poplibs_error("Dimension 1 of 'inputs[1]' must be smaller or equal to 'inputs[0]' dimension 1.");
  }

  // if (num_spikes_int.elementType() != poplar::INT) { // TODO uncomment
  //   throw poputil::poplibs_error("Input 'inputs[2]' must be of type 'int'.");
  // }

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmul"};
  auto prog = poplar::program::Sequence();

  poplar::Tensor matrix = alloc_matrix_manually(graph, matrix_bad_alloc, attributes, true, prog, dnai);
  // TODO should later be a reinterpret cast
  auto spike_ids = popops::cast(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  // TODO should later be a reinterpret cast
  auto num_spikes = popops::cast(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});

  // auto neuronTileMapping = graph.getTileMapping(matrix.dimShuffle({1,0})[0], true);
  // std::cout << "SPARSE OPS"<< std::endl;
  // for (unsigned tile=0; tile<1472; ++tile){
  //   if (neuronTileMapping[tile].size() > 0){
  //     std::cout << tile << ": " << neuronTileMapping[tile].size() << std::endl;
  //   }
  // }

  size_t batchsize = spike_ids.dim(0);
  size_t numNeurons = matrix.dim(0);
  auto dtype = matrix.elementType();
  auto output = graph.addVariable(dtype, {batchsize, numNeurons});
  outputs.push_back(output);

  calcDynDenseSparseProd(graph, matrix, spike_ids, num_spikes, output, prog, {dnai, "calcDynDenseSparseProd"});

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



  // graph.addCodelets("custom_codelet.cpp");

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmulGrads"};
  auto prog = poplar::program::Sequence();

  std::cout << "\nEntering Build_grad" << std::endl;
  std::cout << "\nNUM_TILES: " << graph.getTarget().getNumTiles() << std::endl;

  // auto dLdy = gradients[0];
  // auto weights = fwd_inputs[0];
  // auto spike_ids_fptype = fwd_inputs[1];
  // // TODO should later be a reinterpret cast
  // auto spike_ids = popops::cast(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  // auto num_spikes_int = fwd_inputs[2];
  // // TODO should later be a reinterpret cast
  // auto num_spikes = popops::cast(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});


  poplar::Tensor dLdy = gradients[0];
  poplar::Tensor weights_bad_alloc = fwd_inputs[0];
  // TODO remove and use line above when alloc works properly!
  poplar::Tensor weights = alloc_matrix_manually(graph, weights_bad_alloc, attributes, true, prog, dnai);
  poplar::Tensor spike_ids_fptype = fwd_inputs[1];
  // TODO should later be a reinterpret cast
  poplar::Tensor spike_ids = popops::cast(graph, spike_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  poplar::Tensor num_spikes_int = fwd_inputs[2];
  // TODO should later be a reinterpret cast
  poplar::Tensor num_spikes = popops::cast(graph, num_spikes_int, poplar::UNSIGNED_INT, prog, {dnai, "cast num spikes"});


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

  poplar::Tensor dLdweights = graph.clone(weights, {dnai, "dLdweights"});
  popops::zero(graph, dLdweights, prog, {dnai, "zero dLdweights"});
  // calcWeightsGrad(graph, dLdweights, spike_ids_fptype, num_spikes_int, dLdy, prog, {dnai, "calcWeightsGrad"});
  calcWeightsGrad(graph, dLdweights, spike_ids, num_spikes, dLdy, prog, {dnai, "calcWeightsGrad"});
  outputs.push_back(dLdweights);

  poplar::Tensor dLdx = calcInpSpikesGradRowWise(graph, weights, spike_ids, dLdy, prog, {dnai, "calcInpSpikesGradRowWise"});
  // poplar::Tensor dLdx = graph.clone(spike_ids_fptype, {dnai, "dLdspike_ids"});
  // popops::zero(graph, dLdx, prog, dnai);
  outputs.push_back(dLdx);

  // poplar::Tensor dLdn = graph.clone(num_spikes_int, {dnai, "dLdnum_inp_spikes"}); 
  // outputs.push_back(dLdn);

  return prog;
}