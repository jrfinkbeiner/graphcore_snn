#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <popops/Cast.hpp>

#include <iostream>

// #include "poplar_code.hpp"
// #include "../../string_util.hpp"

// #include <popops>
// #include "custom_dynamic_sparse"
#include "custom_dynamic_sparse/string_util.hpp"
// #include "custom_dynamic_sparse/select_spikes_twoThresh.hpp"
// #include "select_spikes_twoThresh.hpp"
#include "custom_dynamic_sparse/custom_select_spikes/twoThresh/poplar_code.hpp"


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

// !!! TODO !!! place tensors


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  auto state = inputs[0];
  auto thresholds = inputs[1];
  // std::cout << "\nattributes fwd" << std::endl;
  // std::cout << "'" << attributes << "'"  << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  std::cout << atrribute_sizes[0] << std::endl;
  std::cout << atrribute_sizes[1] << std::endl;
  std::cout << atrribute_sizes[2] << std::endl;

  const size_t sparseSize = atrribute_sizes[0];
  const size_t startTile = atrribute_sizes[1];
  const size_t endTile = atrribute_sizes[2];

  if (inputs.size() != 2) {
    throw poputil::poplibs_error("ComputeSpikesTwoThreshs requires 2 inputs");
  }

  if (state.numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (state.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 2.");
  }

  if (thresholds.rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2.");
  }

  const size_t numNeurons = state.dim(1);
  if (sparseSize > numNeurons) {
    throw poputil::poplibs_error("'inputs[0].dim(1)' must be greater or equal sparseSize (first integer in 'attributes').");
  }

  poplar::DebugNameAndId dnai{debug_prefix, "/ComputeSpikesTwoThreshs"};
  auto prog = poplar::program::Sequence();

  auto out_spikes = select_spikes_two_threshs(graph, state, thresholds, sparseSize, startTile, endTile, prog, dnai);
  poplar::Tensor out_spikes_ids = out_spikes.first;
  poplar::Tensor num_out_spikes = out_spikes.second;

  // outputs.push_back(out_spikes_ids);
  // outputs.push_back(num_out_spikes);

  // TODO cast for now
  auto out_spikes_ids_fptype = popops::cast(graph, out_spikes_ids, poplar::FLOAT, prog, {dnai, "cast spikes"});
  auto num_out_spikes_int = popops::cast(graph, num_out_spikes, poplar::INT, prog, {dnai, "cast spikes"});
  outputs.push_back(out_spikes_ids_fptype);
  outputs.push_back(num_out_spikes_int);
  return prog;
}


//---------------------------------------------- backward -----------------------------------------

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

  poplar::DebugNameAndId dnai{debug_prefix, "/ComputeSpikesTwoThreshsGrad"};
  auto prog = poplar::program::Sequence();


  auto state = fwd_inputs[0];
  auto thresholds = fwd_inputs[1];
  // TODO cast for now!!!
  auto out_spikes_ids_fptype = fwd_outputs[0];
  auto out_spikes_ids = popops::cast(graph, out_spikes_ids_fptype, poplar::UNSIGNED_INT, prog, {dnai, "cast spikes"});
  // auto num_out_spikes = fwd_outputs[1];

  auto dLdoutSpikes = gradients[0];

  std::cout << "\nattributes bwd" << std::endl;
  std::cout << "'" << attributes << "'"  << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  printVector(atrribute_sizes);

  // const size_t sparseSize = atrribute_sizes[0];
  const size_t startTile = atrribute_sizes[1];
  const size_t endTile = atrribute_sizes[2];

  // const size_t numNeurons = state.dim(1);
  // const size_t batchsize = state.dim(0);
  auto dtype = state.elementType();



  poplar::Tensor dLdState = graph.clone(state, {dnai, "init dLdState"});
  popops::zero(graph, dLdState, prog, {dnai, "zero dLdState"});
  outputs.push_back(dLdState);

  // if (input_grad_index == 1) { // gradient wrt num_nzelements
  //   throw poputil::poplibs_error("The gradient wrt the thresholds is not implemented.");  
  // } else if (input_grad_index > 1) {
  //   throw poputil::poplibs_error("ComputeSpikesTwoThreshsGrad only has 3 inputs.");
  // }

  select_spikes_two_threshs_dLdState(graph, state, thresholds,
                                      dLdoutSpikes, out_spikes_ids, dLdState,
                                      startTile, endTile, prog, dnai);

  poplar::Tensor dLdthreshs = graph.clone(thresholds, {dnai, "init dLdthreshs"});
  popops::zero(graph, dLdthreshs, prog, {dnai, "zero dLdthreshs"});
  outputs.push_back(dLdthreshs);

  return prog;
}