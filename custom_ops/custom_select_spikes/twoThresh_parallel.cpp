#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>

// #include "poplar_code.hpp"
// #include "../../string_util.hpp"

// #include <popops>
// #include "custom_dynamic_sparse"
#include "custom_dynamic_sparse/string_util.hpp"
#include "custom_dynamic_sparse/vector_util.hpp"
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


  if ((inputs.size() % 2) != 0) {
    throw poputil::poplibs_error("The number of input tensors must be a multiple of two. (Two for each layer, `state` and `thresholds`.)");
  }  
  
  const size_t num_layers = inputs.size() / 2;

  std::vector<poplar::Tensor> states(inputs.begin(),inputs.begin()+num_layers);
  std::vector<poplar::Tensor> thresholds(inputs.begin()+num_layers,inputs.begin()+2*num_layers);

  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  std::cout << atrribute_sizes[0] << std::endl;
  std::cout << atrribute_sizes[1] << std::endl;
  std::cout << atrribute_sizes[2] << std::endl;

  const std::vector<size_t> sparseSizes(atrribute_sizes.begin(), atrribute_sizes.begin()+num_layers);
  const std::vector<size_t> startTiles(atrribute_sizes.begin()+num_layers, atrribute_sizes.begin()+2*num_layers);
  const std::vector<size_t> endTiles(atrribute_sizes.begin()+2*num_layers, atrribute_sizes.begin()+3*num_layers);

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    if (states[ilay].numElements() == 0) {
      return poplar::program::Sequence();
    }

    if (states[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 2.");
    }

    if (thresholds[ilay].rank() != 1) {
      throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2.");
    }

    const size_t numNeurons = states[ilay].dim(1);
    if (sparseSizes[ilay] > numNeurons) {
      throw poputil::poplibs_error("'inputs[0].dim(1)' must be greater or equal sparseSize (first integer in 'attributes').");
    }
  }
  poplar::DebugNameAndId dnai{debug_prefix, "/ComputeSpikesTwoThreshsParallel"};
  auto prog = poplar::program::Sequence();
  std::vector<poplar::Tensor> out_spikes = select_spikes_two_threshs(graph, states, thresholds, sparseSizes, startTiles, endTiles, prog, dnai);
  extend_tensor_vector(out_spikes, outputs);
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

  poplar::DebugNameAndId dnai{debug_prefix, "/ComputeSpikesTwoThreshsParallelGrad"};

  size_t num_layers = fwd_inputs.size() / 2;
  std::vector<poplar::Tensor> state(fwd_inputs.begin(),fwd_inputs.begin()+num_layers);
  std::vector<poplar::Tensor> thresholds(fwd_inputs.begin()+num_layers,fwd_inputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> out_spikes_ids(fwd_outputs.begin(), fwd_outputs.begin()+num_layers);
  std::vector<poplar::Tensor> dLdoutSpikes(gradients.begin(), gradients.begin()+num_layers);

  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  std::cout << atrribute_sizes[0] << std::endl;
  std::cout << atrribute_sizes[1] << std::endl;
  std::cout << atrribute_sizes[2] << std::endl;

  // const std::vector<size_t> sparseSize(atrribute_sizes.begin(), atrribute_sizes.begin()+num_layers);
  const std::vector<size_t> startTile(atrribute_sizes.begin()+num_layers, atrribute_sizes.begin()+2*num_layers);
  const std::vector<size_t> endTile(atrribute_sizes.begin()+2*num_layers, atrribute_sizes.begin()+3*num_layers);

  auto prog = poplar::program::Sequence();

  std::vector<poplar::Tensor> dLdState = clone_tensor_vector(graph, state, {dnai, "init dLdState"});
  zero_tensor_vector(graph, dLdState, prog,  {dnai, "zero dLdState"});

  select_spikes_two_threshs_dLdState(graph, state, thresholds,
                                      dLdoutSpikes, out_spikes_ids, dLdState,
                                      startTile, endTile, prog, dnai);
  extend_tensor_vector(dLdState, outputs);
  return prog;
}