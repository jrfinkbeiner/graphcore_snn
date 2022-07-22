#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>

#include "poplar_code.hpp"

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
    std::vector<poplar::Tensor>& outputs, const std::string& debug_prefix) {

  auto matrix = inputs[0];
  auto spike_ids = inputs[1];
  auto num_spikes = inputs[2];

  if (inputs.size() != 3) {
    throw poputil::poplibs_error("DynDenseBinarySparseProduct requires 3 inputs");
  }

  if (matrix.numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (matrix.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 2.");
  }

  if (spike_ids.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2.");
  }

  if (num_spikes.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be tensor of rank 2.");
  }

  if (spike_ids.dim(1) > matrix.dim(1)) {
    throw poputil::poplibs_error("Dimension 1 of 'inputs[1]' must be smaller or equal to 'inputs[0]' dimension 1.");
  }

  if (num_spikes.elementType() != poplar::INT) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be of type 'int'.");
  }

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmul"};
  auto prog = poplar::program::Sequence();

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
    const std::string& debug_prefix) {


  // graph.addCodelets("custom_codelet.cpp");

  poplar::DebugNameAndId dnai{debug_prefix, "/DynDenseBinarySparseMatmulGrads"};
  auto prog = poplar::program::Sequence();

  std::cout << "\nEntering Build_grad\n" << std::endl;

  auto dLdy = gradients[0];
  auto weights = fwd_inputs[0];
  auto spike_ids = fwd_inputs[1];
  auto num_spikes = fwd_inputs[2];

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
  popops::zero(graph, dLdweights, prog, dnai);
  calcWeightsGrad(graph, dLdweights, spike_ids, num_spikes, dLdy, prog, {dnai, "calcWeightsGrad"});
  outputs.push_back(dLdweights);

  poplar::Tensor dLdx = calcInpSpikesGradRowWise(graph, weights, spike_ids, dLdy, prog, {dnai, "calcInpSpikesGradRowWise"});
  outputs.push_back(dLdx);

  poplar::Tensor dLdn = graph.clone(num_spikes, {dnai, "dLdnum_inp_spikes"}); 
  outputs.push_back(dLdn);

  return prog;
}