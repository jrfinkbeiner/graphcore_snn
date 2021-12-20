#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <iostream>

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


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debug_prefix) {

  if (inputs.size() != 3) {
    throw poputil::poplibs_error("DynDenseBinarySparseProduct requires 3 inputs");
  }

  if (inputs[0].numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (inputs[0].rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be matrix (tensor of rank 2).");
  }

  if (inputs[1].rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be vector (tensor of rank 1).");
  }

  if (inputs[2].rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be vector (tensor of rank 1).");
  }

  if (inputs[1].dim(0) > inputs[0].dim(1)) {
    throw poputil::poplibs_error("Input 'inputs[1]' size must be size smaller or equal to input 'inputs[0]' dimension 1.");
  }

  if (inputs[2].elementType() != poplar::INT) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be of type 'int'.");
  }

  auto dtype = inputs[0].elementType();

  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet(debug_prefix + "/DynDenseBinarySparseMatmul");

  // // TODO Here it really only makes sense to place whole rows of the matrix 
  // // (and the corresponding vertex) on one tile and not use some already existing mapping.
  // // For that the whole LIF should be built as custom layer, not just the sparse op
  // auto tileMapping = graph.getTileMapping(inputs[0]);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // Get the vector width of the particular data type, so that later we can
  // divide the tensor up between workers in an appropriate way.
  const auto vectorWidth = target.getVectorWidth(dtype);

  auto matrix = inputs[0];
  auto sparse_vec = inputs[1];
  auto num_nzelements = inputs[2];
  outputs.push_back(graph.addVariable(dtype, {matrix.dim(0)}));
  auto output = outputs[0];
  for (unsigned i = 0; i < output.dim(0); ++i) {
    graph.setTileMapping(output[i], i);
  }

  for (unsigned irow = 0; irow < matrix.dim(0); ++irow) {
    auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProduct", dtype),
                              {{"matrix_slice", matrix[irow]},
                              {"sparse_vec", sparse_vec},
                              {"num_nzelements", num_nzelements[0]},
                              {"output", output[irow]}});
    // TODO totally bogus tile mapping, must be improved
    graph.setTileMapping(v, irow);
    // Provide a bogus cycle count estimate for the profiler.
    graph.setPerfEstimate(v, 1);
  }
  return poplar::program::Execute(cs);
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

/// Define the gradient op.
extern "C"
poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& debug_prefix) {


  auto cs = graph.addComputeSet(debug_prefix + "/DynDenseBinarySparseMatmulGrads");

  auto dLdy = gradients[0];
  auto weights = fwd_inputs[0]; // TODO not all weights are necessary, 
                                // but only the columns that correspond to non-zero elements
                                // therfore the ones which were used used during forward pass
  auto sparse_vec = fwd_inputs[1];
  auto num_nzelements = fwd_inputs[2];

  auto dtype = weights.elementType();

  auto dLdW = graph.addVariable(dtype, {weights.dim(0), weights.dim(1)}, "dLdW");
  auto dLdx = graph.addVariable(dtype, {sparse_vec.dim(0)}, "dLdx");
  auto dLdn = graph.addConstant<int>(poplar::INT, {1}, {0});
  graph.setTileMapping(dLdn, 0);

  for (unsigned irow = 0; irow < weights.dim(0); ++irow) {
    graph.setTileMapping(dLdW[irow], 1+irow);
    auto vtx = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradWeight", dtype),
                              {{"dLdyi", dLdy[irow]},
                               {"sparse_vec", sparse_vec},
                               {"num_nzelements", num_nzelements[0]},
                               {"dLdW_row", dLdW[irow]}});


    // Map the vertex onto the appropriate tile.
    graph.setTileMapping(vtx, 1+irow);
    // Provide a bogus cycle count estimate for the profiler.
    graph.setPerfEstimate(vtx, 1);
  }

  graph.setTileMapping(dLdx, 0);
  auto vtx = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradInputs", dtype),
                              {{"dLdy", dLdy},
                               {"weights", weights.flatten()},
                               {"num_cols", weights.dim(1)},
                               {"sparse_vec", sparse_vec},
                               {"num_nzelements", num_nzelements[0]},
                               {"dLdx", dLdx}});


  // Map the vertex onto the appropriate tile.
  graph.setTileMapping(vtx, 0);
  // Provide a bogus cycle count estimate for the profiler.
  graph.setPerfEstimate(vtx, 1);

  outputs.push_back(dLdW);
  outputs.push_back(dLdx);
  outputs.push_back(dLdn);

  return poplar::program::Execute(cs);
}