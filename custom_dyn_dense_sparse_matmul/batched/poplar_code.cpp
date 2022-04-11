#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>

template<typename T>
void printVector(std::vector<T> vec) {
  std::cout << "{";
  for (auto val: vec) {
    std::cout << val << ", ";
  }
  std::cout << "}"<< std::endl;
}


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

  size_t batchsize = spike_ids.dim(0);
  size_t numNeurons = matrix.dim(0);
  auto dtype = matrix.elementType();
  auto output = graph.addVariable(dtype, {batchsize, numNeurons});
  outputs.push_back(output);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  const auto numTiles = graph.getTarget().getNumTiles();
  auto neuronTileMapping = graph.getTileMapping(matrix.dimShuffle({1,0})[0], true);

  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet(debug_prefix + "/DynDenseBinarySparseMatmul");

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: thisTileMap) {
      const auto numNeuronsThisTile = neuronRange.size();
      poplar::Tensor neuronWeights = matrix.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronOut = output.slice(neuronRange, 1);
      graph.setTileMapping(neuronOut, tile);

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      for (unsigned ineuron = 0; ineuron < numNeuronsThisTile; ++ineuron){
        for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
          auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProduct", dtype),
                                    // {{"weights", weights[ilay][neuronId]},
                                    {{"matrix_slice", neuronWeights[ineuron]},
                                    {"spike_ids", spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                    {"num_spikes", num_spikes[ibatch][0]},
                                    {"output", neuronOut[ibatch][ineuron]}});
          graph.setTileMapping(v, tile);
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
    }
  }
  return poplar::program::Execute(cs);
}





//---------------------------------------------- backward -----------------------------------------

void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
                        poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  
  auto cs = graph.addComputeSet({dnai, "calcWeightsGrad"});
  const size_t numTiles = graph.getTarget().getNumTiles();

  auto dtype = dLdweights.elementType();
  size_t numRows = dLdweights.dim(0);
  size_t sparse_out_dim = fwdInpSpikeIds.dim(1);

  // auto neuronTileMapping = graph.getTileMapping(dLdweights.slice(0, 1, 1), true); // TODO use this
  auto neuronTileMapping = graph.getTileMapping(dLdweights.dimShuffle({1,0})[0], true);

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronDLdWeights = dLdweights.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronDLdy = dLdy.slice(neuronRange, 1);

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
      // TODO include batch-loop here when figured out how to be thread/parallel safe
      // parallelisms might intruduce probelms due to the += operation...
      for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradWeight", dtype),
                                  // {{"dLdy", neuronDLdy.slice(ineuron, ineuron+1, 1)},
                                  {{"dLdy", neuronDLdy.dimShuffle({1, 0})[0]},
                                  {"fwd_inp_spikes_ids", fwdInpSpikeIds.flatten()}, // TODO flatten here or does a Tensor structure exist for vertex Input ?
                                  {"fwd_num_inp_spikes", fwdNumInpSpikes.flatten()}, // TODO does this copy over for every neuron or once onto the tile ?
                                  {"dLdweights_row", neuronDLdWeights[ineuron]},
                                  {"sparse_out_dim", sparse_out_dim}});
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




poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdInpSpikeIds, 
                                  poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGradRowWise"});
  const size_t numTiles = graph.getTarget().getNumTiles();

  size_t numRows = weights.dim(0);
  size_t batchsize = dLdy.dim(0);
  auto dtype = weights.elementType();

  size_t sparseSize = fwdInpSpikeIds.dim(1);
  poplar::Tensor dLdx = graph.addVariable(dtype, {numRows, batchsize, sparseSize});

  auto neuronTileMapping = graph.getTileMapping(weights.dimShuffle({1,0})[0], true);

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronWeights = weights.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronDLdy = dLdy.slice(neuronRange, 1);
      poplar::Tensor neuronDLdx = dLdx.slice(neuronRange);
      graph.setTileMapping(neuronDLdx, tile);

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
          auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradInputsRowWise", dtype),
                                    {{"matrix_row", neuronWeights[ineuron]},
                                    {"dLdy", neuronDLdy[ibatch][ineuron]},
                                    {"fwd_inp_spike_ids", fwdInpSpikeIds[ibatch]},
                                    {"dLdinp_spike_ids", neuronDLdx[ineuron][ibatch]}});
          // !!! TODO !!! totally bogus tile mapping, must be improved
          // graph.setTileMapping(relevantWeights[irow][ibatch], start_tile+irow/rowsPerTile);
          graph.setTileMapping(v, tile); 
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
    }
  }
  prog.add(poplar::program::Execute(cs));

  // std::cout << "\nBefore Reduce\n" << std::endl;
  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
  poplar::Tensor dLdInpSpikes = reduce(graph, dLdx, {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  // poplar::Tensor dLdInpSpikes = graph.addVariable(dtype, {fwdInpSpikeIds.dim(0), fwdInpSpikeIds.dim(1)});
  // reduceWithOutput(graph, dLdx, dLdInpSpikes, {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  // std::cout << "\nAfter Reduce\n" << std::endl;
  return dLdInpSpikes;
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