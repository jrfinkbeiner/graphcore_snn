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
  std::cout << "\nattributes fwd" << std::endl;
  std::cout << "'" << attributes << "'"  << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  printVector(atrribute_sizes);
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

  const size_t batchsize = state.dim(0);
  auto dtype = state.elementType();
  auto out_spikes_ids = graph.addVariable(dtype, {batchsize, sparseSize});
  auto num_out_spikes = graph.addVariable(poplar::INT, {batchsize, 1});
  outputs.push_back(out_spikes_ids);
  outputs.push_back(num_out_spikes);

  const size_t numTilesToUse{startTile - endTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet(debug_prefix + "/ComputeSpikesTwoThreshs");
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("SpikesTwoThreshs", dtype),
                              {{"state", state[ibatch]},
                              {"thresholds", thresholds},
                              {"out_spikes_ids", out_spikes_ids[ibatch]},
                              {"num_out_spikes", num_out_spikes[ibatch][0]}});
    size_t tile{startTile + ibatch/batchesPerTile};
    graph.setTileMapping(out_spikes_ids[ibatch], tile);
    graph.setTileMapping(num_out_spikes[ibatch], tile);
    graph.setTileMapping(v, tile);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }

  return poplar::program::Execute(cs);
}





//---------------------------------------------- backward -----------------------------------------

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


  poplar::DebugNameAndId dnai{debug_prefix, "/ComputeSpikesTwoThreshsGrad"};

  auto state = fwd_inputs[0];
  auto thresholds = fwd_inputs[1];
  auto out_spikes_ids = fwd_outputs[0];
  auto num_out_spikes = fwd_outputs[1];

  auto dLdoutSpikes = gradients[0];

  std::cout << "\nattributes bwd" << std::endl;
  std::cout << "'" << attributes << "'"  << std::endl;
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  printVector(atrribute_sizes);
  // const size_t sparseSize = atrribute_sizes[0];
  // const size_t startTile = atrribute_sizes[1];
  // const size_t endTile = atrribute_sizes[2];

  const size_t sparseSize = out_spikes_ids.dim(1);
  const size_t startTile = 1;
  const size_t endTile = 25;

  const size_t numNeurons = state.dim(1);
  const size_t batchsize = state.dim(0);
  auto dtype = state.elementType();

  auto prog = poplar::program::Sequence();

  poplar::Tensor dLdState = graph.clone(state, {dnai, "init dLdState"});
  popops::zero(graph, dLdState, prog, {dnai, "zero dLdState"});
  outputs.push_back(dLdState);

  // if (input_grad_index == 1) { // gradient wrt num_nzelements
  //   throw poputil::poplibs_error("The gradient wrt the thresholds is not implemented.");  
  // } else if (input_grad_index > 1) {
  //   throw poputil::poplibs_error("ComputeSpikesTwoThreshsGrad only has 3 inputs.");
  // }

  const size_t numTilesToUse{startTile - endTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  auto cs = graph.addComputeSet({dnai, "computeSpikesTwoThreshsGrad"});
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("StateGrad", dtype),
                              {{"fwdState", state[ibatch]},
                              {"thresholds", thresholds},
                              {"dLdoutSpikes", dLdoutSpikes[ibatch]},
                              {"fwd_out_spikes_ids", out_spikes_ids[ibatch]},
                              //  {"dLdState_inp", dLdState[ibatch]},
                              //  {"fwd_num_out_spikes", fwdOutSpikes.num_spikes[ibatch][0]},
                              //  {"dLdState", dLdState[ibatch]}});
                              {"dLdState", dLdState[ibatch]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // should be based on state mapping
    // graph.setTileMapping(v, (ibatch+1)*32); 
    size_t tile{startTile + ibatch/batchesPerTile};
    graph.setTileMapping(v, tile);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));

  poplar::Tensor dLdthreshs = graph.clone(thresholds, {dnai, "init dLdthreshs"});
  popops::zero(graph, dLdthreshs, prog, {dnai, "zero dLdthreshs"});
  outputs.push_back(dLdthreshs);

  return prog;
}