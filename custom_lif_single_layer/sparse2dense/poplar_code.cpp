#include <iostream>
#include <vector>
#include <boost/optional.hpp>
#include <cmath> // ceil
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Zero.hpp>
// #include <poplibs_support/logging.hpp> // TODO no logging file...
#include <popnn/Rnn.hpp>
#include <popnn/NonLinearityDef.hpp> // TODO delete after sigmoid non-lin was replaced by custom non-lin
// #include "RnnUtil.hpp"
#include <popops/ElementWise.hpp>
#include <popops/TopK.hpp>
#include <popops/SortOrder.hpp>

// #include "RnnUtil.hpp" // only for boost::optional

//---------------------------------------------- Build functions -----------------------------------------

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
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  if (inputs.size() != 2) {
    throw poputil::poplibs_error("Sparse2Dense requires 6 inputs");
  }

  poplar::DebugNameAndId dnai{debug_prefix};

  poplar::Tensor spike_ids = inputs[0];
  poplar::Tensor num_spikes = inputs[1];
  
  if (spike_ids.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be tensor of rank 3 (seq_dim, batch_size, sparse_size).");
  }

  if (num_spikes.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 3 (seq_dim, batch_size, 1).");
  }

  size_t seq_len = spike_ids.dim(0);
  size_t batchsize = spike_ids.dim(1);
  size_t size_sparse = spike_ids.dim(2);
  size_t size_dense;
  sscanf(attributes.c_str(), "%zu", &size_dense);
  auto dtype = spike_ids.elementType();

  std::cout << "size_dense " << size_dense << std::endl;

  if (size_dense < size_sparse) {
    throw poputil::poplibs_error("Dense size must be larger than sparse size.");
  }

  poplar::program::Sequence fwdProg;

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  poplar::Tensor dense_spikes = graph.addVariable(dtype, {seq_len, batchsize, size_dense});
  outputs.push_back(dense_spikes);

  for (unsigned i = 0; i < seq_len; ++i) {
    for (unsigned j = 0; j < batchsize; ++j) {
      graph.setTileMapping(dense_spikes[i][j], i*batchsize+j);
    }
  }

  // set tensor to zeros
  popops::zero(graph, dense_spikes, fwdProg, dnai); 

  
  auto cs = graph.addComputeSet({dnai, "performSparse2Dense"});
  for (unsigned iseq = 0; iseq < seq_len; ++iseq) {
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("Sparse2Dense", dtype),
                              {{"spikeIds", spike_ids[iseq][ibatch]},
                               {"numSpikes", num_spikes[iseq][ibatch][0]},
                               {"denseSpikes", dense_spikes[iseq][ibatch]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      graph.setTileMapping(v, iseq*batchsize+ibatch); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  fwdProg.add(poplar::program::Execute(cs));

  // outputs.push_back(updatedState);
  return fwdProg;
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
    const std::string& attributes,
    const std::string& debug_prefix) {


  // if (input_grad_index != 0) {
  //   throw poputil::poplibs_error("Gradient calculation only defined for weight tensor ('inputs[0]').");
  // }

  poplar::program::Sequence bwdProg;
  poplar::DebugNameAndId dnai{debug_prefix};

  poplar::Tensor spike_ids = fwd_inputs[0];
  poplar::Tensor num_spikes = fwd_inputs[1];
  
  size_t seq_len = spike_ids.dim(0);
  size_t batchsize = spike_ids.dim(1);
  size_t size_sparse = spike_ids.dim(1);
  auto dtype = spike_ids.elementType();

  poplar::Tensor dLdSpikeIds = graph.clone(spike_ids, {dnai, "dLdSpikeIds"});
  poplar::Tensor dLdNumSpikes = graph.clone(num_spikes, {dnai, "dLdNumSpikes"});
  popops::zero(graph, dLdNumSpikes, bwdProg, dnai);
  outputs.push_back(dLdSpikeIds);
  outputs.push_back(dLdNumSpikes);

  poplar::Tensor dLdDenseSpikes = gradients[0];

  auto cs = graph.addComputeSet({dnai, "performSparse2DenseGrad"});
  for (unsigned iseq = 0; iseq < seq_len; ++iseq) {
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("Sparse2DenseGrad", dtype),
                              {{"dLdDenseSpikes", dLdDenseSpikes[iseq][ibatch]},
                               {"spikeIds", spike_ids[iseq][ibatch]},
                               {"dLdSpikeIds", dLdSpikeIds[iseq][ibatch]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      graph.setTileMapping(v, iseq*batchsize+ibatch); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  bwdProg.add(poplar::program::Execute(cs));
 
  return bwdProg;
}