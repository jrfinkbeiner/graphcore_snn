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
#include <popops/Fill.hpp>
// #include <poplibs_support/logging.hpp> // TODO no logging file...
#include <popnn/Rnn.hpp>
#include <popnn/NonLinearityDef.hpp> // TODO delete after sigmoid non-lin was replaced by custom non-lin
// #include "RnnUtil.hpp"
#include <popops/ElementWise.hpp>
#include <popops/TopK.hpp>
#include <popops/SortOrder.hpp>

// #include "RnnUtil.hpp" // only for boost::optional



// TODO use dim and dtype info like `batchsize` or `dtype` from LIFParams 
// TODO instead of obtaining them from input arrays in every function ? 


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

struct LIFOpts {
  bool inferenceOnly;
  poplar::Type partialsType;
  boost::optional<double> availableMemoryProportion;
  boost::optional<std::size_t> numShards;
  boost::optional<bool> rnnCodeReuse;

  LIFOpts();
  LIFOpts(bool inferenceOnly, poplar::Type &partialsType)
    : inferenceOnly{inferenceOnly}
    , partialsType{partialsType}
    {};
};


// TODO this function might need further (slight) reworking/adjustments
// TODO adjust text
// Sharding is relevant for LSTM/GRU models which use significantly fewer
// tiles for storage of sequences than are available on the target. The total
// memory required to store the input and output dimensions is directly
// proportional to the LSTM sequence size. For large sequence sizes the tiles
// on which the sequences have been mapped would run out of memory, even with
// the availability of spare memory on the unmapped tiles on the same IPU.
// Sharding alleviates this problem by mapping the sequences to disjoint
// sets of tiles. The ratio of the total number of tiles on the target to the
// number of tiles that the sequences would be mapped to without sharding
// determines the maximum number of shards. However sharding involves code
// duplication and memory overheads due to additional exchanges. These memory
// usage overheads could become prohibitive when excessive sharding is applied.
// Likewise sharding also adds execution time overheads.
//
// For reasonably sized batch/feature dimensions the maximum number of shards
// is a small enough number which can be used to directly determine the number
// of shards. However this approach does not work well for smaller sized LSTM
// models. For very small input and output layer sizes and small batch sizes
// the maximum number of shards could run into the hundreds or thousands.
//
// To limit sharding when batch/feature dimensions are small, we allow operands
// to occupy up to 10% of total tile memory before sharding further. Layers
// with reasonably large batch/feature dimensions typically utilise enough tiles
// that the maximum shards calculated is small even if memory usage per-tile for
// operands is high. Hence this only really applies to the small cases.
//
// All LSTM passes - Fwd, Bwd & WU passes - must use the same number of shards.
// Hence, operand memory is calculated based on the Fwd pass since it can
// be used as a reasonable approximation for all the passes.
static std::size_t getNumShards(const poplar::Graph &graph, const LIFParams &params,
                                const LIFOpts &opt,
                                const poplar::DebugNameAndId &dnai) {
  auto target = graph.getTarget();
  auto tileMemory = target.getBytesPerTile();
  auto maxShards = params.rnn.getMaxShards(graph);

  size_t dimOutSparse = params.rnn.layerSizes[1];
  size_t dimOut = params.numNeurons;

  auto inputSize = params.rnn.getInputBytesPerTile(graph);
  auto outputSize = params.rnn.getOutputBytesPerTile(graph); // TODO adjust *2 based on actual mem requirement 
  auto stateSize = (outputSize * dimOut) / dimOutSparse;
  auto numIntermediates = 1;
  // auto numIntermediates = getNumFwdIntermediatesToSave(params, opt);
  // *2 due to two tensors in input and output
  // TODO adjust *2 based on actual types in input and ouput tensors
  // TODO one is float32 other is int32, but could less in the furture
  auto operandSingleIteration = 2*inputSize + 2*outputSize + numIntermediates*stateSize;
  auto operandSize = operandSingleIteration * params.rnn.maxTimeSteps;

  // Fraction of total tile memory that is nominally designated for operands
  double operandFraction = 0.1;

  double availableOperandMemory = tileMemory * operandFraction;
  std::size_t estShards = std::ceil(operandSize / availableOperandMemory);
  auto numShards = std::min(estShards, maxShards);
  if (opt.numShards) {
    if ((*opt.numShards < 1) || (*opt.numShards > maxShards)) {
      throw poputil::poplibs_error("LSTM numShards must be within "
                                   "interval [1," +
                                   std::to_string(maxShards) + "]");
    }
    numShards = *opt.numShards;
  }
  // TODO uncomment when imclude logging works
  // poplibs::logging::popnn::debug(
  //     "'{}': inputSize={} outputSize={} operandSize={} numInter={} "
  //     "available={} maxShards={} estimated-shards={} numShards={}",
  //     dnai.getPathName(), inputSize, outputSize, operandSize, numIntermediates,
  //     availableOperandMemory, maxShards, estShards, numShards);
  return numShards;
}


template<typename T>
void printVector(std::vector<T> vec) {
  std::cout << "{";
  for (auto val: vec) {
    std::cout << val << ", ";
  }
  std::cout << "}"<< std::endl;
}


//---------------------------------------------- forward -----------------------------------------

poplar::Tensor performBatchedLIFStateUpdate(poplar::Graph &graph, poplar::Tensor &weights, 
                            poplar::Tensor &state, BatchedSparseSpikes &inp_spikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "performBatchedLIFStateUpdate"});

  auto dtype = weights.elementType();
  size_t batchsize = state.dim(0);
  poplar::Tensor new_state = graph.clone(state, {dnai, "New State"});
  for (unsigned irow = 0; irow < weights.dim(0); ++irow) {
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateUpdate", dtype),
                                {{"weights", weights[irow]},
                                {"state", state[ibatch][irow]},
                                {"inp_spikes_ids", inp_spikes.spike_ids[ibatch]},
                                {"num_inp_spikes", inp_spikes.num_spikes[ibatch][0]},
                                {"decay_constant", decay_constants[irow]},
                                {"threshold", thresholds[irow]},
                                {"new_state", new_state[ibatch][irow]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on weights mapping
      graph.setTileMapping(v, irow);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }

  prog.add(poplar::program::Execute(cs));
  return new_state;
}


void genBatchedLIFOutSpikesOnlySpikes(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, BatchedSparseSpikes &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  auto cs = graph.addComputeSet({dnai, "genBatchedLIFOutSpikes"});

  auto dtype = state.elementType();
  size_t batchsize = state.dim(0);
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes", dtype),
                              {{"state", state[ibatch]},
                               {"thresholds", thresholds},
                               {"out_spikes_ids", out_spikes.spike_ids[ibatch]},
                               {"num_out_spikes", out_spikes.num_spikes[ibatch][0]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // most likely should be based on out_spikes mapping
    graph.setTileMapping(v, ibatch);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }

  prog.add(poplar::program::Execute(cs));
}                    


void genBatchedLIFOutSpikesTopK(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, BatchedSparseSpikes &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  // popops::SortOrder sortOrder = None;
  // popops::SortOrder sortOrder = popops::SortOrder::NONE;
  auto numSparseOutSpikes = out_spikes.spike_ids.dim(1);
  // popops::TopKParams topKparams(numSparseOutSpikes, true, popops::SortOrder::DESCENDING);
  popops::TopKParams topKparams(numSparseOutSpikes, true, popops::SortOrder::NONE);

  std::pair<poplar::Tensor, poplar::Tensor> topKStatesPair{popops::topKWithPermutation(graph, prog, state, topKparams, dnai)};
  poplar::Tensor topKStateVals = topKStatesPair.first;
  poplar::Tensor topKStateIds = topKStatesPair.second;

  auto cs = graph.addComputeSet({dnai, "genBatchedLIFOutSpikesFromTopK"});
  auto dtype = state.elementType();
  size_t batchsize = state.dim(0);
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikesFromTopK", dtype),
                              {{"topKStateVals", topKStateVals[ibatch]},
                               {"topKStateIds", topKStateIds[ibatch]},
                               {"thresholds", thresholds},
                               {"out_spikes_ids", out_spikes.spike_ids[ibatch]},
                               {"num_out_spikes", out_spikes.num_spikes[ibatch][0]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // most likely should be based on out_spikes mapping
    graph.setTileMapping(v, ibatch);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }

  prog.add(poplar::program::Execute(cs));
}                    


poplar::Tensor performLIFStepFworwardPass(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &state, BatchedSparseSpikes &inp_spikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &out_spikes,
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai = {}) {
  
  poplar::Tensor new_state{performBatchedLIFStateUpdate(graph, weights, state, inp_spikes, decay_constants, thresholds, prog, dnai)};
  genBatchedLIFOutSpikesTopK(graph, new_state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikesOnlySpikes(graph, new_state, thresholds, out_spikes, prog, dnai);
  return new_state;
}



//---------------------------------------------- backward -----------------------------------------


// void calcLIFStateGrad(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdState, 
//                             poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &fwdOutSpikes,
//                             poplar::Tensor &dLdState, poplar::Tensor &dLdoutSpikes, 
//                             poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai) {

//   auto dtype = weights.elementType();
//   size_t batchsize = fwdState.dim(0);

//   popops::mulInPlace(graph, dLdState, decay_constants.expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);

//   poplar::Tensor dLdState_clone = graph.clone(dLdState, {dnai, "dLdweights_clone"});
//   popops::zero(graph, dLdState_clone, prog, dnai);

//   auto cs = graph.addComputeSet({dnai, "calcLIFStateOutGrad"});
//   for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
//     auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateOutGrad", dtype),
//                               {{"fwdState", fwdState[ibatch]},
//                                {"thresholds", thresholds},
//                                {"dLdoutSpikes", dLdoutSpikes[ibatch]},
//                                {"fwd_out_spikes_ids", fwdOutSpikes.spike_ids[ibatch]},
//                               //  {"fwd_num_out_spikes", fwdOutSpikes.num_spikes[ibatch][0]},
//                               //  {"dLdState", dLdState[ibatch]}});
//                                {"dLdState", dLdState_clone[ibatch]}});
//     // !!! TODO !!! totally bogus tile mapping, must be improved
//     // should be based on state mapping
//     graph.setTileMapping(v, ibatch); 
//     // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
//     graph.setPerfEstimate(v, 1);
//   }
//   prog.add(poplar::program::Execute(cs));
//   popops::addInPlace(graph, dLdState, dLdState_clone, prog, dnai);
// }

void calcLIFStateGrad(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdState, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &fwdOutSpikes,
                            poplar::Tensor &dLdState, poplar::Tensor &dLdoutSpikes, 
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai) {

  auto dtype = weights.elementType();
  size_t batchsize = fwdState.dim(0);

  popops::mulInPlace(graph, dLdState, decay_constants.expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);

  auto cs = graph.addComputeSet({dnai, "calcLIFStateOutGrad"});
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateOutGrad", dtype),
                              {{"fwdState", fwdState[ibatch]},
                               {"thresholds", thresholds},
                               {"dLdoutSpikes", dLdoutSpikes[ibatch]},
                               {"fwd_out_spikes_ids", fwdOutSpikes.spike_ids[ibatch]},
                              //  {"dLdState_inp", dLdState[ibatch]},
                              //  {"fwd_num_out_spikes", fwdOutSpikes.num_spikes[ibatch][0]},
                              //  {"dLdState", dLdState[ibatch]}});
                               {"dLdState", dLdState[ibatch]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // should be based on state mapping
    graph.setTileMapping(v, ibatch); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFWeightGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &decay_constants, poplar::Tensor &dLdState, 
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai) {

  auto dtype = dLdweights.elementType();
  size_t num_rows = dLdweights.dim(0);
  size_t sparse_out_dim = fwdInpSpikes.spike_ids.dim(1);
  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});

  // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
  // TODO include batch-loop here when figured out how to be thread/parallel safe
  // parallelisms might intruduce probelms due to the += operation...
  for (unsigned irow = 0; irow < num_rows; ++irow) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGrad", dtype),
                              {{"dLdState", dLdState.dimShuffle({1,0})[irow]},
                              {"decay_constant", decay_constants[irow]},
                              {"fwd_inp_spikes_ids", fwdInpSpikes.spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                              {"fwd_num_inp_spikes", fwdInpSpikes.num_spikes.dimShuffle({1,0})[0]},
                              {"sparse_out_dim", sparse_out_dim},
                              {"dLdweights_row", dLdweights[irow]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // should be based on state mapping
    graph.setTileMapping(v, irow); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void selectLIFInpSpikeGrads(poplar::Graph &graph, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &dLdx, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai) {
  size_t batchsize = dLdx.dim(0); 
  auto dtype = dLdx.elementType();
  auto cs = graph.addComputeSet({dnai, "selectLIFInpSpikeGrads"});
  // TODO include batch-loop here when figured out how to be thread/parallel safe
  // parallelisms might intruduce probelms due to the += operation...
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFSelectInpSpikesGrad", dtype),
                              {{"fwd_inp_spike_ids", fwdInpSpikes.spike_ids[ibatch]},
                              //  {"fwd_num_inp_spikes", fwdInpSpikes.num_spikes[ibatch][0]},
                               {"dLdx", dLdx[ibatch]},
                               {"dLdInpSpikes", dLdInpSpikes[ibatch]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // should be based on state mapping
    graph.setTileMapping(v, ibatch); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFInpSpikesGrad(poplar::Graph &graph, poplar::Tensor &weights, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &decay_constants,
                            poplar::Tensor &dLdState, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai) {  
  // TODO IMPORTANT: For backwards bass, weight matrix schould be distributed column-wise to different tiles
  size_t num_cols = weights.dim(1);
  size_t batchsize = dLdState.dim(0);
  auto dtype = weights.elementType();

  poplar::Tensor dLdx = graph.addVariable(weights.elementType(), {batchsize, num_cols});

  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGrad"});
  // TODO include batch-loop here when figured out how to be thread/parallel safe
  // parallelisms might intruduce probelms due to the += operation...  
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned icol = 0; icol < num_cols; ++icol) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFInpSpikesGrad", dtype),
                                {{"weights_column", weights.dimShuffle({1,0})[icol]},
                                {"dLdState", dLdState[ibatch]},
                                {"decay_constants", decay_constants},
                                {"fwd_inp_spike_ids", fwdInpSpikes.spike_ids[ibatch]},
                                // {"fwd_num_inp_spikes", fwdInpSpikes.num_spikes[ibatch][0]},
                                {"col_id", icol},
                                {"dLdx", dLdx[ibatch][icol]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      graph.setTileMapping(dLdx[ibatch][icol], icol);
      graph.setTileMapping(v, icol); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));

  selectLIFInpSpikeGrads(graph, fwdInpSpikes, dLdx, dLdInpSpikes, prog, opt, params, dnai);
}


void performLIFStepBackwardPass(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdState, BatchedSparseSpikes &fwdInpSpikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &fwdOutSpikes,
                            poplar::Tensor &dLdweights, poplar::Tensor &dLdState, poplar::Tensor &dLdOutSpikes, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const LIFOpts &opt, const LIFParams &params, const poplar::DebugNameAndId &dnai = {}) {
  
  calcLIFStateGrad(graph, weights, fwdState, decay_constants, thresholds, fwdOutSpikes, dLdState, dLdOutSpikes, prog, opt, params, dnai);
  calcLIFWeightGrad(graph, dLdweights, fwdInpSpikes, decay_constants, dLdState, prog, opt, params, dnai);
  calcLIFInpSpikesGrad(graph, weights, fwdInpSpikes, decay_constants, dLdState, dLdInpSpikes,  prog, opt, params, dnai);
}


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


poplar::Tensor alloc_perneuron_1d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[0];
  size_t numTiles = graph.getTarget().getNumTiles();
  size_t neuronsPerTile = numNeurons / numTiles + 1;

  for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
    graph.setTileMapping(allocTensor[ineuron], start_id+ineuron/neuronsPerTile);
  }
  return allocTensor;
}

poplar::Tensor alloc_rowwise_2d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numRows = shape[0];
  size_t numTiles = graph.getTarget().getNumTiles();
  size_t rowsPerTile = numRows / numTiles + 1;

  for (unsigned irow = 0; irow < numRows; ++irow) {
    graph.setTileMapping(allocTensor[irow], start_id + irow / rowsPerTile);
  }
  return allocTensor;
}

poplar::Tensor alloc_perneuron_2d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t batchsize = shape[0];
  size_t numNeurons = shape[1];
  size_t numTiles = graph.getTarget().getNumTiles();
  size_t neuronsPerTile = numNeurons / numTiles + 1;

  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
      graph.setTileMapping(allocTensor[ibatch][ineuron], start_id+ineuron/neuronsPerTile);
    }
  }
  return allocTensor;
}

poplar::Tensor alloc_perneuron_3d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t seq_len = shape[0];
  size_t batchsize = shape[1];
  size_t numNeurons = shape[2];
  size_t numTiles = graph.getTarget().getNumTiles();
  size_t neuronsPerTile = numNeurons / numTiles + 1;

  for (unsigned iseq = 0; iseq < seq_len; ++iseq) {
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
        graph.setTileMapping(allocTensor[iseq][ibatch][ineuron], start_id+ineuron/neuronsPerTile);
      }
    }
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
  
  poplar::DebugNameAndId dnai{debug_prefix};

  poplar::Tensor allocTensor;
  switch (operand) {
    case 0: allocTensor = alloc_rowwise_2d(graph, shape, type, 0, {dnai, "weights"}); 
            break;
    case 1: allocTensor = alloc_perneuron_2d(graph, shape, type, 0, {dnai, "init_state"});
            break;
    case 2: allocTensor = alloc_perneuron_3d(graph, shape, type, 0, {dnai, "inp_spike_ids"}); // TODO just do this for now...
            break;  
    case 3: allocTensor = alloc_perneuron_3d(graph, shape, type, 0, {dnai, "num_inp_spikes"}); // TODO just do this for now...
            break;
    case 4: allocTensor = alloc_perneuron_1d(graph, shape, type, 0, {dnai, "decay_constatns"});
            break;
    case 5: allocTensor = alloc_perneuron_1d(graph, shape, type, 0, {dnai, "thresholds"});
            break;
  }
  return allocTensor;
}


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  if (inputs.size() != 6) {
    throw poputil::poplibs_error("LIFSingleLayer requires 6 inputs");
  }

  poplar::DebugNameAndId dnai{debug_prefix};

  poplar::Tensor weights = inputs[0];
  poplar::Tensor init_state = inputs[1];
  poplar::Tensor inp_spike_ids = inputs[2];
  poplar::Tensor num_inp_spikes = inputs[3];
  poplar::Tensor decay_constatns = inputs[4];
  poplar::Tensor thresholds = inputs[5];
  
  if (weights.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be matrix (tensor of rank 2, (size_out, size_in)).");
  }

  if (init_state.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2, (batch_size, size_out)).");
  }

  if (inp_spike_ids.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be tensor of rank 3 (seq_dim, batch_size, inp_dim).");
  }

  if (num_inp_spikes.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[3]' must be tensor of rank 3 (seq_dim, batch_size, 1).");
  }

  if (decay_constatns.rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[4]' must be vector (size_out,).");
  }

  if (thresholds.rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[5]' must be vector (size_out,).");
  }

  size_t seq_len = inp_spike_ids.dim(0);
  size_t batchsize = inp_spike_ids.dim(1);
  size_t size_in = weights.dim(1);
  size_t size_sparse_in = inp_spike_ids.dim(2);
  size_t size_out = weights.dim(0);
  size_t size_sparse_out;
  sscanf(attributes.c_str(), "%zu", &size_sparse_out);
  auto dtype = weights.elementType();

  if (size_in < size_sparse_in) {
    throw poputil::poplibs_error("`inputs[0].shape[1]` (size_in) must be greater or equals `inp_spike_ids.dim(2)` (size_sparse_in).");
  }

  if (size_out < size_sparse_out) {
    throw poputil::poplibs_error("`inputs[0].shape[0]` (size_out) must be greater or equals `int(attributes)` (size_sparse_out).");
  }

  poplar::program::Sequence fwdProg;

  // // TODO Here it really only makes sense to place whole rows of the matrix 
  // // (and the corresponding vertex) on one tile and not use some already existing mapping.
  // // For that the whole LIF should be built as custom layer, not just the sparse op
  // auto tileMapping = graph.getTileMapping(inputs[0]);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // // Get the vector width of the particular data type, so that later we can
  // // divide the tensor up between workers in an appropriate way.
  // const auto vectorWidth = target.getVectorWidth(dtype);

  poplar::Tensor out_spike_ids = graph.addVariable(dtype, {seq_len, batchsize, size_sparse_out});
  poplar::Tensor num_out_spikes = graph.addVariable(dtype, {seq_len, batchsize, 1});
  // poplar::Tensor num_out_spikes = graph.clone(num_inp_spikes, {dnai, "Num Output Spikes"});
  outputs.push_back(out_spike_ids);
  outputs.push_back(num_out_spikes);
  
  // // TODO generate `out_spikes_ids` mapping to tiles
  // // TODO generate `num_out_spikes` mapping to tiles
  for (unsigned i = 0; i < out_spike_ids.dim(0); ++i) {
    graph.setTileMapping(out_spike_ids[i], i);
    graph.setTileMapping(num_out_spikes[i], i);
  }
  // TODO write functions for allocation of inout tensors (weights, state, input, ...)

  //-------------------------------------------- arguments to specify -------------------------------------------------
  // TODO arguments to specify:
  poplar::OptionFlags rnnOptions;
  const popnn::rnn::RnnParams rnnparams(dtype, batchsize, seq_len, {size_sparse_in, size_sparse_out});
  const LIFParams params(rnnparams, size_out);
  const LIFOpts opt(false, dtype);
  // validateParams(params);
  // auto opt = parseOptions(options);

  poplar::Tensor stateSeqOutput = graph.addVariable(dtype, {seq_len, batchsize, params.numNeurons}, {dnai, "stateOutput"});
  popnn::rnn::StateSequence stateSequence{stateSeqOutput, 0};
  // stateSequence = popnn::rnn::StateSequence{popnn::rnn::createOutputTensor(graph, params.rnn, numShards, {dnai, "stateOutput"}), 0};
  for (unsigned i = 0; i < stateSeqOutput.dim(0); ++i) {
    graph.setTileMapping(stateSeqOutput[i], i);
  }

  //------------------------------------------- loop -------------------------------------------------  
  auto numShards = getNumShards(graph, params, opt, {dnai, "numShards"});
  auto loopFwd = [&params, &weights, &decay_constatns, &thresholds, &opt](
                     poplar::Graph &graph, const popnn::rnn::TimeStepState &time,
                     const popnn::rnn::RnnBatchwiseFlags &batchwiseFlags,
                     std::vector<poplar::Tensor> &shardState, const popnn::rnn::RnnSlice &slice,
                     std::vector<poplar::Tensor> &created, poplar::program::Sequence *initProg,
                     const poplar::DebugNameAndId &dnai) {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};
    // debug_tensor(loop, "fwdLoop:", time.seqIdx); // TODO see gru.cpp how to get it to work?
    auto prevState = shardState[0].squeeze({0});
    BatchedSparseSpikes inpSpikes{slice.inputs[0], slice.inputs[1]};
    BatchedSparseSpikes outSpikes{slice.outputs[0].squeeze({0}), slice.outputs[1].squeeze({0})};

    poplar::Tensor newState{performLIFStepFworwardPass(
        graph, weights, prevState, inpSpikes, decay_constatns, thresholds, outSpikes, loop, opt, params, {dnai})};
    loop.add(poplar::program::Copy(newState, prevState, false, {dnai}));

    return loop;
  };
  
  std::vector<poplar::Tensor> rnnInputs{inp_spike_ids, num_inp_spikes};
  
  // TODO what was the puprose of this additional shardingLoop in the Gru example 
  // TODO and why is it not in the LSTM case?
  // const auto shardingLoop = std::bind(
  //     loopFwd, std::placeholders::_1, std::placeholders::_2,
  //     std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
  //     std::placeholders::_6, std::placeholders::_7, std::placeholders::_8);
  auto updatedState =
      popnn::rnn::Rnn(graph, params.rnn, false, {init_state.expand({0})}, stateSequence, rnnInputs,
               nullptr, nullptr, outputs, {}, fwdProg, loopFwd,
               numShards, rnnOptions, {dnai, "rnn"});

  outputs.push_back(stateSequence.output);

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

  poplar::Tensor weights = fwd_inputs[0]; // TODO not all weights are necessary every timestep, 
                                          // but only the columns that correspond to non-zero elements
                                          // therfore the ones which were used used during forward pass
  poplar::Tensor init_state = fwd_inputs[1];
  poplar::Tensor inp_spike_ids = fwd_inputs[2];
  poplar::Tensor num_inp_spikes = fwd_inputs[3];
  poplar::Tensor decay_constatns = fwd_inputs[4];
  poplar::Tensor thresholds = fwd_inputs[5];

  poplar::Tensor out_spike_ids = fwd_outputs[0];
  poplar::Tensor num_out_spikes = fwd_outputs[1];
  poplar::Tensor fwd_states_seq = fwd_outputs[2];

  size_t seq_len = inp_spike_ids.dim(0);
  size_t batchsize = inp_spike_ids.dim(1);
  size_t size_in = weights.dim(1);
  size_t size_sparse_in = inp_spike_ids.dim(2);
  size_t size_out = weights.dim(0);
  size_t size_sparse_out = out_spike_ids.dim(2);
  auto dtype = weights.elementType();

  poplar::Tensor dLdweights = graph.clone(weights, {dnai, "dLdweights"});
  popops::zero(graph, dLdweights, bwdProg, dnai);
  poplar::Tensor dLdinit_state = graph.clone(init_state, {dnai, "dLdinit_state"});
  poplar::Tensor dLdinp_spike_ids = graph.clone(inp_spike_ids, {dnai, "dLdinp_spike_ids"});
  poplar::Tensor dLdnum_inp_spikes = graph.clone(num_inp_spikes, {dnai, "dLdnum_inp_spikes"});
  poplar::Tensor dLddecay_constatns = graph.clone(decay_constatns, {dnai, "dLddecay_constatns"});
  poplar::Tensor dLdthresholds = graph.clone(thresholds, {dnai, "dLdthresholds"});

  poplar::Tensor dLdout_spike_ids = gradients[0];
  // poplar::Tensor dLdnum_out_spikes = gradients[1]; // not needed
  // poplar::Tensor dLdfwd_states_seq = gradients[2]; // Ignore this possibility for now. Essentially assume 0

  poplar::Tensor init_reverse_state = graph.clone(init_state, {dnai, "init_reverse_state"});
  // float one = 1.0;
  // popops::fill(graph, init_reverse_state, bwdProg, one, dnai); // !!! TODO !!! uncomment  
  popops::zero(graph, init_reverse_state, bwdProg, dnai); // set reverse init state to zero
  std::vector<poplar::Tensor> rnnInputs{inp_spike_ids, num_inp_spikes, out_spike_ids, num_out_spikes, fwd_states_seq, dLdout_spike_ids};

  poplar::OptionFlags rnnOptions;
  const popnn::rnn::RnnParams rnnparams(dtype, batchsize, seq_len, {size_sparse_in, size_sparse_out});
  const LIFParams params(rnnparams, size_out);
  const LIFOpts opt(false, dtype);
  popnn::rnn::StateSequence stateSequence; // don't store intermediate state derivative calculations


  // printVector(inp_spike_ids.shape());
  // printVector(num_inp_spikes.shape());
  // printVector(out_spike_ids.shape());
  // printVector(num_out_spikes.shape());
  // printVector(fwd_states_seq.shape());
  // printVector(dLdout_spike_ids.shape());


  //------------------------------------------- loop -------------------------------------------------  
  auto numShards = getNumShards(graph, params, opt, {dnai, "numShards"});
  auto loopBwd = [&params, &weights, &decay_constatns, &thresholds, &dLdweights, &opt](
                     poplar::Graph &graph, const popnn::rnn::TimeStepState &time,
                     const popnn::rnn::RnnBatchwiseFlags &batchwiseFlags,
                     std::vector<poplar::Tensor> &shardState, const popnn::rnn::RnnSlice &slice,
                     std::vector<poplar::Tensor> &created, poplar::program::Sequence *initProg,
                     const poplar::DebugNameAndId &dnai) {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};
    // debug_tensor(loop, "fwdLoop:", time.seqIdx); // TODO see gru.cpp how to get it to work?
    auto dLdnextState = shardState[0].squeeze({0});
    BatchedSparseSpikes fwdInpSpikes{slice.inputs[0], slice.inputs[1]};
    BatchedSparseSpikes fwdOutSpikes{slice.inputs[2], slice.inputs[3]};
    poplar::Tensor fwdState{slice.inputs[4]};
    poplar::Tensor dLdfwdOutSpikes{slice.inputs[5]};
    poplar::Tensor dLdInpSpikes{slice.outputs[0].squeeze({0})};

    // BatchedSparseSpikes fwdOutSpikes{slice.inputs[0].squeeze({0}), slice.inputs[1].squeeze({0})};

    performLIFStepBackwardPass(
        graph, weights, fwdState, fwdInpSpikes, decay_constatns, thresholds, fwdOutSpikes, dLdweights, dLdnextState, dLdfwdOutSpikes, dLdInpSpikes, loop, opt, params, {dnai});
    // poplar::Tensor dLdcurrentState{performLIFStepBackwardPass(
    //     graph, weights, fwdState, fwdInpSpikes, decay_constatns, thresholds, fwdOutSpikes, dLdweights, dLdnextState, dLdfwdOutSpikes, dLdInpSpikes, loop, opt, params, {dnai})};
    // loop.add(poplar::program::Copy(dLdcurrentState, dLdnextState, false, {dnai}));

    return loop;
  };

  auto dLdfirstState =
      popnn::rnn::Rnn(graph, params.rnn, true, {init_reverse_state.expand({0})}, stateSequence, rnnInputs,
               nullptr, nullptr, {dLdinp_spike_ids}, {}, bwdProg, loopBwd,
               numShards, rnnOptions, {dnai, "rnn"});

  outputs.push_back(dLdweights);
  outputs.push_back(dLdfirstState[0].squeeze({0})); // TODO change to dLdinit_state
  // outputs.push_back(dLdinit_state); // only placeholder for now, could easily be calculated from `updatedState` though
  outputs.push_back(dLdinp_spike_ids);
  outputs.push_back(dLdnum_inp_spikes); // placeholder
  outputs.push_back(dLddecay_constatns); // only placeholder for now
  outputs.push_back(dLdthresholds); // only placeholder for now

  // poplar::program::Execute(cs)
  
  return bwdProg;
}