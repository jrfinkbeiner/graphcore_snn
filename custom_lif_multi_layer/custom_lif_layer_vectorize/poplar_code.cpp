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
#include <popops/Loop.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Reduce.hpp>
// #include <popops/Operation.hpp>
#include <popops/Cast.hpp>


// #include "RnnUtil.hpp" // only for boost::optional



// TODO use dim and dtype info like `batchsize` or `dtype` from LIFParams 
// TODO instead of obtaining them from input arrays in every function ? 

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


//---------------------------------------------- forward -----------------------------------------

void performBatchedLIFStateUpdateInPlace(poplar::Graph &graph, poplar::Tensor &weights, 
                            poplar::Tensor &state, BatchedSparseSpikes &inp_spikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "performBatchedLIFStateUpdateInPlace"});

  size_t numTiles = graph.getTarget().getNumTiles();
  size_t numRows = weights.dim(0);
  size_t rowsPerTile = numRows / numTiles + (numRows % numTiles > 0); // integer ceil div ;
  size_t start_tile{1};

  auto dtype = weights.elementType();
  size_t batchsize = state.dim(0);
  for (unsigned irow = 0; irow < numRows; ++irow) {
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateUpdateInPlace", dtype),
                                {{"weights", weights[irow]},
                                {"state", state[ibatch][irow]},
                                {"inp_spikes_ids", inp_spikes.spike_ids[ibatch]},
                                {"num_inp_spikes", inp_spikes.num_spikes[ibatch][0]},
                                {"decay_constant", decay_constants[irow]},
                                {"threshold", thresholds[irow]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on weights mapping
      graph.setTileMapping(v, start_tile+irow/rowsPerTile);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }

  prog.add(poplar::program::Execute(cs));
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


void genBatchedLIFOutSpikes2Threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, BatchedSparseSpikes &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  auto cs = graph.addComputeSet({dnai, "genBatchedLIFOutSpikes2Threshs"});
  auto dtype = state.elementType();
  size_t batchsize = state.dim(0);
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes2Threshs", dtype),
                              {{"state", state[ibatch]},
                               {"thresholds", thresholds},
                               {"out_spikes_ids", out_spikes.spike_ids[ibatch]},
                               {"num_out_spikes", out_spikes.num_spikes[ibatch][0]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // most likely should be based on out_spikes mapping
    // graph.setTileMapping(v, (ibatch+1)*32);
    graph.setTileMapping(v, 1471-ibatch);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }

  prog.add(poplar::program::Execute(cs));
}     

void performLIFStepFworwardPassInPlace(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &state, BatchedSparseSpikes &inp_spikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &out_spikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  performBatchedLIFStateUpdateInPlace(graph, weights, state, inp_spikes, decay_constants, thresholds, prog, dnai);
  // genBatchedLIFOutSpikesTopK(graph, state, thresholds, out_spikes, prog, dnai);
  genBatchedLIFOutSpikes2Threshs(graph, state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikesOnlySpikes(graph, state, thresholds, out_spikes, prog, dnai);
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


// !!! TODO !!! rewrite to just apply operation where tensor elements are at
void mulInPlace_custom(poplar::Graph &graph, poplar::Tensor &tensor2d, poplar::Tensor &tensor1d, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "perf_mulInPlaceCustom"});
  size_t numRows = tensor2d.dim(1);
  auto dtype = tensor2d.elementType();

  size_t numTiles = graph.getTarget().getNumTiles();
  size_t rowsPerTile = numRows / numTiles + (numRows % numTiles > 0); // integer ceil div 
  size_t start_tile{1};

  for (unsigned irow = 0; irow < numRows; ++irow) {
    auto v = graph.addVertex(cs, poputil::templateVertex("MulInPlaceCustom", dtype),
                              {{"vec", tensor2d.dimShuffle({1,0})[irow]},
                               {"val", tensor1d[irow]}});
    graph.setTileMapping(v, start_tile+irow/rowsPerTile); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFStateGrad(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdState, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &fwdOutSpikes,
                            poplar::Tensor &dLdState, poplar::Tensor &dLdoutSpikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto dtype = weights.elementType();
  size_t batchsize = fwdState.dim(0);

  // popops::mulInPlace(graph, dLdState, decay_constants.expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);
  mulInPlace_custom(graph, dLdState, decay_constants, prog, dnai);

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
    // graph.setTileMapping(v, (ibatch+1)*32); 
    graph.setTileMapping(v, 1471-ibatch); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFWeightGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &decay_constants, poplar::Tensor &dLdState, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto dtype = dLdweights.elementType();
  size_t numRows = dLdweights.dim(0);
  size_t sparse_out_dim = fwdInpSpikes.spike_ids.dim(1);

  size_t numTiles = graph.getTarget().getNumTiles();
  size_t rowsPerTile = numRows / numTiles + (numRows % numTiles > 0); // integer ceil div 
  size_t start_tile{1};

  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});
  // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
  // TODO include batch-loop here when figured out how to be thread/parallel safe
  // parallelisms might intruduce probelms due to the += operation...
  for (unsigned irow = 0; irow < numRows; ++irow) {
    auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGrad", dtype),
                              {{"dLdState", dLdState.dimShuffle({1,0})[irow]},
                              {"decay_constant", decay_constants[irow]},
                              {"fwd_inp_spikes_ids", fwdInpSpikes.spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                              {"fwd_num_inp_spikes", fwdInpSpikes.num_spikes.dimShuffle({1,0})[0]},
                              {"sparse_out_dim", sparse_out_dim},
                              {"dLdweights_row", dLdweights[irow]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // should be based on state mapping
    graph.setTileMapping(v, start_tile+irow/rowsPerTile); 
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  prog.add(poplar::program::Execute(cs));
}


void selectLIFInpSpikeGrads(poplar::Graph &graph, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &dLdx, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
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



// !!! TODO !!! possibly rewrite this function to place neuronwise/rowwise on tiles instead of columnwise
void calcLIFInpSpikesGrad(poplar::Graph &graph, poplar::Tensor &weights, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &decay_constants,
                            poplar::Tensor &dLdState, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  // TODO IMPORTANT: For backwards bass, weight matrix schould be distributed column-wise to different tiles
  size_t num_cols = weights.dim(1);
  size_t batchsize = dLdState.dim(0);
  auto dtype = weights.elementType();

  size_t numTiles = graph.getTarget().getNumTiles();
  size_t colsPerTile = num_cols / numTiles + (num_cols % numTiles > 0); // integer ceil div 
  size_t start_tile{1};

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
      graph.setTileMapping(dLdx[ibatch][icol], start_tile+icol/colsPerTile);
      graph.setTileMapping(v, start_tile+icol/colsPerTile); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));

  selectLIFInpSpikeGrads(graph, fwdInpSpikes, dLdx, dLdInpSpikes, prog, dnai);
}



void calcLIFInpSpikesGradRowWise(poplar::Graph &graph, poplar::Tensor &weights, BatchedSparseSpikes &fwdInpSpikes, poplar::Tensor &decay_constants,
                            poplar::Tensor &dLdState, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  // TODO IMPORTANT: For backwards bass, weight matrix schould be distributed column-wise to different tiles
  size_t numRows = weights.dim(0);
  size_t batchsize = dLdState.dim(0);
  auto dtype = weights.elementType();

  size_t numTiles = graph.getTarget().getNumTiles();
  size_t rowsPerTile = numRows / numTiles + (numRows % numTiles > 0); // integer ceil div 
  size_t start_tile{1};

  size_t sparseSize = fwdInpSpikes.spike_ids.dim(1);
  poplar::Tensor dLdx = graph.addVariable(dtype, {numRows, batchsize, sparseSize});
  // poplar::Tensor relevantWeights = graph.addVariable(dtype, {numRows, batchsize, sparseSize});

  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGradRowWise"});
  // TODO include batch-loop here when figured out how to be thread/parallel safe
  // parallelisms might intruduce probelms due to the += operation...  
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned irow = 0; irow < numRows; ++irow) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFInpSpikesGradRowWise", dtype),
                                {{"weights_row", weights[irow]},
                                // {"relevant_weights", relevantWeights[irow][ibatch]},
                                {"dLdState", dLdState[ibatch][irow]},
                                {"decay_constant", decay_constants[irow]},
                                {"fwd_inp_spike_ids", fwdInpSpikes.spike_ids[ibatch]},
                                {"dLdinp_spike_ids", dLdx[irow][ibatch]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      graph.setTileMapping(dLdx[irow][ibatch], start_tile+irow/rowsPerTile);
      // graph.setTileMapping(relevantWeights[irow][ibatch], start_tile+irow/rowsPerTile);
      graph.setTileMapping(v, start_tile+irow/rowsPerTile); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));

  // TODO possibly scale here instead of in `LIFInpSpikesGradRowWise` verstex with (1-decay_constant)
  // std::string operation = "ADD";
  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
  reduceWithOutput(graph, dLdx, dLdInpSpikes, {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
}


void performLIFStepBackwardPass(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdState, BatchedSparseSpikes &fwdInpSpikes, 
                            poplar::Tensor &decay_constants, poplar::Tensor &thresholds, BatchedSparseSpikes &fwdOutSpikes,
                            poplar::Tensor &dLdweights, poplar::Tensor &dLdState, poplar::Tensor &dLdOutSpikes, poplar::Tensor &dLdInpSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  calcLIFStateGrad(graph, weights, fwdState, decay_constants, thresholds, fwdOutSpikes, dLdState, dLdOutSpikes, prog, dnai);
  calcLIFWeightGrad(graph, dLdweights, fwdInpSpikes, decay_constants, dLdState, prog, dnai);
  // calcLIFInpSpikesGrad(graph, weights, fwdInpSpikes, decay_constants, dLdState, dLdInpSpikes,  prog, dnai);
  calcLIFInpSpikesGradRowWise(graph, weights, fwdInpSpikes, decay_constants, dLdState, dLdInpSpikes,  prog, dnai);
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
  allocating_indices = {0, 1, 2, 3, 4, 5};
  is_elementwise = false;
  is_stateless = true;
  num_inputs = 6;
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

  // std::cout << "Build_allocator" << std::endl;
  size_t num_layers = inputs.size() / 6;

  // TODO improve allocation! and figure out ideal mapping right here!

  poplar::Tensor allocTensor;
  switch (operand/num_layers) {
    case 0: allocTensor = alloc_rowwise_2d(graph, shape, type, 1, {dnai, "weights"}); 
            break;
    case 1: allocTensor = alloc_perneuron_2d(graph, shape, type, 1, {dnai, "init_state"});
            break;
    case 2: allocTensor = alloc_perneuron_3d(graph, shape, type, 1, {dnai, "inp_spike_ids"}); // TODO just do this for now...
            break;  
    case 3: allocTensor = alloc_perneuron_3d(graph, shape, type, 1, {dnai, "num_inp_spikes"}); // TODO just do this for now...
            break;
    case 4: allocTensor = alloc_perneuron_1d(graph, shape, type, 1, {dnai, "decay_constatns"});
            break;
    case 5: allocTensor = alloc_perneuron_1d(graph, shape, type, 1, {dnai, "thresholds"});
            break;
  }
  return allocTensor;
}


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {


  if ((inputs.size() % 6) =! 0) {
    throw poputil::poplibs_error("LIFMultiLayer requires that the number of inputs is divisible by 6.");
  }
  size_t num_layers = inputs.size() / 6;


  poplar::DebugNameAndId dnai{debug_prefix};

  std::vector<poplar::Tensor> weights(inputs.begin(),inputs.begin()+num_layers);
  std::vector<poplar::Tensor> init_state(inputs.begin()+1*num_layers,inputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> inp_spike_ids_fptype(inputs.begin()+2*num_layers,inputs.begin()+3*num_layers);
  std::vector<poplar::Tensor> num_inp_spikes_int(inputs.begin()+3*num_layers,inputs.begin()+4*num_layers);
  std::vector<poplar::Tensor> decay_constatns(inputs.begin()+4*num_layers,inputs.begin()+5*num_layers);
  std::vector<poplar::Tensor> thresholds(inputs.begin()+5*num_layers,inputs.begin()+6*num_layers);
  
  // poplar::Tensor weights = alloc_rowwise_2d(graph, weights_inp.shape(), weights_inp.elementType(), 0, {dnai, "weights_alloc"});
  // poplar::Tensor init_state = alloc_perneuron_2d(graph, init_state_inp.shape(), init_state_inp.elementType(), 0, {dnai, "init_state_alloc"});
  // poplar::Tensor inp_spike_ids = alloc_perneuron_3d(graph, inp_spike_ids_inp.shape(), inp_spike_ids_inp.elementType(), 0, {dnai, "inp_spike_ids_alloc"});
  // poplar::Tensor num_inp_spikes = alloc_perneuron_3d(graph, num_inp_spikes_inp.shape(), num_inp_spikes_inp.elementType(), 0, {dnai, "num_inp_spikes_alloc"});
  // poplar::Tensor decay_constatns = alloc_perneuron_1d(graph, decay_constatns_inp.shape(), decay_constatns_inp.elementType(), 0, {dnai, "decay_constatns_alloc"});
  // poplar::Tensor thresholds = alloc_perneuron_1d(graph, thresholds_inp.shape(), thresholds_inp.elementType(), 0, {dnai, "thresholds_alloc"});


  if (weights.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[0]' must be matrix (tensor of rank 2, (size_out, size_in)).");
  }

  if (init_state.rank() != 2) {
    throw poputil::poplibs_error("Input 'inputs[1]' must be tensor of rank 2, (batch_size, size_out)).");
  }

  if (inp_spike_ids_fptype.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[2]' must be tensor of rank 3 (seq_dim, batch_size, inp_dim).");
  }

  if (num_inp_spikes_int.rank() != 3) {
    throw poputil::poplibs_error("Input 'inputs[3]' must be tensor of rank 3 (seq_dim, batch_size, 1).");
  }

  if (decay_constatns.rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[4]' must be vector (size_out,).");
  }

  if (thresholds.rank() != 1) {
    throw poputil::poplibs_error("Input 'inputs[5]' must be vector (size_out,).");
  }

  // size_t seq_len = inp_spike_ids_fptype.dim(0);
  // size_t batchsize = inp_spike_ids_fptype.dim(1);
  // size_t size_in = weights.dim(1);
  // size_t size_sparse_in = inp_spike_ids_fptype.dim(2);
  // size_t size_out = weights.dim(0);
  // size_t size_sparse_out;
  // sscanf(attributes.c_str(), "%zu", &size_sparse_out);
  // auto dtype = weights.elementType();

  size_t seq_len = inp_spike_ids_fptype.dim(0);
  size_t batchsize = inp_spike_ids_fptype.dim(1);
  size_t dense_sizes = weights.dim(1);
  size_t size_out = weights.dim(0);
  std::vector<size_t> sparse_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  auto dtype = weights.elementType();

  // if (size_in < size_sparse_in) {
  //   throw poputil::poplibs_error("`inputs[0].shape[1]` (size_in) must be greater or equals `inp_spike_ids.dim(2)` (size_sparse_in).");
  // }

  // if (size_out < size_sparse_out) {
  //   throw poputil::poplibs_error("`inputs[0].shape[0]` (size_out) must be greater or equals `int(attributes)` (size_sparse_out).");
  // }

  poplar::program::Sequence fwdProg;

  // fwdProg.add(poplar::program::Copy(weights_inp, weights, false, {dnai, "alloc copy weights"}));
  // fwdProg.add(poplar::program::Copy(init_state_inp, init_state, false, {dnai, "alloc copy init_state"}));
  // fwdProg.add(poplar::program::Copy(inp_spike_ids_inp, inp_spike_ids, false, {dnai, "alloc copy inp_spike_ids"}));
  // fwdProg.add(poplar::program::Copy(num_inp_spikes_inp, num_inp_spikes, false, {dnai, "alloc copy num_inp_spikes"}));
  // fwdProg.add(poplar::program::Copy(decay_constatns_inp, decay_constatns, false, {dnai, "alloc copy decay_constatns"}));
  // fwdProg.add(poplar::program::Copy(thresholds_inp, thresholds, false, {dnai, "alloc copy thresholds"}));

  // // TODO Here it really only makes sense to place whole rows of the matrix 
  // // (and the corresponding vertex) on one tile and not use some already existing mapping.
  // // For that the whole LIF should be built as custom layer, not just the sparse op
  // auto tileMapping = graph.getTileMapping(inputs[0]);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // // Get the vector width of the particular data type, so that later we can
  // // divide the tensor up between workers in an appropriate way.
  // const auto vectorWidth = target.getVectorWidth(dtype);

  //-------------------------------------------- arguments to specify -------------------------------------------------

  poplar::Tensor inp_spike_ids{popops::cast(graph, inp_spike_ids_fptype, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast inp_spike_ids"})};
  poplar::Tensor num_inp_spikes{popops::cast(graph, num_inp_spikes_int, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast num_inp_spikes"})};

  poplar::Tensor out_spike_ids = alloc_perneuron_3d(graph, {seq_len, batchsize, sparse_sizes.back()}, poplar::UNSIGNED_INT, 1, {dnai, "init_state"});
  poplar::Tensor num_out_spikes = alloc_perneuron_3d(graph, {seq_len, batchsize, 1}, poplar::UNSIGNED_INT, 1, {dnai, "init_state"});
  poplar::Tensor stateSeqOutput = alloc_perneuron_3d(graph, {seq_len, batchsize, size_out}, dtype, 1, {dnai, "init_state"});


  // //--------------------------------------- manuel  loop -------------------------------------------------  
  // auto loopFwd = [&weights, &decay_constatns, &thresholds](
  //                    poplar::Graph &graph,
  //                    BatchedSparseSpikes inpSpikes,
  //                    BatchedSparseSpikes outSpikes,
  //                    poplar::Tensor &prevState,
  //                    poplar::Tensor &thisState,
  //                    poplar::DebugNameAndId &dnai) {
    
  //   auto loop = poplar::program::Sequence{{}, {dnai}};

  //   performLIFStepFworwardPass(
  //       graph, weights, prevState, thisState, inpSpikes, decay_constatns, thresholds, outSpikes, loop, {dnai});
  //   return loop;
  // };
  
  // for (unsigned itime = 0; itime < seq_len; ++itime) {
  //   poplar::Tensor prevState;
  //   if (itime > 0) {
  //     prevState = stateSeqOutput[itime-1];
  //   } else {
  //     prevState = init_state;
  //   }
  //   poplar::Tensor thisState = stateSeqOutput[itime];
  //   BatchedSparseSpikes inp_spikes{inp_spike_ids[itime], num_inp_spikes[itime]};
  //   BatchedSparseSpikes out_spikes{out_spike_ids[itime], num_out_spikes[itime]};
  //   poplar::program::Sequence loopIter = loopFwd(graph, inp_spikes, out_spikes, prevState, thisState, dnai);
  //   fwdProg.add(loopIter);
  // }


  //----------------------------------------- REPEAT -------------------------------------------------  
  // TODO later maybe better clone and copy to not create unintended behaviour
  poplar::Tensor currentState = init_state;
  
  auto loopFwd = [&graph, &weights, &decay_constatns, &thresholds, &currentState, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &stateSeqOutput, &dnai] (
    poplar::Tensor itime
  ) {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};

    poplar::Tensor slicedInpSpikeIds = popops::dynamicSlice(graph, inp_spike_ids, itime, {0}, {1}, loop, {dnai, "slice inp_spike_ids"});
    // poplar::Tensor slicedOutSpikeIds = popops::dynamicSlice(graph, out_spike_ids, itime, {0}, {1}, loop, {dnai, "slice out_spike_ids"});
    poplar::Tensor slicedNumInpSpikes = popops::dynamicSlice(graph, num_inp_spikes, itime, {0}, {1}, loop, {dnai, "slice num_inp_spikes"});
    // poplar::Tensor slicedNumOutSpikes = popops::dynamicSlice(graph, num_out_spikes, itime, {0}, {1}, loop, {dnai, "slice num_out_spikes"});
    // TODO could just define outside loopFwd as for `currentState`
    poplar::Tensor slicedOutSpikeIds{alloc_perneuron_2d(graph, {out_spike_ids.dim(1), out_spike_ids.dim(2)}, out_spike_ids.elementType(), 0, {dnai, "slicedOutSpikeIds"})};
    poplar::Tensor slicedNumOutSpikes{alloc_perneuron_2d(graph, {num_out_spikes.dim(1), num_out_spikes.dim(2)}, num_out_spikes.elementType(), 0, {dnai, "slicedNumOutSpikes"})};

    BatchedSparseSpikes inpSpikes{slicedInpSpikeIds[0], slicedNumInpSpikes[0]};
    BatchedSparseSpikes outSpikes{slicedOutSpikeIds, slicedNumOutSpikes};
    // poplar::program::Sequence loopIter = loopFwd(graph, inp_spikes, out_spikes, prevState, thisState, dnai);

    performLIFStepFworwardPassInPlace(
        graph, weights, currentState, inpSpikes, decay_constatns, thresholds, outSpikes, loop, {dnai});
    // to record state sequence
    // loop.add(poplar::program::Copy(currentState, thisState, false, {dnai, "copy state"}));
    popops::dynamicUpdate(graph, stateSeqOutput, currentState.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic stateSeqOutput update"});
    popops::dynamicUpdate(graph, out_spike_ids, outSpikes.spike_ids.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic out_spike_ids update"});
    popops::dynamicUpdate(graph, num_out_spikes, outSpikes.num_spikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic num_out_spikes update"});
    return loop;
  };

  poplar::program::Sequence cloop = popops::countedLoop(graph, seq_len, loopFwd, {dnai, "countedLoop"});
  fwdProg.add(cloop);

  poplar::Tensor out_spike_ids_fptype{popops::cast(graph, out_spike_ids, weights.elementType(), fwdProg, {dnai, "cast out_spike_ids"})};
  poplar::Tensor num_out_spikes_int{popops::cast(graph, num_out_spikes, weights.elementType(), fwdProg, {dnai, "cast num_out_spikes"})};

  outputs.push_back(out_spike_ids_fptype);
  outputs.push_back(num_out_spikes_int);
  outputs.push_back(stateSeqOutput);

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

// TODO implement version for only weight gradient  (for first layer of a network)
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
  poplar::Tensor inp_spike_ids_fptype = fwd_inputs[2];
  poplar::Tensor num_inp_spikes_int = fwd_inputs[3];
  poplar::Tensor decay_constatns = fwd_inputs[4];
  poplar::Tensor thresholds = fwd_inputs[5];

  poplar::Tensor out_spike_ids_fptype = fwd_outputs[0];
  poplar::Tensor num_out_spikes_int = fwd_outputs[1];
  poplar::Tensor fwd_states_seq = fwd_outputs[2];

  size_t seq_len = inp_spike_ids_fptype.dim(0);
  size_t batchsize = inp_spike_ids_fptype.dim(1);
  size_t size_in = weights.dim(1);
  size_t size_sparse_in = inp_spike_ids_fptype.dim(2);
  size_t size_out = weights.dim(0);
  size_t size_sparse_out = out_spike_ids_fptype.dim(2);
  auto dtype = weights.elementType();

  poplar::Tensor dLdweights = graph.clone(weights, {dnai, "dLdweights"});
  popops::zero(graph, dLdweights, bwdProg, dnai);
  poplar::Tensor dLdinit_state = graph.clone(init_state, {dnai, "dLdinit_state"});
  poplar::Tensor dLdinp_spike_ids = graph.clone(inp_spike_ids_fptype, {dnai, "dLdinp_spike_ids"}); // how to  set mapping in Reduce operation
  // poplar::Tensor dLdinp_spike_ids = graph.addVariable(inp_spike_ids.elementType(), {inp_spike_ids.dim(0),inp_spike_ids.dim(1), inp_spike_ids.dim(2)}, {dnai, "dLdinp_spike_ids"}); // how to  set mapping in Reduce operation
  poplar::Tensor dLdnum_inp_spikes = graph.clone(num_inp_spikes_int, {dnai, "dLdnum_inp_spikes"});
  poplar::Tensor dLddecay_constatns = graph.clone(decay_constatns, {dnai, "dLddecay_constatns"});
  poplar::Tensor dLdthresholds = graph.clone(thresholds, {dnai, "dLdthresholds"});

  poplar::Tensor dLdout_spike_ids = gradients[0];
  // poplar::Tensor dLdnum_out_spikes = gradients[1]; // not needed
  // poplar::Tensor dLdfwd_states_seq = gradients[2]; // Ignore this possibility for now. Essentially assume 0

  poplar::Tensor init_reverse_state = graph.clone(init_state, {dnai, "init_reverse_state"});
  // float one = 1.0;
  // popops::fill(graph, init_reverse_state, bwdProg, one, dnai); // !!! TODO !!! uncomment  
  popops::zero(graph, init_reverse_state, bwdProg, dnai); // set reverse init state to zero


  poplar::Tensor inp_spike_ids{popops::cast(graph, inp_spike_ids_fptype, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast inp_spike_ids"})};
  poplar::Tensor num_inp_spikes{popops::cast(graph, num_inp_spikes_int, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast num_inp_spikes"})};
  poplar::Tensor out_spike_ids{popops::cast(graph, out_spike_ids_fptype, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast out_spike_ids"})};
  poplar::Tensor num_out_spikes{popops::cast(graph, num_out_spikes_int, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast num_out_spikes"})};

  //------------------------------------------- Repeat -------------------------------------------------  
  
  poplar::Tensor dLdstate = init_reverse_state;
  poplar::Tensor SEQ_LEN = graph.addConstant(poplar::UNSIGNED_INT, {1}, seq_len, {dnai, "step"});
  poplar::Tensor itime = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dnai, "itime"});
  poplar::Tensor step = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {dnai, "step"});
  graph.setTileMapping(itime, 0);
  graph.setTileMapping(SEQ_LEN, graph.getTileMapping(itime));
  graph.setTileMapping(step, graph.getTileMapping(itime));
  bwdProg.add(poplar::program::Copy(SEQ_LEN, itime, false, dnai));

  auto loopBwd = [&graph, &weights, &decay_constatns, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
                  &dLdweights, &dLdinp_spike_ids, &dLdnum_inp_spikes, &dLdout_spike_ids, &dLdstate, &itime, &step, &dnai] () {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};
    
    popops::subInPlace(graph, itime, step, loop, dnai);

    poplar::Tensor slicedInpSpikeIds = popops::dynamicSlice(graph, inp_spike_ids, itime, {0}, {1}, loop, {dnai, "slice inp_spike_ids"})[0];
    poplar::Tensor slicedOutSpikeIds = popops::dynamicSlice(graph, out_spike_ids, itime, {0}, {1}, loop, {dnai, "slice out_spike_ids"})[0];
    poplar::Tensor slicedNumInpSpikes = popops::dynamicSlice(graph, num_inp_spikes, itime, {0}, {1}, loop, {dnai, "slice num_inp_spikes"})[0];
    poplar::Tensor slicedNumOutSpikes = popops::dynamicSlice(graph, num_out_spikes, itime, {0}, {1}, loop, {dnai, "slice num_out_spikes"})[0];
    poplar::Tensor slicedDldOutSPikeIds = popops::dynamicSlice(graph, dLdout_spike_ids, itime, {0}, {1}, loop, {dnai, "slice dLdout_spike_ids"})[0];
    poplar::Tensor slicedFwdState = popops::dynamicSlice(graph, fwd_states_seq, itime, {0}, {1}, loop, {dnai, "slice fwd_states_seq"})[0];

    BatchedSparseSpikes fwdInpSpikes{slicedInpSpikeIds, slicedNumInpSpikes};
    BatchedSparseSpikes fwdOutSpikes{slicedOutSpikeIds, slicedNumOutSpikes};

    poplar::Tensor slicedDLdfwdOutSpikes = popops::dynamicSlice(graph, dLdout_spike_ids, itime, {0}, {1}, loop, {dnai, "slice dLdout_spike_ids"});
    // poplar::Tensor slicedDLdInpSpikes = popops::dynamicSlice(graph, dLdinp_spike_ids, itime, {0}, {1}, loop, {dnai, "slice dLdinp_spike_ids"});
    // how to  set mapping in Reduce operation
    poplar::Tensor slicedDLdInpSpikes{alloc_perneuron_2d(graph, {dLdinp_spike_ids.dim(1), dLdinp_spike_ids.dim(2)}, dLdinp_spike_ids.elementType(), 1, {dnai, "slicedDLdInpSpikes"})};

    performLIFStepBackwardPass(
        graph, weights, slicedFwdState, fwdInpSpikes, decay_constatns, thresholds, fwdOutSpikes, dLdweights, dLdstate, slicedDldOutSPikeIds, slicedDLdInpSpikes, loop, {dnai});

    popops::dynamicUpdate(graph, dLdinp_spike_ids, slicedDLdInpSpikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic dLdInpSpikes update"});

    return loop;
  };

  // for (unsigned itime = 0; itime < seq_len; ++itime) {
  //   BatchedSparseSpikes fwdInpSpikes{inp_spike_ids[seq_len-itime-1], num_inp_spikes[seq_len-itime-1]};
  //   BatchedSparseSpikes fwdOutSpikes{out_spike_ids[seq_len-itime-1], num_out_spikes[seq_len-itime-1]};
  //   poplar::Tensor fwdState = fwd_states_seq[seq_len-itime-1];
  //   poplar::Tensor dLdfwdOutSpikes = dLdout_spike_ids[seq_len-itime-1];
  //   poplar::Tensor dLdInpSpikes = dLdinp_spike_ids[seq_len-itime-1];
  //   poplar::program::Sequence loopIter = loopBwd(graph, fwdInpSpikes, fwdOutSpikes, fwdState, init_reverse_state, dLdfwdOutSpikes, dLdInpSpikes, dnai);
  //   bwdProg.add(loopIter);
  // }
  poplar::program::Sequence bodyProg = {loopBwd()};
  auto repeat = poplar::program::Repeat(seq_len, bodyProg, {dnai, "repeat"});
  bwdProg.add(repeat);


  outputs.push_back(dLdweights);
  outputs.push_back(dLdinit_state); // only placeholder for now, could easily be calculated from `updatedState` though
  outputs.push_back(dLdinp_spike_ids);
  outputs.push_back(dLdnum_inp_spikes); // placeholder
  outputs.push_back(dLddecay_constatns); // only placeholder for now
  outputs.push_back(dLdthresholds); // only placeholder for now

  // poplar::program::Execute(cs)
  
  return bwdProg;
}
