#include <iostream>
#include <algorithm>
#include <vector>
// #include <boost/optional.hpp>
#include <cmath> // ceil
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TargetType.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
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

#include <poplar/StringRef.hpp>



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

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
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

void clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &dnai](const poplar::Tensor &t){return graph.clone(t, dnai);});
}

std::vector<poplar::Tensor> clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  clone_tensor_vector(graph, src, dst, 0, dnai);
  return dst;
}

std::vector<poplar::Tensor> clone_tensor_vector(poplar::Graph& graph, const poplar::Type &type, const std::vector<poplar::Tensor> &src, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &type, &dnai](const poplar::Tensor &t){return graph.clone(type, t, dnai);});
  return dst;
}

std::vector<poplar::Tensor> cast_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, poplar::Type &dtype, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  // std::vector<poplar::Tensor> dst;
  // std::transform(src.begin(), src.end(), dst.begin(), [&graph, &dtype, &prog,  &dnai](const poplar::Tensor &t) -> poplar::Tensor {return popops::cast(graph, t, dtype, prog, dnai);});
  std::vector<poplar::Tensor> dst = clone_tensor_vector(graph, dtype, src, 0);
  for (unsigned i=0; i<src.size(); ++i){
    prog.add(popops::cast(graph, src[i], dst[i], dnai));
  }
  return dst;
}

void zero_tensor_vector(poplar::Graph& graph, std::vector<poplar::Tensor> &vec, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::for_each(vec.begin(), vec.end(), [&graph, &prog, &dnai](poplar::Tensor &t){popops::zero(graph, t, prog, dnai);});
}

void extend_tensor_vector(std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst){
  std::transform(src.begin(), src.end(), std::back_inserter(dst), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
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

// struct LIFOpts {
//   bool inferenceOnly;
//    poplar::Type partialsType;
//    boost::optional<double> availableMemoryProportion;
//    boost::optional<std::size_t> numShards;
//    boost::optional<bool> rnnCodeReuse;

//    LIFOpts();
//    LIFOpts(bool inferenceOnly, poplar::Type &partialsType)
//      : inferenceOnly{inferenceOnly}
//      , partialsType{partialsType}
//      {};
//  };


//---------------------------------------------- forward -----------------------------------------

void performBatchedLIFStateUpdateInPlace(poplar::Graph &graph, std::vector<poplar::Tensor> &weights, 
                            std::vector<poplar::Tensor> &state, std::vector<BatchedSparseSpikes> &inp_spikes, 
                            std::vector<poplar::Tensor> &decay_constants, std::vector<poplar::Tensor> &oneMinus_decay_constants, 
                            std::vector<poplar::Tensor> &thresholds,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "performBatchedLIFStateUpdateInPlace"});
  poplar::TargetType target_type = graph.getTarget().getTargetType();
  const auto numTiles = graph.getTarget().getNumTiles();
  size_t num_layers = weights.size();


  std::vector<poplar::Tensor> syn_input;
  if (target_type != poplar::TargetType::IPU) {
    syn_input = clone_tensor_vector(graph, state, {dnai, "syn_input"});
    zero_tensor_vector(graph, syn_input, prog, {dnai, "zero_syn_input"});
  }

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = weights[ilay].elementType();
    size_t batchsize = state[ilay].dim(0);

    auto neuronTileMapping = graph.getTileMapping(weights[ilay][0], true);

    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();
        // poplar::Tensor neuronWeights = weights[ilay].slice(neuronRange, 1).dimRoll(1, 0); // TODO does this create new tensors ?
        poplar::Tensor neuronStates = state[ilay].slice(neuronRange, 1);
        // std::cout << "neuronStates.shapeToString(): " << neuronStates.shapeToString() << std::endl;
        // std::cout << "state[ilay].shapeToString(): " << state[ilay].shapeToString() << std::endl;
        // std::cout << "neuronRange: " << neuronRange.lower() << ", " << neuronRange.upper() << std::endl; 
        poplar::Tensor neuronDecay_constants = decay_constants[ilay].slice(neuronRange);
        poplar::Tensor neuronOneMinus_decay_constants = oneMinus_decay_constants[ilay].slice(neuronRange);
        poplar::Tensor neuronThresholds = thresholds[ilay].slice(neuronRange);

        // // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
        // // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
        // for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        //   for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        //     auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateUpdateInPlace", dtype),
        //                               // {{"weights", weights[ilay][neuronId]},
        //                               {{"weights", neuronWeights[ineuron]},
        //                               {"state", neuronStates[ibatch][ineuron]},
        //                               {"inp_spikes_ids", inp_spikes[ilay].spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
        //                               {"num_inp_spikes", inp_spikes[ilay].num_spikes[ibatch][0]},
        //                               {"decay_constant", neuronDecay_constants[ineuron]},
        //                               {"decay_constant", neuronOneMinus_decay_constants[ineuron]},
        //                               {"threshold", neuronThresholds[ineuron]}});
        //     // !!! TODO !!! totally bogus tile mapping, must be improved
        //     // should be based on weights mapping
        //     graph.setTileMapping(v, tile);
        //     // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        //     graph.setPerfEstimate(v, 1);
        //   }
        // }

        poplar::Tensor neuronWeights = weights[ilay].slice(neuronRange, 1);
        poplar::Tensor neuronSyn_input;
        if (target_type != poplar::TargetType::IPU) {
          neuronSyn_input = syn_input[ilay].slice(neuronRange, 1);
        }
        

        const auto num_neurons = neuronWeights.dim(1);
        std::string vertexType;
        if (target_type == poplar::TargetType::IPU) {
          if ((num_neurons == 2) && (dtype == poplar::FLOAT)) {
            vertexType = "LIFStateUpdateInPlaceTwoNeuronSIMD";
          } else if (dtype == poplar::FLOAT) {
            // // if ((num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
            //   if (dtype == poplar::FLOAT) {
            vertexType = "LIFStateUpdateInPlaceMultiNeuronSIMD";
          }
          // could vectorize batches for multiple of 6
          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, vertexType,
                                      // {{"weights", weights[ilay][neuronId]},
                                      {{"weights", neuronWeights.flatten()},
                                      {"state", neuronStates[ibatch]},
                                      // {"syn_input", neuronSyn_input[ibatch]}, // TODO necessary for LIFStateUpdateInPlaceMultiNeuron
                                      {"inp_spikes_ids", inp_spikes[ilay].spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                      {"num_inp_spikes", inp_spikes[ilay].num_spikes[ibatch][0]},
                                      {"decay_constant", neuronDecay_constants},
                                      {"oneMinus_decay_constant", neuronOneMinus_decay_constants},
                                      {"threshold", neuronThresholds},
                                      {"num_neurons", neuronWeights.dim(1)}});
            // !!! TODO !!! totally bogus tile mapping, must be improved
            // should be based on weights mapping
            graph.setTileMapping(v, tile);
            // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
            graph.setPerfEstimate(v, 1);
          }
        } else {
          vertexType = poputil::templateVertex("LIFStateUpdateInPlaceMultiNeuron", dtype);

          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, vertexType,
                                      // {{"weights", weights[ilay][neuronId]},
                                      {{"weights", neuronWeights.flatten()},
                                      {"state", neuronStates[ibatch]},
                                      {"syn_input", neuronSyn_input[ibatch]}, // TODO necessary for LIFStateUpdateInPlaceMultiNeuron
                                      {"inp_spikes_ids", inp_spikes[ilay].spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                      {"num_inp_spikes", inp_spikes[ilay].num_spikes[ibatch][0]},
                                      {"decay_constant", neuronDecay_constants},
                                      {"oneMinus_decay_constant", neuronOneMinus_decay_constants},
                                      {"threshold", neuronThresholds},
                                      {"num_neurons", neuronWeights.dim(1)}});
          graph.setTileMapping(v, tile);
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
          }
        }
      }
    }
  }
  std::cout << "DONE" << std::endl;
  prog.add(poplar::program::Execute(cs));
}


void genBatchedLIFOutSpikesTopK(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
              std::vector<BatchedSparseSpikes> &out_spikes, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  // popops::SortOrder sortOrder = None;
  // popops::SortOrder sortOrder = popops::SortOrder::NONE;
  size_t num_layers = state.size();

  std::vector<poplar::Tensor> topKStateVals;
  std::vector<poplar::Tensor> topKStateIds;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto numSparseOutSpikes = out_spikes[ilay].spike_ids.dim(1);
    // popops::TopKParams topKparams(numSparseOutSpikes, true, popops::SortOrder::DESCENDING);
    popops::TopKParams topKparams(numSparseOutSpikes, true, popops::SortOrder::NONE);

    std::pair<poplar::Tensor, poplar::Tensor> topKStatesPair{popops::topKWithPermutation(graph, prog, state[ilay], topKparams, dnai)};
    topKStateVals.push_back(topKStatesPair.first);
    topKStateIds.push_back(topKStatesPair.second);
  }

  auto cs = graph.addComputeSet({dnai, "genBatchedLIFOutSpikesFromTopK"});
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = state[ilay].elementType();
    size_t batchsize = state[ilay].dim(0);
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikesFromTopK", dtype),
                                {{"topKStateVals", topKStateVals[ilay][ibatch]},
                                {"topKStateIds", topKStateIds[ilay][ibatch]},
                                {"thresholds", thresholds[ilay]},
                                {"out_spikes_ids", out_spikes[ilay].spike_ids[ibatch]},
                                {"num_out_spikes", out_spikes[ilay].num_spikes[ibatch][0]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // most likely should be based on out_spikes mapping
      graph.setTileMapping(v, 1471-ibatch-batchsize*ilay);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));
}


// TODO !!! think about tile mapping !!!
void genBatchedLIFOutSpikes2Threshs(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  auto cs = graph.addComputeSet({dnai, "genBatchedLIFOutSpikes2Threshs"});
  size_t num_layers = state.size();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = state[ilay].elementType();
    size_t batchsize = state[ilay].dim(0);
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes2Threshs", dtype),
                                {{"state", state[ilay][ibatch]},
                                {"thresholds", thresholds[ilay]},
                                {"out_spikes_ids", out_spikes[ilay].spike_ids[ibatch]},
                                {"num_out_spikes", out_spikes[ilay].num_spikes[ibatch][0]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // most likely should be based on out_spikes mapping
      // graph.setTileMapping(v, (ibatch+1)*32);
      graph.setTileMapping(v, 1471-ibatch-batchsize*ilay);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));
}     

// TODO !!! think about tile mapping !!!
void genBatchedLIFOutSpikes2ThreshsMutliWorker(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  auto cs = graph.addComputeSet({dnai, "LIFOutSpikes2ThreshsMultiVertex"});
  const size_t num_layers = state.size();
  // std::vector<unsigned> indices;
  // for( unsigned i = 0; i < numWorkers; ++i ) indices.push_back( i );
  // printVector(indices);

  std::vector<poplar::Tensor> repeated_out_spikes_ids;
  std::vector<poplar::Tensor> repeated_num_out_spikes;


  for (unsigned ilay=0; ilay<num_layers; ++ilay){

    auto dtype = state[ilay].elementType();
    const size_t batchsize = state[ilay].dim(0);
    const size_t sparse_size = out_spikes[ilay].spike_ids.dim(1);

    // std::cout << "ilay: " << ilay << std::endl;
    // std::cout << "sparse_size: " << sparse_size << std::endl;
    const size_t denseSpraseRatio = state[ilay].dim(1) / sparse_size;
    const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();; // TODO get this from poplar ?
    const size_t numWorkers = std::min(denseSpraseRatio, numPossibleParallelThreads); // TODO way to get this from poplar?
    // // const size_t numWorkers = 1;
    // std::cout << "numWorkers: " << numWorkers << std::endl;

    repeated_out_spikes_ids.push_back(graph.addVariable(out_spikes[ilay].spike_ids.elementType(), {batchsize, numWorkers*sparse_size}));
    repeated_num_out_spikes.push_back(graph.addVariable(out_spikes[ilay].num_spikes.elementType(), {batchsize, numWorkers}));

    // for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    //   auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes2ThreshsMultiVertex", dtype),
    //                             {{"state", state[ilay][ibatch]},
    //                             {"thresholds", thresholds[ilay]},
    //                             {"sizeSparseOut", sparse_size},
    //                             {"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch]},
    //                             {"repeated_num_out_spikes", repeated_num_out_spikes[ilay][ibatch]}});
    //   // !!! TODO !!! totally bogus tile mapping, must be improved
    //   // most likely should be based on out_spikes mapping
    //   // graph.setTileMapping(v, (ibatch+1)*32);
    //   size_t tile{1471-ibatch-batchsize*ilay};
    //   graph.setTileMapping(repeated_out_spikes_ids[ilay][ibatch], tile);
    //   graph.setTileMapping(repeated_num_out_spikes[ilay][ibatch], tile);
    //   graph.setTileMapping(v, tile);
    //   // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    //   graph.setPerfEstimate(v, 1);
    // }



    size_t worker_start{0};
    size_t worker_end{0};
    for (unsigned iwor = 0; iwor < numWorkers; ++iwor) {
      size_t numStatesThisWorker = state[ilay].dim(1) / numWorkers + ((state[ilay].dim(1) % numWorkers) > iwor);
      worker_end += numStatesThisWorker;
      // std::cout << "state[ilay].dim(1): "<< state[ilay].dim(1) << std::endl;
      // std::cout << "worker_start: "<< worker_start << std::endl;
      // std::cout << "worker_end: "<< worker_end << std::endl;
      // std::cout << "numStatesThisWorker: "<< numStatesThisWorker << std::endl;

      auto state_worker = state[ilay].slice(worker_start, worker_end, 1);
      auto thresholds_worker = thresholds[ilay].slice(worker_start, worker_end, 0);
      auto out_spike_ids_worker = repeated_out_spikes_ids[ilay].slice(iwor*sparse_size, (iwor+1)*sparse_size, 1);

      // printVector(state_worker.shape());
      // printVector(thresholds_worker.shape());
      // printVector(out_spike_ids_worker.shape());

      for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes2ThreshsSplitWorker", dtype),
                                  {{"state", state_worker[ibatch]},
                                  {"thresholds", thresholds_worker},
                                  {"start_id", worker_start},
                                  {"repeated_out_spikes_ids", out_spike_ids_worker[ibatch]},
                                  {"repeated_num_out_spikes", repeated_num_out_spikes[ilay][ibatch][iwor]}});

        size_t tile{1471-ibatch-batchsize*ilay};
        graph.setTileMapping(repeated_out_spikes_ids[ilay][ibatch], tile);
        graph.setTileMapping(repeated_num_out_spikes[ilay][ibatch], tile);
        graph.setTileMapping(v, tile);
        // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        graph.setPerfEstimate(v, 1);
      }
      worker_start = worker_end;
    }
  }

  prog.add(poplar::program::Execute(cs));

  auto cs2 = graph.addComputeSet({dnai, "LIFOutSpikes2ThreshsCombine"});
  for (unsigned ilay=0; ilay<num_layers; ++ilay){

    // // popops::fill(poplar::Graph &graph, const poplar::Tensor &t, poplar::program::Sequence &prog, FillValueType fillValue)
    // popops::fill(graph, repeated_num_out_spikes[ilay], prog, 1);


    auto dtype = state[ilay].elementType();
    const size_t batchsize = state[ilay].dim(0);

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs2, poputil::templateVertex("LIFOutSpikes2ThreshsCombine", dtype),
                                // {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch]},
                                {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch]},
                                {"repeated_num_out_spikes", repeated_num_out_spikes[ilay][ibatch]},
                                {"out_spikes_ids", out_spikes[ilay].spike_ids[ibatch]},
                                {"num_out_spikes", out_spikes[ilay].num_spikes[ibatch][0]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // most likely should be based on out_spikes mapping
      // graph.setTileMapping(v, (ibatch+1)*32);
      size_t tile{1471-ibatch-batchsize*ilay};
      graph.setTileMapping(v, tile);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs2));
} 

void performLIFStepFworwardPassInPlace(poplar::Graph &graph, std::vector<poplar::Tensor> &weights, std::vector<poplar::Tensor> &state, std::vector<BatchedSparseSpikes> &inp_spikes, 
                            std::vector<poplar::Tensor> &decay_constants, std::vector<poplar::Tensor> &oneMinus_decay_constants, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  performBatchedLIFStateUpdateInPlace(graph, weights, state, inp_spikes, decay_constants, oneMinus_decay_constants, thresholds, prog, dnai);
  // genBatchedLIFOutSpikesTopK(graph, state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikes2Threshs(graph, state, thresholds, out_spikes, prog, dnai);
  genBatchedLIFOutSpikes2ThreshsMutliWorker(graph, state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikesOnlySpikes(graph, state, thresholds, out_spikes, prog, dnai);
}



//---------------------------------------------- backward -----------------------------------------

// // !!! TODO !!! rewrite to just apply operation where tensor elements are at
// void mulInPlace_custom(poplar::Graph &graph, poplar::Tensor &tensor2d, poplar::Tensor &tensor1d, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
//   auto cs = graph.addComputeSet({dnai, "perf_mulInPlaceCustom"});
//   size_t numRows = tensor2d.dim(1);
//   auto dtype = tensor2d.elementType();

//   size_t numTiles = graph.getTarget().getNumTiles();
//   size_t rowsPerTile = numRows / numTiles + (numRows % numTiles > 0); // integer ceil div 
//   size_t start_tile{1};

//   for (unsigned irow = 0; irow < numRows; ++irow) {
//     auto v = graph.addVertex(cs, poputil::templateVertex("MulInPlaceCustom", dtype),
//                               {{"vec", tensor2d.dimShuffle({1,0})[irow]},
//                                {"val", tensor1d[irow]}});
//     graph.setTileMapping(v, start_tile+irow/rowsPerTile); 
//     // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
//     graph.setPerfEstimate(v, 1);
//   }
//   prog.add(poplar::program::Execute(cs));
// }



// !!! TODO !!! think about tile mapping !!! 
// !!! TODO !!! maybe rewrite function to local version. every state is conditionally updated ?
void calcLIFStateGrad(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, const std::vector<poplar::Tensor> &fwdState, 
                            const std::vector<poplar::Tensor> &decay_constants, const std::vector<poplar::Tensor> &thresholds, const std::vector<BatchedSparseSpikes> &fwdOutSpikes,
                            std::vector<poplar::Tensor> &dLdState, const std::vector<poplar::Tensor> &dLdoutSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "calcLIFStateOutGrad"});
  size_t num_layers = weights.size();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = weights[ilay].elementType();
    size_t batchsize = fwdState[ilay].dim(0);

    popops::mulInPlace(graph, dLdState[ilay], decay_constants[ilay].expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);
    // mulInPlace_custom(graph, dLdState[ilay], decay_constants[ilay], prog, dnai);

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateOutGrad", dtype),
                                {{"fwdState", fwdState[ilay][ibatch]},
                                {"thresholds", thresholds[ilay]},
                                {"dLdoutSpikes", dLdoutSpikes[ilay][ibatch]},
                                {"fwd_out_spikes_ids", fwdOutSpikes[ilay].spike_ids[ibatch]},
                                //  {"dLdState_inp", dLdState[ibatch]},
                                //  {"fwd_num_out_spikes", fwdOutSpikes.num_spikes[ibatch][0]},
                                //  {"dLdState", dLdState[ibatch]}});
                                {"dLdState", dLdState[ilay][ibatch]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      // graph.setTileMapping(v, (ibatch+1)*32); 
      graph.setTileMapping(v, 1471-ibatch-batchsize*ilay); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFWeightGrad_singleThread(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                        const std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  
  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});
  const size_t num_layers = dLdweights.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = dLdweights[ilay].elementType();
    size_t sparse_out_dim = fwdInpSpikes[ilay].spike_ids.dim(1);
    size_t batchsize = fwdInpSpikes[ilay].spike_ids.dim(0);

    auto neuronTileMapping = graph.getTileMapping(dLdweights[ilay][0], true);

    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();

        // poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 1).dimRoll(1, 0); // TODO does this create new tensors ?
        // poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);

        // std::cout << "neuronDLdWeights.isContiguous(): " << neuronDLdWeights.isContiguous() << std::endl;
        // std::cout << "neuronDLdState.isContiguous(): " << neuronDLdState.isContiguous() << std::endl;

        // // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
        // // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
        // // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
        // // TODO include batch-loop here when figured out how to be thread/parallel safe
        // // parallelisms might intruduce probelms due to the += operation...
        // for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        //   auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGrad", dtype),
        //                             {{"dLdState", neuronDLdState.dimRoll(1, 0)[ineuron]},
        //                             {"fwd_inp_spikes_ids", fwdInpSpikes[ilay].spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
        //                             {"fwd_num_inp_spikes", fwdInpSpikes[ilay].num_spikes.dimRoll(1, 0)[0]},
        //                             {"sparse_out_dim", sparse_out_dim},
        //                             {"dLdweights_row", neuronDLdWeights[ineuron]}});
        //   // !!! TODO !!! totally bogus tile mapping, must be improved
        //   // should be based on state mapping
        //   graph.setTileMapping(v, tile); 
        //   // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        //   graph.setPerfEstimate(v, 1);
        // }

        poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 1); // TODO does this create new tensors ?
        poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);

        // std::cout << "neuronDLdWeights.isContiguous(): " << neuronDLdWeights.isContiguous() << std::endl;
        // std::cout << "neuronDLdState.isContiguous(): " << neuronDLdState.isContiguous() << std::endl;



        const unsigned num_neurons = neuronDLdWeights.dim(1);
        std::string vertexType;
        poplar::TargetType target_type = graph.getTarget().getTargetType();
        if ((target_type == poplar::TargetType::IPU) && (num_neurons == 2) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradTwoNeuronSIMD";
          // vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else if ((target_type == poplar::TargetType::IPU) && (num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else {
          vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype);
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        }

        std::cout << "vertexType: " << vertexType << std::endl;

        // auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype),
        // auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype),
        // auto v = graph.addVertex(cs, "LIFWeightsGradMultiNeuronSIMD",
        // auto v = graph.addVertex(cs, "LIFWeightsGradTwoNeuronSIMD",
        auto v = graph.addVertex(cs, vertexType,
                                  {{"dLdState", neuronDLdState.flatten()},
                                  {"fwd_inp_spikes_ids", fwdInpSpikes[ilay].spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                                  {"fwd_num_inp_spikes", fwdInpSpikes[ilay].num_spikes.dimRoll(1, 0)[0]},
                                  {"sparse_out_dim", sparse_out_dim},
                                  {"batchsize", batchsize},
                                  {"num_neurons", num_neurons},
                                  // {"num_weights_per_neuron", neuronDLdWeights.dim(0)},
                                  {"dLdweights", neuronDLdWeights.flatten()}
                                  });
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


void calcLIFWeightGrad(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                        const std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  
  auto cs = graph.addComputeSet({dnai, "calcLIFWeightGrad"});
  const size_t num_layers = dLdweights.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = dLdweights[ilay].elementType();
    size_t sparse_out_dim = fwdInpSpikes[ilay].spike_ids.dim(1);
    size_t batchsize = fwdInpSpikes[ilay].spike_ids.dim(0);
    size_t num_threads = dLdweights[ilay].dim(0);
    size_t batchsize_per_thread = batchsize / num_threads;

    if (batchsize < num_threads) {
      throw poputil::poplibs_error("For `calcLIFWeightGrad`: `batchsize` must be greater or equal to `num_threads`.");
    }
    
    // [0][0] because of replicated tensor due to multi thread approach
    auto neuronTileMapping = graph.getTileMapping(dLdweights[ilay][0][0], true); 

    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        const auto numNeuronsThisThile = neuronRange.size();

        // poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 1).dimRoll(1, 0); // TODO does this create new tensors ?
        // poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);

        // std::cout << "neuronDLdWeights.isContiguous(): " << neuronDLdWeights.isContiguous() << std::endl;
        // std::cout << "neuronDLdState.isContiguous(): " << neuronDLdState.isContiguous() << std::endl;

        // // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
        // // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
        // // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
        // // TODO include batch-loop here when figured out how to be thread/parallel safe
        // // parallelisms might intruduce probelms due to the += operation...
        // for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        //   auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGrad", dtype),
        //                             {{"dLdState", neuronDLdState.dimRoll(1, 0)[ineuron]},
        //                             {"fwd_inp_spikes_ids", fwdInpSpikes[ilay].spike_ids.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
        //                             {"fwd_num_inp_spikes", fwdInpSpikes[ilay].num_spikes.dimRoll(1, 0)[0]},
        //                             {"sparse_out_dim", sparse_out_dim},
        //                             {"dLdweights_row", neuronDLdWeights[ineuron]}});
        //   // !!! TODO !!! totally bogus tile mapping, must be improved
        //   // should be based on state mapping
        //   graph.setTileMapping(v, tile); 
        //   // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        //   graph.setPerfEstimate(v, 1);
        // }

        poplar::Tensor neuronDLdWeights = dLdweights[ilay].slice(neuronRange, 2); // TODO does this create new tensors ?
        poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);

        const unsigned num_neurons = neuronDLdWeights.dim(2);
        
        std::string vertexType;
        poplar::TargetType target_type = graph.getTarget().getTargetType();
        if ((target_type == poplar::TargetType::IPU) && (num_neurons == 2) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradTwoNeuronSIMD";
          // vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else if  ((target_type == poplar::TargetType::IPU) && (num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
          vertexType = "LIFWeightsGradMultiNeuronSIMD";
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        } else {
          vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype);
          // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype);
        }
        // vertexType = poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype);

        // auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGradMultiNeuron", dtype),
        // auto v = graph.addVertex(cs, poputil::templateVertex("LIFWeightsGradMultiNeuronVectorized", dtype),
        // auto v = graph.addVertex(cs, "LIFWeightsGradMultiNeuronSIMD",
        // auto v = graph.addVertex(cs, "LIFWeightsGradTwoNeuronSIMD",

        for (unsigned thread_id = 0; thread_id < num_threads; ++thread_id) {

          size_t batch_id_start = thread_id*batchsize_per_thread;
          size_t batch_id_end = std::min((thread_id+1)*batchsize_per_thread, batchsize);

          poplar::Tensor neuronDLdWeightsThread = neuronDLdWeights[thread_id];
          poplar::Tensor neuronDLdStateThread = neuronDLdState.slice(batch_id_start, batch_id_end, 0);
          poplar::Tensor fwd_inp_spikes_ids_thread = fwdInpSpikes[ilay].spike_ids.slice(batch_id_start, batch_id_end, 0);
          poplar::Tensor fwd_num_inp_spikes_thread = fwdInpSpikes[ilay].num_spikes.slice(batch_id_start, batch_id_end, 0).dimRoll(1, 0)[0];

          auto v = graph.addVertex(cs, vertexType,
                                    {{"dLdState", neuronDLdStateThread.flatten()},
                                    {"fwd_inp_spikes_ids", fwd_inp_spikes_ids_thread.flatten()}, // TODO flatten here or does a Tneosr structure exist for vertex Input ?
                                    {"fwd_num_inp_spikes", fwd_num_inp_spikes_thread},
                                    {"sparse_out_dim", sparse_out_dim},
                                    {"batchsize", neuronDLdStateThread.dim(0)},
                                    {"num_neurons", num_neurons},
                                    // {"num_weights_per_neuron", neuronDLdWeights.dim(0)},
                                    {"dLdweights", neuronDLdWeightsThread.flatten()}
                                    });
          // !!! TODO !!! totally bogus tile mapping, must be improved
          // should be based on state mapping
          graph.setTileMapping(v, tile); 
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
    }
  }
  prog.add(poplar::program::Execute(cs));
}


unsigned get_num_tiles_of_mapping(const poplar::Graph::TileToTensorMapping neuronTileMapping){
  unsigned num_tiles_total = neuronTileMapping.size();
  unsigned num_tiles_mapping{0};
  for (unsigned tile = 0; tile < num_tiles_total; ++tile) {
    if (!neuronTileMapping[tile].empty()) {
      num_tiles_mapping+=1;
    }
  }
  return num_tiles_mapping;
}


void calcLIFInpSpikesGradRowWise(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                                  const std::vector<poplar::Tensor> &dLdState, std::vector<poplar::Tensor> &dLdInpSpikes,
                                  poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  // TODO IMPORTANT: For backwards bass, weight matrix schould be distributed column-wise to different tiles
  
  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGradRowWise"});
  const size_t num_layers = weights.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  std::vector<poplar::Tensor> dLdx_vec;
  // skip first layer because gradient is not needed (dLdInpSpikes.size() = num_layers-1)
  // if desired in the future, other functions have to be reimplemented
    
  for (unsigned ilay=1; ilay<num_layers; ++ilay){ 
    size_t batchsize = dLdState[ilay].dim(0);
    auto dtype = weights[ilay].elementType();


    auto neuronTileMapping = graph.getTileMapping(weights[ilay][0], true);

    // auto neuronTileMapping = graph.getTileMapping(weights[ilay][0], true);
    const auto numTilesThisLayer = get_num_tiles_of_mapping(neuronTileMapping);

    size_t sparseSize = fwdInpSpikes[ilay].spike_ids.dim(1);
    poplar::Tensor dLdx = graph.addVariable(dtype, {numTilesThisLayer, batchsize, sparseSize});
    
    size_t occupied_tile_counter{0};
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      // printVector(thisTileMap);
      if (thisTileMap.empty()) {
        continue;
      }

      graph.setTileMapping(dLdx[occupied_tile_counter], tile);

      for (const auto &neuronRange: thisTileMap) {
        const auto numNeuronsThisThile = neuronRange.size();
        // std::cout << tile << " " << numNeuronsThisThile << std::endl;
        poplar::Tensor neuronWeights = weights[ilay].slice(neuronRange, 1); // TODO does this create new tensors ?
        poplar::Tensor neuronDLdState = dLdState[ilay].slice(neuronRange, 1);
        
        // std::cout << "ilay: " << ilay << std::endl;
        // std::cout << "neuronWeights.isContiguous(): " << neuronWeights.isContiguous() << std::endl;
        // std::cout << "dLdState[ilay].isContiguous(): " << dLdState[ilay].isContiguous() << std::endl;
        // std::cout << "neuronDLdState.isContiguous(): " << neuronDLdState.isContiguous() << std::endl;

        const auto weights_per_neuron = neuronWeights.dim(0);
        const auto num_neurons = neuronWeights.dim(1);

        poplar::TargetType target_type = graph.getTarget().getTargetType();
        if ((target_type == poplar::TargetType::IPU) && (num_neurons == 2) && (dtype == poplar::FLOAT)) {
        // if (false) {
          // std::cout << "2 ROW SIMD" << std::endl;
          // std::cout << "num_neurons: " << num_neurons << std::endl;
          // std::cout << "dtype: " << dtype << std::endl;
          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, "LIFInpSpikesGradTwoRowSIMD",
                                      {{"weights_rows", neuronWeights.flatten()},
                                      // {{"weights_rows", neuronWeights},
                                      // {"relevant_weights", relevantWeights[irow][ibatch]},
                                      {"dLdStates", neuronDLdState[ibatch]},
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      {"sparse_size", sparseSize}});
            graph.setTileMapping(v, tile); 
            graph.setPerfEstimate(v, 1);
          }
        } else if ((target_type == poplar::TargetType::IPU) && (num_neurons % 2 == 0) && (dtype == poplar::FLOAT)) {
          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, "LIFInpSpikesGradMultiRowSIMD",
                                      {{"weights_rows", neuronWeights.flatten()},
                                      // {{"weights_rows", neuronWeights},
                                      // {"relevant_weights", relevantWeights[irow][ibatch]},
                                      {"dLdStates", neuronDLdState[ibatch]},
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      {"num_iters", num_neurons / 2},
                                      {"sparse_size", sparseSize}});
            graph.setTileMapping(v, tile); 
            graph.setPerfEstimate(v, 1);
          }
        } else {
          // std::cout << "STANDARD" << std::endl;
          // std::cout << "num_neurons: " << num_neurons << std::endl;
          // std::cout << "dtype: " << dtype << std::endl;
          for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
            auto v = graph.addVertex(cs, poputil::templateVertex("LIFInpSpikesGradMultiRow", dtype),
                                      {{"weights_rows", neuronWeights.flatten()},
                                      // {{"weights_rows", neuronWeights},
                                      // {"relevant_weights", relevantWeights[irow][ibatch]},
                                      {"dLdStates", neuronDLdState[ibatch]},
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      {"num_neurons", num_neurons},
                                      {"sparse_size", sparseSize}});
            graph.setTileMapping(v, tile); 
            graph.setPerfEstimate(v, 1);
          }
        }
      }
      ++occupied_tile_counter;
    }
    dLdx_vec.push_back(dLdx);
  }
  prog.add(poplar::program::Execute(cs));

  // std::string operation = "ADD";
  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
  
  // for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
  //   // reduceWithOutput(graph, dLdx_vec[ilay], dLdInpSpikes[ilay], {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  //   auto temp = reduce(graph, dLdx_vec[ilay], {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  //   prog.add(poplar::program::Copy(temp, dLdInpSpikes[ilay]));
  //   // prog.add(poplar::program::Copy(dLdx_vec[ilay][0], dLdInpSpikes[ilay]));
  // }

  std::vector<popops::SingleReduceOp> single_reduce_ops;
  for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
    single_reduce_ops.push_back(
      popops::SingleReduceOp(dLdx_vec[ilay], {0}, reduceParams, "single reduce rowwise inpSpikeGrads")
    );
  }
  std::vector<poplar::Tensor> reduce_outs;
  std::transform(dLdInpSpikes.begin(), dLdInpSpikes.end(), std::back_inserter(reduce_outs), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
  reduceMany(graph, single_reduce_ops, reduce_outs, prog, {dnai, "add rowwise inpSpikeGrads"});
}


const std::vector<poplar::Tensor> performSharedUpdate(poplar::Graph &graph, const std::vector<poplar::Tensor> &oneMinus_decay_constants, 
                                                      std::vector<poplar::Tensor> &dLdState, poplar::program::Sequence &prog, 
                                                      const poplar::DebugNameAndId &dnai = {}){

  std::vector<poplar::Tensor> intermediate_dLdState = clone_tensor_vector(graph, dLdState, {dnai, "intermediate_dLdState"});

  auto tile_map_dLdState = graph.getTileMapping(dLdState[0]);
  auto tile_map_iter_dLdState = graph.getTileMapping(intermediate_dLdState[0]);
  std::cout << "\ntile_map_dLdState[0].size(): " << tile_map_dLdState[0].size() << std::endl;
  std::cout << "tile_map_dLdState[1].size(): " << tile_map_dLdState[1].size() << std::endl;
  std::cout << "tile_map_iter_dLdState[0].size(): " << tile_map_iter_dLdState[0].size() << std::endl;
  std::cout << "tile_map_iter_dLdState[1].size(): " << tile_map_iter_dLdState[1].size() << std::endl;


  const auto num_lays{dLdState.size()};
  for (unsigned ilay=0; ilay<num_lays; ++ilay){
    const auto batchsize{dLdState[ilay].dim(0)};
    prog.add(poplar::program::Copy(dLdState[ilay], intermediate_dLdState[ilay]));
    popops::mulInPlace(graph, intermediate_dLdState[ilay], oneMinus_decay_constants[ilay].expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);
  }
  return intermediate_dLdState;
}



void performLIFStepBackwardPass(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, const std::vector<poplar::Tensor> &fwdState, const std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                            const std::vector<poplar::Tensor> &decay_constants, const std::vector<poplar::Tensor> &oneMinus_decay_constants, const std::vector<poplar::Tensor> &thresholds, const std::vector<BatchedSparseSpikes> &fwdOutSpikes,
                            std::vector<poplar::Tensor> &dLdweights, std::vector<poplar::Tensor> &dLdState, poplar::Tensor &dLdOutSpikes, std::vector<poplar::Tensor> &dLdInpSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  std::vector<poplar::Tensor> allDLdOutSpikes(dLdInpSpikes.begin(), dLdInpSpikes.end());
  allDLdOutSpikes.push_back(dLdOutSpikes);
  calcLIFStateGrad(graph, weights, fwdState, decay_constants, thresholds, fwdOutSpikes, dLdState, allDLdOutSpikes, prog, dnai);
 
  const std::vector<poplar::Tensor> intermediate_dLdState = performSharedUpdate(graph, oneMinus_decay_constants, dLdState, prog, {dnai, "performSharedUpdate"});

  // calcLIFWeightGrad_singleThread(graph, dLdweights, fwdInpSpikes, intermediate_dLdState, prog, dnai);
  calcLIFWeightGrad(graph, dLdweights, fwdInpSpikes, intermediate_dLdState, prog, dnai);
  // calcLIFInpSpikesGrad(graph, weights, fwdInpSpikes, decay_constants, dLdState, dLdInpSpikes,  prog, dnai);
  calcLIFInpSpikesGradRowWise(graph, weights, fwdInpSpikes, intermediate_dLdState, dLdInpSpikes,  prog, dnai);
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
  num_inputs = 6*2;
  // allocating_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (std::int64_t i=0; i<num_inputs; ++i){
    allocating_indices.push_back(i);
  }
  is_elementwise = false;
  is_stateless = true;
  // num_inputs = 6;
}


// poplar::Tensor alloc_perneuron_1d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
//   poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
//   size_t numNeurons = shape[0];
//   size_t numTiles = graph.getTarget().getNumTiles();
//   size_t neuronsPerTile = numNeurons / numTiles + 1;

//   for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
//     graph.setTileMapping(allocTensor[ineuron], start_id+ineuron/neuronsPerTile);
//   }
//   return allocTensor;
// }

// poplar::Tensor alloc_rowwise_2d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
//   poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
//   size_t numRows = shape[0];
//   size_t numTiles = graph.getTarget().getNumTiles();
//   size_t rowsPerTile = numRows / numTiles + 1;

//   for (unsigned irow = 0; irow < numRows; ++irow) {
//     graph.setTileMapping(allocTensor[irow], start_id + irow / rowsPerTile);
//   }
//   return allocTensor;
// }

// poplar::Tensor alloc_perneuron_2d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
//   poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
//   size_t batchsize = shape[0];
//   size_t numNeurons = shape[1];
//   size_t numTiles = graph.getTarget().getNumTiles();
//   size_t neuronsPerTile = numNeurons / numTiles + 1;

//   for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
//     for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
//       graph.setTileMapping(allocTensor[ibatch][ineuron], start_id+ineuron/neuronsPerTile);
//     }
//   }
//   return allocTensor;
// }

// poplar::Tensor alloc_perneuron_3d(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t start_id, const poplar::DebugNameAndId &dnai = {}) {
//   poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
//   size_t seq_len = shape[0];
//   size_t batchsize = shape[1];
//   size_t numNeurons = shape[2];
//   size_t numTiles = graph.getTarget().getNumTiles();
//   size_t neuronsPerTile = numNeurons / numTiles + 1;

//   for (unsigned iseq = 0; iseq < seq_len; ++iseq) {
//     for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
//       for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
//         graph.setTileMapping(allocTensor[iseq][ibatch][ineuron], start_id+ineuron/neuronsPerTile);
//       }
//     }
//   }
//   return allocTensor;
// }

std::vector<size_t> determine_neuron_mapping(size_t num_tiles, size_t layer_id, std::vector<size_t> dense_sizes, std::vector<size_t> sparse_sizes, size_t batchsize) {
  
  const double min_num_neurons_per_tile = 2;

  // std::vector<std::vector<unsigned int>> neuron_mapping;
  std::vector<size_t> neuron_mapping;

  size_t tile_offset = 1;
  size_t max_num_tiles_to_use = num_tiles-tile_offset; // TODO substract batchsize because of mapped operations?
  size_t num_layers = dense_sizes.size()-1;

  size_t num_neurons_total = std::accumulate(dense_sizes.begin()+1, dense_sizes.end(), 0);
  size_t weighted_num_neurons_total{0};
  // for (unsigned int ilay=1; ilay < num_layers+1; ++ilay){
  //   weighted_num_neurons_total += dense_sizes[ilay]*sparse_sizes[ilay];
  // };
  for (unsigned int ilay=0; ilay < num_layers; ++ilay){
    weighted_num_neurons_total += dense_sizes[ilay+1]*sparse_sizes[ilay]; // TODO should be based on num input spikes
  };
  double weighted_num_neurons_total_fptype = (double)weighted_num_neurons_total;
 
  std::vector<size_t> num_tiles_per_layer;
  std::vector<size_t> num_neurons_per_tile;
  for (unsigned int ilay=0; ilay < layer_id+1; ++ilay){
    double num_neurons_fptype = (double)dense_sizes[ilay+1];
    // double size_sparse_fptype = (double)sparse_sizes[ilay+1];
    double size_sparse_fptype = (double)sparse_sizes[ilay]; // TODO should be based on num input spikes
    // if (ilay==0) { // no input spike calculation 
    // }
    // linear scaling in input spikes for gradient calculation and output spikes for forward
    double weight_factor = size_sparse_fptype;
    // double weight_factor = sparse_sizes[ilay+1]+sparse_sizes[ilay+1];

    double max_num_tiles_to_use_fptype = (double)max_num_tiles_to_use;
    double num_tiles_fptype = (num_neurons_fptype * weight_factor * max_num_tiles_to_use_fptype) / weighted_num_neurons_total_fptype;
    size_t num_neurons_per_tile_ilay = std::max(min_num_neurons_per_tile, std::ceil(num_neurons_fptype / num_tiles_fptype));
    size_t num_tiles_ilay = std::ceil(num_neurons_fptype / num_neurons_per_tile_ilay);

    num_tiles_per_layer.push_back(num_tiles_ilay);
    num_neurons_per_tile.push_back(num_neurons_per_tile_ilay);
  }

  size_t num_tiles_prev_layers = std::accumulate(num_tiles_per_layer.begin(), num_tiles_per_layer.begin()+layer_id, 0);
  size_t num_neurons_per_tile_this_layer = num_neurons_per_tile[layer_id];
  size_t layer_tile_offset = num_tiles_prev_layers + tile_offset;
  std::cout << layer_id << "  start: " << layer_tile_offset << ", neurons per tile: " << num_neurons_per_tile_this_layer << std::endl;

  for (unsigned int ineuron=0; ineuron < dense_sizes[layer_id+1]; ++ineuron){
    neuron_mapping.push_back(layer_tile_offset+ineuron/num_neurons_per_tile_this_layer);
  }
  return neuron_mapping;
}

poplar::Tensor alloc_linearly(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, unsigned offset, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  poputil::mapTensorLinearlyWithOffset(graph, allocTensor, offset);
  return allocTensor;
}

poplar::Tensor alloc_linearly(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, unsigned offset, unsigned minElementsPerTile, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  unsigned minGrainSize{1};
  poputil::mapTensorLinearlyWithOffset(graph, allocTensor, minElementsPerTile, minGrainSize, offset);
  return allocTensor;
}

poplar::Tensor alloc_neuronwise(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, std::vector<size_t> neuronMapping, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[neuronDim];
  for (unsigned ineuron = 0; ineuron < numNeurons; ++ineuron) {
    graph.setTileMapping(allocTensor.slice(ineuron, ineuron+1, neuronDim), neuronMapping[ineuron]);
  }
  return allocTensor;
}

poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, std::vector<size_t> neuronMapping, const poplar::DebugNameAndId &dnai = {}) {

  std::vector<size_t> shape_slice;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_slice), [](size_t t) -> size_t {return t;});

  const size_t rank = shape.size();
  const size_t numNeurons = neuronMapping.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  std::vector<size_t> numNeuronsPerTile(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    numNeuronsPerTile[tile] = 0;
  }
  for (unsigned i = 0; i < numNeurons; ++i) {
    numNeuronsPerTile[neuronMapping[i]] += 1;
  }

  std::vector<poplar::Tensor> allocTensorVector;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
  
    const auto numNeuronsThisTile = numNeuronsPerTile[tile];
    // printVector(thisTileMap);
    if (numNeuronsThisTile == 0) {
      continue;
    }

    shape_slice[neuronDim] = numNeuronsThisTile;
    poplar::Tensor allocTensorSlice = graph.addVariable(type, shape_slice, dnai);
    graph.setTileMapping(allocTensorSlice, tile);
    allocTensorVector.push_back(allocTensorSlice);
  }

  poplar::Tensor allocTensor = poplar::concat(allocTensorVector, neuronDim);
  return allocTensor;
}

poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, poplar::Graph::TileToTensorMapping neuronTileMapping, const poplar::DebugNameAndId &dnai = {}) {

  std::vector<size_t> shape_slice;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_slice), [](size_t t) -> size_t {return t;});

  const size_t rank = shape.size();
  const size_t numTiles = graph.getTarget().getNumTiles();

  std::vector<poplar::Tensor> allocTensorVector;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    } else if (neuronTileMapping[tile].size() > 1) {
      throw poputil::poplibs_error("For `alloc_neuronwise_contiguous` the neurons on every tile have to be contiguos, meaning there can be only one neuron interval per tile in `neuronTileMapping`.");
    }
    
    const auto neuronRange = neuronTileMapping[tile][0];
    const auto numNeuronsThisTile = neuronRange.size();

    // This should never happen...
    if (numNeuronsThisTile == 0) {
      continue;
    }

    shape_slice[neuronDim] = numNeuronsThisTile;
    poplar::Tensor allocTensorSlice = graph.addVariable(type, shape_slice, dnai);
    graph.setTileMapping(allocTensorSlice, tile);
    allocTensorVector.push_back(allocTensorSlice);
  }

  poplar::Tensor allocTensor = poplar::concat(allocTensorVector, neuronDim);
  return allocTensor;
}


poplar::Tensor alloc_neuronwise_contiguous(poplar::Graph& graph, const std::vector<size_t>& shape, poplar::Type type, size_t neuronDim, size_t start_tile, size_t end_tile, const poplar::DebugNameAndId &dnai = {}) {
  poplar::Tensor allocTensor = graph.addVariable(type, shape, dnai);
  size_t numNeurons = shape[neuronDim];
  size_t num_tiles_this_tensor = end_tile - start_tile;
  size_t num_neurons_per_tile = (numNeurons + num_tiles_this_tensor - 1) / num_tiles_this_tensor;
  size_t num_full_tiles = (numNeurons % num_neurons_per_tile == 0) ?  num_tiles_this_tensor-1 : num_tiles_this_tensor-1;

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
  
  poplar::DebugNameAndId dnai{debug_prefix};

  // {*dense_sizes, *sparse_sizes, batchsize}
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  size_t num_layers = (atrribute_sizes.size()-3) / 4;
  std::vector<size_t> dense_sizes(atrribute_sizes.begin(), atrribute_sizes.begin()+num_layers+1);
  std::vector<size_t> sparse_sizes(atrribute_sizes.begin()+num_layers+1, atrribute_sizes.begin()+2*(num_layers+1));
  size_t sizes_offset = 2*(num_layers+1)+1;
  auto tiles_iterator = atrribute_sizes.begin() + sizes_offset;
  std::vector<size_t> start_tiles(tiles_iterator, tiles_iterator+num_layers);
  std::vector<size_t> end_tiles(tiles_iterator+num_layers, tiles_iterator+2*num_layers);

  size_t batchsize = atrribute_sizes[2*(num_layers+1)];

  std::cout << "\nBuild_allocator: " << operand << std::endl;
  // std::cout << "num_layers: " << num_layers << std::endl;
  // std::cout << "batchsize: " << batchsize << std::endl;

  auto target = graph.getTarget();

  size_t numTiles = target.getNumTiles();
  size_t numTilesPerIPU = target.getTilesPerIPU();
  
  size_t layer_id = operand % num_layers;
  size_t layer_id_prev = (layer_id==0)? 0 : layer_id-1;
  size_t neuronDim;
  std::vector<size_t> neuron_mapping;
  std::string tensor_name;
  poplar::Tensor allocTensor;

  // std::cout << "\noperand: " << operand << ", operand/num_layers: " << operand/num_layers << std::endl;

  // // size_t num_elements{1};
  // // size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b){return a*b;});
  // size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  // std::cout << "layer_id: " << layer_id << std::endl;
  // std::cout << "layer_id_prev: " << layer_id_prev << std::endl;
  // std::cout << "num_elements: " << num_elements << std::endl;
  // printVector(shape);
  size_t tile_offset = 1;
  // // TODO does this work? is ceil correct as it will require one tile to have less...?
  // size_t minElementsPerTile = num_elements / (numTilesPerIPU-tile_offset) + ((num_elements % (numTilesPerIPU-tile_offset)) > 0) ;
  size_t ipu_id = start_tiles[layer_id] / numTilesPerIPU;
  size_t ipu_id_prev = start_tiles[layer_id_prev] / numTilesPerIPU;
  // size_t start_tile_ipu = ipu_id * numTilesPerIPU + tile_offset;
  // size_t start_tile_ipu_prev = ipu_id_prev * numTilesPerIPU + tile_offset;

  size_t ipu_id_to_use;
  poplar::Graph virtualGraph;

  std::cout << "case " << operand/num_layers << std::endl;

  switch (operand/num_layers) {
    case 0: neuronDim = 1; 
            tensor_name = "weights";
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            break;
    case 1: neuronDim = 1;
            tensor_name = "init_state";
            std::cout << "Build alloc init_state" << std::endl;
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // // allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            // allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            std::cout << "\nalloc_tensor tileMap[0].size(): " << graph.getTileMapping(allocTensor)[0].size() << std::endl;
            std::cout << "alloc_tensor tileMap[1].size(): " << graph.getTileMapping(allocTensor)[1].size() << std::endl;
            break;
    case 2: tensor_name = "inp_spike_ids";
            ipu_id_to_use = (layer_id == 0)? ipu_id : ipu_id_prev;
            virtualGraph = graph.createVirtualGraph(numTilesPerIPU * ipu_id_to_use + tile_offset, numTilesPerIPU * (ipu_id_to_use+1));
            if (layer_id == 0) {
              // allocTensor = popops::createSliceableTensor(graph, type, shape, {0}, {1}, minElementsPerTile, {dnai, tensor_name});
              allocTensor = popops::createSliceableTensor(virtualGraph, type, shape, {0}, {1}, 0, {dnai, tensor_name});
            } else {
              // TODO really put start tiles here ? (primarily as easy fix for multi ipu implementation)
              // TODO if yes, layer_id-1 or layer_id ?
              // allocTensor = alloc_linearly(graph, shape, type, start_tile_ipu_prev, minElementsPerTile, {dnai, tensor_name});
              allocTensor = alloc_linearly(virtualGraph, shape, type, tile_offset, {dnai, tensor_name});
            }
            break;  
    case 3: tensor_name = "num_inp_spikes";
            ipu_id_to_use = (layer_id == 0)? ipu_id : ipu_id_prev;
            virtualGraph = graph.createVirtualGraph(numTilesPerIPU * ipu_id_to_use + tile_offset, numTilesPerIPU * (ipu_id_to_use+1));
            if (layer_id == 0) {
              // allocTensor = popops::createSliceableTensor(graph, type, shape, {0}, {1}, 0, {dnai, tensor_name});
              // allocTensor = popops::createSliceableTensor(graph, type, shape, {0}, {1}, minElementsPerTile, {dnai, tensor_name});
              allocTensor = popops::createSliceableTensor(virtualGraph, type, shape, {0}, {1}, 0, {dnai, tensor_name});
            } else {
              // allocTensor = alloc_linearly(graph, shape, type, 0, {dnai, tensor_name});
              // TODO really put start tiles here ? (primarily as easy fix for multi ipu implementation)
              // TODO if yes, layer_id-1 or layer_id ?
              // allocTensor = alloc_linearly(graph, shape, type, start_tile_ipu_prev, 0, {dnai, tensor_name});
              // allocTensor = alloc_linearly(graph, shape, type, start_tile_ipu_prev, minElementsPerTile, {dnai, tensor_name});
              allocTensor = alloc_linearly(virtualGraph, shape, type, tile_offset, {dnai, tensor_name});
            }
            break;
    case 4: neuronDim = 0;
            tensor_name = "decay_constants";
            neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            // alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            break;
    case 5: neuronDim = 0;
            tensor_name = "thresholds";
            neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            // allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            break;
  }
  return allocTensor;
}


// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  if ((inputs.size() % 6) != 0) {
    throw poputil::poplibs_error("LIFMultiLayer requires that the number of inputs plus 1 is divisible by 6.");
  }
  size_t num_layers = inputs.size() / 6;

  poplar::DebugNameAndId dnai{debug_prefix};

  std::vector<poplar::Tensor> weights(inputs.begin(),inputs.begin()+num_layers);
  std::vector<poplar::Tensor> init_state(inputs.begin()+1*num_layers,inputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> inp_spike_ids_fptype(inputs.begin()+2*num_layers,inputs.begin()+3*num_layers);
  std::vector<poplar::Tensor> num_inp_spikes_int(inputs.begin()+3*num_layers,inputs.begin()+4*num_layers);
  std::vector<poplar::Tensor> decay_constants(inputs.begin()+4*num_layers,inputs.begin()+5*num_layers);
  std::vector<poplar::Tensor> thresholds(inputs.begin()+5*num_layers,inputs.begin()+6*num_layers);
  

  std::cout << "RUN Build" << std::endl;

  auto tile_map_init_state0 = graph.getTileMapping(init_state[0]);
  auto tile_map_init_state1 = graph.getTileMapping(init_state[1]);
  std::cout << "\ntile_map_init_state0[0].size(): " << tile_map_init_state0[0].size() << std::endl;
  std::cout << "tile_map_init_state0[1].size(): " << tile_map_init_state0[1].size() << std::endl;
  std::cout << "tile_map_init_state1[0].size(): " << tile_map_init_state1[0].size() << std::endl;
  std::cout << "tile_map_init_state1[1].size(): " << tile_map_init_state1[1].size() << std::endl;


  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    if (weights[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[0]' must be matrices (tensor of rank 2, (size_out, size_in)).");
    }

    if (init_state[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[1]' must be tensors of rank 2, (batch_size, size_out)).");
    }

    if (ilay == 0) {
      if (inp_spike_ids_fptype[ilay].rank() != 3) {
        throw poputil::poplibs_error("Input 'inputs[2]' for the first layer must be tensors of rank 3 (seq_dim, batch_size, inp_dim).");
      }
      if (num_inp_spikes_int[ilay].rank() != 3) {
        throw poputil::poplibs_error("Input 'inputs[3]' for the first layer must be tensors of rank 3 (seq_dim, batch_size, 1).");
      }

    } else {
      if (inp_spike_ids_fptype[ilay].rank() != 2) {
        throw poputil::poplibs_error("Input 'inputs[2]' for every except the first layer must be tensors of rank 2 (batch_size, inp_dim).");
      }
      if (num_inp_spikes_int[ilay].rank() != 2) {
        throw poputil::poplibs_error("Input 'inputs[2]' for every except the first layer must be tensors of rank 3 (batch_size, 1).");
      }
    }


    if (decay_constants[ilay].rank() != 1) {
      throw poputil::poplibs_error("Input 'inputs[4]' must be vectors (size_out,).");
    }

    if (thresholds[ilay].rank() != 1) {
      throw poputil::poplibs_error("Input 'inputs[5]' must be vectors (size_out,).");
    }
  }

  std::cout << "\ncustom_lif_multi_layer_vec_transpose NUM_TILES: " << graph.getTarget().getNumTiles() << std::endl;

  size_t seq_len = inp_spike_ids_fptype[0].dim(0);
  size_t batchsize = inp_spike_ids_fptype[0].dim(1);
  std::vector<size_t> dense_sizes = {weights[0].dim(0)};
  std::transform(weights.begin(), weights.end(), std::back_inserter(dense_sizes), [](poplar::Tensor &t) -> size_t {return t.dim(1);});
  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  std::vector<size_t> sparse_sizes(atrribute_sizes.begin()+num_layers+1, atrribute_sizes.begin()+2*(num_layers+1));
  size_t sizes_offset = 2*(num_layers+1)+1;
  auto tiles_iterator = atrribute_sizes.begin() + sizes_offset;
  std::vector<size_t> start_tiles(tiles_iterator, tiles_iterator+num_layers);
  std::vector<size_t> end_tiles(tiles_iterator+num_layers, tiles_iterator+2*num_layers);

  std::cout << "start_tiles: ";
  printVector(start_tiles);
  std::cout << "end_tiles: ";
  printVector(end_tiles);

  auto dtype = weights[0].elementType();

  for (unsigned i=0; i<=num_layers; ++i){
    if (dense_sizes[i] != atrribute_sizes[i]) {
      throw poputil::poplibs_error("The dense size obtained from weight tensor shapes and from attributes variable are different.");
    }

    if (dense_sizes[i] < sparse_sizes[i]) {
      throw poputil::poplibs_error("The dense size of every layer must be greater or equal to the corresponding sparse size.");
    }
  }
  
  poplar::program::Sequence fwdProg;

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();
  size_t numTiles = target.getNumTiles();
  size_t numTilesPerIPU = target.getTilesPerIPU();

  poplar::TargetType target_type = poplar::TargetType::IPU;
  std::cout << "target.getTargetType(): " << poplar::toString(target.getTargetType()) << std::endl;
  std::cout << "target.getTargetSystemString(): " << target.getTargetSystemString() << std::endl;
  // std::cout << "target.getTargetArchString(): " << target.getTargetArchString() << std::endl;

  // // Get the vector width of the particular data type, so that later we can
  // // divide the tensor up between workers in an appropriate way.
  // const auto vectorWidth = target.getVectorWidth(dtype);

  std::vector<poplar::Tensor> oneMinus_decay_constants;
  for (unsigned i=0; i<num_layers ; ++i) {
    auto ones = graph.addConstant(decay_constants[i].elementType(), decay_constants[i].shape(), 1.0, {dnai, "ones"});
    graph.setTileMapping(ones, graph.getTileMapping(decay_constants[i]));
    oneMinus_decay_constants.push_back(popops::sub(graph, ones, decay_constants[i], fwdProg, {dnai, "itime"}));
  }

  //-------------------------------------------- arguments to specify -------------------------------------------------
  std::vector<size_t> arange_vec;
  for (unsigned i=0; i<num_layers ; ++i) {
    arange_vec.push_back(i);
  }
  
  // auto castVecElements = [&graph, &fwdProg, &dnai](poplar::Tensor &t) -> poplar::Tensor { return popops::cast(graph, t, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast spikes"}); };
  // std::vector<poplar::Tensor> inp_spike_ids;
  // std::vector<poplar::Tensor> num_inp_spikes;
  // std::transform(inp_spike_ids_fptype.begin(), inp_spike_ids_fptype.end(), std::back_inserter(inp_spike_ids), castVecElements);
  // std::transform(num_inp_spikes_int.begin(), num_inp_spikes_int.end(), std::back_inserter(num_inp_spikes), castVecElements);

  std::vector<poplar::Tensor> inp_spike_ids = cast_tensor_vector(graph, inp_spike_ids_fptype, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast inp_spike_ids"});
  std::vector<poplar::Tensor> num_inp_spikes = cast_tensor_vector(graph, num_inp_spikes_int, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast num_inp_spikes"});


  std::vector<size_t> layer_to_ipu_id;
  std::transform(start_tiles.begin(), start_tiles.end(), std::back_inserter(layer_to_ipu_id), [&numTilesPerIPU](size_t &start_tile){return start_tile / numTilesPerIPU;});
  
  std::vector<poplar::Graph> ipu_to_virtualGraph;
  size_t num_ipus = target.getNumIPUs();
  size_t num_tiles_per_ipu = target.getTilesPerIPU();
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    ipu_to_virtualGraph.push_back(graph.createVirtualGraph(numTilesPerIPU * iipu + 1, numTilesPerIPU * (iipu+1)));
  }


  std::vector<poplar::Tensor> out_spike_ids;
  std::vector<poplar::Tensor> num_out_spikes;
  std::vector<poplar::Tensor> stateSeqOutput;
  std::vector<poplar::Tensor> slicedOutSpikeIds;
  std::vector<poplar::Tensor> slicedNumOutSpikes;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    // poplar::Graph graph_layer_ipu = ipu_to_virtualGraph[layer_to_ipu_id[ilay]];

    out_spike_ids.push_back(popops::createSliceableTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], poplar::UNSIGNED_INT, {seq_len, batchsize, sparse_sizes[ilay+1]}, {0}, {1}, 0, {dnai, "alloc out_spike_ids"}));
    num_out_spikes.push_back(popops::createSliceableTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]],  poplar::UNSIGNED_INT, {seq_len, batchsize, 1}, {0}, {1}, 0, {dnai, "alloc  num_out_spikes"}));
    stateSeqOutput.push_back(alloc_neuronwise_contiguous(graph, {seq_len, batchsize, dense_sizes[ilay+1]}, dtype, 2, graph.getTileMapping(weights[ilay][0]), {dnai, "alloc stateSeqOutput"}));

    slicedOutSpikeIds.push_back(popops::createSliceTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], out_spike_ids.back(), {0}, {1}, 1, {dnai, "initial createSliceTensor slicedOutSpikeIds"})[0][0]);
    slicedNumOutSpikes.push_back(popops::createSliceTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], num_out_spikes.back(), {0}, {1}, 1, {dnai, "initial createSliceTensor slicedNumOutSpikes"})[0][0]);
  } 

  // /// !!! TODO !!! improve mapping ! these tensors can really be distributed evenly over the ipu
  // std::vector<poplar::Tensor> out_spike_ids;
  // /// TODO alloc_linearly linearly best choice here ? think about copying between this and per timestep sliced tensors...
  // std::transform(sparse_sizes.begin()+1,sparse_sizes.end(), std::back_inserter(out_spike_ids), 
  //                 [&graph, &dnai, &seq_len, &batchsize](size_t sparse_size) 
  //                   -> poplar::Tensor {return popops::createSliceableTensor(graph, poplar::UNSIGNED_INT, {seq_len, batchsize, sparse_size}, {0}, {1}, 0, {dnai, "alloc out_spike_ids"});});
  //                   // -> poplar::Tensor {return alloc_linearly(graph, {seq_len, batchsize, sparse_size}, poplar::UNSIGNED_INT, 0, {dnai, "alloc out_spike_ids"});});


  // std::vector<poplar::Tensor> num_out_spikes;
  // /// TODO alloc_linearly linearly best choice here ? think about copying between this and per timestep sliced tensors...
  // std::transform(sparse_sizes.begin()+1,sparse_sizes.end(), std::back_inserter(num_out_spikes), 
  //                 [&graph, &dnai, &seq_len, &batchsize](size_t sparse_size) 
  //                   -> poplar::Tensor {return popops::createSliceableTensor(graph,  poplar::UNSIGNED_INT, {seq_len, batchsize, 1}, {0}, {1}, 0, {dnai, "alloc  num_out_spikes"});});
  //                   // -> poplar::Tensor {return alloc_linearly(graph, {seq_len, batchsize, 1}, poplar::UNSIGNED_INT, 0, {dnai, "alloc  num_out_spikes"});});
  // std::vector<poplar::Tensor> stateSeqOutput;
  // for (unsigned ilay=0; ilay<num_layers; ++ilay){
  //   // TODO also here popops::createSliceableTensor, otherwise memory bottleneck might be an issue...
  //   // std::vector<size_t> neuron_mapping = determine_neuron_mapping(numTiles, ilay, dense_sizes, sparse_sizes, batchsize);
  //   // stateSeqOutput.push_back(alloc_neuronwise(graph, {seq_len, batchsize, dense_sizes[ilay+1]}, dtype, 2, neuron_mapping, {dnai, "alloc stateSeqOutput"}));
  //   stateSeqOutput.push_back(alloc_neuronwise_contiguous(graph, {seq_len, batchsize, dense_sizes[ilay+1]}, dtype, 2, graph.getTileMapping(weights[ilay][0]), {dnai, "alloc stateSeqOutput"}));
    
  // }
  // // std::transform(dense_sizes.begin()+1,dense_sizes.end(), std::back_inserter(stateSeqOutput), 
  // //                 [&graph, &dnai, &seq_len, &batchsize, &dtype](size_t dense_size) 
  // //                   -> poplar::Tensor {
  // //                   -> poplar::Tensor {return alloc_perneuron_3d(graph, {seq_len, batchsize, dense_size}, dtype, 1, {dnai, "alloc stateSeqOutput"});});

  //----------------------------------------- Prepare inital state for REPEAT -------------------------------------------------  
  // TODO later maybe better clone and copy to not create unintended behaviour
  std::vector<poplar::Tensor> currentState = clone_tensor_vector(graph, init_state, {dnai, "clone_into_currentState"});
  for (unsigned i=0; i<num_layers; ++i){
    fwdProg.add(poplar::program::Copy(init_state[i], currentState[i], false, {dnai, "copy_to_currenState"}));
  }

  // auto tile_map_currentState0 = graph.getTileMapping(currentState[0]);
  // auto tile_map_currentState1 = graph.getTileMapping(currentState[1]);
  // std::cout << "\ntile_map_currentState0[0].size(): " << tile_map_currentState0[0].size() << std::endl;
  // std::cout << "tile_map_currentState0[1].size(): " << tile_map_currentState0[1].size() << std::endl;
  // std::cout << "tile_map_currentState1[0].size(): " << tile_map_currentState1[0].size() << std::endl;
  // std::cout << "tile_map_currentState1[1].size(): " << tile_map_currentState1[1].size() << std::endl;


  // // As the netowrk is purely feed forward, generate the output spikes tensors from the initial input spikes tensors of the next layer
  // std::vector<poplar::Tensor> slicedOutSpikeIds;
  // // std::transform(inp_spike_ids.begin()+1, inp_spike_ids.end(), std::back_inserter(slicedOutSpikeIds), 
  // //                 [&graph, &dnai](const poplar::Tensor &t) -> poplar::Tensor {return graph.clone(t, {dnai, "initial clone slicedOutSpikeIds"});});
  // // slicedOutSpikeIds.push_back(alloc_linearly(graph, {batchsize, sparse_sizes.back()}, out_spike_ids.back().elementType(), 0, {dnai, "slicedNumOutSpikes"})); // TODO improve alloc
  // std::transform(out_spike_ids.begin(), out_spike_ids.end(), std::back_inserter(slicedOutSpikeIds), 
  //                 [&graph, &dnai](const poplar::Tensor &t) -> poplar::Tensor {return popops::createSliceTensor(graph, t, {0}, {1}, 1, {dnai, "initial createSliceTensor slicedOutSpikeIds"})[0][0];});

  // std::vector<poplar::Tensor> slicedNumOutSpikes;
  // // std::transform(num_inp_spikes.begin()+1, num_inp_spikes.end(), std::back_inserter(slicedNumOutSpikes), 
  // //                 [&graph, &dnai](poplar::Tensor &t) -> poplar::Tensor {return graph.clone(t, {dnai, "initial clone slicedOutSpikeIds"});});
  // // slicedNumOutSpikes.push_back(alloc_linearly(graph, {batchsize, 1}, num_out_spikes.back().elementType(), 0, {dnai, "slicedNumOutSpikes"}));  // TODO improve alloc
  // std::transform(num_out_spikes.begin(), num_out_spikes.end(), std::back_inserter(slicedNumOutSpikes), 
  //                 [&graph, &dnai](const poplar::Tensor &t) -> poplar::Tensor {return popops::createSliceTensor(graph, t, {0}, {1}, 1, {dnai, "initial createSliceTensor slicedNumOutSpikes"})[0][0];});


  for (unsigned i=0; i<num_layers-1; ++i){
    fwdProg.add(poplar::program::Copy(inp_spike_ids[i+1], slicedOutSpikeIds[i], false, dnai));
    fwdProg.add(poplar::program::Copy(num_inp_spikes[i+1], slicedNumOutSpikes[i], false, dnai));
  }

  // input spikes
  std::vector<poplar::Tensor> slicedInpSpikeIds(1); // TODO does this do what I want ?
  // slicedInpSpikeIds.push_back(alloc_perneuron_2d(graph, {batchsize, sparse_sizes[0]}, inp_spike_ids[0].elementType(), 1, {dnai, "slicedInpSpikeIds"})); // TODO improve alloc
  std::transform(inp_spike_ids.begin()+1, inp_spike_ids.end(), std::back_inserter(slicedInpSpikeIds), 
                  [&graph, &dnai](poplar::Tensor &t) -> poplar::Tensor {return graph.clone(t, {dnai, "initial clone inp_spike_ids"});});

  std::vector<poplar::Tensor> slicedNumInpSpikes(1); // TODO does this do what I want ?
  // slicedNumInpSpikes.push_back(alloc_perneuron_2d(graph, {batchsize, 1}, num_out_spikes.back().elementType(), 1, {dnai, "slicedNumOutSpikes"}));  // TODO improve alloc
  std::transform(num_inp_spikes.begin()+1, num_inp_spikes.end(), std::back_inserter(slicedNumInpSpikes), 
                  [&graph, &dnai](poplar::Tensor &t) -> poplar::Tensor {return graph.clone(t, {dnai, "initial clone num_inp_spikes"});});


  std::cout << "tile mappings" << std::endl;
  std::cout << "inp_spike_ids" << std::endl;
  

  auto tile_map_inp_ids_fptype = graph.getTileMapping(inp_spike_ids_fptype[1]);
  auto tile_map_num_ins_int = graph.getTileMapping(num_inp_spikes_int[1]);
  std::cout << "\ntile_map_inp_ids_fptype[0].size(): " << tile_map_inp_ids_fptype[0].size() << std::endl;
  std::cout << "tile_map_inp_ids_fptype[1].size(): " << tile_map_inp_ids_fptype[1].size() << std::endl;
  std::cout << "tile_map_num_ins_int[0].size(): " << tile_map_num_ins_int[0].size() << std::endl;
  std::cout << "tile_map_num_ins_int[1].size(): " << tile_map_num_ins_int[1].size() << std::endl;

  auto tile_map_inp_ids = graph.getTileMapping(inp_spike_ids[1]);
  auto tile_map_num_ins = graph.getTileMapping(num_inp_spikes[1]);
  std::cout << "\ntile_map_inp_ids[0].size(): " << tile_map_inp_ids[0].size() << std::endl;
  std::cout << "tile_map_inp_ids[1].size(): " << tile_map_inp_ids[1].size() << std::endl;
  std::cout << "tile_map_num_ins[0].size(): " << tile_map_num_ins[0].size() << std::endl;
  std::cout << "tile_map_num_ins[1].size(): " << tile_map_num_ins[1].size() << std::endl;

  auto tile_map_sl_inp_ids = graph.getTileMapping(slicedInpSpikeIds[1]);
  auto tile_map_sl_num_ins = graph.getTileMapping(slicedNumInpSpikes[1]);
  std::cout << "\ntile_map_sl_inp_ids[0].size(): " << tile_map_sl_inp_ids[0].size() << std::endl;
  std::cout << "tile_map_sl_inp_ids[1].size(): " << tile_map_sl_inp_ids[1].size() << std::endl;
  std::cout << "tile_map_sl_num_ins[0].size(): " << tile_map_sl_num_ins[0].size() << std::endl;
  std::cout << "tile_map_sl_num_ins[1].size(): " << tile_map_sl_num_ins[1].size() << std::endl;

  //----------------------------------------- REPEAT -------------------------------------------------  
  auto loopFwd = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &currentState, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, 
                  &stateSeqOutput, &dnai, &slicedInpSpikeIds, &slicedNumInpSpikes, &slicedOutSpikeIds, &slicedNumOutSpikes] (
    poplar::Tensor itime
  ) {
    auto loop = poplar::program::Sequence{{}, {dnai}};
    size_t num_layers = weights.size();

    slicedInpSpikeIds[0] = popops::dynamicSlice(graph, inp_spike_ids[0], itime, {0}, {1}, loop, {dnai, "slice_inp_spike_ids"})[0];
    slicedNumInpSpikes[0] = popops::dynamicSlice(graph, num_inp_spikes[0], itime, {0}, {1}, loop, {dnai, "slice_num_inp_spikes"})[0];
    for (unsigned i=0; i < num_layers-1; ++i){
      loop.add(poplar::program::Copy(slicedOutSpikeIds[i], slicedInpSpikeIds[i+1], false, dnai));
      loop.add(poplar::program::Copy(slicedNumOutSpikes[i], slicedNumInpSpikes[i+1], false, dnai));
    }

    std::vector<BatchedSparseSpikes> inpSpikes;
    std::vector<BatchedSparseSpikes> outSpikes;
    // std::cout << "\nslicedOutSpikeIds[i].shape()" << std::endl;
    for (unsigned i=0; i < num_layers; ++i){
      inpSpikes.push_back({slicedInpSpikeIds[i], slicedNumInpSpikes[i]});
      outSpikes.push_back({slicedOutSpikeIds[i], slicedNumOutSpikes[i]});
    }

    performLIFStepFworwardPassInPlace(
        graph, weights, currentState, inpSpikes, decay_constants, oneMinus_decay_constants, thresholds, outSpikes, loop, {dnai});
    // to record state sequence
    // loop.add(poplar::program::Copy(currentState, thisState, false, {dnai, "copy state"}));

    // TODO vectorize this and stateSeqOutput in the first place
    for (unsigned i=0; i<num_layers; ++i){
      popops::dynamicUpdate(graph, stateSeqOutput[i], currentState[i].expand({0}), itime, {0}, {1}, loop, {dnai, "dynamicUpdate_stateSeqOutput"});
      popops::dynamicUpdate(graph, out_spike_ids[i], outSpikes[i].spike_ids.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamicUpdate_out_spike_ids"});
      popops::dynamicUpdate(graph, num_out_spikes[i], outSpikes[i].num_spikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamicUpdate_num_out_spikes"});
    }
    return loop;
  };

  poplar::program::Sequence cloop = popops::countedLoop(graph, seq_len, loopFwd, {dnai, "countedLoop"});
  fwdProg.add(cloop);

  // poplar::Tensor out_spike_ids_fptype{popops::cast(graph, out_spike_ids, weights.elementType(), fwdProg, {dnai, "cast out_spike_ids"})};
  // poplar::Tensor num_out_spikes_int{popops::cast(graph, num_out_spikes, weights.elementType(), fwdProg, {dnai, "cast num_out_spikes"})};

  std::vector<poplar::Tensor> out_spike_ids_fptype;
  std::vector<poplar::Tensor> num_out_spikes_int;
  std::transform(out_spike_ids.begin(), out_spike_ids.end(), std::back_inserter(out_spike_ids_fptype), 
    [&graph, &fwdProg, &dtype, &dnai](poplar::Tensor &t) -> poplar::Tensor { return popops::cast(graph, t, dtype, fwdProg, {dnai, "cast spikes"});});
  std::transform(num_out_spikes.begin(), num_out_spikes.end(), std::back_inserter(num_out_spikes_int), 
    [&graph, &fwdProg, &dnai](poplar::Tensor &t) -> poplar::Tensor { return popops::cast(graph, t, poplar::FLOAT, fwdProg, {dnai, "cast spikes"});});
    // // TODO change back to int!!
    // [&graph, &fwdProg, &dnai](poplar::Tensor &t) -> poplar::Tensor { return popops::cast(graph, t, poplar::INT, fwdProg, {dnai, "cast spikes"});});

  // // // append to outputs
  // std::transform(out_spike_ids_fptype.begin(), out_spike_ids_fptype.end(), std::back_inserter(outputs), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
  // std::transform(num_out_spikes_int.begin(), num_out_spikes_int.end(), std::back_inserter(outputs), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
  // std::transform(stateSeqOutput.begin(), stateSeqOutput.end(), std::back_inserter(outputs), [](poplar::Tensor &t) -> poplar::Tensor {return t;});

  std::cout << "\nDONE FORWARD\n";

  // append to outputs
  extend_tensor_vector(out_spike_ids_fptype, outputs);
  extend_tensor_vector(num_out_spikes_int, outputs);
  extend_tensor_vector(stateSeqOutput, outputs);
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

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();
  size_t numTiles = target.getNumTiles();
  size_t numTilesPerIPU = target.getTilesPerIPU();

  poplar::program::Sequence bwdProg;
  poplar::DebugNameAndId dnai{debug_prefix};

  size_t num_layers = fwd_inputs.size() / 6;
  const std::vector<poplar::Tensor> weights(fwd_inputs.begin(),fwd_inputs.begin()+num_layers);
  const std::vector<poplar::Tensor> init_state_preReAlloc(fwd_inputs.begin()+1*num_layers,fwd_inputs.begin()+2*num_layers);
  const std::vector<poplar::Tensor> inp_spike_ids_fptype_preReAlloc(fwd_inputs.begin()+2*num_layers,fwd_inputs.begin()+3*num_layers);
  const std::vector<poplar::Tensor> num_inp_spikes_int_preReAlloc(fwd_inputs.begin()+3*num_layers,fwd_inputs.begin()+4*num_layers);
  const std::vector<poplar::Tensor> decay_constants(fwd_inputs.begin()+4*num_layers,fwd_inputs.begin()+5*num_layers);
  const std::vector<poplar::Tensor> thresholds(fwd_inputs.begin()+5*num_layers,fwd_inputs.begin()+6*num_layers);

  std::vector<size_t> atrribute_sizes = convert_vecOfStr_to_vecOfSizet(attributes, '_');
  // size_t num_layers = (atrribute_sizes.size()-3) / 4;
  std::vector<size_t> dense_sizes(atrribute_sizes.begin(), atrribute_sizes.begin()+num_layers+1);
  std::vector<size_t> sparse_sizes(atrribute_sizes.begin()+num_layers+1, atrribute_sizes.begin()+2*(num_layers+1));
  size_t sizes_offset = 2*(num_layers+1)+1;
  auto tiles_iterator = atrribute_sizes.begin() + sizes_offset;
  std::vector<size_t> start_tiles(tiles_iterator, tiles_iterator+num_layers);
  std::vector<size_t> end_tiles(tiles_iterator+num_layers, tiles_iterator+2*num_layers);

  std::vector<poplar::Tensor> init_state;
  for (unsigned i=0; i<num_layers; ++i){
    init_state.push_back(alloc_neuronwise_contiguous(graph, init_state_preReAlloc[i].shape(), init_state_preReAlloc[i].elementType(), 1, start_tiles[i], end_tiles[i], {dnai, "alloc_init_state_bwd"}));
  }
  for (unsigned i=0; i<num_layers; ++i){
    bwdProg.add(poplar::program::Copy(init_state_preReAlloc[i], init_state[i], false, {dnai, "copy_to_currenState"}));
  }

  // auto tile_map_init_state0 = graph.getTileMapping(init_state[0]);
  // auto tile_map_init_state1 = graph.getTileMapping(init_state[1]);
  // std::cout << "\ntile_map_init_state0[0].size(): " << tile_map_init_state0[0].size() << std::endl;
  // std::cout << "tile_map_init_state0[1].size(): " << tile_map_init_state0[1].size() << std::endl;
  // std::cout << "tile_map_init_state1[0].size(): " << tile_map_init_state1[0].size() << std::endl;
  // std::cout << "tile_map_init_state1[1].size(): " << tile_map_init_state1[1].size() << std::endl;

  std::vector<size_t> layer_to_ipu_id;
  std::transform(start_tiles.begin(), start_tiles.end(), std::back_inserter(layer_to_ipu_id), [&numTilesPerIPU](size_t &start_tile){return start_tile / numTilesPerIPU;});
  
  std::vector<poplar::Graph> ipu_to_virtualGraph;
  size_t num_ipus = target.getNumIPUs();
  size_t num_tiles_per_ipu = target.getTilesPerIPU();
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    ipu_to_virtualGraph.push_back(graph.createVirtualGraph(numTilesPerIPU * iipu + 1, numTilesPerIPU * (iipu+1)));
  }

  std::vector<poplar::Tensor> inp_spike_ids_fptype = {inp_spike_ids_fptype_preReAlloc[0], };
  // std::vector<poplar::Tensor> num_inp_spikes_int = {num_inp_spikes_int_preReAlloc[0], };
  for (unsigned ilay=1; ilay<num_layers; ++ilay){
    inp_spike_ids_fptype.push_back(alloc_linearly(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], inp_spike_ids_fptype_preReAlloc[ilay].shape(), inp_spike_ids_fptype_preReAlloc[ilay].elementType(), 1, {dnai, "alloc_inp_spike_ids_fptype_bwd"}));
    // num_inp_spikes_int.push_back(alloc_linearly(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], num_inp_spikes_int_preReAlloc[ilay].shape(), num_inp_spikes_int_preReAlloc[ilay].elementType(), 1, {dnai, "alloc_num_inp_spikes_int_bwd"}));
  }
  for (unsigned i=1; i<num_layers; ++i){
    bwdProg.add(poplar::program::Copy(inp_spike_ids_fptype_preReAlloc[i], inp_spike_ids_fptype[i], false, {dnai, "copy_to_inp_spike_ids_fptype_bwd"}));
    // bwdProg.add(poplar::program::Copy(num_inp_spikes_int_preReAlloc[i], num_inp_spikes_int[i], false, {dnai, "copy_to_num_inp_spikes_int_bwd"}));
  }
  std::vector<poplar::Tensor> num_inp_spikes_int = num_inp_spikes_int_preReAlloc;

  auto tile_map_inp_spike_ids_fptype1 = graph.getTileMapping(inp_spike_ids_fptype[1]);
  auto tile_map_inp_spike_ids_fptype_preAlloc = graph.getTileMapping(inp_spike_ids_fptype_preReAlloc[1]);
  std::cout << "\ntile_map_inp_spike_ids_fptype[0].size(): " << tile_map_inp_spike_ids_fptype1[0].size() << std::endl;
  std::cout << "tile_map_inp_spike_ids_fptype[1].size(): " << tile_map_inp_spike_ids_fptype1[1].size() << std::endl;
  std::cout << "tile_map_inp_spike_ids_fptype_preAlloc[0].size(): " << tile_map_inp_spike_ids_fptype_preAlloc[0].size() << std::endl;
  std::cout << "tile_map_inp_spike_ids_fptype_preAlloc[1].size(): " << tile_map_inp_spike_ids_fptype_preAlloc[1].size() << std::endl;

  std::vector<poplar::Tensor> oneMinus_decay_constants;
  for (unsigned i=0; i<num_layers ; ++i) {
    auto ones = graph.addConstant(decay_constants[i].elementType(), decay_constants[i].shape(), 1.0, {dnai, "ones"});
    graph.setTileMapping(ones, graph.getTileMapping(decay_constants[i]));
    oneMinus_decay_constants.push_back(popops::sub(graph, ones, decay_constants[i], bwdProg, {dnai, "itime"}));
  }

  std::vector<poplar::Tensor> out_spike_ids_fptype(fwd_outputs.begin(),fwd_outputs.begin()+num_layers);
  std::vector<poplar::Tensor> num_out_spikes_int(fwd_outputs.begin()+1*num_layers,fwd_outputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> fwd_states_seq(fwd_outputs.begin()+2*num_layers,fwd_outputs.begin()+3*num_layers);

  // std::vector<poplar::Tensor> dLdweights = clone_tensor_vector(graph, weights, {dnai, "dLdweights"});

  // std::vector<poplar::Tensor> dLdweights_temp = clone_tensor_vector(graph, weights, {dnai, "dLdweights"});
  // std::vector<poplar::Tensor> dLdweights;
  // for (unsigned i=0; i<num_layers ; ++i) {
  //     dLdweights.push_back(dLdweights_temp[i].expand({0}));
  // }

  const size_t num_threads = 6;
  std::vector<poplar::Tensor> dLdweights;
  for (unsigned i=0; i<num_layers ; ++i) {
    auto tileMap_debug = graph.getTileMapping(weights[i][0]); 
    // std::cout << "tileMap_debug[0][0]: " << tileMap_debug[0][0].lower() << ", " << tileMap_debug[0][0].upper() << std::endl;
    // std::cout << tileMap_debug.size() << std::endl;
    // std::cout << tileMap_debug[0].size() << std::endl;
    // std::cout << "tileMap_debug[1][0]: " << tileMap_debug[1][0].lower() << ", " << tileMap_debug[1][0].upper() << std::endl;
    // std::cout << "tileMap_debug[2][0]: " << tileMap_debug[2][0].lower() << ", " << tileMap_debug[2][0].upper() << std::endl;
    dLdweights.push_back(alloc_neuronwise_contiguous(graph, {num_threads, weights[i].dim(0), weights[i].dim(1)}, weights[i].elementType(), 2, graph.getTileMapping(weights[i][0]), {dnai, "alloc dLdweights"}));
  }
  zero_tensor_vector(graph, dLdweights, bwdProg, dnai);
  std::vector<poplar::Tensor> dLdinit_state = clone_tensor_vector(graph, init_state, {dnai, "dLdinit_state"});
  std::vector<poplar::Tensor> dLdinp_spike_ids = clone_tensor_vector(graph, inp_spike_ids_fptype, {dnai, "dLdinp_spike_ids"}); // how to  set mapping in Reduce operation
  std::vector<poplar::Tensor> dLdnum_inp_spikes = clone_tensor_vector(graph, num_inp_spikes_int, {dnai, "dLdnum_inp_spikes"});
  std::vector<poplar::Tensor> dLddecay_constatns = clone_tensor_vector(graph, decay_constants, {dnai, "dLddecay_constatns"});
  std::vector<poplar::Tensor> dLdthresholds = clone_tensor_vector(graph, thresholds, {dnai, "dLdthresholds"});

  // only account for gradients though last layers spikes (as it should be for feed forward network)
  poplar::Tensor dLdout_spike_ids = gradients[num_layers-1]; //essentailly assume all others are 0
  // poplar::Tensor dLdnum_out_spikes = gradients[1]; // not needed
  // poplar::Tensor dLdfwd_states_seq = gradients[2]; // Ignore this possibility for now. Essentially assume 0

  // init reverse state
  std::vector<poplar::Tensor> dLdstate = clone_tensor_vector(graph, init_state, {dnai, "dLdstate_clone"});
  zero_tensor_vector(graph, dLdstate, bwdProg, dnai);

  
  auto tile_map_init_state = graph.getTileMapping(init_state[0]);
  auto tile_map_dLdState = graph.getTileMapping(dLdstate[0]);
  std::cout << "\ntile_map_init_state[0].size(): " << tile_map_init_state[0].size() << std::endl;
  std::cout << "tile_map_init_state[1].size(): " << tile_map_init_state[1].size() << std::endl;
  std::cout << "tile_map_dLdState[0].size(): " << tile_map_dLdState[0].size() << std::endl;
  std::cout << "tile_map_dLdState[1].size(): " << tile_map_dLdState[1].size() << std::endl;


  std::vector<poplar::Tensor> inp_spike_ids = cast_tensor_vector(graph, inp_spike_ids_fptype, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast inp_spike_ids"});
  std::vector<poplar::Tensor> num_inp_spikes = cast_tensor_vector(graph, num_inp_spikes_int, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast num_inp_spikes"});
  std::vector<poplar::Tensor> out_spike_ids = cast_tensor_vector(graph, out_spike_ids_fptype, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast out_spike_ids"});
  std::vector<poplar::Tensor> num_out_spikes = cast_tensor_vector(graph, num_out_spikes_int, poplar::UNSIGNED_INT, bwdProg, {dnai, "cast num_out_spikes"});

  // TODO first layer grad never used except when gradient with respect to input tensors is desired  = {graph.clone(dLdinp_spike_ids[0][0], {dnai, "clone slicedDLdInpSpikes"})};
  std::vector<poplar::Tensor> slicedDLdInpSpikes; 
  clone_tensor_vector(graph, inp_spike_ids_fptype, slicedDLdInpSpikes, 1, {dnai, "clone slicedDLdInpSpikes"});
  zero_tensor_vector(graph, slicedDLdInpSpikes, bwdProg, dnai);
  // std::vector<poplar::Tensor> slicedDLdOutSpikes(num_layers);
  // std::transform(slicedDLdInpSpikes.begin()+1, slicedDLdInpSpikes.end(), slicedDLdOutSpikes.begin(), [&graph, &dnai](const poplar::Tensor &t){return graph.clone(t, dnai);});

  auto tile_map_inp_spike_ids_fptype = graph.getTileMapping(inp_spike_ids_fptype[1]);
  auto tile_map_slicedDLdinpSpikes = graph.getTileMapping(slicedDLdInpSpikes[0]);
  std::cout << "\ntile_map_inp_spike_ids_fptype[0].size(): " << tile_map_inp_spike_ids_fptype[0].size() << std::endl;
  std::cout << "tile_map_inp_spike_ids_fptype[1].size(): " << tile_map_inp_spike_ids_fptype[1].size() << std::endl;
  std::cout << "tile_map_slicedDLdinpSpikes[0].size(): " << tile_map_slicedDLdinpSpikes[0].size() << std::endl;
  std::cout << "tile_map_slicedDLdinpSpikes[1].size(): " << tile_map_slicedDLdinpSpikes[1].size() << std::endl;

  //------------------------------------------- Repeat -------------------------------------------------  
  // poplar::Tensor dLdstate = init_reverse_state;
  const size_t seq_len = inp_spike_ids_fptype[0].dim(0);
  poplar::Tensor SEQ_LEN = graph.addConstant(poplar::UNSIGNED_INT, {1}, seq_len, {dnai, "step"});
  poplar::Tensor itime = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dnai, "itime"});
  poplar::Tensor step = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {dnai, "step"});
  graph.setTileMapping(itime, 0);
  graph.setTileMapping(SEQ_LEN, graph.getTileMapping(itime));
  graph.setTileMapping(step, graph.getTileMapping(itime));
  bwdProg.add(poplar::program::Copy(SEQ_LEN, itime, false, dnai));

  auto loopBwd = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
                  &dLdweights, &dLdinp_spike_ids, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};
    
    popops::subInPlace(graph, itime, step, loop, dnai);
    poplar::Tensor itimeMinOne = popops::sub(graph, itime, step, loop, dnai);

    std::vector<BatchedSparseSpikes> fwdInpSpikes;
    std::vector<BatchedSparseSpikes> fwdOutSpikes;
    std::vector<poplar::Tensor> slicedFwdState;

    poplar::Tensor slicedInpSpikeIds = popops::dynamicSlice(graph, inp_spike_ids[0], itime, {0}, {1}, loop, {dnai, "slice inp_spike_ids"})[0];
    poplar::Tensor slicedNumInpSpikes = popops::dynamicSlice(graph, num_inp_spikes[0], itime, {0}, {1}, loop, {dnai, "slice num_inp_spikes"})[0];
    fwdInpSpikes.push_back({slicedInpSpikeIds, slicedNumInpSpikes});
    for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
      // input spikes are prev layers outspikes at prev timestep
      poplar::Tensor slicedInpSpikeIds = popops::dynamicSlice(graph, out_spike_ids[ilay], itimeMinOne, {0}, {1}, loop, {dnai, "slice inp_spike_ids"})[0];
      poplar::Tensor slicedNumInpSpikes = popops::dynamicSlice(graph, num_out_spikes[ilay], itimeMinOne, {0}, {1}, loop, {dnai, "slice num_inp_spikes"})[0];
      fwdInpSpikes.push_back({slicedInpSpikeIds, slicedNumInpSpikes});

      // fwdProg.add(poplar::program::Copy(slicedDLdInpSpikes[i+1], slicedDLdOutSpikes[i], false, dnai));
    }
    poplar::Tensor slicedDLdOutSpikes = popops::dynamicSlice(graph, dLdout_spike_ids, itime, {0}, {1}, loop, {dnai, "slice dLdout_spike_ids"})[0];

    for (unsigned ilay=0; ilay<num_layers; ++ilay){
      poplar::Tensor slicedOutSpikeIds = popops::dynamicSlice(graph, out_spike_ids[ilay], itime, {0}, {1}, loop, {dnai, "slice out_spike_ids"})[0];
      poplar::Tensor slicedNumOutSpikes = popops::dynamicSlice(graph, num_out_spikes[ilay], itime, {0}, {1}, loop, {dnai, "slice num_out_spikes"})[0];
      fwdOutSpikes.push_back({slicedOutSpikeIds, slicedNumOutSpikes});
      slicedFwdState.push_back(popops::dynamicSlice(graph, fwd_states_seq[ilay], itime, {0}, {1}, loop, {dnai, "slice fwd_states_seq"})[0]);
    }

    performLIFStepBackwardPass(
        graph, weights, slicedFwdState, fwdInpSpikes, decay_constants, oneMinus_decay_constants, thresholds, fwdOutSpikes, dLdweights, dLdstate, slicedDLdOutSpikes, slicedDLdInpSpikes, loop, {dnai});

    // TODO if gradient with respect to input spike tensors is desired uncomment this and funcs above and rewrite 
    // (also in `performLIFStepBackwardPass` first layer grad has to be calculated) 
    // popops::dynamicUpdate(graph, dLdinp_spike_ids, slicedDLdInpSpikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic dLdInpSpikes update"});
    return loop;
  };

  poplar::program::Sequence bodyProg = {loopBwd()};
  auto repeat = poplar::program::Repeat(seq_len-1, bodyProg, {dnai, "repeat"});
  bwdProg.add(repeat);


  //----------------------------------- this is just for the first timestep because inp_spikes can not be determined from out_spikes_tensor ----------------------------- 
  auto loopBwdFirstTimestep = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
                  &dLdweights, &dLdinp_spike_ids, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
    
    auto loop = poplar::program::Sequence{{}, {dnai}};
    
    popops::subInPlace(graph, itime, step, loop, dnai);
    poplar::Tensor itimeMinOne = popops::sub(graph, itime, step, loop, dnai);

    std::vector<BatchedSparseSpikes> fwdInpSpikes;
    std::vector<BatchedSparseSpikes> fwdOutSpikes;
    std::vector<poplar::Tensor> slicedFwdState;

    poplar::Tensor slicedInpSpikeIds = popops::dynamicSlice(graph, inp_spike_ids[0], itime, {0}, {1}, loop, {dnai, "slice inp_spike_ids"})[0];
    poplar::Tensor slicedNumInpSpikes = popops::dynamicSlice(graph, num_inp_spikes[0], itime, {0}, {1}, loop, {dnai, "slice num_inp_spikes"})[0];
    fwdInpSpikes.push_back({slicedInpSpikeIds, slicedNumInpSpikes});
    for (unsigned ilay=1; ilay<num_layers; ++ilay){ // TODO this here is why the separate loop program for timestep 0 is necessary
      // input spikes are prev layers outspikes at prev timestep
      poplar::Tensor slicedInpSpikeIds = inp_spike_ids[ilay];
      poplar::Tensor slicedNumInpSpikes = num_inp_spikes[ilay];
      fwdInpSpikes.push_back({slicedInpSpikeIds, slicedNumInpSpikes});
    }
    poplar::Tensor slicedDLdOutSpikes = popops::dynamicSlice(graph, dLdout_spike_ids, itime, {0}, {1}, loop, {dnai, "slice dLdout_spike_ids"})[0];

    for (unsigned ilay=0; ilay<num_layers; ++ilay){
      poplar::Tensor slicedOutSpikeIds = popops::dynamicSlice(graph, out_spike_ids[ilay], itime, {0}, {1}, loop, {dnai, "slice out_spike_ids"})[0];
      poplar::Tensor slicedNumOutSpikes = popops::dynamicSlice(graph, num_out_spikes[ilay], itime, {0}, {1}, loop, {dnai, "slice num_out_spikes"})[0];
      fwdOutSpikes.push_back({slicedOutSpikeIds, slicedNumOutSpikes});
      slicedFwdState.push_back(popops::dynamicSlice(graph, fwd_states_seq[ilay], itime, {0}, {1}, loop, {dnai, "slice fwd_states_seq"})[0]);
    }

    performLIFStepBackwardPass(
        graph, weights, slicedFwdState, fwdInpSpikes, decay_constants, oneMinus_decay_constants, thresholds, fwdOutSpikes, dLdweights, dLdstate, slicedDLdOutSpikes, slicedDLdInpSpikes, loop, {dnai});

    // TODO if gradient with respect to input spike tensors is desired uncomment this and funcs above and rewrite 
    // (also in `performLIFStepBackwardPass` first layer grad has to be calculated) 
    // popops::dynamicUpdate(graph, dLdinp_spike_ids, slicedDLdInpSpikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic dLdInpSpikes update"});
    return loop;
  };

  bwdProg.add(loopBwdFirstTimestep());


  std::vector<poplar::Tensor> dLdweights_final;
  if (num_threads > 1) {
    popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
    std::vector<popops::SingleReduceOp> single_reduce_ops;
    for (unsigned ilay=0; ilay<num_layers; ++ilay){
      single_reduce_ops.push_back(
        popops::SingleReduceOp(dLdweights[ilay], {0}, reduceParams, "single reduce dLdweights")
      );
    }

    // std::vector<poplar::Tensor> dLdweights_final; // TODO alloc with or without tileMapping ?
    dLdweights_final = clone_tensor_vector(graph, weights, {dnai, "dLdweights"});
    std::vector<poplar::Tensor> dLdweights_final_expanded;
    std::transform(dLdweights_final.begin(), dLdweights_final.end(), std::back_inserter(dLdweights_final_expanded), [](poplar::Tensor &t){return t.expand({0});});
    reduceMany(graph, single_reduce_ops, dLdweights_final_expanded, bwdProg, {dnai, "add dLdweights"});

  } else {
    std::transform(dLdweights.begin(), dLdweights.end(), std::back_inserter(dLdweights_final), [](poplar::Tensor &t){return t[0];});
  }

  // std::vector<poplar::Tensor> dLdweights_final;
  // std::transform(dLdweights.begin(), dLdweights.end(), std::back_inserter(dLdweights_final), [](poplar::Tensor &t){return t[0];});


  //----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

  extend_tensor_vector(dLdweights_final, outputs);
  extend_tensor_vector(dLdinit_state, outputs); // only placeholder for now, could easily be calculated from `updatedState` though
  extend_tensor_vector(dLdinp_spike_ids, outputs);
  extend_tensor_vector(dLdnum_inp_spikes, outputs); // placeholder
  extend_tensor_vector(dLddecay_constatns, outputs); // only placeholder for now
  extend_tensor_vector(dLdthresholds, outputs); // only placeholder for now

  return bwdProg;
}
