#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poplar/StringRef.hpp>
#include <poprand/RandomGen.hpp>

#include <iostream>

#include "poplar_code.hpp"
// #include "../../custom_codelet_path.hpp"
// #include "../../string_util.hpp"
#include "custom_dynamic_sparse/custom_codelet_path.hpp"
#include "custom_dynamic_sparse/string_util.hpp"


std::pair<poplar::Tensor, poplar::Tensor> combine_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &repeated_out_spikes_ids, poplar::Tensor &repeated_num_out_spikes,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::ComputeSet &cs) {

  const size_t batchsize = repeated_out_spikes_ids.dim(0);

  const size_t numTilesToUse{endTile - startTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  auto out_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, sparseSize});
  auto num_out_spikes = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, 1});

  poplar::StringRef vertex_name = "SpikesTwoThreshsCombine";
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, vertex_name,
                              // {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch]},
                              {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ibatch]},
                              {"repeated_num_out_spikes", repeated_num_out_spikes[ibatch]},
                              {"out_spikes_ids", out_spike_ids[ibatch]},
                              {"num_out_spikes", num_out_spikes[ibatch][0]}});
    // !!! TODO !!! totally bogus tile mapping, must be improved
    // most likely should be based on out_spikes mapping
    // graph.setTileMapping(v, (ibatch+1)*32);
    size_t tile{startTile + ibatch/batchesPerTile};
    graph.setTileMapping(out_spike_ids[ibatch], tile);
    graph.setTileMapping(num_out_spikes[ibatch], tile);
    graph.setTileMapping(v, tile);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
  return std::make_pair(out_spike_ids, num_out_spikes);
}


// std::vector<poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
//                                                   const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
//   auto cs = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel"});
//   size_t num_layers = state.size();

//   std::vector<poplar::Tensor> spike_ids;
//   std::vector<poplar::Tensor> num_spikes;
//   std::vector<poplar::Tensor> spikes_tensors(2*num_layers);
//   // TODO include checks for same vector lengths
//   for (unsigned i = 0; i < num_layers; ++i){
//     auto sparse_spikes = select_spikes_two_threshs(graph, state[i], thresholds[i], sparseSize[i], startTile[i], endTile[i], cs);
//     spikes_tensors[i] = std::move(sparse_spikes.first);
//     spikes_tensors[i+num_layers] = std::move(sparse_spikes.second);
//   }
//   prog.add(poplar::program::Execute(cs));
//   return spikes_tensors;
// }


std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &randomInds, poplar::Tensor &thresholds,
                                                  const size_t &sparseSize, const size_t &startTile, const size_t &endTile, poplar::ComputeSet &cs) {

  const size_t batchsize = state.dim(0);
  const auto dtype = state.elementType();

  const size_t denseSpraseRatio = state.dim(1) / sparseSize;
  const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();; // TODO get this from poplar ?
  const size_t numWorkers = std::min(denseSpraseRatio, numPossibleParallelThreads); // TODO way to get this from poplar?

  auto repeated_out_spikes_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, numWorkers*sparseSize});
  auto repeated_num_out_spikes = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, numWorkers});

  std::string custom_codelet_path = get_custom_codelet_path_string({"custom_select_spikes", "twoThresh", "custom_codelet.gp"});
  graph.addCodelets(custom_codelet_path);

  const size_t numTilesToUse{endTile - startTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < numWorkers; ++iwor) {
    size_t numStatesThisWorker = state.dim(1) / numWorkers + ((state.dim(1) % numWorkers) > iwor);
    worker_end += numStatesThisWorker;

    auto state_worker = state.slice(worker_start, worker_end, 1);
    auto thresholds_worker = thresholds.slice(worker_start, worker_end, 0);
    auto out_spike_ids_worker = repeated_out_spikes_ids.slice(iwor*sparseSize, (iwor+1)*sparseSize, 1);

    // printVector(state_worker.shape());
    // printVector(thresholds_worker.shape());
    // printVector(out_spike_ids_worker.shape());

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("SpikesTwoThreshsSplitWorkerRandOffset", dtype),
                                {{"state", state_worker[ibatch]},
                                {"thresholds", thresholds_worker},
                                {"start_id", worker_start},
                                {"random_offset", randomInds[ibatch][iwor]},
                                {"repeated_out_spikes_ids", out_spike_ids_worker[ibatch]},
                                {"repeated_num_out_spikes", repeated_num_out_spikes[ibatch][iwor]}});

      size_t tile{startTile + ibatch/batchesPerTile};
      graph.setTileMapping(repeated_out_spikes_ids[ibatch], tile);
      graph.setTileMapping(repeated_num_out_spikes[ibatch], tile);
      graph.setTileMapping(v, tile);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
    worker_start = worker_end;
  }
  return std::make_pair(repeated_out_spikes_ids, repeated_num_out_spikes);
}



std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  const size_t &sparseSize, const size_t &startTile, const size_t &endTile, poplar::ComputeSet &cs) {

  const size_t batchsize = state.dim(0);
  const auto dtype = state.elementType();

  const size_t denseSpraseRatio = state.dim(1) / sparseSize;
  const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();; // TODO get this from poplar ?
  const size_t numWorkers = std::min(denseSpraseRatio, numPossibleParallelThreads); // TODO way to get this from poplar?

  auto repeated_out_spikes_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, numWorkers*sparseSize});
  auto repeated_num_out_spikes = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, numWorkers});

  std::string custom_codelet_path = get_custom_codelet_path_string({"custom_select_spikes", "twoThresh", "custom_codelet.gp"});
  graph.addCodelets(custom_codelet_path);

  const size_t numTilesToUse{endTile - startTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < numWorkers; ++iwor) {
    size_t numStatesThisWorker = state.dim(1) / numWorkers + ((state.dim(1) % numWorkers) > iwor);
    worker_end += numStatesThisWorker;

    auto state_worker = state.slice(worker_start, worker_end, 1);
    auto thresholds_worker = thresholds.slice(worker_start, worker_end, 0);
    auto out_spike_ids_worker = repeated_out_spikes_ids.slice(iwor*sparseSize, (iwor+1)*sparseSize, 1);

    // printVector(state_worker.shape());
    // printVector(thresholds_worker.shape());
    // printVector(out_spike_ids_worker.shape());

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("SpikesTwoThreshsSplitWorker", dtype),
                                {{"state", state_worker[ibatch]},
                                {"thresholds", thresholds_worker},
                                {"start_id", worker_start},
                                {"repeated_out_spikes_ids", out_spike_ids_worker[ibatch]},
                                {"repeated_num_out_spikes", repeated_num_out_spikes[ibatch][iwor]}});

      size_t tile{startTile + ibatch/batchesPerTile};
      graph.setTileMapping(repeated_out_spikes_ids[ibatch], tile);
      graph.setTileMapping(repeated_num_out_spikes[ibatch], tile);
      graph.setTileMapping(v, tile);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
    worker_start = worker_end;
  }
  return std::make_pair(repeated_out_spikes_ids, repeated_num_out_spikes);
}


std::vector<poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
  size_t num_layers = state.size();

  auto cs1 = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel_multiThread_selectSpikes"});
  std::vector<poplar::Tensor> repeated_out_spikes_ids(num_layers);
  std::vector<poplar::Tensor> repeated_num_out_spikes(num_layers);
  // TODO include checks for same vector lengths
  
  for (unsigned i = 0; i < num_layers; ++i){

    const size_t denseSpraseRatio = state[i].dim(1) / sparseSize[i];
    const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();; // TODO get this from poplar ?
    const size_t numWorkers = std::min(denseSpraseRatio, numPossibleParallelThreads); // TODO way to get this from poplar?
    const size_t batchsize = state[i].dim(0);

    const size_t numTilesToUse{endTile[i] - startTile[i]};
    const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

    poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, numWorkers}, {dnai, "reference_tensor"});

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      size_t tile{startTile[i] + ibatch/batchesPerTile};
      graph.setTileMapping(reference, tile);
    }
    poplar::Tensor randomInds = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, state[i].dim(1) / numWorkers, prog, {dnai, "randomInds"});

    auto repeated_sparse_spikes = select_spikes_two_threshs(graph, state[i], randomInds, thresholds[i], sparseSize[i], startTile[i], endTile[i], cs1);
    // auto repeated_sparse_spikes = select_spikes_two_threshs(graph, state[i], thresholds[i], sparseSize[i], startTile[i], endTile[i], cs1);
    repeated_out_spikes_ids[i] = std::move(repeated_sparse_spikes.first);
    repeated_num_out_spikes[i] = std::move(repeated_sparse_spikes.second);
  }
  prog.add(poplar::program::Execute(cs1));

  auto cs2 = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel_multiThread_combineSpikes"});
  std::vector<poplar::Tensor> spikes_tensors(2*num_layers);
  for (unsigned i = 0; i < num_layers; ++i){
    auto sparse_spikes = combine_spikes_two_threshs(graph, repeated_out_spikes_ids[i], repeated_num_out_spikes[i], sparseSize[i], startTile[i], endTile[i], cs2);
    spikes_tensors[i] = sparse_spikes.first;
    spikes_tensors[i+num_layers] = sparse_spikes.second;    
  }
  prog.add(poplar::program::Execute(cs2));
  return spikes_tensors;
}

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  const size_t sparseSize, const size_t startTile, const size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
  auto cs1 = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel_multiThread_selectSpikes"});
  auto repeated_sparse_spikes = select_spikes_two_threshs(graph, state, thresholds, sparseSize, startTile, endTile, cs1);

  auto cs2 = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel_multiThread_combineSpikes"});
  auto sparse_spikes = combine_spikes_two_threshs(graph, repeated_sparse_spikes.first, repeated_sparse_spikes.second, sparseSize, startTile, endTile, cs2);
  prog.add(poplar::program::Execute(cs2));
  return sparse_spikes;
}


std::vector<poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
  auto cs = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel_singleThread"});
  size_t num_layers = state.size();

  std::vector<poplar::Tensor> spikes_tensors(2*num_layers);
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < num_layers; ++i){
    auto sparse_spikes = select_spikes_two_threshs_singleThread(graph, state[i], thresholds[i], sparseSize[i], startTile[i], endTile[i], cs);
    spikes_tensors[i] = std::move(sparse_spikes.first);
    spikes_tensors[i+num_layers] = std::move(sparse_spikes.second);
  }
  prog.add(poplar::program::Execute(cs));
  return spikes_tensors;
}

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshs_singleThread"});
  auto sparse_spikes = select_spikes_two_threshs_singleThread(graph, state, thresholds, sparseSize, startTile, endTile, cs);
  prog.add(poplar::program::Execute(cs));
  return sparse_spikes;
}

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::ComputeSet &cs) {

  const size_t batchsize = state.dim(0);
  const auto dtype = state.elementType();
  auto out_spikes_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, sparseSize});
  auto num_out_spikes = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, 1});

  std::string custom_codelet_path = get_custom_codelet_path_string({"custom_select_spikes", "twoThresh", "custom_codelet.gp"});
  graph.addCodelets(custom_codelet_path);

  const size_t numTilesToUse{endTile - startTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();  

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

  return std::make_pair(out_spikes_ids, num_out_spikes);
}


void select_spikes_two_threshs_dLdState(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  std::vector<poplar::Tensor> &dLdoutSpikes, std::vector<poplar::Tensor> &out_spikes_ids, std::vector<poplar::Tensor> &dLdState,
                                                  const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "computeSpikesTwoThreshsParallelGrad"});
  size_t num_layers = state.size();

  std::vector<poplar::Tensor> spike_ids;
  std::vector<poplar::Tensor> num_spikes;
  std::vector<poplar::Tensor> spikes_tensors(2*num_layers);
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < num_layers; ++i){
    select_spikes_two_threshs_dLdState(graph, state[i], thresholds[i], dLdoutSpikes[i], out_spikes_ids[i], dLdState[i], startTile[i], endTile[i], cs);
  }
  prog.add(poplar::program::Execute(cs));
}


void select_spikes_two_threshs_dLdState(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  poplar::Tensor &dLdoutSpikes, poplar::Tensor &out_spikes_ids, poplar::Tensor &dLdState,
                                                  size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "computeSpikesTwoThreshsGrad"});
  select_spikes_two_threshs_dLdState(graph, state, thresholds, dLdoutSpikes, out_spikes_ids, dLdState, startTile, endTile, cs);
  prog.add(poplar::program::Execute(cs));
}


void select_spikes_two_threshs_dLdState(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  poplar::Tensor &dLdoutSpikes, poplar::Tensor &out_spikes_ids, poplar::Tensor &dLdState,
                                                  size_t startTile, size_t endTile, poplar::ComputeSet &cs) {

  std::string custom_codelet_path = get_custom_codelet_path_string({"custom_select_spikes", "twoThresh", "custom_codelet.gp"});
  graph.addCodelets(custom_codelet_path);

  const size_t batchsize = state.dim(0);
  const auto dtype = state.elementType();                                        
  const size_t numTilesToUse{endTile - startTile};
  const size_t batchesPerTile = batchsize / numTilesToUse + (batchsize % numTilesToUse > 0); // integer ceil div 


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
}

