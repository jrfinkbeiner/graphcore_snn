#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>

#include "poplar_code.hpp"
// #include "../../custom_codelet_path.hpp"
// #include "../../string_util.hpp"
#include "custom_dynamic_sparse/custom_codelet_path.hpp"
#include "custom_dynamic_sparse/string_util.hpp"


std::vector<poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
  auto cs = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshsParallel"});
  size_t num_layers = state.size();

  std::vector<poplar::Tensor> spike_ids;
  std::vector<poplar::Tensor> num_spikes;
  std::vector<poplar::Tensor> spikes_tensors(2*num_layers);
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < num_layers; ++i){
    auto sparse_spikes = select_spikes_two_threshs(graph, state[i], thresholds[i], sparseSize[i], startTile[i], endTile[i], cs);
    spikes_tensors[i] = std::move(sparse_spikes.first);
    spikes_tensors[i+num_layers] = std::move(sparse_spikes.second);
  }
  prog.add(poplar::program::Execute(cs));
  return spikes_tensors;
}

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "/ComputeSpikesTwoThreshs"});
  auto sparse_spikes = select_spikes_two_threshs(graph, state, thresholds, sparseSize, startTile, endTile, cs);
  prog.add(poplar::program::Execute(cs));
  return sparse_spikes;
}

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::ComputeSet &cs) {

  const size_t batchsize = state.dim(0);
  const auto dtype = state.elementType();
  auto out_spikes_ids = graph.addVariable(dtype, {batchsize, sparseSize});
  auto num_out_spikes = graph.addVariable(poplar::INT, {batchsize, 1});

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

