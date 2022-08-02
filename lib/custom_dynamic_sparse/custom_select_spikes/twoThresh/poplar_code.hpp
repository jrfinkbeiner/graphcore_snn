#pragma once

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

std::vector<poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

// std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
//                                                   size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);
std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  const size_t sparseSize, const size_t startTile, const size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);



std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  const size_t &sparseSize, const size_t &startTile, const size_t &endTile, poplar::ComputeSet &cs);

std::vector<poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  const std::vector<size_t> sparseSize, const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs_singleThread(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::ComputeSet &cs);

// std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
//                                                   size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);


void select_spikes_two_threshs_dLdState(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                                                  std::vector<poplar::Tensor> &dLdoutSpikes, std::vector<poplar::Tensor> &out_spikes_ids, std::vector<poplar::Tensor> &dLdState,
                                                  const std::vector<size_t> startTile, const std::vector<size_t> endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

void select_spikes_two_threshs_dLdState(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  poplar::Tensor &dLdoutSpikes, poplar::Tensor &out_spikes_ids, poplar::Tensor &dLdState,
                                                  size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);                                                 

void select_spikes_two_threshs_dLdState(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  poplar::Tensor &dLdoutSpikes, poplar::Tensor &out_spikes_ids, poplar::Tensor &dLdState,
                                                  size_t startTile, size_t endTile, poplar::ComputeSet &cs);
