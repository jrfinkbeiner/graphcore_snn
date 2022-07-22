#pragma once

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <iostream>


std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);
// std::pair<poplar::Tensor, poplar::Tensor> select_spikes_two_threshs(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
//                                                   size_t sparseSize, size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);


void select_spikes_two_threshs_dLdState(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                                                  poplar::Tensor &dLdoutSpikes, poplar::Tensor &out_spikes_ids, poplar::Tensor &dLdState,
                                                  size_t startTile, size_t endTile, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);