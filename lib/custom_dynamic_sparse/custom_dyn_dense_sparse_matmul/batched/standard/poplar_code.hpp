#pragma once

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>
#include <vector>

#include "custom_dynamic_sparse/sparse_spikes.hpp"

//---------------------------------------------- forward -----------------------------------------

void calcDynDenseSparseProd(poplar::Graph &graph, std::vector<poplar::Tensor> &matrices, std::vector<poplar::Tensor>  &spike_ids, std::vector<poplar::Tensor>  &num_spikes, 
                        std::vector<poplar::Tensor> &output, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

void calcDynDenseSparseProd(poplar::Graph &graph, poplar::Tensor &matrix, poplar::Tensor &spike_ids, poplar::Tensor &num_spikes, 
                        poplar::Tensor &output, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

void calcDynDenseSparseProd(poplar::Graph &graph, poplar::Tensor &matrix, poplar::Tensor &spike_ids, poplar::Tensor  &num_spikes, 
                        poplar::Tensor &output, poplar::ComputeSet &cs);

// void calcDynDenseSparseProd(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor  &spike_ids, poplar::Tensor  &num_spikes, 
//                         poplar::Tensor &output, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);


//---------------------------------------------- backward -----------------------------------------

void calcWeightsGrad(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, std::vector<poplar::Tensor> &fwdInpSpikeIds, std::vector<poplar::Tensor> &fwdNumInpSpikes,
                        std::vector<poplar::Tensor> &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
                        poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
                        poplar::Tensor &dLdy, poplar::ComputeSet &cs);

// void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
//                         poplar::Tensor &dLdy, poplar::ComputeSet &cs, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

// void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
//                         poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

std::vector<poplar::Tensor> calcInpSpikesGradRowWise(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, const std::vector<poplar::Tensor> &fwdInpSpikeIds,
                                  const std::vector<poplar::Tensor> &dLdy, const bool &grad_for_first_inp,
                                  poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdInpSpikeIds, 
                                  poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);

poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, const poplar::Tensor &weights, const poplar::Tensor &fwdInpSpikeIds, 
                                  const poplar::Tensor &dLdy, poplar::ComputeSet &cs);

// poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdInpSpikeIds, 
//                                   poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai);