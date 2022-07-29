#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

#include <iostream>

#include "poplar_code.hpp"
// #include "../../../custom_codelet_path.hpp"
#include "custom_dynamic_sparse/custom_codelet_path.hpp"
#include "custom_dynamic_sparse/sparse_spikes.hpp"


//---------------------------------------------- forward -----------------------------------------

void calcDynDenseSparseProd(poplar::Graph &graph, std::vector<poplar::Tensor> &matrices, std::vector<poplar::Tensor>  &spike_ids, std::vector<poplar::Tensor>  &num_spikes, 
                        std::vector<poplar::Tensor> &output, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet({dnai, "/DynDenseBinarySparseMatmul"});
  size_t vec_len = matrices.size();
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < vec_len; ++i){
    calcDynDenseSparseProd(graph, matrices[i], spike_ids[i], num_spikes[i], output[i], cs);
  }
  prog.add(poplar::program::Execute(cs));
}

void calcDynDenseSparseProd(poplar::Graph &graph, poplar::Tensor &matrix, poplar::Tensor &spike_ids, poplar::Tensor &num_spikes, 
                        poplar::Tensor &output, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet({dnai, "/DynDenseBinarySparseMatmul"});
  calcDynDenseSparseProd(graph, matrix, spike_ids, num_spikes, output, cs);
  prog.add(poplar::program::Execute(cs));
}

void calcDynDenseSparseProd(poplar::Graph &graph, poplar::Tensor &matrix, poplar::Tensor &spike_ids, poplar::Tensor  &num_spikes, 
                        poplar::Tensor &output, poplar::ComputeSet &cs) {

  size_t batchsize = spike_ids.dim(0);
  auto dtype = matrix.elementType();

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  const auto numTiles = graph.getTarget().getNumTiles();
  auto neuronTileMapping = graph.getTileMapping(matrix.dimShuffle({1,0})[0], true);

  // graph.addCodelets("./custom_codelet.gp");
  std::string custom_codelet_path = get_custom_codelet_path_string({"custom_dyn_dense_sparse_matmul", "batched", "standard", "custom_codelet.gp"});
  graph.addCodelets(custom_codelet_path);


  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: thisTileMap) {
      const auto numNeuronsThisTile = neuronRange.size();
      poplar::Tensor neuronWeights = matrix.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronOut = output.slice(neuronRange, 1);
      graph.setTileMapping(neuronOut, tile);

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      for (unsigned ineuron = 0; ineuron < numNeuronsThisTile; ++ineuron){
        for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
          auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProduct", dtype),
                                    // {{"weights", weights[ilay][neuronId]},
                                    {{"matrix_slice", neuronWeights[ineuron]},
                                    {"spike_ids", spike_ids[ibatch]}, // TODO does this move the tensors for every vertex operation or once for all vertices on the tile ?
                                    {"num_spikes", num_spikes[ibatch][0]},
                                    {"output", neuronOut[ibatch][ineuron]}});
          graph.setTileMapping(v, tile);
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
    }
  }
}

//---------------------------------------------- backward -----------------------------------------

void calcWeightsGrad(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdweights, std::vector<poplar::Tensor> &fwdInpSpikeIds, std::vector<poplar::Tensor> &fwdNumInpSpikes,
                        std::vector<poplar::Tensor> &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "calcWeightsGrad"});
  size_t vec_len = dLdweights.size();
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < vec_len; ++i){
    calcWeightsGrad(graph, dLdweights[i], fwdInpSpikeIds[i], fwdNumInpSpikes[i], dLdy[i], cs);
  }
  prog.add(poplar::program::Execute(cs));
}

void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
                        poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "calcWeightsGrad"});
  calcWeightsGrad(graph, dLdweights, fwdInpSpikeIds, fwdNumInpSpikes, dLdy, cs);
  prog.add(poplar::program::Execute(cs));
}

// void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
//                         poplar::Tensor &dLdy, poplar::ComputeSet &cs, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
void calcWeightsGrad(poplar::Graph &graph, poplar::Tensor &dLdweights, poplar::Tensor  &fwdInpSpikeIds, poplar::Tensor  &fwdNumInpSpikes, 
                        poplar::Tensor &dLdy, poplar::ComputeSet &cs) {

  // graph.addCodelets("custom_codelet.gp");
  const size_t numTiles = graph.getTarget().getNumTiles();

  auto dtype = dLdweights.elementType();
  size_t sparse_out_dim = fwdInpSpikeIds.dim(1);

  // auto neuronTileMapping = graph.getTileMapping(dLdweights.slice(0, 1, 1), true); // TODO use this
  auto neuronTileMapping = graph.getTileMapping(dLdweights.dimShuffle({1,0})[0], true);

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronDLdWeights = dLdweights.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronDLdy = dLdy.slice(neuronRange, 1);

      // popops::zero(graph, neuronDLdWeights, prog, {dnai, "zero neuronDLdWeights"}); // debug, zero should happen in the custom op

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      // !!! TODO !!! really row wise or just column wise as in `calcLIFInpSpikesGrad` case ?
      // TODO include batch-loop here when figured out how to be thread/parallel safe
      // parallelisms might intruduce probelms due to the += operation...
      for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
        auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradWeight", dtype),
                                  // {{"dLdy", neuronDLdy.slice(ineuron, ineuron+1, 1)},
                                  {{"dLdy", neuronDLdy.dimShuffle({1, 0})[ineuron]},
                                  {"fwd_inp_spikes_ids", fwdInpSpikeIds.flatten()}, // TODO flatten here or does a Tensor structure exist for vertex Input ?
                                  {"fwd_num_inp_spikes", fwdNumInpSpikes.flatten()}, // TODO does this copy over for every neuron or once onto the tile ?
                                  {"dLdweights_row", neuronDLdWeights[ineuron]},
                                  {"sparse_out_dim", sparse_out_dim}});
        // !!! TODO !!! totally bogus tile mapping, must be improved
        // should be based on state mapping
        graph.setTileMapping(v, tile); 
        // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        graph.setPerfEstimate(v, 1);
      }
    }
  }
}


void calcInpSpikesGradRowWise(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, const std::vector<poplar::Tensor> &fwdInpSpikeIds,
                                  const std::vector<poplar::Tensor> &dLdy, std::vector<poplar::Tensor> &dLdInpSpikes,
                                  poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGradRowWise"});
  size_t num_layers = weights.size();
  std::vector<poplar::Tensor> dLdx_vec;
  // TODO include checks for same vector lengths
  for (unsigned i = 0; i < num_layers; ++i){
    poplar::Tensor dLdx = calcInpSpikesGradRowWise(graph, weights[i], fwdInpSpikeIds[i], dLdy[i], cs);
    popops::zero(graph, dLdx, prog, {dnai, "zero dLdx"});
    dLdx_vec.push_back(dLdx);
  }
  prog.add(poplar::program::Execute(cs));

  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
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


poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, poplar::Tensor &weights, poplar::Tensor &fwdInpSpikeIds, 
                                  poplar::Tensor &dLdy, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {  
  auto cs = graph.addComputeSet({dnai, "calcLIFInpSpikesGradRowWise"});
  poplar::Tensor dLdx = calcInpSpikesGradRowWise(graph, weights, fwdInpSpikeIds, dLdy, cs);
  popops::zero(graph, dLdx, prog, {dnai, "zero dLdx"});
  prog.add(poplar::program::Execute(cs));

  // std::cout << "\nBefore Reduce\n" << std::endl;
  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
  poplar::Tensor dLdInpSpikes = reduce(graph, dLdx, {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  // poplar::Tensor dLdInpSpikes = graph.addVariable(dtype, {fwdInpSpikeIds.dim(0), fwdInpSpikeIds.dim(1)});
  // reduceWithOutput(graph, dLdx, dLdInpSpikes, {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  // std::cout << "\nAfter Reduce\n" << std::endl;
  return dLdInpSpikes;
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


poplar::Tensor calcInpSpikesGradRowWise(poplar::Graph &graph, const poplar::Tensor &weights, const poplar::Tensor &fwdInpSpikeIds, 
                                  const poplar::Tensor &dLdy, poplar::ComputeSet &cs) {  

  const size_t numTiles = graph.getTarget().getNumTiles();

  // graph.addCodelets("custom_codelet.gp");

  // size_t numRows = weights.dim(0);
  size_t numCols = weights.dim(1);
  size_t batchsize = dLdy.dim(0);
  auto dtype = weights.elementType();

  size_t sparseSize = fwdInpSpikeIds.dim(1);
  

  auto neuronTileMapping = graph.getTileMapping(weights.dimShuffle({1,0})[0], true);

  // poplar::Tensor dLdx = graph.addVariable(dtype, {numRows, batchsize, sparseSize});
  // for (unsigned tile = 0; tile < numTiles; ++tile) {
  //   // If a tile contains no elements of the tensor then do not create any
  //   // vertices for it.
  //   const auto thisTileMap = neuronTileMapping[tile];
  //   if (thisTileMap.empty()) {
  //     continue;
  //   }

  //   for (const auto &neuronRange: neuronTileMapping[tile]) {
  //     const auto numNeuronsThisThile = neuronRange.size();
  //     poplar::Tensor neuronWeights = weights.slice(neuronRange); // TODO does this create new tensors ?
  //     poplar::Tensor neuronDLdy = dLdy.slice(nenumColse and potentially overhead for spawning vertices ?
  //     for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
  //       for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
  //         auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradInputsRowWise", dtype),
  //                                   {{"matrix_row", neuronWeights[ineuron]},
  //                                   {"dLdy", neuronDLdy[ibatch][ineuron]},
  //                                   {"fwd_inp_spike_ids", fwdInpSpikeIds[ibatch]},
  //                                   {"dLdinp_spike_ids", neuronDLdx[ineuron][ibatch]}});
  //         // !!! TODO !!! totally bogus tile mapping, must be improved
  //         // graph.setTileMapping(relevantWeights[irow][ibatch], start_tile+irow/rowsPerTile);
  //         graph.setTileMapping(v, tile); 
  //         // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
  //         graph.setPerfEstimate(v, 1);
  //       }
  //     }
  //   }
  // }


  const auto numTilesThisLayer = get_num_tiles_of_mapping(neuronTileMapping);
  poplar::Tensor dLdx = graph.addVariable(dtype, {numTilesThisLayer, batchsize, sparseSize});
  size_t occupied_tile_counter{0};
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }

    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronWeights = weights.slice(neuronRange); // TODO does this create new tensors ?
      poplar::Tensor neuronDLdy = dLdy.slice(neuronRange, 1);
      // poplar::Tensor neuronDLdx = dLdx.slice(neuronRange);
      poplar::Tensor neuronDLdx = dLdx[occupied_tile_counter];
      graph.setTileMapping(neuronDLdx, tile);

      // TODO ? should perform worker spilt and rewrite Vertex code to take multiple neurons ?
      // TODO ? does that reduce memory for code and potentially overhead for spawning vertices ?
      // for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron){
      for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        // auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradInputsRowWise", dtype),
        auto v = graph.addVertex(cs, poputil::templateVertex("DynDenseBinarySparseProductGradInputsMultiRow", dtype),
                                  {{"matrix_rows", neuronWeights.flatten()},
                                  {"dLdy", neuronDLdy[ibatch]},
                                  {"fwd_inp_spike_ids", fwdInpSpikeIds[ibatch]},
                                  {"dLdinp_spike_ids", neuronDLdx[ibatch]},
                                  {"sparse_size", sparseSize},
                                  {"num_rows", numNeuronsThisThile},
                                  {"row_size", numCols}});
        // !!! TODO !!! totally bogus tile mapping, must be improved
        // graph.setTileMapping(relevantWeights[irow][ibatch], start_tile+irow/rowsPerTile);
        graph.setTileMapping(v, tile); 
        // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
        graph.setPerfEstimate(v, 1);
      }
      // }
    }
    ++occupied_tile_counter;
  }
  return dLdx;
}








