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
#include <poprand/RandomGen.hpp>

#include <poplar/StringRef.hpp>

#include "poplar_functions.cpp"

// #include "RnnUtil.hpp" // only for boost::optional



// TODO use dim and dtype info like `batchsize` or `dtype` from LIFParams 
// TODO instead of obtaining them from input arrays in every function ? 



// TODO !!! think about tile mapping !!!
void genBatchedLIFOutSpikes2ThreshsMutliWorker(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  auto cs = graph.addComputeSet({dnai, "LIFOutSpikes2ThreshsMultiVertex"});
  const size_t num_layers = state.size();
  const unsigned num_tiles_per_ipu = graph.getTarget().getTilesPerIPU();
  // std::vector<unsigned> indices;
  // for( unsigned i = 0; i < numWorkers; ++i ) indices.push_back( i );
  // printVector(indices);

  std::vector<poplar::Tensor> repeated_out_spikes_ids;
  std::vector<poplar::Tensor> repeated_num_out_spikes;


  const std::vector<unsigned> layers_to_ipu_mapping(get_tensor_ipu_id(graph, state));
  const std::vector<unsigned> layer_ids_per_ipu(get_relative_layer_id_on_ipu(layers_to_ipu_mapping));

  std::cout << "\nlayers_to_ipu_mapping" << std::endl;
  printVector(layers_to_ipu_mapping);
  printVector(layer_ids_per_ipu);


  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = state[ilay].elementType();
    const size_t batchsize = state[ilay].dim(0);
    const size_t sparse_size = out_spikes[ilay].spike_ids.dim(1);

    const unsigned layer_vertex_start_tile = determine_start_tile_spike_gen(layers_to_ipu_mapping[ilay], layer_ids_per_ipu[ilay], batchsize, num_tiles_per_ipu);
    std::cout << ilay << ": layer_vertex_start_tile" << layer_vertex_start_tile << std::endl; 

    // std::cout << "ilay: " << ilay << std::endl;
    // std::cout << "sparse_size: " << sparse_size << std::endl;
    const size_t denseSpraseRatio = state[ilay].dim(1) / sparse_size;
    const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();; // TODO get this from poplar ?
    const size_t numWorkers = std::min(denseSpraseRatio, numPossibleParallelThreads); // TODO way to get this from poplar?
    // // const size_t numWorkers = 1;
    // std::cout << "numWorkers: " << numWorkers << std::endl;

    // const size_t num_threshs = out_spikes[ilay].num_spikes.dim(1);
    const size_t num_threshs = 2;
    repeated_out_spikes_ids.push_back(graph.addVariable(out_spikes[ilay].spike_ids.elementType(), {batchsize, num_threshs, numWorkers*sparse_size}));
    repeated_num_out_spikes.push_back(graph.addVariable(out_spikes[ilay].num_spikes.elementType(), {batchsize, num_threshs, numWorkers}));

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


    poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, numWorkers}, {dnai, "reference_tensor"});

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      size_t tile{layer_vertex_start_tile+ibatch};
      graph.setTileMapping(reference[ibatch], tile);
    }
    // max_val not perfect and will slihgtly bias first neurons
    poplar::Tensor random_offset = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, state[ilay].dim(1) / numWorkers, prog, {dnai, "randomInds"});

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
      auto thresholds_worker = thresholds[ilay].slice(worker_start, worker_end, 1);
      auto out_spike_ids_worker = repeated_out_spikes_ids[ilay].slice(iwor*sparse_size, (iwor+1)*sparse_size, 2);

      // printVector(state_worker.shape());
      // printVector(thresholds_worker.shape());
      // printVector(out_spike_ids_worker.shape());

      for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        // auto v = graph.addVertex(cs, poputil::templateVertex("LIFOutSpikes2ThreshsSplitWorker", dtype),
        // auto v = graph.addVertex(cs, poputil::templateVertex("Spikes2ThreshsSplitWorkerRandOffset", dtype),
        auto v = graph.addVertex(cs, poputil::templateVertex("SpikesMultiThreshsSplitWorkerRandOffset", dtype),
                                  {{"state", state_worker[ibatch]},
                                  {"first_thresh", thresholds_worker[0]},
                                  {"second_thresh", thresholds_worker[1]},
                                  {"start_id", worker_start},
                                  {"random_offset", random_offset[ibatch][iwor]},
                                  {"repeated_out_spikes_ids", out_spike_ids_worker[ibatch][0]},
                                  {"repeated_out_spikes_ids_grads", out_spike_ids_worker[ibatch][1]},
                                  {"repeated_num_out_spikes_first", repeated_num_out_spikes[ilay][ibatch][0][iwor]},
                                  {"repeated_num_out_spikes_second", repeated_num_out_spikes[ilay][ibatch][1][iwor]}});

        size_t tile{layer_vertex_start_tile+ibatch};
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

    const unsigned layer_vertex_start_tile = determine_start_tile_spike_gen(layers_to_ipu_mapping[ilay], layer_ids_per_ipu[ilay], batchsize, num_tiles_per_ipu);
    std::cout << ilay << ": layer_vertex_start_tile" << layer_vertex_start_tile << std::endl;
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs2, "LIFOutSpikesMultiThreshsCombine",
                                // {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch]},
                                {{"repeated_out_spikes_ids", repeated_out_spikes_ids[ilay][ibatch][0]},
                                {"repeated_out_spikes_ids_grads", repeated_out_spikes_ids[ilay][ibatch][1]},
                                {"repeated_num_out_spikes_first", repeated_num_out_spikes[ilay][ibatch][0]},
                                {"repeated_num_out_spikes_second", repeated_num_out_spikes[ilay][ibatch][1]},
                                {"out_spikes_ids", out_spikes[ilay].spike_ids[ibatch]},
                                {"num_out_spikes", out_spikes[ilay].num_spikes[ibatch]},
                                {"num_workers", repeated_num_out_spikes[ilay].dim(2)},
                                {"sparse_size", out_spikes[ilay].spike_ids.dim(1)},
                                {"worker_size", out_spikes[ilay].spike_ids.dim(1)}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // most likely should be based on out_spikes mapping
      // graph.setTileMapping(v, (ibatch+1)*32);
      // size_t tile{1471-ibatch-batchsize*ilay};
      size_t tile{layer_vertex_start_tile+ibatch};
      graph.setTileMapping(v, tile);
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs2));
} 


BatchedSparseSpikes gen_sparseBatchSpikes(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds,
                            poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}) {

  const unsigned num_neurons = state.dim(1);
  const unsigned batchsize = state.dim(0);
  const unsigned num_repeats_batch = 3;
  const unsigned num_thresholds = thresholds.dim(0);
  const unsigned per_repeat_batchsize = batchsize / num_repeats_batch;
  const auto dtype = state.elementType();

  if (batchsize % num_repeats_batch){
    throw poputil::poplibs_error("Currently only batchsizes that are a multiple of 3 are supprted for the `gen_sparseBatchSpikes` operation.");
  }

  poplar::Tensor repeatedBatchSpikeIds = graph.addVariable(poplar::UNSIGNED_INT, {num_neurons, num_repeats_batch, num_thresholds, per_repeat_batchsize}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeatedBatchSpikeNums = graph.addVariable(poplar::UNSIGNED_INT, {num_neurons, num_repeats_batch, num_thresholds}, {dnai, "repeatedNeuronSpikeNums"});


  const auto numTiles = graph.getTarget().getNumTiles();
  auto neuronTileMapping = graph.getTileMapping(state[0], true);

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }
    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronStates = state.slice(neuronRange, 1);
      poplar::Tensor neuronThresholds = thresholds.slice(neuronRange, 1);
      poplar::Tensor neuronRepeatedBatchSpikeIds = repeatedBatchSpikeIds.slice(neuronRange, 0);
      poplar::Tensor neuronRepeatedBatchSpikeNums = repeatedBatchSpikeNums.slice(neuronRange, 0);

      graph.setTileMapping(neuronRepeatedBatchSpikeIds, tile);
      graph.setTileMapping(neuronRepeatedBatchSpikeNums, tile);

      std::cout << "\ngen_sparseBatchSpikes, " << "tile: " << tile << std::endl;
      std::cout << "neuronStates: " << neuronStates.shapeToString() << std::endl;
      std::cout << "first_thresh: " << neuronThresholds.shapeToString() << std::endl;
      std::cout << "numStates: " << per_repeat_batchsize << std::endl;
      std::cout << "repeated_out_spikes_ids: " << neuronRepeatedBatchSpikeIds.shapeToString() << std::endl;
      std::cout << "repeated_num_out_spikes_first: " << neuronRepeatedBatchSpikeNums.shapeToString() << std::endl;


      if ((num_repeats_batch==3) && (numNeuronsThisThile==2)) {
        auto v = graph.addVertex(cs, poputil::templateVertex("SpikesMultiThreshsSplitWorkerBatchSpikesMultiVertexb3n2", dtype),
                                    {{"state", neuronStates.flatten()},
                                    {"first_thresh", neuronThresholds[0]},
                                    {"second_thresh", neuronThresholds[1]},
                                    {"numNeurons", numNeuronsThisThile},
                                    {"numBatchReps", num_repeats_batch},
                                    {"numStates", per_repeat_batchsize},
                                    {"repeated_out_spikes_ids", neuronRepeatedBatchSpikeIds.flatten()},
                                    {"repeated_num_out_spikes", neuronRepeatedBatchSpikeNums.flatten()}});
          graph.setTileMapping(v, tile);
          graph.setPerfEstimate(v, 1);

      } else {
        for (unsigned iwor = 0; iwor < num_repeats_batch; ++iwor) {
          const unsigned worker_start = iwor*per_repeat_batchsize;
          const unsigned worker_end = (iwor+1)*per_repeat_batchsize;
          poplar::Tensor neuronStatesWorker = neuronStates.slice(worker_start, worker_end, 0).dimShuffle({1,0});
          std::cout << "neuronStatesWorker: " << neuronStatesWorker.shapeToString() << std::endl;
          for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron) {
            auto v = graph.addVertex(cs, poputil::templateVertex("SpikesMultiThreshsSplitWorkerBatchSpikes", dtype),
                                        {{"state", neuronStatesWorker[ineuron]},
                                        {"first_thresh", neuronThresholds[0][ineuron]},
                                        {"second_thresh", neuronThresholds[1][ineuron]},
                                        {"numStates", per_repeat_batchsize},
                                        {"repeated_out_spikes_ids", neuronRepeatedBatchSpikeIds[ineuron][iwor][0]},
                                        {"repeated_out_spikes_ids_grads", neuronRepeatedBatchSpikeIds[ineuron][iwor][1]},
                                        {"repeated_num_out_spikes_first", neuronRepeatedBatchSpikeNums[ineuron][iwor][0]},
                                        {"repeated_num_out_spikes_second", neuronRepeatedBatchSpikeNums[ineuron][iwor][1]}});
              graph.setTileMapping(v, tile);
              graph.setPerfEstimate(v, 1);
          }
        }
      }
    }  
  }
  return {repeatedBatchSpikeIds, repeatedBatchSpikeNums}; 
}

std::vector<BatchedSparseSpikes> gen_sparseBatchSpikes(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  unsigned num_layers = state.size();
  auto cs = graph.addComputeSet({dnai, "gen_sparseBatchSpikes"});
  std::vector<BatchedSparseSpikes> repeatedNeuronSpikeIds;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    repeatedNeuronSpikeIds.push_back(gen_sparseBatchSpikes(graph, state[ilay], thresholds[ilay], cs, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  return repeatedNeuronSpikeIds;
}

BatchedSparseSpikes repeatedBatchSpikeIds_to_repeatedNeuronSpikeIds(poplar::Graph &graph, const unsigned &sparse_size, BatchedSparseSpikes &repeatedBacthSpikeIdsMultiThresh, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){
  
  std::vector<unsigned> tensor_tile_ids = get_tensor_tile_ids(graph, repeatedBacthSpikeIdsMultiThresh.num_spikes);

  const unsigned num_occupied_tiles = tensor_tile_ids.size();
  const unsigned num_neurons = repeatedBacthSpikeIdsMultiThresh.spike_ids.dim(0);
  const unsigned num_repeats_batch = repeatedBacthSpikeIdsMultiThresh.spike_ids.dim(1);
  const unsigned num_thresholds = repeatedBacthSpikeIdsMultiThresh.spike_ids.dim(2);
  const unsigned per_repeat_batchsize = repeatedBacthSpikeIdsMultiThresh.spike_ids.dim(3);

  const unsigned batchsize = num_repeats_batch * per_repeat_batchsize;
  auto dtype_ids = repeatedBacthSpikeIdsMultiThresh.spike_ids.elementType();
  auto dtype_nums = repeatedBacthSpikeIdsMultiThresh.num_spikes.elementType();
  // poplar::TargetType target_type = graph.getTarget().getTargetType();

  std::cout << "num_occupied_tiles: " << num_occupied_tiles << std::endl;
  std::cout << "num_repeats_batch: " << num_repeats_batch << std::endl;
  std::cout << "batchsize: " << batchsize << std::endl;
  std::cout << "sparse_size: " << sparse_size << std::endl;


  unsigned num_repeats_neurons = num_neurons / sparse_size + ((num_neurons % sparse_size) > 0);
  unsigned repeat_neuron_size = num_neurons / num_repeats_neurons + ((num_neurons % num_repeats_neurons) > 0);
  
  std::cout << "num_repeats_neurons: " << num_repeats_neurons << std::endl;
  std::cout << "repeat_neuron_size: " << repeat_neuron_size << std::endl;

  printVector(tensor_tile_ids);

  unsigned num_vertices = num_repeats_neurons * num_repeats_batch * num_thresholds;

  poplar::Tensor repeatedNeuronSpikeIds = graph.addVariable(dtype_ids, {num_repeats_neurons, num_thresholds, num_repeats_batch, per_repeat_batchsize, sparse_size}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeatedNeuronSpikeNums = graph.addVariable(dtype_nums, {num_repeats_neurons, num_thresholds, num_repeats_batch, per_repeat_batchsize}, {dnai, "repeatedNeuronSpikeNums"});

  unsigned iiter{0};
  for (unsigned irepneuron=0; irepneuron<num_repeats_neurons; ++irepneuron){
    unsigned neuron_start = repeat_neuron_size*irepneuron;
    unsigned neuron_end = std::min(num_neurons, neuron_start+repeat_neuron_size);
    poplar::Tensor bacthSpikeIds_neuronRange = repeatedBacthSpikeIdsMultiThresh.spike_ids.slice(neuron_start, neuron_end, 0).dimShuffle({1,2,0,3});
    poplar::Tensor bacthSpikeNums_neuronRange = repeatedBacthSpikeIdsMultiThresh.num_spikes.slice(neuron_start, neuron_end, 0).dimShuffle({1,2,0});

    for (unsigned irepbatch=0; irepbatch<num_repeats_batch; ++irepbatch){
      for (unsigned ithr=0; ithr<num_thresholds; ++ithr){

      double vertex_occ_tile_scaling = (double)num_occupied_tiles / (double)num_vertices;
      unsigned occ_tile_id = vertex_occ_tile_scaling * (double)iiter; // TODO improve ?
      // unsigned occ_tile_id{occ_tile_id_fp};
      unsigned tile_id = tensor_tile_ids[occ_tile_id];
      std::cout << "occ_tile_id: " << occ_tile_id << ", tile_id: " << tile_id << std::endl;
      
      graph.setTileMapping(repeatedNeuronSpikeIds[irepneuron][ithr][irepbatch], tile_id);
      graph.setTileMapping(repeatedNeuronSpikeNums[irepneuron][ithr][irepbatch], tile_id);
      
      std::cout << "\nbatch_spike_ids: " << bacthSpikeIds_neuronRange[irepbatch][ithr].flatten().shapeToString() << std::endl;
      std::cout << "batch_spike_ids non flat: " << bacthSpikeIds_neuronRange[irepbatch][ithr].shapeToString() << std::endl;
      std::cout << "batch_num_spikes: " << bacthSpikeNums_neuronRange[irepbatch][ithr].shapeToString() << std::endl;
      std::cout << "per_repeat_batchsize: " << per_repeat_batchsize << std::endl;
      std::cout << "num_neurons: " << neuron_end-neuron_start << std::endl;
      std::cout << "neuron_offset: " << neuron_start << std::endl;
      std::cout << "sparse_size: " << sparse_size << std::endl;
      std::cout << "neuron_spike_ids non flat: " << repeatedNeuronSpikeIds[irepneuron][ithr][irepbatch].shapeToString() << std::endl;
      std::cout << "neuron_spike_ids: " << repeatedNeuronSpikeIds[irepneuron][ithr][irepbatch].flatten().shapeToString() << std::endl;
      std::cout << "neuron_num_spikes: " << repeatedNeuronSpikeNums[irepneuron][ithr][irepbatch].shapeToString() << std::endl;

      auto v = graph.addVertex(cs, "BatchSpikeIds2NeuronSpikeIds",
                                {{"batch_spike_ids", bacthSpikeIds_neuronRange[irepbatch][ithr].flatten()},
                                {"batch_num_spikes", bacthSpikeNums_neuronRange[irepbatch][ithr]},
                                {"per_repeat_batchsize", per_repeat_batchsize},
                                {"num_neurons", neuron_end-neuron_start},
                                {"neuron_offset", neuron_start},
                                {"sparse_size", sparse_size},
                                {"neuron_spike_ids", repeatedNeuronSpikeIds[irepneuron][ithr][irepbatch].flatten()},
                                {"neuron_num_spikes", repeatedNeuronSpikeNums[irepneuron][ithr][irepbatch]}});
      graph.setTileMapping(v, tile_id); 
      graph.setPerfEstimate(v, 1);
      ++iiter;
      }
    }
  }
  return {repeatedNeuronSpikeIds, repeatedNeuronSpikeNums};
}



std::vector<BatchedSparseSpikes> repeatedBatchSpikeIds_to_repeatedNeuronSpikeIds(poplar::Graph &graph, const std::vector<unsigned> &sparse_size, std::vector<BatchedSparseSpikes> &repeatedBacthSpikeIdsMultiThresh,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  
  unsigned num_layers = repeatedBacthSpikeIdsMultiThresh.size();
  auto cs = graph.addComputeSet({dnai, "batchSpikeIds_to_neuronSpikeIds"});
  
  std::vector<BatchedSparseSpikes> repeatedNeuronSpikeIds;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    repeatedNeuronSpikeIds.push_back(repeatedBatchSpikeIds_to_repeatedNeuronSpikeIds(graph, sparse_size[ilay], repeatedBacthSpikeIdsMultiThresh[ilay], cs, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  return repeatedNeuronSpikeIds;
}


void combineRepeatedNeuronSpikeIds(poplar::Graph &graph, BatchedSparseSpikes &repeatedNeuronSparseSpikes, BatchedSparseSpikes &outSpikes,
                            std::vector<unsigned> state_tile_ids, poplar::ComputeSet &cs){

  poplar::Tensor repeatedNeuronSpikeIds = repeatedNeuronSparseSpikes.spike_ids.dimShuffle({1, 2, 3, 0, 4});
  poplar::Tensor repeatedNeuronSpikeNums = repeatedNeuronSparseSpikes.num_spikes.dimShuffle({1, 2, 3, 0});

  const unsigned num_repeats_neurons = repeatedNeuronSpikeNums.dim(3);
  const unsigned num_thresholds = repeatedNeuronSpikeNums.dim(0);
  const unsigned num_repeats_batch = repeatedNeuronSpikeNums.dim(1);
  const unsigned per_repeat_batchsize = repeatedNeuronSpikeNums.dim(2);
  const unsigned batchsize = num_repeats_batch*per_repeat_batchsize;
  const unsigned num_occupied_tiles_layer = state_tile_ids.size();

  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    unsigned inrb = ibatch / per_repeat_batchsize;
    unsigned iprb = ibatch % num_repeats_batch;

    double vertex_occ_tile_scaling = num_occupied_tiles_layer / batchsize;
    unsigned occ_tile_id = vertex_occ_tile_scaling * ibatch; // TODO improve ?
    unsigned tile_id = state_tile_ids[occ_tile_id];
    std::cout << "\ncombineRepeatedNeuronSpikeIds" << std::endl;
    std::cout << "occ_tile_id: " << occ_tile_id << ", tile_id: " << tile_id << std::endl;

    auto v = graph.addVertex(cs, "LIFOutSpikesMultiThreshsCombine",
                              {{"repeated_out_spikes_ids", repeatedNeuronSpikeIds[0][inrb][iprb].flatten()},
                              {"repeated_out_spikes_ids_grads", repeatedNeuronSpikeIds[1][inrb][iprb].flatten()},
                              {"repeated_num_out_spikes_first", repeatedNeuronSpikeNums[0][inrb][iprb]},
                              {"repeated_num_out_spikes_second", repeatedNeuronSpikeNums[1][inrb][iprb]},
                              {"out_spikes_ids", outSpikes.spike_ids[ibatch]},
                              {"num_out_spikes", outSpikes.num_spikes[ibatch]},
                              {"num_workers", repeatedNeuronSpikeNums.dim(3)},
                              {"sparse_size", outSpikes.spike_ids.dim(1)},
                              {"worker_size", outSpikes.spike_ids.dim(1)}});
    graph.setTileMapping(v, tile_id);
    // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
    graph.setPerfEstimate(v, 1);
  }
}


void combineRepeatedNeuronSpikeIds(poplar::Graph &graph, std::vector<BatchedSparseSpikes> &repeatedNeuronSpikeIds, std::vector<BatchedSparseSpikes> &outSpikes,
                            std::vector<std::vector<unsigned>> state_tile_ids, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  
  unsigned num_layers = outSpikes.size();
  auto cs = graph.addComputeSet({dnai, "combineRepeatedNeuronSpikeIds"});
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    combineRepeatedNeuronSpikeIds(graph, repeatedNeuronSpikeIds[ilay], outSpikes[ilay], state_tile_ids[ilay], cs);
  }
  prog.add(poplar::program::Execute(cs));
}



void genBatchedLIFOutSpikesMultiThreshBatchIds(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  std::vector<BatchedSparseSpikes> repeatedBatchSpikeIds_vec = gen_sparseBatchSpikes(graph, state, thresholds, prog, {dnai, "gen_sparseBatchSpikes"});

  std::vector<unsigned> sparse_sizes;
  std::transform(out_spikes.begin(), out_spikes.end(), std::back_inserter(sparse_sizes), [](BatchedSparseSpikes &sparseSpikes){return sparseSpikes.spike_ids.dim(1);});
  std::vector<BatchedSparseSpikes> repeatedNeuronSpikeIds_vec = repeatedBatchSpikeIds_to_repeatedNeuronSpikeIds(graph, sparse_sizes, repeatedBatchSpikeIds_vec, prog, {dnai, "repeatedBatchSpikeIds_to_repeatedNeuronSpikeIds"});

  std::vector<std::vector<unsigned>> state_tile_ids;
  std::transform(state.begin(), state.end(), std::back_inserter(state_tile_ids), [&graph](poplar::Tensor &t){return get_tensor_tile_ids(graph, t);});
  combineRepeatedNeuronSpikeIds(graph, repeatedNeuronSpikeIds_vec, out_spikes, state_tile_ids, prog, {dnai, "combineRepeatedNeuronSpikeIds"});
}


poplar::Tensor gen_dense_spikes(poplar::Graph &graph, poplar::Tensor &state, poplar::Tensor &thresholds, 
                              poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){
  
  const unsigned num_neurons = state.dim(1);
  const unsigned batchsize = state.dim(0);
  const unsigned num_thresholds = thresholds.dim(0);
  const auto dtype = state.elementType();

  // TODO ideally smaller data type (bool or char)
  poplar::Tensor denseSpikes = graph.addVariable(poplar::UNSIGNED_INT, {num_neurons, batchsize, num_thresholds}, {dnai, "denseSpikes"});

  const auto numTiles = graph.getTarget().getNumTiles();
  auto neuronTileMapping = graph.getTileMapping(state[0], true);

  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }
    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronStates = state.slice(neuronRange, 1);
      poplar::Tensor neuronThresholds = thresholds.slice(neuronRange, 1);
      poplar::Tensor neuronDenseSpikes = denseSpikes.slice(neuronRange, 0);

      graph.setTileMapping(neuronDenseSpikes, tile);

      for (unsigned ineuron = 0; ineuron < numNeuronsThisThile; ++ineuron) {
        for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
          auto v = graph.addVertex(cs, poputil::templateVertex("DenseSpikesMultiThresh", dtype), // TODO could this be vectorized over the states ?
                                      {{"state", neuronStates[ibatch][ineuron]},
                                      {"first_thresh", neuronThresholds[0][ineuron]},
                                      {"second_thresh", neuronThresholds[1][ineuron]},
                                      {"dense_spikes_thresh0", neuronDenseSpikes[ineuron][ibatch][0]},
                                      {"dense_spikes_thresh1", neuronDenseSpikes[ineuron][ibatch][1]}});
                                      // {"dense_spikes_thresh0", neuronDenseSpikes[ineuron][ibatch][0]},
                                      // {"dense_spikes_thresh1", neuronDenseSpikes[ineuron][ibatch][1]}});
            graph.setTileMapping(v, tile);
            graph.setPerfEstimate(v, 1);
        }
      }
    }  
  }
  return denseSpikes; 
}

std::vector<poplar::Tensor> gen_dense_spikes(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  unsigned num_layers = state.size();
  auto cs = graph.addComputeSet({dnai, "gen_dense_spikes"});
  std::vector<poplar::Tensor> dense_spikes;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    dense_spikes.push_back(gen_dense_spikes(graph, state[ilay], thresholds[ilay], cs, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  return dense_spikes;
}

BatchedSparseSpikes dense_to_sparse_spikes(poplar::Graph &graph, poplar::Tensor &dense_spikes, const std::vector<unsigned> &tiles_to_use, const unsigned &sparse_size,
                            poplar::program::Sequence &prog, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){
  
  const unsigned num_neurons = dense_spikes.dim(0);
  const unsigned batchsize = dense_spikes.dim(1);
  const unsigned num_thresholds = dense_spikes.dim(2);
  const auto dtype = dense_spikes.elementType();

  const size_t denseSpraseRatio = num_neurons / sparse_size;
  const size_t numPossibleParallelThreads = graph.getTarget().getNumWorkerContexts();
  const size_t num_workers = std::min(denseSpraseRatio, numPossibleParallelThreads);

  poplar::Tensor repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers, sparse_size}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers}, {dnai, "repeatedNeuronSpikeIds"});

  // for random offset (more even sampling, less bias in selection of spikes?)
  poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, num_thresholds, num_workers}, {dnai, "reference_tensor"});
  unsigned vertex_id{0};
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
      graph.setTileMapping(reference[ibatch][ithresh], tiles_to_use[vertex_id]);
      ++vertex_id;
    }
  }
  // max_val not perfect and will slihgtly bias first neurons
  poplar::Tensor random_offset = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, num_neurons / num_workers, prog, {dnai, "randomInds"});

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < num_workers; ++iwor) {
    size_t numStatesThisWorker = num_neurons / num_workers + ((num_neurons % num_workers) > iwor);
    worker_end += numStatesThisWorker;
    poplar::Tensor dense_spikes_worker = dense_spikes.slice(worker_start, worker_end, 0).dimShuffle({1,2,0});

    unsigned vertex_id{0};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
        unsigned tile_id{tiles_to_use[vertex_id]};
        
        graph.setTileMapping(repeated_sparse_spike_ids[ibatch][ithresh][iwor], tile_id);
        graph.setTileMapping(repeated_sparse_spike_nums[ibatch][ithresh][iwor], tile_id);        

        auto v = graph.addVertex(cs, poputil::templateVertex("DenseToSparseSpikes", dtype),
                                    {{"dense_spikes", dense_spikes_worker[ibatch][ithresh]},
                                    {"num_dense_spikes", numStatesThisWorker},
                                    {"sparse_size", sparse_size},
                                    {"start_id", worker_start},
                                    {"random_offset", random_offset[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_ids", repeated_sparse_spike_ids[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_nums", repeated_sparse_spike_nums[ibatch][ithresh][iwor]}});
        graph.setTileMapping(v, tile_id);
        graph.setPerfEstimate(v, 1);
        ++vertex_id;
      }
    }
    worker_start = worker_end;
  }
  return {repeated_sparse_spike_ids, repeated_sparse_spike_nums}; 
}

BatchedSparseSpikes dense_to_sparse_spikes_multi_tile_spread(poplar::Graph &graph, poplar::Tensor &dense_spikes, const std::vector<unsigned> &tiles_to_use, const unsigned &sparse_size,
                            poplar::program::Sequence &prog, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){
  
  const unsigned num_neurons = dense_spikes.dim(0);
  const unsigned batchsize = dense_spikes.dim(1);
  const unsigned num_thresholds = dense_spikes.dim(2);
  const auto dtype = dense_spikes.elementType();
  const unsigned num_tiles_to_use = tiles_to_use.size();
  const unsigned num_tile_spread_fac = num_tiles_to_use / (batchsize * num_thresholds);

  const unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  const size_t num_workers = std::min(num_tile_spread_fac * num_workers_per_tile, num_neurons);

  const unsigned max_num_states_per_worker = num_neurons / num_workers + ((num_neurons % num_workers) > 0);

  const bool sparseSmallerStatesPerWorker = sparse_size < max_num_states_per_worker;
  const unsigned sparse_size_to_use = (sparseSmallerStatesPerWorker) ? sparse_size: max_num_states_per_worker;

  std::cout << "\ndense_to_sparse_spikes_multi_tile_spread" << std::endl;
  std::cout << "num_tile_spread_fac: " << num_tile_spread_fac << std::endl;
  std::cout << "num_workers: " << num_workers << std::endl;
  std::cout << "max_num_states_per_worker: " << max_num_states_per_worker << std::endl;
  std::cout << "sparse_size_to_use: " << sparse_size_to_use << std::endl;
  std::cout << "sparseSmallerStatesPerWorker: " << sparseSmallerStatesPerWorker << std::endl;



  poplar::Tensor repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers, sparse_size_to_use}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers}, {dnai, "repeatedNeuronSpikeIds"});

  // for random offset (more even sampling, less bias in selection of spikes?)
  poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, num_thresholds, num_workers}, {dnai, "reference_tensor"});
  unsigned vertex_id{0};
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
      graph.setTileMapping(reference[ibatch][ithresh], tiles_to_use[vertex_id]);
      ++vertex_id;
    }
  }
  // max_val not perfect and will slihgtly bias first neurons
  poplar::Tensor random_offset = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, num_neurons / num_workers, prog, {dnai, "randomInds"});

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < num_workers; ++iwor) {
    size_t numStatesThisWorker = num_neurons / num_workers + ((num_neurons % num_workers) > iwor); // TODO creates imbalance if spread over multiple tiles
    worker_end += numStatesThisWorker;
    poplar::Tensor dense_spikes_worker = dense_spikes.slice(worker_start, worker_end, 0).dimShuffle({1,2,0});

    std::cout << "numStatesThisWorker: " << numStatesThisWorker << std::endl;

    unsigned vertex_id{0 + iwor/num_workers_per_tile};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
        unsigned tile_id{tiles_to_use[vertex_id]};
        
        graph.setTileMapping(repeated_sparse_spike_ids[ibatch][ithresh][iwor], tile_id);
        graph.setTileMapping(repeated_sparse_spike_nums[ibatch][ithresh][iwor], tile_id);        

        // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
        auto v = graph.addVertex(cs, poputil::templateVertex("DenseToSparseSpikes", dtype),
                                    {{"dense_spikes", dense_spikes_worker[ibatch][ithresh]},
                                    {"num_dense_spikes", numStatesThisWorker},
                                    {"sparse_size", sparse_size_to_use},
                                    {"start_id", worker_start},
                                    {"random_offset", random_offset[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_ids", repeated_sparse_spike_ids[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_nums", repeated_sparse_spike_nums[ibatch][ithresh][iwor]}});
        graph.setTileMapping(v, tile_id);
        graph.setPerfEstimate(v, 1);
        vertex_id += num_tile_spread_fac;
      }
    }
    worker_start = worker_end;
  }
  return {repeated_sparse_spike_ids, repeated_sparse_spike_nums}; 
}


BatchedSparseSpikes dense_to_sparse_spikes_multi_tile_spread_with_combine(poplar::Graph &graph, poplar::Tensor &dense_spikes, const std::vector<unsigned> &tiles_to_use, const unsigned &sparse_size,
                            poplar::program::Sequence &prog, poplar::ComputeSet &cs1, poplar::ComputeSet &cs2, const poplar::DebugNameAndId &dnai = {}){
  
  const unsigned num_neurons = dense_spikes.dim(0);
  const unsigned batchsize = dense_spikes.dim(1);
  const unsigned num_thresholds = dense_spikes.dim(2);
  const auto dtype = dense_spikes.elementType();
  const unsigned num_tiles_to_use = tiles_to_use.size();
  const unsigned num_tile_spread_fac = num_tiles_to_use / (batchsize * num_thresholds);

  const unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  const unsigned num_workers = std::min(num_tile_spread_fac * num_workers_per_tile, num_neurons);

  const unsigned max_num_states_per_worker = num_neurons / num_workers + ((num_neurons % num_workers) > 0);
  const unsigned max_num_states_per_tile = num_neurons / num_tile_spread_fac + ((num_neurons % num_tile_spread_fac) > 0);

  const bool sparseSmallerStatesPerWorker = sparse_size < max_num_states_per_worker;
  const bool sparseSmallerStatesPerTile = sparse_size < max_num_states_per_tile;
  const unsigned sparse_size_to_use = (sparseSmallerStatesPerWorker) ? sparse_size: max_num_states_per_worker;
  const unsigned sparse_size_combined = (sparseSmallerStatesPerTile) ? sparse_size: max_num_states_per_tile;

  std::cout << "\ndense_to_sparse_spikes_multi_tile_spread" << std::endl;
  std::cout << "num_tile_spread_fac: " << num_tile_spread_fac << std::endl;
  std::cout << "num_workers: " << num_workers << std::endl;
  std::cout << "max_num_states_per_worker: " << max_num_states_per_worker << std::endl;
  std::cout << "sparse_size_to_use: " << sparse_size_to_use << std::endl;
  std::cout << "sparseSmallerStatesPerWorker: " << sparseSmallerStatesPerWorker << std::endl;

  poplar::Tensor repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers, sparse_size_to_use}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers}, {dnai, "repeatedNeuronSpikeIds"});

  poplar::Tensor combined_repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tile_spread_fac, sparse_size_combined}, {dnai, "combinedRepeatedNeuronSpikeIds"});
  poplar::Tensor combined_repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tile_spread_fac}, {dnai, "combinedRepeatedNeuronSpikeIds"});

  // TODO random offset for combine!
  // for random offset (more even sampling, less bias in selection of spikes?)
  poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, num_thresholds, num_workers}, {dnai, "reference_tensor"});
  unsigned vertex_id{0};
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
      graph.setTileMapping(reference[ibatch][ithresh], tiles_to_use[vertex_id]);
      ++vertex_id;
    }
  }
  // max_val not perfect and will slihgtly bias first neurons
  poplar::Tensor random_offset = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, num_neurons / num_workers, prog, {dnai, "randomInds"});

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < num_workers; ++iwor) {

    const unsigned itileSpreadFac = iwor / num_workers_per_tile;
    const unsigned iworPerTile = iwor % num_workers_per_tile;

    size_t numStatesThisWorker = num_neurons / num_workers + ((num_neurons % num_workers) > iwor); // TODO creates imbalance if spread over multiple tiles
    worker_end += numStatesThisWorker;
    poplar::Tensor dense_spikes_worker = dense_spikes.slice(worker_start, worker_end, 0).dimShuffle({1,2,0});

    std::cout << "numStatesThisWorker: " << numStatesThisWorker << std::endl;

    // unsigned vertex_id{0 + iwor/num_workers_per_tile};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
        // unsigned tile_id{tiles_to_use[vertex_id]};
        unsigned tile_id{tiles_to_use[ibatch*num_thresholds*num_tile_spread_fac + ithresh*num_tile_spread_fac + itileSpreadFac]};
        
        graph.setTileMapping(repeated_sparse_spike_ids[ibatch][ithresh][iwor], tile_id);
        graph.setTileMapping(repeated_sparse_spike_nums[ibatch][ithresh][iwor], tile_id);        

        // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
        auto v1 = graph.addVertex(cs1, poputil::templateVertex("DenseToSparseSpikes", dtype),
                                    {{"dense_spikes", dense_spikes_worker[ibatch][ithresh]},
                                    {"num_dense_spikes", numStatesThisWorker},
                                    {"sparse_size", sparse_size_to_use},
                                    {"start_id", worker_start},
                                    {"random_offset", random_offset[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_ids", repeated_sparse_spike_ids[ibatch][ithresh][iwor]},
                                    {"repeated_sparse_spike_nums", repeated_sparse_spike_nums[ibatch][ithresh][iwor]}});
        graph.setTileMapping(v1, tile_id);
        graph.setPerfEstimate(v1, 1);

        if  (iworPerTile == 0){
          graph.setTileMapping(combined_repeated_sparse_spike_ids[ibatch][ithresh][itileSpreadFac], tile_id);
          graph.setTileMapping(combined_repeated_sparse_spike_nums[ibatch][ithresh][itileSpreadFac], tile_id);        

          const unsigned workers_tile_start = itileSpreadFac*num_workers_per_tile;
          const unsigned workers_tile_end = std::min(workers_tile_start+num_workers_per_tile,num_workers);

          // std::cout << "\nworkers_tile_start: " << workers_tile_start<< std::endl;
          // std::cout << "workers_tile_end: " << workers_tile_start<< std::endl;
          // std::cout << "repeated_sparse_spike_ids[ibatch][ithresh].shapeToString(): " << repeated_sparse_spike_ids[ibatch][ithresh].shapeToString() << std::endl;
          // std::cout << "repeated_sparse_spike_nums[ibatch][ithresh].shapeToString(): " << repeated_sparse_spike_ids[ibatch][ithresh].shapeToString() << std::endl;


          // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
          auto v2 = graph.addVertex(cs2, "LIFOutSpikesMultiThreshsCombine_singleThresh",
                                      {{"repeated_out_spikes_ids", repeated_sparse_spike_ids[ibatch][ithresh].slice(workers_tile_start, workers_tile_end, 0).flatten()},
                                      {"repeated_num_out_spikes", repeated_sparse_spike_nums[ibatch][ithresh].slice(workers_tile_start, workers_tile_end, 0)},
                                      {"out_spikes_ids", combined_repeated_sparse_spike_ids[ibatch][ithresh][itileSpreadFac]},
                                      {"num_out_spikes", combined_repeated_sparse_spike_nums[ibatch][ithresh][itileSpreadFac]},
                                      {"num_workers", workers_tile_end-workers_tile_start},
                                      {"sparse_size", sparse_size_combined},
                                      {"worker_size", sparse_size_to_use}});
          graph.setTileMapping(v2, tile_id);
          graph.setPerfEstimate(v2, 1);
        }
      }
    }
    worker_start = worker_end;
  }
  return {combined_repeated_sparse_spike_ids, combined_repeated_sparse_spike_nums}; 
}



std::vector<BatchedSparseSpikes> dense_to_sparse_spikes(poplar::Graph &graph, std::vector<poplar::Tensor> &dense_spikes,
                            const std::vector<std::vector<unsigned>> &tiles_to_use, const std::vector<unsigned> &sparse_sizes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  unsigned num_layers = dense_spikes.size();
  auto cs = graph.addComputeSet({dnai, "dense_to_sparse_spikes"});
  auto cs2 = graph.addComputeSet({dnai, "dense_to_sparse_spikes_combine"});
  std::vector<BatchedSparseSpikes> repeated_sprase_spikes;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    // // // TODO don't give program, but random offset tensors ?
    // // repeated_sprase_spikes.push_back(dense_to_sparse_spikes(graph, dense_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, dnai));
    // repeated_sprase_spikes.push_back(dense_to_sparse_spikes_multi_tile_spread(graph, dense_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, dnai));
    
    
    // TODO use the one or the other where it makes more sense depending on dense and sparse size of the layer
    repeated_sprase_spikes.push_back(dense_to_sparse_spikes_multi_tile_spread_with_combine(graph, dense_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, cs2, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  prog.add(poplar::program::Execute(cs2));
  return repeated_sprase_spikes;
}















BatchedSparseSpikes gen_sparse_spikes(poplar::Graph &graph, const poplar::Tensor &state, const poplar::Tensor &thresholds, 
                              const unsigned sparse_size, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){
  
  const unsigned num_neurons = state.dim(1);
  const unsigned batchsize = state.dim(0);
  const unsigned num_thresholds = thresholds.dim(0);
  const auto dtype = state.elementType();
  const unsigned num_tiles_state = get_num_tiles_of_mapping(graph.getTileMapping(state));
  const unsigned max_num_neurons_per_tile = get_max_elements_per_tile(graph.getTileMapping(state[0]));
  const unsigned sparse_size_combined = max_num_neurons_per_tile; // std::min(sparse_size, max_num_neurons_per_tile); // TODO adjust code
  std::cout << "\nmax_num_neurons_per_tile: " << max_num_neurons_per_tile << std::endl;

  poplar::Tensor combined_repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tiles_state, sparse_size_combined}, {dnai, "combinedRepeatedNeuronSpikeIds"});
  poplar::Tensor combined_repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tiles_state}, {dnai, "combinedRepeatedNeuronSpikeIds"});

  const auto numTiles = graph.getTarget().getNumTiles();
  auto neuronTileMapping = graph.getTileMapping(state[0], true);
  unsigned num_occupied_tiles{0};
  unsigned start_id{0};
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    const auto thisTileMap = neuronTileMapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }
    for (const auto &neuronRange: neuronTileMapping[tile]) {
      const auto numNeuronsThisThile = neuronRange.size();
      poplar::Tensor neuronStates = state.slice(neuronRange, 1);
      poplar::Tensor neuronThresholds = thresholds.slice(neuronRange, 1);

      poplar::Tensor tile_sparse_spike_ids = combined_repeated_sparse_spike_ids.slice(num_occupied_tiles, num_occupied_tiles+1, 2);
      poplar::Tensor tile_sparse_spike_nums = combined_repeated_sparse_spike_nums.slice(num_occupied_tiles, num_occupied_tiles+1, 2);

      graph.setTileMapping(tile_sparse_spike_ids, tile);
      graph.setTileMapping(tile_sparse_spike_nums, tile);

      for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        auto v = graph.addVertex(cs, poputil::templateVertex("GenSparseSpikesMultiThresh", dtype), // TODO could this be vectorized over the states ?
                                    {{"state", neuronStates[ibatch]},
                                    {"first_thresh", neuronThresholds[0]},
                                    {"second_thresh", neuronThresholds[1]},
                                    {"num_states", numNeuronsThisThile},
                                    {"start_id", start_id},
                                    {"repeated_sparse_spike_ids_thresh0", tile_sparse_spike_ids[ibatch][0].flatten()},
                                    {"repeated_sparse_spike_ids_thresh1", tile_sparse_spike_ids[ibatch][1].flatten()},
                                    {"repeated_sparse_spike_nums_thresh0", tile_sparse_spike_nums[ibatch][0][0]},
                                    {"repeated_sparse_spike_nums_thresh1", tile_sparse_spike_nums[ibatch][1][0]}});
        graph.setTileMapping(v, tile);
        graph.setPerfEstimate(v, 1);
      }
      start_id += numNeuronsThisThile;
    }
    ++num_occupied_tiles;
  }
  return {combined_repeated_sparse_spike_ids, combined_repeated_sparse_spike_nums}; 
}






std::vector<BatchedSparseSpikes> gen_sparse_spikes(poplar::Graph &graph,  const std::vector<poplar::Tensor> &state, 
                            const std::vector<poplar::Tensor> &thresholds, const std::vector<unsigned> &sparse_sizes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  unsigned num_layers = state.size();
  auto cs = graph.addComputeSet({dnai, "gen_sparse_spikes"});
  std::vector<BatchedSparseSpikes> repeated_sprase_spikes;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    repeated_sprase_spikes.push_back(gen_sparse_spikes(graph, state[ilay], thresholds[ilay], sparse_sizes[ilay], cs, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  return repeated_sprase_spikes;
}



BatchedSparseSpikes sparse_spikes_combine_stage1(poplar::Graph &graph, BatchedSparseSpikes &original_tile_sparse_spikes, const std::vector<unsigned> &tiles_to_use, const unsigned &sparse_size,
                            poplar::program::Sequence &prog, poplar::ComputeSet &cs1, poplar::ComputeSet &cs2, const poplar::DebugNameAndId &dnai = {}){
  

  poplar::Tensor orig_tile_spike_ids = original_tile_sparse_spikes.spike_ids;
  poplar::Tensor orig_tile_num_spikes = original_tile_sparse_spikes.num_spikes;


  const unsigned num_orig_tiles = orig_tile_spike_ids.dim(2);
  const unsigned num_neurons = num_orig_tiles*orig_tile_spike_ids.dim(3);
  const unsigned batchsize = orig_tile_spike_ids.dim(0);
  const unsigned num_thresholds = orig_tile_spike_ids.dim(1);
  const auto dtype = orig_tile_spike_ids.elementType();
  const unsigned num_tiles_to_use = tiles_to_use.size();
  const unsigned num_tile_spread_fac = num_tiles_to_use / (batchsize * num_thresholds);

  const unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  const unsigned num_workers = std::min(num_tile_spread_fac * num_workers_per_tile, num_orig_tiles);

  const unsigned max_num_states_per_worker = num_neurons / num_workers + ((num_neurons % num_workers) > 0);
  const unsigned max_num_states_per_tile = num_neurons / num_tile_spread_fac + ((num_neurons % num_tile_spread_fac) > 0);
  const unsigned max_num_origTiles_per_worker = num_orig_tiles / num_workers + ((num_orig_tiles % num_workers) > 0);
  const unsigned max_num_origTiles_per_tile = num_orig_tiles / num_tile_spread_fac + ((num_orig_tiles % num_tile_spread_fac) > 0);



  const bool sparseSmallerStatesPerWorker = sparse_size < max_num_states_per_worker;
  const bool sparseSmallerStatesPerTile = sparse_size < max_num_states_per_tile;
  const unsigned sparse_size_to_use = (sparseSmallerStatesPerWorker) ? sparse_size: max_num_states_per_worker;
  const unsigned sparse_size_combined = (sparseSmallerStatesPerTile) ? sparse_size: max_num_states_per_tile;

  std::cout << "\ndense_to_sparse_spikes_multi_tile_spread" << std::endl;
  std::cout << "num_tile_spread_fac: " << num_tile_spread_fac << std::endl;
  std::cout << "num_workers: " << num_workers << std::endl;
  std::cout << "max_num_states_per_worker: " << max_num_states_per_worker << std::endl;
  std::cout << "sparse_size_to_use: " << sparse_size_to_use << std::endl;
  std::cout << "sparseSmallerStatesPerWorker: " << sparseSmallerStatesPerWorker << std::endl;

  poplar::Tensor repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers, sparse_size_to_use}, {dnai, "repeatedNeuronSpikeIds"});
  poplar::Tensor repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_workers}, {dnai, "repeatedNeuronSpikeIds"});

  poplar::Tensor combined_repeated_sparse_spike_ids = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tile_spread_fac, sparse_size_combined}, {dnai, "combinedRepeatedNeuronSpikeIds"});
  poplar::Tensor combined_repeated_sparse_spike_nums = graph.addVariable(poplar::UNSIGNED_INT, {batchsize, num_thresholds, num_tile_spread_fac}, {dnai, "combinedRepeatedNeuronSpikeIds"});

  // TODO random offset for combine!
  // for random offset (more even sampling, less bias in selection of spikes?)
  poplar::Tensor reference = graph.addVariable(poplar::INT, {batchsize, num_thresholds, num_workers}, {dnai, "reference_tensor"});
  unsigned vertex_id{0};
  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
      graph.setTileMapping(reference[ibatch][ithresh], tiles_to_use[vertex_id]);
      ++vertex_id;
    }
  }
  // max_val not perfect and will slihgtly bias first neurons
  poplar::Tensor random_offset = poprand::uniform(graph, NULL, 0, reference, poplar::INT, 0, num_orig_tiles / num_workers, prog, {dnai, "randomInds"});

  size_t worker_start{0};
  size_t worker_end{0};
  for (unsigned iwor = 0; iwor < num_workers; ++iwor) {

    const unsigned itileSpreadFac = iwor / num_workers_per_tile;
    const unsigned iworPerTile = iwor % num_workers_per_tile;

    size_t numStatesThisWorker = num_orig_tiles / num_workers + ((num_orig_tiles % num_workers) > iwor); // TODO creates imbalance if spread over multiple tiles
    worker_end += numStatesThisWorker;
    std::cout << "numStatesThisWorker: " << numStatesThisWorker << std::endl;

    // unsigned vertex_id{0 + iwor/num_workers_per_tile};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      for (unsigned ithresh = 0; ithresh < num_thresholds; ++ithresh) {
        // unsigned tile_id{tiles_to_use[vertex_id]};
        unsigned tile_id{tiles_to_use[ibatch*num_thresholds*num_tile_spread_fac + ithresh*num_tile_spread_fac + itileSpreadFac]};
        
        graph.setTileMapping(repeated_sparse_spike_ids[ibatch][ithresh][iwor], tile_id);
        graph.setTileMapping(repeated_sparse_spike_nums[ibatch][ithresh][iwor], tile_id);        

        // // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
        // auto v1 = graph.addVertex(cs1, poputil::templateVertex("DenseToSparseSpikes", dtype),
        //                             {{"dense_spikes", dense_spikes_worker[ibatch][ithresh]},
        //                             {"num_dense_spikes", numStatesThisWorker},
        //                             {"sparse_size", sparse_size_to_use},
        //                             {"start_id", worker_start},
        //                             {"random_offset", random_offset[ibatch][ithresh][iwor]},
        //                             {"repeated_sparse_spike_ids", repeated_sparse_spike_ids[ibatch][ithresh][iwor]},
        //                             {"repeated_sparse_spike_nums", repeated_sparse_spike_nums[ibatch][ithresh][iwor]}});
        // graph.setTileMapping(v1, tile_id);
        // graph.setPerfEstimate(v1, 1);

        // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
        auto v1 = graph.addVertex(cs1, "LIFOutSpikesMultiThreshsCombine_singleThresh",
                                    {{"repeated_out_spikes_ids", orig_tile_spike_ids[ibatch][ithresh].slice(worker_start, worker_end, 0).flatten()},
                                    {"repeated_num_out_spikes", orig_tile_num_spikes[ibatch][ithresh].slice(worker_start, worker_end, 0)},
                                    {"out_spikes_ids", repeated_sparse_spike_ids[ibatch][ithresh][iwor]},
                                    {"num_out_spikes", repeated_sparse_spike_nums[ibatch][ithresh][iwor]},
                                    {"num_workers", worker_end-worker_start},
                                    {"sparse_size", sparse_size_to_use},
                                    {"worker_size", orig_tile_spike_ids.dim(3)}});
        graph.setTileMapping(v1, tile_id);
        graph.setPerfEstimate(v1, 1);

        if  (iworPerTile == 0){
          graph.setTileMapping(combined_repeated_sparse_spike_ids[ibatch][ithresh][itileSpreadFac], tile_id);
          graph.setTileMapping(combined_repeated_sparse_spike_nums[ibatch][ithresh][itileSpreadFac], tile_id);        

          const unsigned workers_tile_start = itileSpreadFac*num_workers_per_tile;
          const unsigned workers_tile_end = std::min(workers_tile_start+num_workers_per_tile,num_workers);

          // std::cout << "\nworkers_tile_start: " << workers_tile_start<< std::endl;
          // std::cout << "workers_tile_end: " << workers_tile_start<< std::endl;
          // std::cout << "repeated_sparse_spike_ids[ibatch][ithresh].shapeToString(): " << repeated_sparse_spike_ids[ibatch][ithresh].shapeToString() << std::endl;
          // std::cout << "repeated_sparse_spike_nums[ibatch][ithresh].shapeToString(): " << repeated_sparse_spike_ids[ibatch][ithresh].shapeToString() << std::endl;


          // TODO for not sparseSmallerStatesPerWorker could have a separate vertex (without checks)
          auto v2 = graph.addVertex(cs2, "LIFOutSpikesMultiThreshsCombine_singleThresh",
                                      {{"repeated_out_spikes_ids", repeated_sparse_spike_ids[ibatch][ithresh].slice(workers_tile_start, workers_tile_end, 0).flatten()},
                                      {"repeated_num_out_spikes", repeated_sparse_spike_nums[ibatch][ithresh].slice(workers_tile_start, workers_tile_end, 0)},
                                      {"out_spikes_ids", combined_repeated_sparse_spike_ids[ibatch][ithresh][itileSpreadFac]},
                                      {"num_out_spikes", combined_repeated_sparse_spike_nums[ibatch][ithresh][itileSpreadFac]},
                                      {"num_workers", workers_tile_end-workers_tile_start},
                                      {"sparse_size", sparse_size_combined},
                                      {"worker_size", sparse_size_to_use}});
          graph.setTileMapping(v2, tile_id);
          graph.setPerfEstimate(v2, 1);
        }
      }
    }
    worker_start = worker_end;
  }
  return {combined_repeated_sparse_spike_ids, combined_repeated_sparse_spike_nums}; 
}



std::vector<BatchedSparseSpikes> sparse_spikes_combine_stage1(poplar::Graph &graph, std::vector<BatchedSparseSpikes> &orig_tiles_sparse_spikes,
                            const std::vector<std::vector<unsigned>> &tiles_to_use, const std::vector<unsigned> &sparse_sizes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  unsigned num_layers = orig_tiles_sparse_spikes.size();
  auto cs = graph.addComputeSet({dnai, "sparse_spikes_combine_stage1"});
  auto cs2 = graph.addComputeSet({dnai, "sparse_spikes_combine_stage1_combine"});
  std::vector<BatchedSparseSpikes> repeated_sprase_spikes;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    // // // TODO don't give program, but random offset tensors ?
    // // repeated_sprase_spikes.push_back(dense_to_sparse_spikes(graph, dense_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, dnai));
    // repeated_sprase_spikes.push_back(dense_to_sparse_spikes_multi_tile_spread(graph, dense_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, dnai));
    
    
    // TODO use the one or the other where it makes more sense depending on dense and sparse size of the layer
    repeated_sprase_spikes.push_back(sparse_spikes_combine_stage1(graph, orig_tiles_sparse_spikes[ilay], tiles_to_use[ilay], sparse_sizes[ilay], prog, cs, cs2, dnai));
  }
  prog.add(poplar::program::Execute(cs));
  prog.add(poplar::program::Execute(cs2));
  return repeated_sprase_spikes;
}






void combine_repeated_sparse_spikes_multi_thresh(poplar::Graph &graph, BatchedSparseSpikes &repeated_sparse_spikes,
                            BatchedSparseSpikes &out_spikes, const std::vector<unsigned> &tiles_to_use,
                            poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai = {}){

  poplar::Tensor repeated_sparse_spike_ids = repeated_sparse_spikes.spike_ids;
  poplar::Tensor repeated_sparse_spike_nums = repeated_sparse_spikes.num_spikes;

  const unsigned batchsize = repeated_sparse_spike_nums.dim(0);
  const unsigned num_workers = repeated_sparse_spike_nums.dim(2);
  const unsigned dim_per_worker = repeated_sparse_spike_ids.dim(3);
  std::cout << " repeated_sparse_spike_ids.shapeToString(): " << repeated_sparse_spike_ids.shapeToString() << std::endl;


  for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
    auto v = graph.addVertex(cs, "LIFOutSpikesMultiThreshsCombine",
                                {{"repeated_out_spikes_ids", repeated_sparse_spike_ids[ibatch][0].flatten()},
                                {"repeated_out_spikes_ids_grads", repeated_sparse_spike_ids[ibatch][1].flatten()},
                                {"repeated_num_out_spikes_first", repeated_sparse_spike_nums[ibatch][0].flatten()},
                                {"repeated_num_out_spikes_second", repeated_sparse_spike_nums[ibatch][1].flatten()},
                                {"out_spikes_ids", out_spikes.spike_ids[ibatch]},
                                {"num_out_spikes", out_spikes.num_spikes[ibatch]},
                                {"num_workers", num_workers},
                                {"sparse_size", out_spikes.spike_ids.dim(1)},
                                {"worker_size", dim_per_worker}});
    graph.setTileMapping(v, tiles_to_use[ibatch]);
    graph.setPerfEstimate(v, 1);
  }
}

void combine_repeated_sparse_spikes_multi_thresh(poplar::Graph &graph, std::vector<BatchedSparseSpikes> &repeated_sparse_spikes,
                            std::vector<BatchedSparseSpikes> &out_spikes, const std::vector<std::vector<unsigned>> &tiles_to_use,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}){
  unsigned num_layers = repeated_sparse_spikes.size();
  auto cs = graph.addComputeSet({dnai, "combine_repeated_sparse_spikes_multi_thresh"});
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    combine_repeated_sparse_spikes_multi_thresh(graph, repeated_sparse_spikes[ilay], out_spikes[ilay], tiles_to_use[ilay], cs);  
  }
  prog.add(poplar::program::Execute(cs));
}





std::vector<unsigned> determine_layerwise_tile_spread_factor(const unsigned &desired_per_worker_size, const unsigned num_possible_threads, const std::vector<std::vector<unsigned>> layer_ids_per_ipu, std::vector<unsigned> num_neurons, std::vector<unsigned> sparse_sizes, std::vector<unsigned> batchsize, std::vector<unsigned> num_thresholds){
  const unsigned num_layers = num_neurons.size();
  std::vector<unsigned> tile_spread_fac(num_layers);
  for (unsigned iipu=0; iipu<layer_ids_per_ipu.size(); ++iipu){
    for (auto &layer_id: layer_ids_per_ipu[iipu]){

      // // TODO rewrite this here ! using desired_per_worker_size
      // const unsigned denseSpraseRatio = num_neurons[layer_id] / sparse_sizes[layer_id];
      // const unsigned num_workers = std::min(denseSpraseRatio, num_possible_threads); 

      // TODO improve here!
      const unsigned num_workers = num_neurons[layer_id] / desired_per_worker_size + ((num_neurons[layer_id] % desired_per_worker_size) > (desired_per_worker_size / 2));
      const unsigned min_workers_per_tile{2};
      const unsigned min_num_workers{1};
      tile_spread_fac[layer_id] = std::max(num_workers / num_possible_threads + ((num_workers % num_possible_threads) >= min_workers_per_tile), min_num_workers);
    }
  }
  return tile_spread_fac;
}


std::vector<std::vector<std::vector<unsigned>>> determine_tile_mapping(const std::vector<std::vector<unsigned>> &layer_ids_per_ipu, const unsigned &num_tiles_per_ipu, const std::vector<unsigned> &layerwise_tile_spread_factor, 
                                                                    const std::vector<unsigned> &batchsize, const std::vector<unsigned> &num_thresholds) {

  unsigned num_ipus = layer_ids_per_ipu.size();
  unsigned num_layers = layerwise_tile_spread_factor.size();
  std::vector<std::vector<unsigned>> tiles_to_use_dense_to_sparse(num_layers);
  std::vector<std::vector<unsigned>> tiles_to_use_combine(num_layers);

  std::vector<unsigned> num_tiles_to_use_per_ipu;
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    unsigned num_tiles_to_use_this_ipu{0};
    for (auto &layer_id: layer_ids_per_ipu[iipu]){
      num_tiles_to_use_this_ipu += batchsize[layer_id] * num_thresholds[layer_id] * layerwise_tile_spread_factor[layer_id];
    }
    if (num_tiles_per_ipu < num_tiles_to_use_this_ipu){
      // TODO How to thorw/raise warning? stderr ?
      std::cout << "WARNING: Potentially non-optimal performance in `genBatchedLIFOutSpikes2ThreshsNeuronSpikes`."
                      "For optimal performance make sure that `tiles_to_use_this_ipu < num_tiles_per_ipu`. Got: " << "num_tiles_per_ipu = " << num_tiles_per_ipu << "num_tiles_to_use_this_ipu = " << num_tiles_to_use_this_ipu << " ."
                      " To achieve that you should probably choose a smaller batchsize or fewer layers per IPU. (Or recompile with larger `desired_per_worker_size`.)" << std::endl;
    }
    num_tiles_to_use_per_ipu.push_back(num_tiles_to_use_this_ipu);
  }

  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    const double mul_fac = (double)num_tiles_per_ipu / (double)num_tiles_to_use_per_ipu[iipu];
    const unsigned tile_offset = iipu * num_tiles_per_ipu;

    unsigned num_vertices_iiter{0};
    for (auto &layer_id: layer_ids_per_ipu[iipu]){
      std::vector<unsigned> tiles_to_use_dense_to_sparse_this_layer;
      std::vector<unsigned> tiles_to_use_combine_this_layer;
      unsigned vertex_id_iiter_this_layer{0};
      for (unsigned ibatch=0; ibatch<batchsize[layer_id]; ++ibatch){
        for (unsigned ithresh=0; ithresh<num_thresholds[layer_id]; ++ithresh){ // TODO thresholds on tiles next to each other ?
          for (unsigned ispreadfac=0; ispreadfac<layerwise_tile_spread_factor[layer_id]; ++ispreadfac){ // TODO thresholds on tiles next to each other ?
            unsigned tile_id = mul_fac * (double)num_vertices_iiter + tile_offset;
            tiles_to_use_dense_to_sparse_this_layer.push_back(tile_id);
            ++num_vertices_iiter;
            ++vertex_id_iiter_this_layer;
          }
        }
        tiles_to_use_combine_this_layer.push_back(tiles_to_use_dense_to_sparse_this_layer[vertex_id_iiter_this_layer - (num_thresholds[layer_id] * layerwise_tile_spread_factor[layer_id])]);
      }
      printVector(tiles_to_use_dense_to_sparse_this_layer);
      printVector(tiles_to_use_combine_this_layer);
      tiles_to_use_dense_to_sparse[layer_id] = tiles_to_use_dense_to_sparse_this_layer;
      tiles_to_use_combine[layer_id] = tiles_to_use_combine_this_layer;
    }
  }
  return {tiles_to_use_dense_to_sparse, tiles_to_use_combine};
}

std::vector<std::vector<std::vector<unsigned>>> determine_tile_mapping_sinegleTilePerOp(const std::vector<std::vector<unsigned>> &layer_ids_per_ipu, const unsigned &num_tiles_per_ipu, const unsigned &num_layers, const unsigned &batchsize, const unsigned &num_thresholds) {

  unsigned num_ipus = layer_ids_per_ipu.size();
  std::vector<std::vector<unsigned>> tiles_to_use_dense_to_sparse(num_layers);
  std::vector<std::vector<unsigned>> tiles_to_use_combine(num_layers);
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    std::cout << "001000" << std::endl;
    const unsigned num_layers_this_ipu = layer_ids_per_ipu[iipu].size();
    const unsigned num_tiles_to_use_this_ipu = batchsize * num_thresholds * num_layers_this_ipu;  
    const double mul_fac = (double)num_tiles_per_ipu / (double)num_tiles_to_use_this_ipu;
    const unsigned tile_offset = iipu * num_tiles_per_ipu;

    std::cout << "num_layers_this_ipu: " << num_layers_this_ipu << std::endl;
    std::cout << "num_tiles_to_use_this_ipu: " << num_tiles_to_use_this_ipu << std::endl;
    std::cout << "mul_fac: " << mul_fac << std::endl;
    std::cout << "tile_offset: " << tile_offset << std::endl;

    if (num_tiles_per_ipu < num_tiles_to_use_this_ipu){
      // TODO How to thorw/raise warning? stderr ?
      std::cout << "WARNING: Potentially non-optimal performance in `genBatchedLIFOutSpikes2ThreshsNeuronSpikes`."
                      "For optimal performance make sure that `batchsize*num_thresholds*num_layers_cuurent_ipu < num_tiles_per_ipu`."
                      " To achieve that you should probably choose a smaller batchsize or fewer layers per IPU." << std::endl;
    }

    unsigned num_vertices_iiter{0};
    for (auto &layer_id: layer_ids_per_ipu[iipu]){
      std::cout << "001100" << std::endl;
      // unsigned batchsize = state[layer_id].dim(0);
      // unsigned num_thresholds = thresholds[layer_id].dim(0);

      std::vector<unsigned> tiles_to_use_dense_to_sparse_this_layer;
      std::vector<unsigned> tiles_to_use_combine_this_layer;
      unsigned vertex_id_iiter_this_layer{0};
      for (unsigned ibatch=0; ibatch<batchsize; ++ibatch){
        for (unsigned ithresh=0; ithresh<num_thresholds; ++ithresh){ // TODO thresholds on tiles next to each other ?
          unsigned tile_id = mul_fac * (double)num_vertices_iiter + tile_offset;
          tiles_to_use_dense_to_sparse_this_layer.push_back(tile_id);
          ++num_vertices_iiter;
          ++vertex_id_iiter_this_layer;
        }
        tiles_to_use_combine_this_layer.push_back(tiles_to_use_dense_to_sparse_this_layer[vertex_id_iiter_this_layer-num_thresholds]);
      }
      printVector(tiles_to_use_dense_to_sparse_this_layer);
      printVector(tiles_to_use_combine_this_layer);
      tiles_to_use_dense_to_sparse[layer_id] = tiles_to_use_dense_to_sparse_this_layer;
      tiles_to_use_combine[layer_id] = tiles_to_use_combine_this_layer;
    }
  }
  return {tiles_to_use_dense_to_sparse, tiles_to_use_combine};
}


void genBatchedLIFOutSpikes2ThreshsNeuronSpikes(poplar::Graph &graph, std::vector<poplar::Tensor> &state, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, 
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {

  std::cout << "\nSTART genBatchedLIFOutSpikes2ThreshsNeuronSpikes" << std::endl;


  size_t num_layers = state.size();

  // std::vector<unsigned> sparse_sizes;
  // std::transform(out_spikes.begin(), out_spikes.end(), std::back_inserter(sparse_sizes), [](BatchedSparseSpikes &sparseSpikes){return sparseSpikes.spike_ids.dim(1);});

  const unsigned num_tiles_per_ipu{graph.getTarget().getTilesPerIPU()};
  const unsigned num_ipus{graph.getTarget().getNumIPUs()};
  const std::vector<std::vector<unsigned>> layer_ids_per_ipu(get_layer_ids_per_ipu(graph, state));

 
  // TODO just use the sparse size ? or some factor of it if sparse / dense too small, meaning too many states per worker?
  const unsigned desired_per_worker_size = 64; 
  std::vector<unsigned> num_neurons; 
  std::vector<unsigned> sparse_sizes; 
  std::vector<unsigned> batchsize; 
  std::vector<unsigned> num_thresholds;
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    num_neurons.push_back(state[ilay].dim(1));
    sparse_sizes.push_back(out_spikes[ilay].spike_ids.dim(1));
    batchsize.push_back(state[ilay].dim(0));
    num_thresholds.push_back(thresholds[ilay].dim(0));
  }
  std::cout << "\nDONE 00100" << std::endl;
  std::vector<unsigned> layerwise_tile_spread_factor = determine_layerwise_tile_spread_factor(desired_per_worker_size, graph.getTarget().getNumWorkerContexts(), 
                                                                                          layer_ids_per_ipu, num_neurons, sparse_sizes, batchsize, num_thresholds);
  printVector(layerwise_tile_spread_factor);
  std::cout << "\nDONE determine_layerwise_tile_spread_factor" << std::endl;
  auto vertex_tile_mappings = determine_tile_mapping(layer_ids_per_ipu, num_tiles_per_ipu, layerwise_tile_spread_factor, batchsize, num_thresholds);
  std::vector<std::vector<unsigned>> tiles_to_use_dense_to_sparse = vertex_tile_mappings[0];
  std::vector<std::vector<unsigned>> tiles_to_use_combine = vertex_tile_mappings[1];
  std::cout << "\nDONE determine_tile_mapping" << std::endl;

  // const unsigned batchsize = state[0].dim(0); // TODO do it layerwise inside the loop
  // const unsigned num_thresholds = thresholds[0].dim(0);
  // auto vertex_tile_mappings = determine_tile_mapping_sinegleTilePerOp(layer_ids_per_ipu, num_tiles_per_ipu, num_layers, batchsize, num_thresholds);
  // std::vector<std::vector<unsigned>> tiles_to_use_dense_to_sparse = vertex_tile_mappings[0];
  // std::vector<std::vector<unsigned>> tiles_to_use_combine = vertex_tile_mappings[1];


  std::cout << "\nDONE setup" << std::endl;
  std::vector<poplar::Tensor> dense_spikes = gen_dense_spikes(graph, state, thresholds, prog, {dnai, "gen_dense_spikes"});
  std::cout << "\nDONE gen_dense_spikes" << std::endl;
  std::vector<BatchedSparseSpikes> repeated_sparse_spikes = dense_to_sparse_spikes(graph, dense_spikes, tiles_to_use_dense_to_sparse, sparse_sizes, prog, {dnai, "dense_to_sparse_spikes"});
  std::cout << "\nDONE dense_to_sparse_spikes" << std::endl;
  combine_repeated_sparse_spikes_multi_thresh(graph, repeated_sparse_spikes, out_spikes, tiles_to_use_combine, prog, {dnai, "combine_repeated_sparse_spikes_multi_thresh"});
  std::cout << "\nDONE combine_repeated_sparse_spikes_multi_thresh" << std::endl;

  // std::vector<BatchedSparseSpikes> repeated_sparse_spikes_orig_tiles = gen_sparse_spikes(graph, state, thresholds, sparse_sizes, prog, {dnai, "gen_sparse_spikes"});
  // std::cout << "\nDONE gen_sparse_spikes" << std::endl;
  // std::vector<BatchedSparseSpikes> repeated_sparse_spikes = sparse_spikes_combine_stage1(graph, repeated_sparse_spikes_orig_tiles, tiles_to_use_dense_to_sparse, sparse_sizes, prog, {dnai, "sparse_spikes_combine_stage1"});
  // std::cout << "\nDONE sparse_spikes_combine_stage1" << std::endl;
  // combine_repeated_sparse_spikes_multi_thresh(graph, repeated_sparse_spikes, out_spikes, tiles_to_use_combine, prog, {dnai, "combine_repeated_sparse_spikes_multi_thresh"});
  // std::cout << "\nDONE combine_repeated_sparse_spikes_multi_thresh" << std::endl;
}

void performLIFStepFworwardPassInPlace(poplar::Graph &graph, std::vector<poplar::Tensor> &weights, std::vector<poplar::Tensor> &state, std::vector<BatchedSparseSpikes> &inp_spikes, 
                            std::vector<poplar::Tensor> &decay_constants, std::vector<poplar::Tensor> &oneMinus_decay_constants, std::vector<poplar::Tensor> &thresholds, 
                            std::vector<BatchedSparseSpikes> &out_spikes, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  performBatchedLIFStateUpdateInPlace(graph, weights, state, inp_spikes, decay_constants, oneMinus_decay_constants, thresholds, prog, dnai);
  // // genBatchedLIFOutSpikesTopK(graph, state, thresholds, out_spikes, prog, dnai);
  // // genBatchedLIFOutSpikes2Threshs(graph, state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikes2ThreshsMutliWorker(graph, state, thresholds, out_spikes, prog, dnai);
  // // genBatchedLIFOutSpikesOnlySpikes(graph, state, thresholds, out_spikes, prog, dnai);
  
  genBatchedLIFOutSpikes2ThreshsNeuronSpikes(graph, state, thresholds, out_spikes, prog, dnai);
  // genBatchedLIFOutSpikesMultiThreshBatchIds(graph, state, thresholds, out_spikes, prog, dnai);
}


//---------------------------------------------- Backward functions -----------------------------------------

// !!! TODO !!! think about tile mapping !!! 
// !!! TODO !!! maybe rewrite function to local version. every state is conditionally updated ?
void calcLIFStateGrad(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, std::vector<poplar::Tensor> &fwdState, 
                            const std::vector<poplar::Tensor> &decay_constants, const std::vector<poplar::Tensor> &thresholds, const std::vector<BatchedSparseSpikes> &fwdOutSpikes,
                            std::vector<poplar::Tensor> &dLdState, const std::vector<poplar::Tensor> &dLdoutSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  auto cs = graph.addComputeSet({dnai, "calcLIFStateOutGrad"});
  size_t num_layers = weights.size();

  // TODO change allocation, see genBatchedLIFOutSpikes2ThreshsNeuronSpikes
  const std::vector<unsigned> layers_to_ipu_mapping(get_tensor_ipu_id(graph, fwdState));
  const std::vector<unsigned> layer_ids_per_ipu(get_relative_layer_id_on_ipu(layers_to_ipu_mapping));
  unsigned num_tiles_per_ipu = graph.getTarget().getTilesPerIPU();

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = weights[ilay].elementType();
    size_t batchsize = fwdState[ilay].dim(0);
    const unsigned layer_vertex_start_tile = determine_start_tile_spike_gen(layers_to_ipu_mapping[ilay], layer_ids_per_ipu[ilay], batchsize, num_tiles_per_ipu);

    popops::mulInPlace(graph, dLdState[ilay], decay_constants[ilay].expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);
    // mulInPlace_custom(graph, dLdState[ilay], decay_constants[ilay], prog, dnai);

    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto v = graph.addVertex(cs, poputil::templateVertex("LIFStateOutGrad", dtype),
                                {{"fwdState", fwdState[ilay][ibatch]},
                                {"thresholds", thresholds[ilay]},
                                {"dLdoutSpikes", dLdoutSpikes[ilay][ibatch]},
                                {"fwd_out_spikes_ids", fwdOutSpikes[ilay].spike_ids[ibatch]},
                                //  {"dLdState_inp", dLdState[ibatch]},
                                {"end", fwdOutSpikes[ilay].num_spikes[ibatch][1]},
                                //  {"dLdState", dLdState[ibatch]}}); 
                                {"dLdState", dLdState[ilay][ibatch]}});
      // !!! TODO !!! totally bogus tile mapping, must be improved
      // should be based on state mapping
      // graph.setTileMapping(v, (ibatch+1)*32); 
      // graph.setTileMapping(v, 1471-ibatch-batchsize*ilay); 
      graph.setTileMapping(v, layer_vertex_start_tile+ibatch); 
      // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
      graph.setPerfEstimate(v, 1);
    }
  }
  prog.add(poplar::program::Execute(cs));
}


void calcLIFStateGrad_stateWise(poplar::Graph &graph, const std::vector<poplar::Tensor> &fwdState, const std::vector<poplar::Tensor> &decay_constants, 
                            const std::vector<poplar::Tensor> &thresholds, const std::vector<BatchedSparseSpikes> &fwdOutSpikes,
                            std::vector<poplar::Tensor> &dLdState, const std::vector<poplar::Tensor> &dLdoutSpikes,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {

  // // bring spike tensor to tiles
  // std::vector<BatchedSparseSpikes> fwdOutSpikes_tileReplicated;
  // std::vector<poplar::Tensor> spike_grads_tileReplicated;
  size_t num_layers = fwdState.size();
  // for (unsigned ilay=0; ilay<num_layers; ++ilay){
  //   auto tileMapping = graph.getTileMapping(dLdState[ilay][0], true);
  //   poplar::Tensor fwdOut_ids_replicated = replicate_and_alloc_tensor(graph, fwdOutSpikes[ilay].spike_ids, tileMapping, prog, {dnai, "create_fwdOut_ids_replicated"});
  //   poplar::Tensor fwdOut_nums_replicated = replicate_and_alloc_tensor(graph, fwdOutSpikes[ilay].num_spikes.slice(1, 2, 1), tileMapping, prog, {dnai, "create_fwdOut_nums_replicated"});
  //   poplar::Tensor spike_grads_repl = replicate_and_alloc_tensor(graph, dLdoutSpikes[ilay], tileMapping, prog, {dnai, "create_spike_grads_replicated"});
  //   BatchedSparseSpikes fwdOuts_replicated = {fwdOut_ids_replicated, fwdOut_nums_replicated};
  //   fwdOutSpikes_tileReplicated.push_back(fwdOuts_replicated);
  //   spike_grads_tileReplicated.push_back(spike_grads_repl);
  // }

  auto cs = graph.addComputeSet({dnai, "calcLIFStateGrad_stateWise"});

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto dtype = fwdState[ilay].elementType();
    size_t batchsize = fwdState[ilay].dim(0);

    popops::mulInPlace(graph, dLdState[ilay], decay_constants[ilay].expand({0}).upsample(batchsize, 0, poplar::UpsampleMethod::REPEAT), prog, dnai);

    auto neuronTileMapping = graph.getTileMapping(fwdState[ilay][0], true);
    const auto numTiles = graph.getTarget().getNumTiles();
    unsigned num_occupied_tiles{0};
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      // If a tile contains no elements of the tensor then do not create any
      // vertices for it.
      const auto thisTileMap = neuronTileMapping[tile];
      if (thisTileMap.empty()) {
        continue;
      }

      std::cout << "fwdState[ilay]: " << fwdState[ilay].shapeToString() << std::endl;
      std::cout << "dLdState[ilay]: " << dLdState[ilay].shapeToString() << std::endl;
      std::cout << "thresholds[ilay]: " << thresholds[ilay].shapeToString() << std::endl;
      // std::cout << "fwdOutSpikes_tileReplicated[ilay].spike_ids: " << fwdOutSpikes_tileReplicated[ilay].spike_ids.shapeToString() << std::endl;
      // std::cout << "fwdOutSpikes_tileReplicated[ilay].num_spikes: " << fwdOutSpikes_tileReplicated[ilay].num_spikes.shapeToString() << std::endl;

      for (const auto &neuronRange: neuronTileMapping[tile]) {
        std::cout << "neuron_start_id: " << neuronRange.lower()  << std::endl;
        std::cout << "neuron_end_id: " << neuronRange.upper() << std::endl;
        std::cout << "num_neurons: " << neuronRange.size() << std::endl;

        poplar::Tensor neuronStates = fwdState[ilay].slice(neuronRange, 1);
        poplar::Tensor neurondLdStates = dLdState[ilay].slice(neuronRange, 1);
        poplar::Tensor neuronThresholds = thresholds[ilay].slice(neuronRange);
        for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
          auto v = graph.addVertex(cs, poputil::templateVertex("calcLIFStateGrad_stateWise", dtype),
                                    // {{"weights", weights[ilay][neuronId]},
                                    {{"fwdState", neuronStates[ibatch]},
                                    // {"fwd_spikes_ids", fwdOutSpikes_tileReplicated[ilay].spike_ids[num_occupied_tiles][ibatch]},
                                    // {"fwd_num_spikes", fwdOutSpikes_tileReplicated[ilay].num_spikes[num_occupied_tiles][ibatch][0]},
                                    // {"spike_grads", spike_grads_tileReplicated[ilay][num_occupied_tiles][ibatch]},
                                    {"fwd_spikes_ids", fwdOutSpikes[ilay].spike_ids[ibatch]},
                                    {"fwd_num_spikes", fwdOutSpikes[ilay].num_spikes[ibatch][1]},
                                    {"spike_grads", dLdoutSpikes[ilay][ibatch]},
                                    {"thresholds", neuronThresholds},
                                    {"neuron_start_id", neuronRange.lower()},
                                    {"neuron_end_id", neuronRange.upper()},
                                    {"num_neurons", neuronRange.size()},
                                    {"dLdState", neurondLdStates[ibatch]}});
          graph.setTileMapping(v, tile);
          // Provide a cycle count estimate for the profiler. // TODO make educated guess/provide equation
          graph.setPerfEstimate(v, 1);
        }
      }
      ++num_occupied_tiles;
    }
  }
  prog.add(poplar::program::Execute(cs));
}




void poplarReduceInpSpikesGrad(poplar::Graph &graph, const unsigned &num_layers, std::vector<poplar::Tensor> &dLdx_vec, std::vector<poplar::Tensor> &dLdInpSpikes,
                                  poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){

  // std::string operation = "ADD";
  popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 
  
  // for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
  //   // reduceWithOutput(graph, dLdx_vec[ilay], dLdInpSpikes[ilay], {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  //   auto temp = reduce(graph, dLdx_vec[ilay], {0}, reduceParams, prog, {dnai, "add rowwise inpSpikeGrads"});
  //   prog.add(poplar::program::Copy(temp, dLdInpSpikes[ilay]));
  //   // prog.add(poplar::program::Copy(dLdx_vec[ilay][0], dLdInpSpikes[ilay]));
  // }

  unsigned num_ipus = graph.getTarget().getNumIPUs();
  if (num_ipus > 0){

    std::vector<poplar::Graph> ipu_to_virtualGraph;
    size_t num_tiles_per_ipu = graph.getTarget().getTilesPerIPU();
    for (unsigned iipu=0; iipu<num_ipus; ++iipu){
      unsigned tile_offset = (iipu == 0) ? 1 : 0;
      ipu_to_virtualGraph.push_back(graph.createVirtualGraph(num_tiles_per_ipu * iipu + tile_offset, num_tiles_per_ipu * (iipu+1)));
    }
    std::vector<unsigned> dLdx_vec_to_ipu_id = get_ipu_ids_from_tensor_vec(graph, dLdx_vec);
    std::vector<std::vector<unsigned>> ipu_layer_ids(num_ipus);
    for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
      ipu_layer_ids[dLdx_vec_to_ipu_id[ilay]].push_back(ilay);
    }
    for (unsigned ipu_id=0; ipu_id<num_ipus; ++ipu_id){
      if (ipu_layer_ids[ipu_id].size()>0){
        std::vector<poplar::Tensor> reduce_outs_this_ipu;
        std::vector<popops::SingleReduceOp> single_reduce_ops;
        for (unsigned &layer_id: ipu_layer_ids[ipu_id]){
          reduce_outs_this_ipu.push_back(dLdInpSpikes[layer_id]);
          single_reduce_ops.push_back(popops::SingleReduceOp(dLdx_vec[layer_id], {0}, reduceParams, "single_reduce_rowwise_inpSpikeGrads"));
        }
        reduceMany(ipu_to_virtualGraph[ipu_id], single_reduce_ops, reduce_outs_this_ipu, prog, {dnai, "add_rowwise_inpSpikeGrads"});
      }
    }

  } else {
    std::vector<popops::SingleReduceOp> single_reduce_ops;
    for (unsigned ilay=0; ilay<num_layers-1; ++ilay){
      single_reduce_ops.push_back(
        popops::SingleReduceOp(dLdx_vec[ilay], {0}, reduceParams, "single_reduce_rowwise_inpSpikeGrads")
      );
    }
    std::vector<poplar::Tensor> reduce_outs;
    std::transform(dLdInpSpikes.begin(), dLdInpSpikes.end(), std::back_inserter(reduce_outs), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
    reduceMany(graph, single_reduce_ops, reduce_outs, prog, {dnai, "add_rowwise_inpSpikeGrads"});
  }
}


poplar::Tensor customReduceSparseInpSpikesGrad_stage1_layer(poplar::Graph &graph, poplar::Tensor &dLdx, poplar::Tensor &replicated_numGrads, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai){
  
  // // dLdx_reallocated.shape() = {occupied_tiles, exchangeBatchSize, sparse_size, exchangeGropuSize}
  // auto [start_tile, end_tile, is_contiguous] = get_start_end_is_contigious(graph, dLdx_reduced);

  // TODO not sure whether to do in here or give dLdx_reduced_ilay as input
  unsigned num_occupied_tiles = dLdx.dim(0);
  unsigned batchsize = dLdx.dim(1);
  unsigned sparse_size = dLdx.dim(2);
  auto dtype = dLdx.elementType();
  poplar::TargetType target_type = graph.getTarget().getTargetType();

  std::cout << "num_occupied_tiles: " << num_occupied_tiles << std::endl;
  std::cout << "batchsize: " << batchsize << std::endl;
  std::cout << "sparse_size: " << sparse_size << std::endl;
  std::cout << "dLdx.shapeToString(): " << dLdx.shapeToString() << std::endl;
  std::cout << "replicated_numGrads.shapeToString(): " << replicated_numGrads.shapeToString() << std::endl;

  unsigned defaultExchangeGroupSize{batchsize/2};
  unsigned exchangeGroupSize = std::min(defaultExchangeGroupSize, num_occupied_tiles);
  unsigned numExchangeGroups = num_occupied_tiles / exchangeGroupSize;
  unsigned exchangeBatchSize = batchsize / exchangeGroupSize + ((batchsize % exchangeGroupSize) > 0);
  // if (batchsize % exchangeGroupSize){
  //   throw poputil::poplibs_error("Currently only batchsizes that are a multiple of 8 are supprted for the `customReduceSparseInpSpikesGrad_stage1` operation.");
  // }
  bool lastExchangeGroupLarger = num_occupied_tiles % exchangeGroupSize;

  std::cout << "exchangeGroupSize: " << exchangeGroupSize << std::endl;
  std::cout << "numExchangeGroups: " << numExchangeGroups << std::endl;
  std::cout << "exchangeBatchSize: " << exchangeBatchSize << std::endl;
  std::cout << "lastExchangeGroupLarger: " << lastExchangeGroupLarger << std::endl;

  std::vector<unsigned> tensor_tile_ids = get_tensor_tile_ids(graph, dLdx);

  printVector(tensor_tile_ids);

  // poplar::Tensor dLdx_reduced = graph.addVariable(dtype, {, sparseSize, }, {dnai, "customReduce_stage1/dLdx_realloc"});
  poplar::Tensor dLdx_reduced_ilay = graph.addVariable(dtype, {numExchangeGroups, batchsize, sparse_size}, {dnai, "customReduce_stage1/dLdx_reduce"});

  unsigned start_id{0};
  unsigned end_id{0};

  for (unsigned iex=0; iex<(numExchangeGroups-lastExchangeGroupLarger); ++iex){
    end_id = start_id+exchangeGroupSize;
    std::cout << "start_id: " << start_id << ", end_id: " << end_id << std::endl;
    auto dLdx_exchangeGroundSlice = dLdx.slice(start_id, end_id, 0).dimShuffle({1,2,0});
    
    // std::vector<unsigned> batch_id_to_tiles_ilay;
    // std::vector<unsigned> batch_id_to_row_id_ilay;
    for (unsigned ibatch=0; ibatch<batchsize; ++ibatch){
      unsigned row_id{ibatch / exchangeBatchSize + start_id};
      unsigned tile_id = tensor_tile_ids[row_id];
      // batch_id_to_row_id_ilay.push_back(row_id);
      // batch_id_to_tiles_ilay.push_back(tile_id);

      // std::cout << "row_id: " << row_id << ", tile_id: " << tile_id << std::endl;

      graph.setTileMapping(dLdx_reduced_ilay[iex][ibatch], tile_id);

      for (unsigned isparse=0; isparse<sparse_size; ++isparse){

        if ((target_type == poplar::TargetType::IPU) && (exchangeGroupSize == 8) && (dtype == poplar::FLOAT)) {
          auto v = graph.addVertex(cs, "CustomSparseReduceStage1FloatSIMD8",
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    // {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        } else if ((target_type == poplar::TargetType::IPU) && (exchangeGroupSize == 5) && (dtype == poplar::FLOAT)) { // TODO doesn't do anything auto vec seems to work
          auto v = graph.addVertex(cs, "CustomSparseReduceStage1FloatSIMD5",
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    // {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        } else {
          auto v = graph.addVertex(cs, poputil::templateVertex("CustomSparseReduceStage1", dtype),
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        }
      }
      // std::cout << std::endl;
    }
    start_id = end_id;
  }

  if (lastExchangeGroupLarger){
    unsigned end_id = dLdx.dim(0);
    std::cout << "lastExchangeGroupLarger" << std::endl;
    std::cout << "start_id: " << start_id << ", end_id: " << end_id << std::endl;
    unsigned iex = numExchangeGroups-1;
    unsigned exchangeGroupSizeLast = end_id-start_id;
    unsigned exchangeBatchSizeLast = batchsize / exchangeGroupSizeLast + ((batchsize % exchangeGroupSizeLast) > 0);

    std::cout << "exchangeGroupSizeLast: " << exchangeGroupSizeLast << std::endl;
    std::cout << "exchangeBatchSizeLast: " << exchangeBatchSizeLast << std::endl;

    auto dLdx_exchangeGroundSlice = dLdx.slice(start_id, end_id, 0).dimShuffle({1,2,0});

    for (unsigned ibatch=0; ibatch<batchsize; ++ibatch){
      unsigned row_id{ibatch / exchangeBatchSizeLast + start_id};
      unsigned tile_id = tensor_tile_ids[row_id];
      
      std::cout << "row_id: " << row_id << ", tile_id: " << tile_id << std::endl;

      graph.setTileMapping(dLdx_reduced_ilay[iex][ibatch], tile_id);

      for (unsigned isparse=0; isparse<sparse_size; ++isparse){
        auto v = graph.addVertex(cs, poputil::templateVertex("CustomSparseReduceStage1", dtype),
                                  {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                  {"num_grads", replicated_numGrads[row_id][ibatch]},
                                  {"isparse", isparse},
                                  {"end", exchangeGroupSizeLast},
                                  {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
        graph.setTileMapping(v, tile_id); 
        graph.setPerfEstimate(v, 1);
      }
    }
  }
  return dLdx_reduced_ilay;
}

poplar::Tensor customReduceSparseInpSpikesGrad_stage1_layer_evenAlloc(poplar::Graph &graph, poplar::Tensor &dLdx, poplar::Tensor &replicated_numGrads, const unsigned &start_tile, const unsigned &num_tiles_to_use, poplar::ComputeSet &cs, const poplar::DebugNameAndId &dnai){
  
  // // dLdx_reallocated.shape() = {occupied_tiles, exchangeBatchSize, sparse_size, exchangeGropuSize}
  // auto [start_tile, end_tile, is_contiguous] = get_start_end_is_contigious(graph, dLdx_reduced);

  // TODO not sure whether to do in here or give dLdx_reduced_ilay as input
  unsigned num_occupied_tiles = dLdx.dim(0);
  // unsigned num_occupied_tiles = num_tiles_to_use;
  unsigned batchsize = dLdx.dim(1);
  unsigned sparse_size = dLdx.dim(2);
  auto dtype = dLdx.elementType();
  poplar::TargetType target_type = graph.getTarget().getTargetType();

  std::cout << "num_occupied_tiles: " << num_occupied_tiles << std::endl;
  std::cout << "batchsize: " << batchsize << std::endl;
  std::cout << "sparse_size: " << sparse_size << std::endl;
  std::cout << "dLdx.shapeToString(): " << dLdx.shapeToString() << std::endl;
  std::cout << "replicated_numGrads.shapeToString(): " << replicated_numGrads.shapeToString() << std::endl;

  unsigned defaultExchangeGroupSize{batchsize/2};
  unsigned exchangeGroupSize = std::min(defaultExchangeGroupSize, num_occupied_tiles);
  unsigned numExchangeGroups = num_occupied_tiles / exchangeGroupSize;
  unsigned exchangeBatchSize = batchsize / exchangeGroupSize + ((batchsize % exchangeGroupSize) > 0);

  // if (batchsize % defaultExchangeBatchSize){
  //   throw poputil::poplibs_error("Currently only batchsizes that are a multiple of 2 are supprted for the `customReduceSparseInpSpikesGrad_stage1_layer_evenAlloc` operation.");
  // }

  bool lastExchangeGroupLarger = num_occupied_tiles % exchangeGroupSize;

  std::cout << "exchangeGroupSize: " << exchangeGroupSize << std::endl;
  std::cout << "numExchangeGroups: " << numExchangeGroups << std::endl;
  std::cout << "exchangeBatchSize: " << exchangeBatchSize << std::endl;
  std::cout << "lastExchangeGroupLarger: " << lastExchangeGroupLarger << std::endl;

  // std::vector<unsigned> tensor_tile_ids = get_tensor_tile_ids(graph, dLdx);
  // printVector(tensor_tile_ids);

  const unsigned num_vertices = numExchangeGroups*batchsize*sparse_size;
  const unsigned vertices_per_tile = num_vertices / num_tiles_to_use + ((num_vertices % num_tiles_to_use) > 0);
  unsigned vertex_id{0};

  // poplar::Tensor dLdx_reduced = graph.addVariable(dtype, {, sparseSize, }, {dnai, "customReduce_stage1/dLdx_realloc"});
  poplar::Tensor dLdx_reduced_ilay = graph.addVariable(dtype, {numExchangeGroups, batchsize, sparse_size}, {dnai, "customReduce_stage1/dLdx_reduce"});

  unsigned start_id{0};
  unsigned end_id{0};
  for (unsigned iex=0; iex<(numExchangeGroups-lastExchangeGroupLarger); ++iex){
    end_id = start_id+exchangeGroupSize;
    std::cout << "start_id: " << start_id << ", end_id: " << end_id << std::endl;
    auto dLdx_exchangeGroundSlice = dLdx.slice(start_id, end_id, 0).dimShuffle({1,2,0});
    
    for (unsigned ibatch=0; ibatch<batchsize; ++ibatch){
      unsigned row_id{ibatch / exchangeBatchSize + start_id};
      for (unsigned isparse=0; isparse<sparse_size; ++isparse){
        unsigned tile_id = start_tile + (vertex_id / vertices_per_tile);

        if ((target_type == poplar::TargetType::IPU) && (exchangeGroupSize == 8) && (dtype == poplar::FLOAT)) {
          auto v = graph.addVertex(cs, "CustomSparseReduceStage1FloatSIMD8",
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    // {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        } else if ((target_type == poplar::TargetType::IPU) && (exchangeGroupSize == 5) && (dtype == poplar::FLOAT)) { // TODO doesn't do anything auto vec seems to work
          auto v = graph.addVertex(cs, "CustomSparseReduceStage1FloatSIMD5",
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    // {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        } else {
          auto v = graph.addVertex(cs, poputil::templateVertex("CustomSparseReduceStage1", dtype),
                                    {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                    {"num_grads", replicated_numGrads[row_id][ibatch]},
                                    {"isparse", isparse},
                                    {"end", exchangeGroupSize},
                                    {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
          graph.setTileMapping(v, tile_id); 
          graph.setPerfEstimate(v, 1);
        }
        graph.setTileMapping(dLdx_reduced_ilay[iex][ibatch][isparse], tile_id);
        ++vertex_id; 
      }
      // std::cout << std::endl;
    }
    start_id = end_id;
  }

  if (lastExchangeGroupLarger){
    unsigned end_id = dLdx.dim(0);
    std::cout << "lastExchangeGroupLarger" << std::endl;
    std::cout << "start_id: " << start_id << ", end_id: " << end_id << std::endl;
    unsigned iex = numExchangeGroups-1;
    unsigned exchangeGroupSizeLast = end_id-start_id;
    unsigned exchangeBatchSizeLast = batchsize / exchangeGroupSizeLast + ((batchsize % exchangeGroupSizeLast) > 0);

    std::cout << "exchangeGroupSizeLast: " << exchangeGroupSizeLast << std::endl;
    std::cout << "exchangeBatchSizeLast: " << exchangeBatchSizeLast << std::endl;

    auto dLdx_exchangeGroundSlice = dLdx.slice(start_id, end_id, 0).dimShuffle({1,2,0});

    for (unsigned ibatch=0; ibatch<batchsize; ++ibatch){
      unsigned row_id{ibatch / exchangeBatchSizeLast + start_id};
      for (unsigned isparse=0; isparse<sparse_size; ++isparse){
        unsigned tile_id = start_tile + (vertex_id / vertices_per_tile);
        auto v = graph.addVertex(cs, poputil::templateVertex("CustomSparseReduceStage1", dtype),
                                  {{"dLdx", dLdx_exchangeGroundSlice[ibatch][isparse]},
                                  {"num_grads", replicated_numGrads[row_id][ibatch]},
                                  {"isparse", isparse},
                                  {"end", exchangeGroupSizeLast},
                                  {"dLdx_reduced", dLdx_reduced_ilay[iex][ibatch][isparse]}});
        graph.setTileMapping(dLdx_reduced_ilay[iex][ibatch][isparse], tile_id);
        graph.setTileMapping(v, tile_id); 
        graph.setPerfEstimate(v, 1);
        ++vertex_id; 
      }
    }
  }
  return dLdx_reduced_ilay;
}

std::vector<poplar::Tensor> customReduceSparseInpSpikesGrad_stage1(poplar::Graph &graph, std::vector<poplar::Tensor> &dLdx_vec, std::vector<poplar::Tensor> &replicated_numGrads,
                                  poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){
  auto cs = graph.addComputeSet({dnai, "customReduceSparseInpSpikesGrad_stage1"});                                    
  // std::vector<poplar::Tensor> dLdx_reduced;
  // for (unsigned ilay=0; ilay<dLdx_vec.size(); ++ilay){
  //   std::cout << "\ncustomReduceSparseInpSpikesGrad_stage1_layer: " << ilay << std::endl;
  //   dLdx_reduced.push_back(customReduceSparseInpSpikesGrad_stage1_layer(graph, dLdx_vec[ilay], replicated_numGrads[ilay], cs, dnai));
  // }


  const std::vector<std::vector<unsigned>> layer_ids_per_ipu(get_layer_ids_per_ipu(graph, dLdx_vec));

  unsigned num_ipus =  graph.getTarget().getNumIPUs();
  const unsigned num_tiles_per_ipu = graph.getTarget().getTilesPerIPU();
  std::vector<poplar::Tensor> dLdx_reduced;
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    unsigned tile_offset = (iipu==0)? 1 : 0;
    unsigned num_tiles_per_ipu_usable = num_tiles_per_ipu-tile_offset;
    unsigned num_operations_this_ipu{0};
    std::vector<unsigned> num_operations_per_layer;
    for (auto &layer_id: layer_ids_per_ipu[iipu]){
      unsigned batchsize = dLdx_vec[layer_id].dim(1);
      unsigned sparse_size = dLdx_vec[layer_id].dim(2);
      unsigned num_occupied_tiles = dLdx_vec[layer_id].dim(0);
            
      double num_operations_this_layer_fp = std::pow((double)num_occupied_tiles, 0.8) * (double)sparse_size * (double)batchsize;
      num_operations_this_ipu += (unsigned)num_operations_this_layer_fp;
      num_operations_per_layer.push_back((unsigned)num_operations_this_layer_fp);
    }
    std::vector<unsigned> num_tiles_per_layer;
    for (unsigned ilay=0; ilay<layer_ids_per_ipu[iipu].size(); ++ilay){
      double mul_fac = (double)num_operations_per_layer[ilay] / (double)num_operations_this_ipu;
      double num_tiles_this_layer_fp = mul_fac * (double)num_tiles_per_ipu_usable;
      num_tiles_per_layer.push_back((unsigned)num_tiles_this_layer_fp);
    }

    printVector(num_tiles_per_layer);
    unsigned start_tile{num_tiles_per_ipu*iipu + tile_offset};
    for (unsigned ilay=0; ilay<layer_ids_per_ipu[iipu].size(); ++ilay){
      unsigned layer_id = layer_ids_per_ipu[iipu][ilay];
      std::cout << "start_tile: " << start_tile << std::endl;
      std::cout << "num_tiles_per_layer[ilay]: " << num_tiles_per_layer[ilay] << std::endl;
      std::cout << "start_tile + num_tiles_per_layer[ilay]: " << start_tile + num_tiles_per_layer[ilay] << std::endl;
      dLdx_reduced.push_back(customReduceSparseInpSpikesGrad_stage1_layer_evenAlloc(graph, dLdx_vec[layer_id], replicated_numGrads[layer_id], start_tile, num_tiles_per_layer[ilay], cs, dnai));
      start_tile += num_tiles_per_layer[ilay];
    }
  }

  prog.add(poplar::program::Execute(cs));
  return dLdx_reduced;
}

void customReduceSparseInpSpikesGrad(poplar::Graph &graph, const unsigned &num_layers, std::vector<poplar::Tensor> &dLdx_vec, std::vector<poplar::Tensor> &dLdInpSpikes,
                                  std::vector<poplar::Tensor> &replicated_numGrads, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai){

  std::vector<poplar::Tensor> dLdx_reduced_stage1 = customReduceSparseInpSpikesGrad_stage1(graph, dLdx_vec, replicated_numGrads, prog, dnai);

  poplarReduceInpSpikesGrad(graph, num_layers, dLdx_reduced_stage1, dLdInpSpikes, prog, dnai);
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

    size_t sparseSize = fwdInpSpikes[ilay].spike_ids.dim(2);
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
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[occupied_tile_counter][ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      // {"end", 0}});
                                      {"end", fwdInpSpikes[ilay].num_spikes[occupied_tile_counter][ibatch][1]}});
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
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[occupied_tile_counter][ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      {"num_iters", num_neurons / 2},
                                      {"end", fwdInpSpikes[ilay].num_spikes[occupied_tile_counter][ibatch][1]}});
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
                                      {"fwd_inp_spike_ids", fwdInpSpikes[ilay].spike_ids[occupied_tile_counter][ibatch]},
                                      {"dLdinp_spike_ids", dLdx[occupied_tile_counter][ibatch]},
                                      {"num_neurons", num_neurons},
                                      {"end", fwdInpSpikes[ilay].num_spikes[occupied_tile_counter][ibatch][1]}});
            graph.setTileMapping(v, tile); 
            graph.setPerfEstimate(v, 1);
          }
        }
      }
      ++occupied_tile_counter;
    }
    dLdx_vec.push_back(dLdx);
  }
  // zero_tensor_vector(graph, dLdx_vec, prog, {dnai, "zero_dLdx_vec"});
  prog.add(poplar::program::Execute(cs));

  // poplarReduceInpSpikesGrad(graph, num_layers, dLdx_vec, dLdInpSpikes, prog, dnai);

  std::vector<poplar::Tensor> replicated_numGrads;
  for (unsigned ilay=1; ilay<num_layers; ++ilay){
    replicated_numGrads.push_back(fwdInpSpikes[ilay].num_spikes.dimShuffle({2,0,1})[1]);
  }
  // std::transform(fwdInpSpikes.begin()+1, fwdInpSpikes.end(), std::back_inserter(replicated_numGrads), [](BatchedSparseSpikes &sparseSpikes){return sparseSpikes.num_spikes.dimShuffle({2,0,1})[1];})
  customReduceSparseInpSpikesGrad(graph, num_layers, dLdx_vec, dLdInpSpikes, replicated_numGrads, prog, dnai);
}


void performLIFStepBackwardPass(poplar::Graph &graph, const std::vector<poplar::Tensor> &weights, std::vector<poplar::Tensor> &fwdState, std::vector<BatchedSparseSpikes> &fwdInpSpikes, 
                            const std::vector<poplar::Tensor> &decay_constants, const std::vector<poplar::Tensor> &oneMinus_decay_constants, const std::vector<poplar::Tensor> &thresholds, const std::vector<BatchedSparseSpikes> &fwdOutSpikes,
                            std::vector<poplar::Tensor> &dLdweights, std::vector<poplar::Tensor> &dLdState, poplar::Tensor &dLdOutSpikes, std::vector<poplar::Tensor> &dLdInpSpikes, bool calcInpSpikeGrads,
                            poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  
  size_t num_layers = weights.size();
  std::vector<poplar::Tensor> allDLdOutSpikes(dLdInpSpikes.begin(), dLdInpSpikes.end());
  allDLdOutSpikes.push_back(dLdOutSpikes);
  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    std::cout << "thresholds[ilay]: " << thresholds[ilay].shapeToString() << std::endl;
  }
  // calcLIFStateGrad(graph, weights, fwdState, decay_constants, thresholds, fwdOutSpikes, dLdState, allDLdOutSpikes, prog, dnai);
  calcLIFStateGrad_stateWise(graph, fwdState, decay_constants, thresholds, fwdOutSpikes, dLdState, allDLdOutSpikes, prog, dnai);

  const std::vector<poplar::Tensor> intermediate_dLdState = performSharedUpdate(graph, oneMinus_decay_constants, dLdState, prog, {dnai, "performSharedUpdate"});

  std::vector<BatchedSparseSpikes> fwdInpSpikes_tileReplicated;

  for (unsigned ilay=0; ilay<num_layers; ++ilay){
    auto tileMapping = graph.getTileMapping(weights[ilay][0], true);
    // unsigned batchsize = fwdInpSpikes[ilay].dim(0);
    // unsigned sparse_size = fwdInpSpikes[ilay].dim(1);
    // unsigned num_tiles_this_layer = std::accumulate(neuronTileMapping.begin(), neuronTileMapping.end(), 0, [](unsigned a; std::vector<poplar::Inverval>& tileMap){return a + (tileMap>0);})
    // poplar::Tensor fwdIps_replicated = graph.addVariable(fwdInpSpikes[ilay].elementType(), {num_tiles_this_layer, batchsize, sparse_size}, {dnai, "alloc_fwdInpSpikes_tileReplicated"});

    poplar::Tensor fwdIps_ids_replicated = replicate_and_alloc_tensor(graph, fwdInpSpikes[ilay].spike_ids, tileMapping, prog, {dnai, "create_fwdIps_ids_replicated"});
    poplar::Tensor fwdIps_nums_replicated = replicate_and_alloc_tensor(graph, fwdInpSpikes[ilay].num_spikes, tileMapping, prog, {dnai, "create_fwdIps_nums_replicated"});
    // poplar::Tensor fwdIps_ids_replicated = replicate_and_alloc_tensor(graph, fwdInpSpikes[ilay].spike_ids, tileMapping, prog, dnai);
    // poplar::Tensor fwdIps_nums_replicated = replicate_and_alloc_tensor(graph, fwdInpSpikes[ilay].num_spikes, tileMapping, prog, dnai);
    BatchedSparseSpikes fwdIps_replicated = {fwdIps_ids_replicated, fwdIps_nums_replicated};
    fwdInpSpikes_tileReplicated.push_back(fwdIps_replicated); //alloc_neuronwise_contiguous(graph, {num_tiles_this_layer, batchsize, sparse_size}, fwdInpSpikes[ilay].elementType(), 2, neuronTileMapping, {dnai, "alloc_fwdInpSpikes_tileReplicated"}));
  }

  // calcLIFWeightGrad_singleThread(graph, dLdweights, fwdInpSpikes, intermediate_dLdState, prog, dnai);
  calcLIFWeightGrad(graph, dLdweights, fwdInpSpikes_tileReplicated, intermediate_dLdState, prog, dnai);

  if (calcInpSpikeGrads){
    // calcLIFInpSpikesGrad(graph, weights, fwdInpSpikes, decay_constants, dLdState, dLdInpSpikes,  prog, dnai);
    calcLIFInpSpikesGradRowWise(graph, weights, fwdInpSpikes_tileReplicated, intermediate_dLdState, dLdInpSpikes,  prog, dnai);
  }
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
  num_inputs = 6*38;
  // allocating_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (std::int64_t i=0; i<num_inputs; ++i){
    allocating_indices.push_back(i);
  }
  is_hashable = true;
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


  // printVector(start_tiles);
  // printVector(end_tiles);
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

  std::cout << "Build_allocator: " << operand << ", " << operand/num_layers << ", " <<  layer_id << std::endl;

  // std::cout << "\noperand: " << operand << ", operand/num_layers: " << operand/num_layers << std::endl;
  // std::cout << "layer_id: " << layer_id << std::endl;

  // size_t num_elements{1};
  // size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b){return a*b;});
  size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  // std::cout << "layer_id: " << layer_id << std::endl;
  // std::cout << "layer_id_prev: " << layer_id_prev << std::endl;
  // std::cout << "num_elements: " << num_elements << std::endl;
  // printVector(shape);

  // // TODO does this work? is ceil correct as it will require one tile to have less...?
  // size_t minElementsPerTile = num_elements / (numTilesPerIPU-tile_offset) + ((num_elements % (numTilesPerIPU-tile_offset)) > 0) ;
  size_t ipu_id = start_tiles[layer_id] / numTilesPerIPU;
  size_t ipu_id_prev = start_tiles[layer_id_prev] / numTilesPerIPU;
  // size_t start_tile_ipu = ipu_id * numTilesPerIPU + tile_offset;
  // size_t start_tile_ipu_prev = ipu_id_prev * numTilesPerIPU + tile_offset;

  size_t tile_offset = (ipu_id == 0) ? 1 : 0;

  size_t minElementsPerTile = num_elements / (end_tiles[layer_id]-start_tiles[layer_id]) + ((num_elements % (end_tiles[layer_id]-start_tiles[layer_id])) > 0);

  size_t ipu_id_to_use;
  poplar::Graph virtualGraph;

  // std::cout << "case " << operand/num_layers << std::endl;

  std::vector<std::vector<poplar::Interval>> tile_map_threshs;
  std::vector<size_t> tile_map_threshs_sizes;
  std::tuple<unsigned, unsigned, bool> start_end_is_contig;

  switch (operand/num_layers) {
    case 0: neuronDim = 1; 
            tensor_name = "weights";
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            break;
    case 1: neuronDim = 1;
            tensor_name = "init_state";
            // std::cout << "Build alloc init_state" << std::endl;
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // // allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            // allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});

            allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            // std::cout << "\nalloc_tensor tileMap[0].size(): " << graph.getTileMapping(allocTensor)[0].size() << std::endl;
            // std::cout << "alloc_tensor tileMap[1].size(): " << graph.getTileMapping(allocTensor)[1].size() << std::endl;

            tile_map_threshs = graph.getTileMapping(allocTensor);
            
            std::transform(tile_map_threshs.begin(), tile_map_threshs.end(), std::back_inserter(tile_map_threshs_sizes), [](std::vector<poplar::Interval> &vec) {return vec.size();});
            // std::cout << "\n" << ilay << std::endl;
            // printVector(tile_map_threshs_sizes);
            start_end_is_contig = get_start_end_is_contigious(tile_map_threshs_sizes);
            // std::cout << layer_id << ": is_contig: " << std::get<2>(start_end_is_contig) << ", start_tile: " << std::get<0>(start_end_is_contig) << ", end_tile: " << std::get<1>(start_end_is_contig) << std::endl;
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
              allocTensor = alloc_linearly(graph, shape, type, tile_offset, {dnai, tensor_name});
            }
            break;
    case 4: neuronDim = 0;
            tensor_name = "decay_constants";
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            // alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            allocTensor = alloc_linearly(graph, shape, type, start_tiles[layer_id], minElementsPerTile, {dnai, tensor_name}); // TODO I don't want minElementsPerTile though but exactly!
            break;
    case 5: neuronDim = 1;
            tensor_name = "thresholds";
            // neuron_mapping = determine_neuron_mapping(numTiles, layer_id, dense_sizes, sparse_sizes, batchsize);
            // allocTensor = alloc_neuronwise(graph, shape, type, neuronDim, neuron_mapping, {dnai, tensor_name});
            allocTensor = alloc_neuronwise_contiguous(graph, shape, type, neuronDim, start_tiles[layer_id], end_tiles[layer_id], {dnai, tensor_name});
            // allocTensor = alloc_linearly(graph, shape, type, start_tiles[layer_id], minElementsPerTile, {dnai, tensor_name}); // TODO I don't want minElementsPerTile though but exactly!
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

  if (num_layers > 38){
    throw poputil::poplibs_error("Program compiled with max 38 layers. For more adjust `Build_metadata`.");
  }

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

  for (unsigned ilay=0; ilay<init_state.size(); ++ilay){
    std::cout << ilay << ": &init_state[ilay]" << &init_state[ilay] << std::endl;
  }

  for (unsigned ilay=0; ilay<init_state.size()-1; ++ilay){
    bool eqaul = init_state[ilay]==init_state[ilay+1];
    std::cout << ilay << ": init_state[ilay]==init_state[ilay+1]: " << eqaul << std::endl;
  }

  std::cout << "\ninp_spike_ids_fptype" << std::endl;
  for (unsigned ilay=0; ilay<inp_spike_ids_fptype.size()-1; ++ilay){
    bool eqaul = inp_spike_ids_fptype[ilay]==inp_spike_ids_fptype[ilay+1];
    std::cout << ilay << ": inp_spike_ids_fptype[ilay]==inp_spike_ids_fptype[ilay+1]: " << eqaul << std::endl;
  }


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

    if (thresholds[ilay].rank() != 2) {
      throw poputil::poplibs_error("Input 'inputs[5]' must be vectors (2, size_out).");
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
    auto mulFac = graph.addConstant(decay_constants[i].elementType(), decay_constants[i].shape(), 10.0, {dnai, "mulFac"});
    graph.setTileMapping(ones, graph.getTileMapping(decay_constants[i]));
    graph.setTileMapping(mulFac, graph.getTileMapping(decay_constants[i]));
    poplar::Tensor oneMinus_decay_constant = graph.clone(decay_constants[i], {dnai, "alloc_oneMinus_decay_constant"}, poplar::TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES);
    popops::subWithOutput(graph, ones, decay_constants[i], oneMinus_decay_constant, fwdProg, {dnai, "fill_oneMinus_decay_constant"});
    popops::mulInPlace(graph, oneMinus_decay_constant, mulFac, fwdProg, {dnai, "mul_oneMinus_decay_constant"});
    oneMinus_decay_constants.push_back(oneMinus_decay_constant);
  }

  // std::vector<poplar::Tensor> second_thresholds;


  // for (unsigned i=0; i<num_layers ; ++i) {
  //   std::cout << "\nthresholds[i].shapeToString(): " << i << std::endl;
  //   std::cout << thresholds[i].shapeToString() << std::endl;
  //   thresholds[i] = thresholds[i][0];
  //   std::cout << thresholds[i].shapeToString() << std::endl;
  // }
  // for (unsigned i=0; i<num_layers ; ++i) {
  //   auto mulFac = graph.addConstant(thresholds[i].elementType(), thresholds[i].shape(), 0.9, {dnai, "mulFac"});
  //   graph.setTileMapping(mulFac, graph.getTileMapping(thresholds[i]));
  //   poplar::Tensor second_threshold = graph.clone(thresholds[i], {dnai, "alloc_second_thresholds"});
  //   popops::mulWithOutput(graph, oneMinus_decay_constant, mulFac, fwdProg, {dnai, "mul_second_thresholds"});
  //   oneMinus_decay_constants.push_back(oneMinus_decay_constant);
  // }

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

  // std::vector<poplar::Tensor> inp_spike_ids = cast_tensor_vector(graph, {inp_spike_ids_fptype[0],}, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast inp_spike_ids 0"});
  // std::vector<poplar::Tensor> num_inp_spikes = cast_tensor_vector(graph, {num_inp_spikes_int[0],}, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast num_inp_spikes 0"});

  // std::vector<poplar::Tensor> inp_spike_ids_dl;
  // std::vector<poplar::Tensor> num_inp_spikes_dl;

  // clone_tensor_vector(graph, poplar::UNSIGNED_INT, inp_spike_ids_fptype, inp_spike_ids_dl, 1, {dnai, "clone_inp_spike_ids"});
  // clone_tensor_vector(graph, poplar::UNSIGNED_INT, num_inp_spikes_int, num_inp_spikes_dl, 1, {dnai, "clone_num_inp_spikes"});

  // for (unsigned ilay=0; ilay<num_layers; ++ilay){

  // }

  std::vector<poplar::Tensor> inp_spike_ids = cast_tensor_vector(graph, inp_spike_ids_fptype, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast inp_spike_ids"});
  std::vector<poplar::Tensor> num_inp_spikes = cast_tensor_vector(graph, num_inp_spikes_int, poplar::UNSIGNED_INT, fwdProg, {dnai, "cast num_inp_spikes"});

  std::vector<size_t> layer_to_ipu_id;
  std::transform(start_tiles.begin(), start_tiles.end(), std::back_inserter(layer_to_ipu_id), [&numTilesPerIPU](size_t &start_tile){return start_tile / numTilesPerIPU;});
  
  std::vector<poplar::Graph> ipu_to_virtualGraph;
  size_t num_ipus = target.getNumIPUs();
  for (unsigned iipu=0; iipu<num_ipus; ++iipu){
    unsigned tile_offset = (iipu == 0) ? 1 : 0;
    ipu_to_virtualGraph.push_back(graph.createVirtualGraph(numTilesPerIPU * iipu + tile_offset, numTilesPerIPU * (iipu+1)));
  }


  std::vector<poplar::Tensor> out_spike_ids;
  std::vector<poplar::Tensor> num_out_spikes;
  std::vector<poplar::Tensor> stateSeqOutput;
  std::vector<poplar::Tensor> slicedOutSpikeIds;
  std::vector<poplar::Tensor> slicedNumOutSpikes;
  std::vector<poplar::Tensor> currentState;
  std::vector<poplar::Tensor> slicedInpSpikeIds(1);
  std::vector<poplar::Tensor> slicedNumInpSpikes(1);
  for (unsigned ilay=0; ilay<num_layers; ++ilay){

    // const size_t num_thresholds{thresholds[ilay].dim(0)};
    const size_t num_thresholds{2};

    out_spike_ids.push_back(popops::createSliceableTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], poplar::UNSIGNED_INT, {seq_len, batchsize, sparse_sizes[ilay+1]}, {0}, {1}, 0, {dnai, "alloc out_spike_ids"}));
    num_out_spikes.push_back(popops::createSliceableTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]],  poplar::UNSIGNED_INT, {seq_len, batchsize, num_thresholds}, {0}, {1}, 0, {dnai, "alloc  num_out_spikes"}));
    stateSeqOutput.push_back(alloc_neuronwise_contiguous(graph, {seq_len, batchsize, dense_sizes[ilay+1]}, dtype, 2, graph.getTileMapping(weights[ilay][0]), {dnai, "alloc stateSeqOutput"}));

    slicedOutSpikeIds.push_back(popops::createSliceTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], out_spike_ids.back(), {0}, {1}, 1, {dnai, "initial createSliceTensor slicedOutSpikeIds"})[0][0]);
    slicedNumOutSpikes.push_back(popops::createSliceTensor(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], num_out_spikes.back(), {0}, {1}, 1, {dnai, "initial createSliceTensor slicedNumOutSpikes"})[0][0]);

    currentState.push_back(alloc_neuronwise_contiguous(graph, init_state[ilay].shape(), init_state[ilay].elementType(), 1, start_tiles[ilay], end_tiles[ilay], {dnai, "current_state"}));
    fwdProg.add(poplar::program::Copy(init_state[ilay], currentState[ilay], false, {dnai, "copy_to_currenState"}));
  } 
  for (unsigned ilay=1; ilay<num_layers; ++ilay){
    slicedInpSpikeIds.push_back(alloc_linearly(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], inp_spike_ids[ilay].shape(), poplar::UNSIGNED_INT, 0, {dnai, "slicedInpSpikeIds"}));
    slicedNumInpSpikes.push_back(alloc_linearly(ipu_to_virtualGraph[layer_to_ipu_id[ilay]], num_inp_spikes[ilay].shape(), poplar::UNSIGNED_INT, 0, {dnai, "slicedNumInpSpikes"}));
  }

  for (unsigned i=0; i<num_layers-1; ++i){
    fwdProg.add(poplar::program::Copy(inp_spike_ids[i+1], slicedOutSpikeIds[i], false, dnai));
    fwdProg.add(poplar::program::Copy(num_inp_spikes[i+1], slicedNumOutSpikes[i].slice(0, 1, 1), false, dnai));
  }


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
      loop.add(poplar::program::Copy(slicedNumOutSpikes[i].slice(0, 1, 1), slicedNumInpSpikes[i+1], false, dnai));
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
  std::vector<poplar::Tensor> multi_thresholds(fwd_inputs.begin()+5*num_layers,fwd_inputs.begin()+6*num_layers);

  std::vector<poplar::Tensor> thresholds;
  std::transform(multi_thresholds.begin(), multi_thresholds.end(), std::back_inserter(thresholds), [](poplar::Tensor &t){return t[0];});

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
    unsigned tile_offset = (iipu == 0) ? 1 : 0;
    ipu_to_virtualGraph.push_back(graph.createVirtualGraph(numTilesPerIPU * iipu + tile_offset, numTilesPerIPU * (iipu+1)));
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

  std::vector<poplar::Tensor> oneMinus_decay_constants;
  for (unsigned i=0; i<num_layers ; ++i) {
    auto ones = graph.addConstant(decay_constants[i].elementType(), decay_constants[i].shape(), 1.0, {dnai, "ones"});
    auto mulFac = graph.addConstant(decay_constants[i].elementType(), decay_constants[i].shape(), 10.0, {dnai, "mulFac"});
    graph.setTileMapping(ones, graph.getTileMapping(decay_constants[i]));
    graph.setTileMapping(mulFac, graph.getTileMapping(decay_constants[i]));
    poplar::Tensor oneMinus_decay_constant = graph.clone(decay_constants[i], {dnai, "alloc_oneMinus_decay_constant"});
    popops::subWithOutput(graph, ones, decay_constants[i], oneMinus_decay_constant, bwdProg, {dnai, "fill_oneMinus_decay_constant"});
    popops::mulInPlace(graph, oneMinus_decay_constant, mulFac, bwdProg, {dnai, "mul_oneMinus_decay_constant"});
    oneMinus_decay_constants.push_back(oneMinus_decay_constant);
  }


  std::vector<poplar::Tensor> out_spike_ids_fptype(fwd_outputs.begin(),fwd_outputs.begin()+num_layers);
  std::vector<poplar::Tensor> num_out_spikes_int(fwd_outputs.begin()+1*num_layers,fwd_outputs.begin()+2*num_layers);
  std::vector<poplar::Tensor> fwd_states_seq(fwd_outputs.begin()+2*num_layers,fwd_outputs.begin()+3*num_layers);

  for (unsigned i=0; i<num_layers ; ++i) {
    std::cout << "num_out_spikes_int[i].spike_ids.shapeToString(): " << num_out_spikes_int[i].shapeToString() << std::endl;
  }


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
  // std::vector<poplar::Tensor> dLdinit_state = clone_tensor_vector(graph, init_state, {dnai, "dLdinit_state"});
  // std::vector<poplar::Tensor> dLdinp_spike_ids = clone_tensor_vector(graph, inp_spike_ids_fptype, {dnai, "dLdinp_spike_ids"}); // how to  set mapping in Reduce operation
  // std::vector<poplar::Tensor> dLdnum_inp_spikes = clone_tensor_vector(graph, num_inp_spikes_int, {dnai, "dLdnum_inp_spikes"});
  // std::vector<poplar::Tensor> dLddecay_constatns = clone_tensor_vector(graph, decay_constants, {dnai, "dLddecay_constatns"});
  // std::vector<poplar::Tensor> dLdthresholds = clone_tensor_vector(graph, thresholds, {dnai, "dLdthresholds"});

  // only account for gradients though last layers spikes (as it should be for feed forward network)
  poplar::Tensor dLdout_spike_ids = gradients[num_layers-1]; //essentailly assume all others are 0
  // poplar::Tensor dLdnum_out_spikes = gradients[1]; // not needed
  // poplar::Tensor dLdfwd_states_seq = gradients[2]; // Ignore this possibility for now. Essentially assume 0

  // init reverse state
  std::vector<poplar::Tensor> dLdstate = clone_tensor_vector(graph, init_state, {dnai, "dLdstate_clone"});
  zero_tensor_vector(graph, dLdstate, bwdProg, dnai);

  for (unsigned ilay=0; ilay<dLdstate.size(); ++ilay){
    std::cout << "\nilay: " << ilay << std::endl;
    std::cout << "init_state[ilay][0].isContiguous().slice(0, 2, 0): " << init_state[ilay][0].slice(0, 2, 0).isContiguous() << std::endl;
    std::cout << "dLdstate[ilay][0].isContiguous().slice(0, 2, 0): " << dLdstate[ilay][0].slice(0, 2, 0).isContiguous() << std::endl;
  }

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

  // auto loopBwd = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
  //                 &dLdweights, &dLdinp_spike_ids, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
  auto loopBwd = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
                  &dLdweights, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
    
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

      std::cout << "\n" << ilay << std::endl;
      std::cout << "out_spike_ids[ilay].shapeToString(): " << out_spike_ids[ilay].shapeToString() << std::endl;
      std::cout << "slicedInpSpikeIds.shapeToString(): " << slicedInpSpikeIds.shapeToString() << std::endl;
      std::cout << "num_out_spikes[ilay].shapeToString(): " << num_out_spikes[ilay].shapeToString() << std::endl;
      std::cout << "slicedNumInpSpikes.shapeToString(): " << slicedNumInpSpikes.shapeToString() << std::endl;

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
        graph, weights, slicedFwdState, fwdInpSpikes, decay_constants, oneMinus_decay_constants, thresholds, fwdOutSpikes, dLdweights, dLdstate, slicedDLdOutSpikes, slicedDLdInpSpikes, true, loop, {dnai});

    // TODO if gradient with respect to input spike tensors is desired uncomment this and funcs above and rewrite 
    // (also in `performLIFStepBackwardPass` first layer grad has to be calculated) 
    // popops::dynamicUpdate(graph, dLdinp_spike_ids, slicedDLdInpSpikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic dLdInpSpikes update"});
    return loop;
  };

  poplar::program::Sequence bodyProg = {loopBwd()};
  auto repeat = poplar::program::Repeat(seq_len-1, bodyProg, {dnai, "repeat"});
  bwdProg.add(repeat);


  //----------------------------------- this is just for the first timestep because inp_spikes can not be determined from out_spikes_tensor ----------------------------- 
  // auto loopBwdFirstTimestep = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
  //                 &dLdweights, &dLdinp_spike_ids, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
  auto loopBwdFirstTimestep = [&graph, &weights, &decay_constants, &oneMinus_decay_constants, &thresholds, &inp_spike_ids, &num_inp_spikes, &out_spike_ids, &num_out_spikes, &fwd_states_seq, 
                  &dLdweights, &dLdout_spike_ids, &dLdstate, &slicedDLdInpSpikes, &itime, &num_layers, &step, &dnai] () {
    
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
        graph, weights, slicedFwdState, fwdInpSpikes, decay_constants, oneMinus_decay_constants, thresholds, fwdOutSpikes, dLdweights, dLdstate, slicedDLdOutSpikes, slicedDLdInpSpikes, false, loop, {dnai});

    // TODO if gradient with respect to input spike tensors is desired uncomment this and funcs above and rewrite 
    // (also in `performLIFStepBackwardPass` first layer grad has to be calculated) 
    // popops::dynamicUpdate(graph, dLdinp_spike_ids, slicedDLdInpSpikes.expand({0}), itime, {0}, {1}, loop, {dnai, "dynamic dLdInpSpikes update"});
    return loop;
  };

  bwdProg.add(loopBwdFirstTimestep());


  std::vector<poplar::Tensor> dLdweights_final;
  if (num_threads > 1) {
    popops::ReduceParams reduceParams = popops::ReduceParams(popops::Operation::ADD, false); 

    // std::vector<poplar::Tensor> dLdweights_final; // TODO alloc with or without tileMapping ?
    dLdweights_final = clone_tensor_vector(graph, weights, {dnai, "dLdweights"});
    std::vector<poplar::Tensor> dLdweights_final_expanded;
    std::transform(dLdweights_final.begin(), dLdweights_final.end(), std::back_inserter(dLdweights_final_expanded), [](poplar::Tensor &t){return t.expand({0});});
    
    if (num_ipus>0){
      std::vector<std::vector<unsigned>> ipu_layer_ids(num_ipus);
      for (unsigned ilay=0; ilay<num_layers; ++ilay){
        ipu_layer_ids[layer_to_ipu_id[ilay]].push_back(ilay);
      }
      for (unsigned ipu_id=0; ipu_id<num_ipus; ++ipu_id){
        if (ipu_layer_ids[ipu_id].size()>0){
          std::vector<poplar::Tensor> dLdweights_final_expanded_this_ipu;
          std::vector<popops::SingleReduceOp> single_reduce_ops;
          for (unsigned &layer_id: ipu_layer_ids[ipu_id]){
            dLdweights_final_expanded_this_ipu.push_back(dLdweights_final_expanded[layer_id]);
            single_reduce_ops.push_back(popops::SingleReduceOp(dLdweights[layer_id], {0}, reduceParams, "single reduce dLdweights"));
          }
          reduceMany(ipu_to_virtualGraph[ipu_id], single_reduce_ops, dLdweights_final_expanded_this_ipu, bwdProg, {dnai, "add dLdweights"});
        }
      }
    } else {
      std::vector<popops::SingleReduceOp> single_reduce_ops;
      for (unsigned ilay=0; ilay<num_layers; ++ilay){
        single_reduce_ops.push_back(
          popops::SingleReduceOp(dLdweights[ilay], {0}, reduceParams, "single reduce dLdweights")
        );
      }
      reduceMany(graph, single_reduce_ops, dLdweights_final_expanded, bwdProg, {dnai, "add dLdweights"});
    }
  } else {
    std::transform(dLdweights.begin(), dLdweights.end(), std::back_inserter(dLdweights_final), [](poplar::Tensor &t){return t[0];});
  }

  // std::vector<poplar::Tensor> dLdweights_final;
  // std::transform(dLdweights.begin(), dLdweights.end(), std::back_inserter(dLdweights_final), [](poplar::Tensor &t){return t[0];});


  //----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

  extend_tensor_vector(dLdweights_final, outputs);
  // extend_tensor_vector(dLdinit_state, outputs); // only placeholder for now, could easily be calculated from `updatedState` though
  // extend_tensor_vector(dLdinp_spike_ids, outputs);
  // extend_tensor_vector(dLdnum_inp_spikes, outputs); // placeholder
  // extend_tensor_vector(dLddecay_constatns, outputs); // only placeholder for now
  // extend_tensor_vector(dLdthresholds, outputs); // only placeholder for now

  return bwdProg;
}
