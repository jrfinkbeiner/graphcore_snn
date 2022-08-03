#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath> // for std::abs, std::pow


//---------------------------------------------- forward -----------------------------------------

template <typename FPType>
class LIFStateUpdateInPlace : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> weights;
  poplar::InOut<FPType> state;
  poplar::Input<poplar::Vector<unsigned>> inp_spikes_ids;
  poplar::Input<unsigned> num_inp_spikes;
  poplar::Input<FPType> decay_constant;
  poplar::Input<FPType> threshold;

  bool compute() {
    FPType sum{0.0};
    FPType state_{state};
    const FPType threshold_{threshold};
    const FPType one{1.0};
    const auto end{num_inp_spikes};
    for (unsigned i = 0; i < end; ++i) {
      sum += weights[inp_spikes_ids[i]];
    }
    if (state_ > threshold_) {
      // *new_state = sum;
      *state = (one - decay_constant) * sum;
    } else {
      // *new_state = decay_constant * state + sum;
      *state = decay_constant * state + (one - decay_constant) * sum;
    }
    return true;
  }
};
template class LIFStateUpdateInPlace<float>;
// template class LIFStateUpdateInPlace<half>;

template <typename FPType>
class LIFStateUpdateInPlaceMultiNeuron : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> weights;
  poplar::InOut<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> state;
  poplar::InOut<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> syn_input;
  poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> inp_spikes_ids;
  poplar::Input<unsigned> num_inp_spikes;
  poplar::Input<poplar::Vector<FPType>> decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<FPType>> oneMinus_decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<FPType>> threshold; // TODO make sure it's contiguous
  poplar::Input<unsigned> num_neurons;

  bool compute() {
    for (unsigned i = 0; i < num_inp_spikes; ++i) {
      const auto idx = num_neurons*inp_spikes_ids[i];
      #pragma clang loop vectorize(enable) interleave(enable)
      #pragma clang loop unroll(disable)
      for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron) {
        syn_input[ineuron] += weights[idx+ineuron];
      }
    }

    // TODO vectorize this loop ?
    for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron) {
      if (state[ineuron] > threshold[ineuron]) {
        // *new_state = sum;
        state[ineuron] = oneMinus_decay_constant[ineuron] * syn_input[ineuron];
      } else {
        // *new_state = decay_constant * state + sum;
        state[ineuron] = decay_constant[ineuron] * state[ineuron] + oneMinus_decay_constant[ineuron] * syn_input[ineuron];
      }
    }
    return true;
  }
};
template class LIFStateUpdateInPlaceMultiNeuron<float>;
// template class LIFStateUpdateInPlaceMultiNeuron<half>;


template <typename FPType>
class LIFOutSpikesFromTopK : public poplar::Vertex {
public:

  poplar::Input<poplar::Vector<FPType>> topKStateVals;
  poplar::Input<poplar::Vector<unsigned>> topKStateIds;
  poplar::Input<poplar::Vector<FPType>> thresholds;

  poplar::Output<poplar::Vector<unsigned>> out_spikes_ids;
  poplar::Output<unsigned> num_out_spikes;
  // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

  bool compute() {
    unsigned numSpikesCounter{0};
    const size_t sizeSparseOut = out_spikes_ids.size();
    for (unsigned i = 0; i < topKStateVals.size(); ++i) {
      unsigned origId = topKStateIds[i];
      if (topKStateVals[i] > thresholds[origId]) { // TODO order and slice thersholds in poplar_code.cpp, not here?
        out_spikes_ids[numSpikesCounter] = origId;
        ++numSpikesCounter;
      } else {
        // Fill up the array with non-spike values in reverse from behind 
        out_spikes_ids[sizeSparseOut-1-i+numSpikesCounter] = origId;
      }
    }
    *num_out_spikes = numSpikesCounter;

    return true;
  }
};
template class LIFOutSpikesFromTopK<float>;
// template class LIFOutSpikesFromTopK<half>;

template <typename FPType>
class LIFOutSpikes2Threshs : public poplar::Vertex {
public:

  poplar::Input<poplar::Vector<FPType>> state;
  poplar::Input<poplar::Vector<FPType>> thresholds;

  poplar::Output<poplar::Vector<unsigned>> out_spikes_ids;
  poplar::Output<unsigned> num_out_spikes;
  // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

  bool compute() {
    unsigned numSpikesCounter{0};
    unsigned numGradsCounter{0};
    const FPType secThreshMul{0.9};
    const size_t numStates = state.size();
    const size_t sizeSparseOut = out_spikes_ids.size();
    for (unsigned i = 0; i < numStates; ++i) {
      // TODO reformulate to only use +
      if ((state[i] > thresholds[i]*secThreshMul) || (numStates - i <= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
        if (state[i] > thresholds[i]) {
          out_spikes_ids[numSpikesCounter] = i;
          ++numSpikesCounter;
        } else {
          // Fill up the array with non-spike values in reverse from behind 
          out_spikes_ids[sizeSparseOut-1-numGradsCounter] = i;
          ++numGradsCounter;
        }
      }
      if (numSpikesCounter+numGradsCounter >= sizeSparseOut) break; // TODO just implement as while
    }
    *num_out_spikes = numSpikesCounter;
    // *num_out_spikes = sum; // TODO uncomment when int is supported

    return true;
  }
};
template class LIFOutSpikes2Threshs<float>;
// template class LIFOutSpikes2Threshs<half>;


template <typename FPType>
class LIFOutSpikes2ThreshsMultiVertex : public poplar::MultiVertex {
public:

  poplar::Input<poplar::Vector<FPType>> state;
  poplar::Input<poplar::Vector<FPType>> thresholds;
  poplar::Input<unsigned> sizeSparseOut;
  poplar::Output<poplar::Vector<unsigned>> repeated_out_spikes_ids;
  poplar::Output<poplar::Vector<unsigned>> repeated_num_out_spikes;
  
  // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

  bool compute(unsigned workerId) {
    const unsigned numWorkers = MultiVertex::numWorkers();
    unsigned numSpikesCounter{0};
    unsigned numGradsCounter{0};
    const FPType secThreshMul{0.9};
    const size_t numStatesThisWorker = state.size() / numWorkers + ((state.size() % numWorkers) > workerId);
    unsigned i{0};
    unsigned workerStartId{workerId*sizeSparseOut};
    unsigned istat{0};
    for (unsigned i = 0; i < numStatesThisWorker; ++i) {
      // TODO reformulate to only use +
    // while (numSpikesCounter+numGradsCounter < sizeSparseOut) {
      if ((state[istat] > thresholds[istat]*secThreshMul) || (numStatesThisWorker - i <= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
        if (state[istat] > thresholds[istat]) {
          repeated_out_spikes_ids[workerStartId+numSpikesCounter] = istat;
          ++numSpikesCounter;
        } else {
          // Fill up the array with non-spike values in reverse from behind 
          repeated_out_spikes_ids[workerStartId+sizeSparseOut-1-numGradsCounter] = istat;
          ++numGradsCounter;
        }
      }
      // i+=numWorkers;
      if (numSpikesCounter+numGradsCounter >= sizeSparseOut) break; // TODO just implement as while

      istat+=numWorkers;
    }
    repeated_num_out_spikes[workerId] = numSpikesCounter;
    
    
    

    // for (unsigned i = 0; i < sizeSparseOut; ++i) {
    //   repeated_out_spikes_ids[workerStartId+i] = workerStartId+i;
    // }

    // repeated_num_out_spikes[workerId] = sizeSparseOut / 4;

    return true;
  }
};
template class LIFOutSpikes2ThreshsMultiVertex<float>;
// template class LIFOutSpikes2ThreshsMultiVertex<half>;


template <typename FPType>
class LIFOutSpikes2ThreshsSplitWorker : public poplar::Vertex {
public:

  poplar::Input<poplar::Vector<FPType>> state;
  poplar::Input<poplar::Vector<FPType>> thresholds;
  poplar::Input<unsigned> start_id;
  poplar::Output<poplar::Vector<unsigned>> repeated_out_spikes_ids;
  poplar::Output<unsigned> repeated_num_out_spikes;


  bool compute() {
    unsigned numSpikesCounter{0};
    unsigned numGradsCounter{0};
    const FPType secThreshMul{0.9};
    const size_t numStates = state.size();
    unsigned state_id{start_id};
    auto sizeSparseOut = repeated_out_spikes_ids.size();

    for (unsigned i = 0; i < numStates; ++i) {
      if ((state[i] > thresholds[i]*secThreshMul) || (numStates - i <= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
        if (state[i] > thresholds[i]) {
          repeated_out_spikes_ids[numSpikesCounter] = state_id;
          ++numSpikesCounter;
        } else {
          // Fill up the array with non-spike values in reverse from behind 
          repeated_out_spikes_ids[sizeSparseOut-1-numGradsCounter] = state_id;
          ++numGradsCounter;
        }
      }
      if (numSpikesCounter+numGradsCounter >= sizeSparseOut) break; // TODO just implement as while
      ++state_id;
    }
    *repeated_num_out_spikes = numSpikesCounter;
    return true;
  }
};
template class LIFOutSpikes2ThreshsSplitWorker<float>;
// template class LIFOutSpikes2ThreshsSplitWorker<half>;

template <typename FPType>
class LIFOutSpikes2ThreshsCombine : public poplar::Vertex {
public:

  poplar::Input<poplar::Vector<unsigned>> repeated_out_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> repeated_num_out_spikes;
  poplar::Output<poplar::Vector<unsigned>> out_spikes_ids;
  poplar::Output<unsigned> num_out_spikes;

  bool compute() {

    const size_t numWorkers = repeated_num_out_spikes.size();
    const unsigned sparse_size = out_spikes_ids.size();
    unsigned numSpikesCounter{0};

    unsigned workerStartId{0};
    for (unsigned iwor=0; iwor<numWorkers; ++iwor){
      const unsigned num_out_spikes_ilay = repeated_num_out_spikes[iwor];
      unsigned numIter = ((num_out_spikes_ilay+numSpikesCounter) <= sparse_size) ? num_out_spikes_ilay : (sparse_size-numSpikesCounter);
      for (unsigned i=0; i<numIter; ++i){
        out_spikes_ids[numSpikesCounter] = repeated_out_spikes_ids[workerStartId+i];
        ++numSpikesCounter;
      }
      workerStartId+=sparse_size;
    }
    *num_out_spikes = numSpikesCounter;

    unsigned restNumSpikesCounter = numSpikesCounter;
    const unsigned numMissingSpikes = sparse_size-restNumSpikesCounter;
    const unsigned numIters = numMissingSpikes / numWorkers;
    const unsigned restVals = numMissingSpikes % numWorkers;

    for (unsigned iiter=0; iiter<numIters; ++iiter){
      for (unsigned i=0; i<numWorkers; ++i){
        out_spikes_ids[numSpikesCounter] = repeated_out_spikes_ids[(i+1)*sparse_size-(iiter+1)];
        ++numSpikesCounter;
      }
    }
    unsigned restId = sparse_size-(numIters+1);
    for (unsigned i=0; i<restVals; ++i){
      out_spikes_ids[numSpikesCounter] = repeated_out_spikes_ids[restId];
      ++numSpikesCounter;
      restId += sparse_size;
    }

    return true;
  }
};
template class LIFOutSpikes2ThreshsCombine<float>;
// template class LIFOutSpikes2ThreshsCombine<half>;



template <typename FPType>
class MulInPlaceCustom : public poplar::Vertex {
public:
  
  poplar::InOut<poplar::Vector<FPType>> vec;
  poplar::Input<FPType> val;

  bool compute() {
    for (unsigned i = 0; i < vec.size(); ++i) {
      vec[i] = vec[i]*val;
    }
  return true;
  }
};
template class MulInPlaceCustom<float>;
// template class MulInPlaceCustom<half>;


//---------------------------------------------- backward -----------------------------------------

template <typename FPType>
FPType superspike_surrogate(FPType x, FPType beta) {
  FPType one{1.0};
  return one / std::pow((beta * std::abs(x) + one), 2);
}

template <typename FPType> 
class LIFStateOutGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> fwdState;
  poplar::Input<poplar::Vector<FPType>> thresholds;
  poplar::Input<poplar::Vector<FPType>> dLdoutSpikes;
  poplar::Input<poplar::Vector<unsigned>> fwd_out_spikes_ids;
    
  poplar::InOut<poplar::Vector<FPType>> dLdState;

  bool compute() {
    const FPType beta = 10.0; // TODO  don't hardcode here but give as input
    int sum{0};
    const size_t size_sparse_out = fwd_out_spikes_ids.size();
    for (unsigned i = 0; i < size_sparse_out; ++i) { 
      unsigned idx = fwd_out_spikes_ids[i];
      // TODO is += here correct ?
      dLdState[idx] += dLdoutSpikes[i] * superspike_surrogate(fwdState[idx] - thresholds[idx], beta);
    }
    return true;
  }
};
template class LIFStateOutGrad<float>;
// template class LIFStateOutGrad<half>;


template <typename FPType>
class [[poplar::constraint("elem(*dLdweights_row) != elem(*dLdState)")]] LIFWeightsGrad : public poplar::Vertex { // TODO maybe unnecessary here
// class LIFWeightsGrad : public poplar::Vertex {
// class LIFWeightsGrad : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdState;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;

  poplar::InOut<poplar::Vector<FPType>> dLdweights_row;

  // bool compute(unsigned workerId) {
  bool compute() {
    const size_t batchsize = dLdState.size();
    // unsigned numWorkers = MultiVertex::numWorkers();
    unsigned start_idx{0};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto dLdS = dLdState[ibatch];
      const auto end{fwd_num_inp_spikes[ibatch]};
      // TODO this loop could use multiple threads: It is guarantted that a single elemnent is only touched once!
      // for (unsigned i = workerId; i < end; i+=numWorkers) {
      for (unsigned i = 0; i < end; ++i) {
        dLdweights_row[fwd_inp_spikes_ids[start_idx+i]] += dLdS;
      }
      start_idx += sparse_out_dim;
    }
    return true;
  }
};
template class LIFWeightsGrad<float>;
// template class LIFWeightsGrad<half>;

template <typename FPType>
// class LIFWeightsGradMultiNeuron : public poplar::MultiVertex {
// class LIFWeightsGradMultiNeuron : public poplar::Vertex {
class [[poplar::constraint("elem(*dLdweights) != elem(*dLdState)")]] LIFWeightsGradMultiNeuron : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdState;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;
  poplar::Input<unsigned> batchsize;
  poplar::Input<unsigned> num_neurons;
  // poplar::Input<unsigned> num_weights_per_neuron;

  poplar::InOut<poplar::Vector<FPType>> dLdweights;

  bool compute(unsigned workerId) { // TODO vectorize operations instead of different threads for different neurons
    unsigned numWorkers = MultiVertex::numWorkers();
    for (unsigned ineuron = workerId; ineuron < num_neurons; ineuron+=numWorkers) {
  // bool compute() {
  //   for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron) {
      unsigned start_idx_spikes{0};
      unsigned start_idx_state{ineuron};
      for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
        auto dLdS = dLdState[start_idx_state];
        const auto end{fwd_num_inp_spikes[ibatch]};
        for (unsigned i = 0; i < end; ++i) {
          dLdweights[fwd_inp_spikes_ids[start_idx_spikes+i]*num_neurons+ineuron] += dLdS;
        }
        start_idx_spikes += sparse_out_dim;
        start_idx_state += num_neurons;
      }
    }
    return true;
  }
};
template class LIFWeightsGradMultiNeuron<float>;
// template class LIFWeightsGradMultiNeuron<half>;


// template <typename FPType>
// class LIFWeightsGradMultiNeuronVectorized : public poplar::Vertex {
template <typename FPType>
class [[poplar::constraint("elem(*dLdweights) != elem(*dLdState)")]] LIFWeightsGradMultiNeuronVectorized : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> dLdState;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;
  poplar::Input<unsigned> batchsize;
  poplar::Input<unsigned> num_neurons;
  // poplar::Input<unsigned> num_weights_per_neuron;

  poplar::InOut<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> dLdweights;

  bool compute() { // TODO vectorize operations instead of different threads for different neurons

    unsigned start_idx_spikes{0};
    unsigned neuron_state_id{0};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      const auto end{fwd_num_inp_spikes[ibatch]};
      for (unsigned i = 0; i < end; ++i) {
        const auto spike_idx = fwd_inp_spikes_ids[start_idx_spikes+i]*num_neurons;
        // #pragma clang loop vectorize(enable) interleave(enable) // or -X -ffast-math to popc
        #pragma clang loop unroll(enable)
        for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron) {
          dLdweights[spike_idx+ineuron] += dLdState[neuron_state_id+ineuron];
          // ++neuron_state_id;
        }
      }
      start_idx_spikes += sparse_out_dim;
      neuron_state_id += num_neurons;
    }
    return true;
  }
};
template class LIFWeightsGradMultiNeuronVectorized<float>;
// template class LIFWeightsGradMultiNeuronVectorized<half>;


template <typename FPType>
// class [[poplar::constraint("elem(*weights_rows) != elem(*dLdinp_spike_ids)")]] LIFInpSpikesGradMultiRow : public poplar::Vertex {
class LIFInpSpikesGradMultiRow : public poplar::Vertex {
public:
  // poplar::Input<poplar::Vector<FPType>> weights_rows;
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> weights_rows;
  // poplar::Input<poplar::VectorList<FPType, poplar::VectorListLayout::COMPACT_DELTAN>> weights_rows;
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> dLdStates;
  //  poplar::Input<poplar::Vector<FPType>> dLdStates;
 
  poplar::Input<unsigned> num_neurons;
  poplar::Input<unsigned> sparse_size;

  // poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> fwd_inp_spike_ids; // TODO test out performace gains with vectorization ?
  // poplar::InOut<poplar::Vector<FPType,  poplar::VectorLayout::ONE_PTR, 8>> dLdinp_spike_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
  poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;

  // unsigned end;

  // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
  bool compute() {
    
    // // #pragma clang loop vectorize_width(4) interleave(enable)
    // #pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < sparse_size; ++i) {
      // const auto spikeId = fwd_inp_spike_ids[i];
      FPType sum{0};
      
      #pragma clang loop vectorize(enable) interleave(enable) // or -X -ffast-math to popc
      #pragma clang loop unroll(disable)
      for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron){
        // sum += dLdStates[ineuron] * weights_rows[ineuron];
        // sum += dLdStates[ineuron] * weights_rows[spikeId+ineuron]; // TODO faster like this with flatten and start_idx, or with VectorList ? 
        // sum += dLdStates[ineuron] * weights_rows[fwd_inp_spike_ids[i]][ineuron];
        sum += dLdStates[ineuron] * weights_rows[fwd_inp_spike_ids[i]+ineuron]; // TODO faster like this with flatten and start_idx, or with VectorList ? 
      }
      dLdinp_spike_ids[i] = sum; 
    }
  
    return true;
  }
};
template class LIFInpSpikesGradMultiRow<float>;
// template class LIFInpSpikesGradRowWise<half>;


#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

// class LIFWeightsGradTwoNeuronSIMD : public poplar::Vertex {
class [[poplar::constraint("elem(*dLdweights) != elem(*dLdState)")]] LIFWeightsGradTwoNeuronSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdState;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;
  poplar::Input<unsigned> batchsize;
  poplar::Input<unsigned> num_neurons;
  // poplar::Input<unsigned> num_weights_per_neuron;

  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdweights;

  bool compute() { // TODO vectorize operations instead of different threads for different neurons

    auto dLdweightsAsFloat2 = reinterpret_cast<float2 *>(&dLdweights[0]);
    auto dLdStateFloat2 = reinterpret_cast<const float2 *>(&dLdState[0]);

    unsigned start_idx_spikes{0};
    unsigned neuron_state_id{0};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      const auto end{fwd_num_inp_spikes[ibatch]};
      for (unsigned i = 0; i < end; ++i) {
        const auto spike_idx = fwd_inp_spikes_ids[start_idx_spikes+i];
        dLdweightsAsFloat2[spike_idx] += dLdStateFloat2[neuron_state_id];
      }
      start_idx_spikes += sparse_out_dim;
      neuron_state_id += 1;
    }
    return true;
  }
};

// class LIFWeightsGradMultiNeuronSIMD : public poplar::Vertex {
class [[poplar::constraint("elem(*dLdweights) != elem(*dLdState)")]] LIFWeightsGradMultiNeuronSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdState;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;
  poplar::Input<unsigned> batchsize;
  poplar::Input<unsigned> num_neurons;
  // poplar::Input<unsigned> num_weights_per_neuron;

  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdweights;

  bool compute() { // TODO vectorize operations instead of different threads for different neurons

    auto dLdweightsAsFloat2 = reinterpret_cast<float2 *>(&dLdweights[0]);
    auto dLdStateFloat2 = reinterpret_cast<const float2 *>(&dLdState[0]);

    const auto num_neurons_div2 = num_neurons / 2;

    unsigned start_idx_spikes{0};
    unsigned neuron_state_id{0};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      const auto end{fwd_num_inp_spikes[ibatch]};
      for (unsigned i = 0; i < end; ++i) {
        const auto spike_idx = fwd_inp_spikes_ids[start_idx_spikes+i]*num_neurons_div2;

        // #pragma clang loop unroll(enable)
        // #pragma clang loop unroll_count(4)
        // #pragma clang loop unroll(full)
        for (unsigned ineuron = 0; ineuron < num_neurons_div2; ++ineuron) {
          dLdweightsAsFloat2[spike_idx+ineuron] += dLdStateFloat2[neuron_state_id+ineuron];
          // ++neuron_state_id;
        }
      }
      start_idx_spikes += sparse_out_dim;
      neuron_state_id += num_neurons_div2;
    }
    return true;
  }
};

class LIFInpSpikesGradTwoRowSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> weights_rows;
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdStates;
 
  poplar::Input<unsigned> sparse_size;

  // poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> fwd_inp_spike_ids;
  poplar::InOut<poplar::Vector<float,  poplar::VectorLayout::ONE_PTR, 8>> dLdinp_spike_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
  // poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;

  bool compute() {

    auto weightsAsFloat2 = reinterpret_cast<const float2 *>(&weights_rows[0]);
    auto dLdStatesFloat2 = reinterpret_cast<const float2 *>(&dLdStates[0]);

    for (unsigned i = 0; i < sparse_size; ++i) {
      float2 sumFloat2 = dLdStatesFloat2[0] * weightsAsFloat2[fwd_inp_spike_ids[i]]; // TODO faster like this with flatten and start_idx, or with VectorList ? 
      auto sum = reinterpret_cast<float*>(&sumFloat2);
      dLdinp_spike_ids[i] = sum[0]+sum[1]; 
    }
  
    return true;
  }
};

class LIFInpSpikesGradMultiRowSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> weights_rows;
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> dLdStates;
 
  poplar::Input<unsigned> num_iters;
  poplar::Input<unsigned> sparse_size;

  // poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> fwd_inp_spike_ids;
  poplar::InOut<poplar::Vector<float,  poplar::VectorLayout::ONE_PTR, 8>> dLdinp_spike_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
  // poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;

  bool compute() {

    auto weightsAsFloat2 = reinterpret_cast<const float2 *>(&weights_rows[0]);
    auto dLdStatesFloat2 = reinterpret_cast<const float2 *>(&dLdStates[0]);

    for (unsigned i = 0; i < sparse_size; ++i) {
      float2 sumFloat2 = dLdStatesFloat2[0] * weightsAsFloat2[fwd_inp_spike_ids[i]];
      for (unsigned ineuron = 1; ineuron < num_iters; ++ineuron){
        sumFloat2 += dLdStatesFloat2[0] * weightsAsFloat2[fwd_inp_spike_ids[i]+ineuron];
      }
      auto sum = reinterpret_cast<float*>(&sumFloat2);
      dLdinp_spike_ids[i] = sum[0]+sum[1]; 
    }
    return true;
  }
};


class LIFStateUpdateInPlaceTwoNeuronSIMD : public poplar::Vertex {
// class [[poplar::constraint("elem(*weights) != elem(*syn_input)")]] LIFStateUpdateInPlaceTwoNeuronSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> weights;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> state;
  // poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> syn_input;
  poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> inp_spikes_ids;
  poplar::Input<unsigned> num_inp_spikes;
  poplar::Input<poplar::Vector<float>> decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<float>> oneMinus_decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<float>> threshold; // TODO make sure it's contiguous
  poplar::Input<unsigned> num_neurons;

  bool compute() {
    float zero{0.0};
    auto weightsAsFloat2 = reinterpret_cast<const float2 *>(&weights[0]);
    // auto synInpFloat2 = reinterpret_cast<float2 *>(&syn_input[0]);
    float2 synInpFloat2 = {zero, zero};

    const auto end{num_inp_spikes}; // TODO WTF WHY DOES THIS MAKE A DIFFERENCE ?!
    for (unsigned i = 0; i < end; ++i) {
      // TODO what is
      // synInpFloat2[0] += weightsAsFloat2[2*inp_spikes_ids[i]];
      synInpFloat2 += weightsAsFloat2[inp_spikes_ids[i]];
    }

    float* syn_input = reinterpret_cast<float *>(&synInpFloat2);

    // TODO vectorize this loop ?
    // #pragma clang loop unroll_count(2)
    #pragma clang loop unroll(enable)
    for (unsigned ineuron = 0; ineuron < num_neurons; ++ineuron) {
      if (state[ineuron] > threshold[ineuron]) {
        // *new_state = sum;
        state[ineuron] = oneMinus_decay_constant[ineuron] * syn_input[ineuron];
      } else {
        // *new_state = decay_constant * state + sum;
        state[ineuron] = decay_constant[ineuron] * state[ineuron] + oneMinus_decay_constant[ineuron] * syn_input[ineuron];
      }
    }
    return true;
  }
};

class LIFStateUpdateInPlaceMultiNeuronSIMD : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> weights;
  poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> state;
  // poplar::InOut<poplar::Vector<float, poplar::VectorLayout::ONE_PTR, 8>> syn_input;
  poplar::Input<poplar::Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> inp_spikes_ids;
  poplar::Input<unsigned> num_inp_spikes;
  poplar::Input<poplar::Vector<float>> decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<float>> oneMinus_decay_constant; // TODO make sure it's contiguous
  poplar::Input<poplar::Vector<float>> threshold; // TODO make sure it's contiguous
  poplar::Input<unsigned> num_neurons;

  bool compute() {
    float zero{0.0};
    auto weightsAsFloat2 = reinterpret_cast<const float2 *>(&weights[0]);
    // auto synInpFloat2 = reinterpret_cast<float2 *>(&syn_input[0]);
    unsigned num_neurons_div2 = num_neurons / 2;

    const auto end{num_inp_spikes}; // TODO WTF WHY DOES THIS MAKE A DIFFERENCE ?!
    for (unsigned ineuron2 = 0; ineuron2 < num_neurons_div2; ++ineuron2) {
      float2 synInpFloat2 = {zero, zero};
      for (unsigned i = 0; i < end; ++i) {
        // store num_neurons_div2*inp_spikes_ids[i] somewhere to not always recompute it 
        synInpFloat2 += weightsAsFloat2[num_neurons_div2*inp_spikes_ids[i]+ineuron2];
      }

      float* syn_input = reinterpret_cast<float *>(&synInpFloat2);

      #pragma clang loop unroll(enable)
      for (unsigned ineuron = 2*ineuron2; ineuron < 2*ineuron2+2; ++ineuron) {
        if (state[ineuron] > threshold[ineuron]) {
          // *new_state = sum;
          state[ineuron] = oneMinus_decay_constant[ineuron] * syn_input[ineuron];
        } else {
          // *new_state = decay_constant * state + sum;
          state[ineuron] = decay_constant[ineuron] * state[ineuron] + oneMinus_decay_constant[ineuron] * syn_input[ineuron];
        }
      }
    }

    // process remaining neuron if number of neurons not divisible by two
    if (num_neurons % 2 != 0) {
      const auto ineuron = num_neurons - 1;
      float synInp = zero;
      for (unsigned i = 0; i < end; ++i) {
        synInp += weights[num_neurons*inp_spikes_ids[i]+ineuron];
      }
      if (state[ineuron] > threshold[ineuron]) {
        // *new_state = sum;
        state[ineuron] = oneMinus_decay_constant[ineuron] * synInp;
      } else {
        // *new_state = decay_constant * state + sum;
        state[ineuron] = decay_constant[ineuron] * state[ineuron] + oneMinus_decay_constant[ineuron] * synInp;
      }
    }

    return true;
  }
};

#endif // __IPU__