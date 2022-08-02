#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath> // for std::abs, std::pow

// template <typename FPType>
// class SpikesTwoThreshs : public poplar::Vertex {
// public:

//   poplar::Input<poplar::Vector<FPType>> state;
//   poplar::Input<poplar::Vector<FPType>> thresholds;

//   poplar::Output<poplar::Vector<FPType>> out_spikes_ids; // TODO change to unsigned ?
//   poplar::Output<int> num_out_spikes; // TODO unsigned
//   // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

//   bool compute() {
//     unsigned numSpikesCounter{0};
//     unsigned numGradsCounter{0};
//     FPType secThreshMul{0.9};
//     const size_t numStates = state.size();
//     const size_t sizeSparseOut = out_spikes_ids.size();
//     for (unsigned i = 0; i < numStates; ++i) {
//       // TODO reformulate to only use +
//       if ((state[i] > thresholds[i]*secThreshMul) || (numStates - i >= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
//         if (state[i] > thresholds[i]) {
//           out_spikes_ids[numSpikesCounter] = i;
//           ++numSpikesCounter;
//         } else {
//           // Fill up the array with non-spike values in reverse from behind 
//           out_spikes_ids[sizeSparseOut-1-numGradsCounter] = i;
//           ++numGradsCounter;
//         }
//       }
//       if (numSpikesCounter+numGradsCounter >= sizeSparseOut) break; // TODO just implement as while
//     }
//     *num_out_spikes = numSpikesCounter;
//     // *num_out_spikes = sum; // TODO uncomment when int is supported

//     return true;
//   }
// };
// template class SpikesTwoThreshs<float>;
// // template class SpikesTwoThreshs<half>;

template <typename FPType>
class SpikesTwoThreshs : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> state;
  poplar::Input<poplar::Vector<FPType>> thresholds;

  poplar::Output<poplar::Vector<unsigned>> out_spikes_ids; // TODO change to unsigned ?
  poplar::Output<unsigned> num_out_spikes; // TODO unsigned
  // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

  bool compute() {
    unsigned numSpikesCounter{0};
    unsigned numGradsCounter{0};
    FPType secThreshMul{0.9};
    const size_t numStates = state.size();
    const size_t sizeSparseOut = out_spikes_ids.size();
    for (unsigned i = 0; i < numStates; ++i) {
      // TODO reformulate to only use +
      if ((state[i] > secThreshMul) || (numStates - i <= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
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
template class SpikesTwoThreshs<float>;
// template class SpikesTwoThreshs<half>;

// the only real difference to `SpikesTwoThreshs` is the additional counter `state_id`, 
// as in the parallel/splitWorker case the loop iteration `i` is not equal to the state id. 
// TODO for code readability could just remove the base call and only kepp the 
// (slightly less efficient) one with the additional `state_id` counter
template <typename FPType>
class SpikesTwoThreshsSplitWorker : public poplar::Vertex {
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
      // TODO reformulate to not use + instead of - operations 
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
template class SpikesTwoThreshsSplitWorker<float>;
// template class SpikesTwoThreshsSplitWorker<half>;


class SpikesTwoThreshsCombine : public poplar::Vertex {
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

template <typename FPType>
FPType superspike_surrogate(FPType x, FPType beta) {
  FPType one{1.0};
  return one / std::pow((beta * std::abs(x) + one), 2);
  // return one;
}

template <typename FPType>
class StateGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> fwdState;
  poplar::Input<poplar::Vector<FPType>> thresholds;
  poplar::Input<poplar::Vector<FPType>> dLdoutSpikes;
  poplar::Input<poplar::Vector<unsigned>> fwd_out_spikes_ids; // TODO unsigned ?
    
  poplar::Output<poplar::Vector<FPType>> dLdState;

  bool compute() {
    FPType beta = 10.0; // TODO  don't hardcode here but give as input
    int sum{0};
    size_t size_sparse_out = fwd_out_spikes_ids.size();
    for (unsigned i = 0; i < size_sparse_out; ++i) { 
      unsigned idx = fwd_out_spikes_ids[i];
      dLdState[idx] = dLdoutSpikes[i] * superspike_surrogate(fwdState[idx] - thresholds[idx], beta);
    }
    return true;
  }
};
template class StateGrad<float>;
// template class StateGrad<half>;