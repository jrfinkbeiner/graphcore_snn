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

  poplar::Output<poplar::Vector<FPType>> out_spikes_ids; // TODO change to unsigned ?
  poplar::Output<int> num_out_spikes; // TODO unsigned
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
  poplar::Input<poplar::Vector<FPType>> fwd_out_spikes_ids; // TODO unsigned ?
    
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