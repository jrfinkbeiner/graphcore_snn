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
    FPType threshold_{threshold};
    FPType one{1.0};
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
    FPType secThreshMul{0.9};
    size_t numStates = state.size();
    size_t sizeSparseOut = out_spikes_ids.size();
    for (unsigned i = 0; i < numStates; ++i) {
      // TODO reformulate to only use +
      if ((state[i] > thresholds[i]*secThreshMul) || (numStates - i >= (sizeSparseOut-(numSpikesCounter+numGradsCounter)))) {
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
    FPType beta = 10.0; // TODO  don't hardcode here but give as input
    int sum{0};
    size_t size_sparse_out = fwd_out_spikes_ids.size();
    for (unsigned i = 0; i < size_sparse_out; ++i) { 
      unsigned idx = fwd_out_spikes_ids[i];
      dLdState[idx] += dLdoutSpikes[i] * superspike_surrogate(fwdState[idx] - thresholds[idx], beta);
    }
    return true;
  }
};
template class LIFStateOutGrad<float>;
// template class LIFStateOutGrad<half>;


template <typename FPType>
class LIFWeightsGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdState;
  poplar::Input<FPType> decay_constant;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes; // TODO to int when possible
  poplar::Input<unsigned> sparse_out_dim;

  poplar::InOut<poplar::Vector<FPType>> dLdweights_row;
  

  bool compute() {
    size_t batchsize = dLdState.size();
    FPType one{1.0};
    FPType decay_const{decay_constant};
    FPType decay_val{one-decay_const};
    unsigned start_idx;
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto dLdS = dLdState[ibatch];
      start_idx = sparse_out_dim*ibatch;
      const auto end{fwd_num_inp_spikes[ibatch]};
      // TODO this loop could use multiple threads: It is guarantted that a single elemnent is only touched once!
      for (unsigned i = 0; i < end; ++i) {
        dLdweights_row[fwd_inp_spikes_ids[start_idx+i]] += decay_val * dLdS;
      }
    }
    return true;
  }
};
template class LIFWeightsGrad<float>;
// template class LIFWeightsGrad<half>;


// template <typename FPType>
// class LIFInpSpikesGradRowWise : public poplar::Vertex {
// public:
//   poplar::Input<poplar::Vector<FPType>> weights_row;
//   poplar::Input<FPType> dLdState;
//   poplar::Input<FPType> decay_constant;
//   poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
 
//   poplar::Output<poplar::Vector<FPType>> dLdinp_spike_ids;

//   // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
//   bool compute() {
//     FPType one{1.0};    
//     const auto end{fwd_inp_spike_ids.size()};
//     for (unsigned i = 0; i < end; ++i) {
//        dLdinp_spike_ids[i] = dLdState * (one-decay_constant) * weights_row[fwd_inp_spike_ids[i]];
//     }
//     return true;
//   }
// };
// template class LIFInpSpikesGradRowWise<float>;
// // template class LIFInpSpikesGradRowWise<half>;

template <typename FPType>
class LIFInpSpikesGradRowWise : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> weights_row;
  // poplar::InOut<poplar::Vector<FPType>> relevant_weights;
  poplar::Input<FPType> dLdState;
  poplar::Input<FPType> decay_constant;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
 
  poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;
  // poplar::Output<poplar::Vector<FPType>> dLdinp_spike_ids;

  // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
  bool compute() {
    FPType one{1.0};
    FPType decay_const{decay_constant};
    FPType decay_val{one-decay_const};
    FPType dLdStat{dLdState};
    const auto end{fwd_inp_spike_ids.size()};
    // TODO this sneakily use `dLdinp_spike_ids` tensor as intermediate storage for relevant weights
    // TODO should probably be removed for readibility and correct future behaviour when `weights_row`
    // TODO and `dLdinp_spike_ids` can have a different type
    for (unsigned i = 0; i < end; ++i) {
      dLdinp_spike_ids[i] = weights_row[fwd_inp_spike_ids[i]];
    }
        
    // #pragma clang loop vectorize_width(4) interleave(enable)
    #pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < end; ++i) {
      //  dLdinp_spike_ids[i] = dLdStat * decay_val * weights_row[fwd_inp_spike_ids[i]];
      dLdinp_spike_ids[i] = dLdStat * decay_val * dLdinp_spike_ids[i];
    }
    return true;
  }
};
template class LIFInpSpikesGradRowWise<float>;
// template class LIFInpSpikesGradRowWise<half>;