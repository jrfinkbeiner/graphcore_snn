#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath> // for std::abs


//---------------------------------------------- forward -----------------------------------------

template <typename FPType>
class LIFStateUpdate : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> weights;
  poplar::Input<FPType> state;
  poplar::Input<poplar::Vector<FPType>> inp_spikes_ids;
  poplar::Input<int> num_inp_spikes;
  poplar::Input<FPType> decay_constant;
  poplar::Input<FPType> threshold;

  poplar::Output<FPType> new_state;
  bool compute() {
    FPType sum{0.0};
    FPType state_{state};
    FPType threshold_{threshold};
    FPType one{1.0};

    for (unsigned i = 0; i < num_inp_spikes; ++i) {
      sum += weights[inp_spikes_ids[i]];
    }
    if (state_ > threshold_) {
      // *new_state = sum;
      *new_state = (one - decay_constant) * sum;
    } else {
      // *new_state = decay_constant * state + sum;
      *new_state = decay_constant * state + (one - decay_constant) * sum;
    }
    return true;
  }
};
template class LIFStateUpdate<float>;
// template class LIFStateUpdate<half>;


template <typename FPType>
class LIFOutSpikes : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> state;
  poplar::Input<poplar::Vector<FPType>> thresholds;

  poplar::Output<poplar::Vector<FPType>> out_spikes_ids;
  poplar::Output<FPType> num_out_spikes;
  // poplar::Output<int> num_out_spikes; // TODO uncomment this should be int longterm...

  bool compute() {
    int sum{0};
    size_t size_sparse_out = out_spikes_ids.size();
    for (unsigned i = 0; i < state.size(); ++i) {
      if (state[i] > thresholds[i]) {
        out_spikes_ids[sum] = i;
        ++sum;
        if (sum >= size_sparse_out) break;
      }
    }
    *num_out_spikes = static_cast<FPType>(sum);
    // *num_out_spikes = sum; // TODO uncomment when int is supported
    return true;
  }
};
template class LIFOutSpikes<float>;
// template class LIFOutSpikes<half>;


//---------------------------------------------- backward -----------------------------------------

template <typename FPType>
FPType superspike_surrogate(FPType x, FPType beta) {
  FPType one{1.0};
  return one / std::pow(beta * std::abs(x) + one, 2);
}

template <typename FPType>
class LIFStateOutGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> fwdState;
  poplar::Input<poplar::Vector<FPType>> thresholds;
  poplar::Input<poplar::Vector<FPType>> dLdoutSpikes;
  poplar::Input<poplar::Vector<FPType>> fwd_out_spikes_ids;
  poplar::Input<FPType> fwd_num_out_spikes;
  // poplar::Input<int> fwd_num_out_spikes; // TODO uncomment and change to int when possible
  
  poplar::Output<poplar::Vector<FPType>> dLdState;

  bool compute() {
    FPType beta = 10.0; // TODO  don't hardcode here but give as input
    int sum{0};
    size_t size_sparse_out = fwd_out_spikes_ids.size();
    // !!! TODO !!! uncomment when gradient computation for entries > fwd_out_spikes_ids is figured out
    // for (unsigned i = 0; i < size_sparse_out; ++i) { 
    size_t num_out_spikes = static_cast<size_t>(fwd_num_out_spikes);
    for (unsigned i = 0; i < num_out_spikes; ++i) {
      size_t idx = fwd_out_spikes_ids[i];
      dLdState[idx] += dLdoutSpikes[i] * superspike_surrogate(fwdState[idx] - thresholds[idx], beta);
    }
    return true;
  }
};
template class LIFStateOutGrad<float>;
// template class LIFStateGrad<half>;


template <typename FPType>
class LIFWeightsGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdState;
  poplar::Input<poplar::Vector<FPType>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<int>> fwd_num_inp_spikes;
  poplar::Input<unsigned> sparse_out_dim;

  poplar::Output<poplar::Vector<FPType>> dLdweights_row;
  

  bool compute() {
    size_t batchsize = dLdState.size();
    // unsigned sparse_out_dim_ = sparse_out_dim;
    unsigned start_idx;
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto dLdS = dLdState[ibatch];
      start_idx = sparse_out_dim*ibatch;
      unsigned num_inp_spikes = static_cast<unsigned>(fwd_num_inp_spikes[ibatch]);
      for (unsigned i = 0; i < num_inp_spikes; ++i) {
        dLdweights_row[fwd_inp_spikes_ids[start_idx+i]] += dLdS;
      }
    }
    return true;
  }
};
template class LIFWeightsGrad<float>;
// template class LIFStateGrad<half>;


template <typename FPType>
class LIFInpSpikesGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> weights_column;
  poplar::Input<poplar::Vector<FPType>> dLdState;
  poplar::Input<poplar::Vector<FPType>> fwd_inp_spike_ids;
  poplar::Input<int> fwd_num_inp_spikes;
  poplar::Input<int> col_id; // TODO FPType to match inp_spike_ids ? or just int/size_t ?
 
  poplar::Output<FPType> dLdx;

  bool compute() {
    bool id_in_inp_spikes{false};
    // !!! TODO !!! iter over all fwd_inp_spike_ids when figured out how to include non spiked gradients
    for (unsigned i = 0; i < fwd_num_inp_spikes; ++i) {
      if (col_id == fwd_inp_spike_ids[i]) {
        id_in_inp_spikes = true;
        break;
      }
    }
    if (id_in_inp_spikes) {
      FPType val{0.0};
      for (unsigned i = 0; i < weights_column.size(); ++i) {
        val += dLdState[i] * weights_column[i];
      }
      *dLdx = val;
    }
    return true;
  }
};
template class LIFInpSpikesGrad<float>;
// template class LIFStateGrad<half>;


template <typename FPType>
class LIFSelectInpSpikesGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> fwd_inp_spike_ids;
  poplar::Input<int> fwd_num_inp_spikes;
  poplar::Input<poplar::Vector<FPType>> dLdx;

  poplar::Output<poplar::Vector<FPType>> dLdInpSpikes;

  bool compute() {
    // !!! TODO !!! iter over all fwd_inp_spike_ids when figured out how to include non spiked gradients
    for (unsigned i = 0; i < fwd_num_inp_spikes; ++i) {
      dLdInpSpikes[i] = dLdx[fwd_inp_spike_ids[i]];
    }
    return true;
  }
};
template class LIFSelectInpSpikesGrad<float>;
// template class LIFStateGrad<half>;