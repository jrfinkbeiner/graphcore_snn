#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

template <typename FPType>
class DynDenseBinarySparseProduct : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> matrix_slice;
  poplar::Input<poplar::Vector<FPType>> spike_ids; // Cast beforehand to int/unsigned might be more efficient ?
  poplar::Input<int> num_spikes; // Cast beforehand to unsigned might be more efficient ?
  poplar::Output<FPType> output;

  bool compute() {
    FPType sum{0.0};
    for (unsigned i = 0; i < num_spikes; ++i)
      sum += matrix_slice[spike_ids[i]];
    *output = sum;
    return true;
  }
};
template class DynDenseBinarySparseProduct<float>;
template class DynDenseBinarySparseProduct<half>;



template <typename FPType>
class DynDenseBinarySparseProductGradInputsRowWise : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> matrix_row;
  poplar::Input<FPType> dLdy;
  poplar::Input<poplar::Vector<FPType>> fwd_inp_spike_ids; // TODO use cast again ?
 
  poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;
  // poplar::Output<poplar::Vector<FPType>> dLdinp_spike_ids;

  // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
  bool compute() {
    const auto end{fwd_inp_spike_ids.size()};
    // TODO this sneakily use `dLdinp_spike_ids` tensor as intermediate storage for relevant weights
    // TODO should probably be removed for readibility and correct future behaviour when `weights_row`
    // TODO and `dLdinp_spike_ids` can have a different type
    for (unsigned i = 0; i < end; ++i) {
      dLdinp_spike_ids[i] = matrix_row[fwd_inp_spike_ids[i]];
    }
        
    // #pragma clang loop vectorize_width(4) interleave(enable)
    #pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < end; ++i) {
      //  dLdinp_spike_ids[i] = dLdy * matrix_row[fwd_inp_spike_ids[i]];
      dLdinp_spike_ids[i] = dLdy * dLdinp_spike_ids[i];
    }
    return true;
  }
};
template class DynDenseBinarySparseProductGradInputsRowWise<float>;
// template class DynDenseBinarySparseProductGradInputsRowWise<half>;


template <typename FPType>
class DynDenseBinarySparseProductGradInputs : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdy;
  poplar::Input<poplar::Vector<FPType>> weights;
  poplar::Input<int> num_cols;
  poplar::Input<poplar::Vector<FPType>> sparse_vec;
  poplar::Input<int> num_nzelements;
  poplar::Output<poplar::Vector<FPType>> dLdx;

  bool compute() {
    size_t num_rows = dLdy.size();
    for (unsigned icol = 0; icol < num_nzelements; ++icol) {
      auto col_idx = sparse_vec[icol];
      FPType sum = 0;
      for (unsigned irow = 0; irow < num_rows; ++irow) {
        sum += weights[irow*num_cols+col_idx] * dLdy[irow];
      }
      dLdx[icol] = sum;
    }
    // TODO could set the rest to zero. But these should't be used anyways...?
    for (unsigned i = num_nzelements; i < dLdx.size(); ++i) {
      dLdx[i] = 0.0;
    }
    return true;
  }
};
template class DynDenseBinarySparseProductGradInputs<float>;
template class DynDenseBinarySparseProductGradInputs<half>;


template <typename FPType>
class DynDenseBinarySparseProductGradWeight : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdy;
  poplar::Input<poplar::Vector<FPType>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<int>> fwd_num_inp_spikes; // TODO what is the type here ?
  poplar::Output<poplar::Vector<FPType>> dLdweights_row;
  poplar::Input<unsigned> sparse_out_dim;

  bool compute() {
    size_t batchsize{dLdy.size()};
    for (unsigned ibatch = 0; ibatch < batchsize; ++ibatch) {
      auto dLdyi = dLdy[ibatch];
      const unsigned start_idx{sparse_out_dim*ibatch};
      const auto end{fwd_num_inp_spikes[ibatch]};
      // TODO this loop could use multiple threads: It is guarantted that a single elemnent is only touched once! 
      // but for that batches have to be handled sequentially
      // for (unsigned i = workerId; i < end; i+=numWorkers) {
      for (unsigned i = 0; i < end; ++i) {
        dLdweights_row[fwd_inp_spikes_ids[start_idx+i]] += dLdyi;
      }
    }
    return true;
  }
};
template class DynDenseBinarySparseProductGradWeight<float>;
template class DynDenseBinarySparseProductGradWeight<half>;