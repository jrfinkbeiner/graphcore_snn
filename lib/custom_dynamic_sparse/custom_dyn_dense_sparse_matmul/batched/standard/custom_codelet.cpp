#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

template <typename FPType>
class DynDenseBinarySparseProduct : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> matrix_slice;
  poplar::Input<poplar::Vector<unsigned>> spike_ids; // Cast beforehand to int/unsigned might be more efficient ?
  poplar::Input<unsigned> num_spikes; // Cast beforehand to unsigned might be more efficient ?
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
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
  poplar::Input<unsigned> sparse_size;

  poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;
  // poplar::Output<poplar::Vector<FPType>> dLdinp_spike_ids;

  // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
  bool compute() {
    // TODO this sneakily use `dLdinp_spike_ids` tensor as intermediate storage for relevant weights
    // TODO should probably be removed for readibility and correct future behaviour when `weights_row`
    // TODO and `dLdinp_spike_ids` can have a different type
    for (unsigned i = 0; i < sparse_size; ++i) {
      dLdinp_spike_ids[i] = matrix_row[fwd_inp_spike_ids[i]];
    }
        
    // #pragma clang loop vectorize_width(4) interleave(enable)
    #pragma clang loop vectorize(enable) interleave(enable)
    for (unsigned i = 0; i < sparse_size; ++i) {
      //  dLdinp_spike_ids[i] = dLdy * matrix_row[fwd_inp_spike_ids[i]];
      dLdinp_spike_ids[i] *= dLdy;
    }
    return true;
  }
};
template class DynDenseBinarySparseProductGradInputsRowWise<float>;
// template class DynDenseBinarySparseProductGradInputsRowWise<half>;


template <typename FPType>
class DynDenseBinarySparseProductGradInputsMultiRow : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> matrix_rows;
  poplar::Input<poplar::Vector<FPType>> dLdy;
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spike_ids;
  poplar::Input<unsigned> sparse_size;
  poplar::Input<unsigned> num_rows;
  poplar::Input<unsigned> row_size;

  poplar::InOut<poplar::Vector<FPType>> dLdinp_spike_ids;
  // poplar::Output<poplar::Vector<FPType>> dLdinp_spike_ids;

  // TODO this could use multiple threads: It is guarantted that a single elemnent is only touched once!
  bool compute() {
    unsigned start_idx{0};

    for (unsigned irow = 0; irow < num_rows; ++irow){
      const auto dLdyi = dLdy[irow];
      for (unsigned i = 0; i < sparse_size; ++i) {
        dLdinp_spike_ids[i] += dLdyi * matrix_rows[start_idx+fwd_inp_spike_ids[i]]; // TODO faster like this with flatten and start_idx, or with VectorList ? 
      }
      start_idx += row_size;
    }
    return true;
  }
};
template class DynDenseBinarySparseProductGradInputsMultiRow<float>;
// template class DynDenseBinarySparseProductGradInputsMultiRow<half>;



template <typename FPType>
class DynDenseBinarySparseProductGradInputs : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdy;
  poplar::Input<poplar::Vector<FPType>> weights;
  poplar::Input<poplar::Vector<unsigned>> sparse_vec;
  poplar::Input<unsigned> num_cols;
  poplar::Input<unsigned> num_rows;
  poplar::Input<unsigned> num_nzelements;
  poplar::Output<poplar::Vector<FPType>> dLdx;

  bool compute() {
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
  poplar::Input<poplar::Vector<unsigned>> fwd_inp_spikes_ids;
  poplar::Input<poplar::Vector<unsigned>> fwd_num_inp_spikes; // TODO what is the type here ?
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