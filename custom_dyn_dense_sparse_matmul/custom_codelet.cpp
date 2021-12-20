#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

template <typename FPType>
class DynDenseBinarySparseProduct : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> matrix_slice;
  poplar::Input<poplar::Vector<FPType>> sparse_vec;
  poplar::Input<int> num_nzelements;
  poplar::Output<FPType> output;

  bool compute() {
    FPType sum = 0;
    for (unsigned i = 0; i < num_nzelements; ++i)
      sum += matrix_slice[sparse_vec[i]];
    *output = sum;
    return true;
  }
};

template class DynDenseBinarySparseProduct<float>;
template class DynDenseBinarySparseProduct<half>;


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
  poplar::Input<FPType> dLdyi;
  poplar::Input<poplar::Vector<FPType>> sparse_vec;
  poplar::Input<int> num_nzelements;
  poplar::Output<poplar::Vector<FPType>> dLdW_row;

  bool compute() {
    // initialize to zeros
    for (unsigned i; i < dLdW_row.size(); ++i) {
      dLdW_row[i] = 0.0;
    }
    // overrwrite sparse elements
    for (unsigned i = 0; i < num_nzelements; ++i) {
      dLdW_row[sparse_vec[i]] = dLdyi;
    }
    return true;
  }
};

template class DynDenseBinarySparseProductGradWeight<float>;
template class DynDenseBinarySparseProductGradWeight<half>;