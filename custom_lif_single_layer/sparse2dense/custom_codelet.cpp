#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cmath> // for std::abs

//---------------------------------------------- forward -----------------------------------------

template <typename FPType>
class Sparse2Dense : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> spikeIds;
  poplar::Input<int> numSpikes;

  poplar::Output<poplar::Vector<FPType>> denseSpikes;
  bool compute() {
    FPType one{1.0};

    for (unsigned i = 0; i < numSpikes; ++i) {
      denseSpikes[spikeIds[i]] = one;
    }
    return true;
  }
};
template class Sparse2Dense<float>;
// template class Sparse2Dense<half>;


//---------------------------------------------- backward -----------------------------------------

template <typename FPType>
class Sparse2DenseGrad : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<FPType>> dLdDenseSpikes;
  poplar::Input<poplar::Vector<FPType>> spikeIds;

  poplar::Output<poplar::Vector<FPType>> dLdSpikeIds;
  bool compute() {
    for (unsigned i = 0; i < spikeIds.size(); ++i) {
      dLdSpikeIds[i] = dLdDenseSpikes[spikeIds[i]];
    }
    return true;
  }
};
template class Sparse2DenseGrad<float>;
// template class Sparse2DenseGrad<half>;