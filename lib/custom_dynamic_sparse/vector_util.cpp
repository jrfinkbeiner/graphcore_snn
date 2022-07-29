#include <vector>
#include <iostream>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Zero.hpp>
#include <popops/Cast.hpp>


void clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst, size_t offset, const poplar::DebugNameAndId &dnai = {}) {
  std::transform(src.begin()+offset, src.end(), std::back_inserter(dst), [&graph, &dnai](const poplar::Tensor &t){return graph.clone(t, dnai);});
}

std::vector<poplar::Tensor> clone_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  clone_tensor_vector(graph, src, dst, 0, dnai);
  return dst;
}

std::vector<poplar::Tensor> cast_tensor_vector(poplar::Graph& graph, const std::vector<poplar::Tensor> &src, poplar::Type &dtype, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::vector<poplar::Tensor> dst;
  std::transform(src.begin(), src.end(), std::back_inserter(dst), [&graph, &dtype, &prog,  &dnai](const poplar::Tensor &t) -> poplar::Tensor {return popops::cast(graph, t, dtype, prog, dnai);});
  return dst;
}

void zero_tensor_vector(poplar::Graph& graph, std::vector<poplar::Tensor> &vec, poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai = {}) {
  std::for_each(vec.begin(), vec.end(), [&graph, &prog, &dnai](poplar::Tensor &t){popops::zero(graph, t, prog, dnai);});
}

void extend_tensor_vector(std::vector<poplar::Tensor> &src, std::vector<poplar::Tensor> &dst){
  std::transform(src.begin(), src.end(), std::back_inserter(dst), [](poplar::Tensor &t) -> poplar::Tensor {return t;});
}
