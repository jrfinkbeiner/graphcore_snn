#include <iostream>
#include <vector>
#include <string> 

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
// #include <poputil/Util.hpp>
#include <popops/Zero.hpp>
#include <popops/ElementWise.hpp>


/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 5; // 5 neccessary for IPU model?
}

extern "C" void Build_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {
  std::cout << "\nBuild_metadata" << std::endl;
  std::cout << "num_inputs: " << num_inputs << std::endl;
  std::cout << "allocating_indices.size(): " << allocating_indices.size() << std::endl;
  num_inputs = 10;
  for (std::int64_t i=0; i<num_inputs; ++i){
    allocating_indices.push_back(i);
  }
  std::cout << "allocating_indices.size(): " << allocating_indices.size() << std::endl;
  // is_elementwise = true;
  is_stateless = true;
}



extern "C" poplar::Tensor Build_allocator(
    poplar::Graph& graph,
    std::uint32_t operand,
    const std::vector<size_t>& shape,
    poplar::Type type,
    const std::string& attributes,
    const std::string& debug_prefix) {
    
  std::cout << "\nBuild_allocator: " << operand << std::endl;
  
  poplar::DebugNameAndId dnai{debug_prefix};
  std::string tensor_name = "alloc_inp_" + std::to_string(operand);
  poplar::Tensor allocTensor = graph.addVariable(type, shape, {dnai, tensor_name});
  graph.setTileMapping(allocTensor, 1);
  return allocTensor;
}




// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& attributes, const std::string& debug_prefix) {

  poplar::DebugNameAndId dnai{debug_prefix};
  poplar::program::Sequence prog;

  std::cout << "\nRUN forward" << std::endl;

  for (unsigned i=0; i<inputs.size(); ++i) {
    auto tileMap = graph.getTileMapping(inputs[i]);
    std:: cout << std::endl;
    std::cout << i << ": tileMap[0].size(): " << tileMap[0].size() << std::endl; 
    std::cout << i << ": tileMap[1].size(): " << tileMap[1].size() << std::endl; 
    std::cout << i << ": tileMap[2].size(): " << tileMap[2].size() << std::endl; 
  }
  // poplar::Tensor dense_spikes = graph.addVariable(dtype, {seq_len, batchsize, size_dense});
//   for (unsigned i = 0; i < seq_len; ++i) {
//     for (unsigned j = 0; j < batchsize; ++j) {
//       graph.setTileMapping(dense_spikes[i][j], i*batchsize/batchesPerTile+j/batchesPerTile);
//     }
//   }

  poplar::Tensor out = graph.clone(inputs[0], {dnai, "init_out_tensor"});
  // do something
  popops::addWithOutput(graph, inputs[0], inputs[1], out, prog, {dnai, "add_inp_0_and_1"});

  // popops::zero(graph, out, prog, dnai); 

  outputs.push_back(out);

  std::cout << "\nDONE forward" << std::endl;

  return prog;
}
