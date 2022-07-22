#include <vector>
#include <iostream>

#include "custom_dynamic_sparse/string_util.hpp"

std::vector<std::string> split_string(const std::string& s, char seperator)
{
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(seperator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );
        output.push_back(substring);
        prev_pos = ++pos;
    }
    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word
    return output;
}

std::vector<size_t> convert_vecOfStr_to_vecOfSizet(const std::string& s, char seperator) {
  std::vector<std::string> sparse_sizes_strs = split_string(s, seperator);
  auto num_layers = sparse_sizes_strs.size();
  std::vector<size_t> outputs;
  for (unsigned i=0; i<num_layers; ++i){
    size_t size_sparse_out;
    sscanf(sparse_sizes_strs[i].c_str(), "%zu", &size_sparse_out);
    outputs.push_back(size_sparse_out);
  }
  return outputs; 
}

// template<typename T>
// void printVector(std::vector<T> vec) {
//   std::cout << "{";
//   for (auto val: vec) {
//     std::cout << val << ", ";
//   }
//   std::cout << "}"<< std::endl;
// }