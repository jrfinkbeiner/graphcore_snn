#pragma once

#include <vector>

std::vector<std::string> split_string(const std::string& s, char seperator);

std::vector<size_t> convert_vecOfStr_to_vecOfSizet(const std::string& s, char seperator);

// template<typename T>
// void printVector(std::vector<T> vec);
template<typename T>
void printVector(std::vector<T> vec) {
  std::cout << "{";
  for (auto val: vec) {
    std::cout << val << ", ";
  }
  std::cout << "}"<< std::endl;
}