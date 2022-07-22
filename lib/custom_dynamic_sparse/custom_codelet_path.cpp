#include <experimental/filesystem>

#include "custom_dynamic_sparse/custom_codelet_path.hpp"

namespace fs = std::experimental::filesystem;

fs::path get_custom_codelet_path(std::vector<std::string> dirnames){
    fs::path custom_codelet_path(CUSTOM_CODELET_BASE_PATH_STRING);
    for (auto &dirname: dirnames){
        custom_codelet_path /= dirname;
    }
    return custom_codelet_path;
}

std::string get_custom_codelet_path_string(std::vector<std::string> dirnames){
    fs::path custom_codelet_path = get_custom_codelet_path(dirnames);
    std::string custom_codelet_path_string = custom_codelet_path.generic_string();
    return custom_codelet_path_string;
}