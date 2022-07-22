#pragma once

#include <experimental/filesystem>

#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define CUSTOM_CODELET_BASE_PATH_STRING STR_VALUE(CUSTOM_CODELET_BASE_PATH)


namespace fs = std::experimental::filesystem;

fs::path get_custom_codelet_path(std::vector<std::string> dirnames);
std::string get_custom_codelet_path_string(std::vector<std::string> dirnames);

// #define CUSTOM_CODELET_BASE_PATH // std::getenv("CUSTOM_CODELET_BASE_PATH")
// const std::string custom_codelet_base_path = str(CUSTOM_CODELET_BASE_PATH);
// const std::string custom_codelet_base_path = CUSTOM_CODELET_BASE_PATH;
// const std::string custom_codelet_base_path = std::getenv("CUSTOM_CODELET_BASE_PATH");

// const std::string custom_codelet_base_path[CUSTOM_CODELET_BASE_PATH];
// const std::string custom_codelet_base_path(CUSTOM_CODELET_BASE_PATH);

// std::string get_gp_path(){
//   auto env = std::getenv("CUSTOM_CODELET_BASE_PATH");
//   gp_path = boost::filesystem
// }