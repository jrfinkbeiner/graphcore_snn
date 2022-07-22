# include(GNUInstallDirs)
# get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

add_gp_executables(
  NAME
    some_name
  CPP_SOURCES
    custom_dynamic_sparse/custom_dyn_dense_sparse_matmul/batched/standard/custom_codelet.cpp
    custom_dynamic_sparse/custom_select_spikes/twoThresh/custom_codelet.cpp
)

add_library(custom_dynamic_sparse SHARED
  custom_dynamic_sparse/custom_dyn_dense_sparse_matmul/batched/standard/poplar_code.cpp
  custom_dynamic_sparse/custom_dyn_dense_sparse_matmul/batched/standard/poplar_code.hpp
  custom_dynamic_sparse/custom_select_spikes/twoThresh/poplar_code.cpp
  custom_dynamic_sparse/custom_select_spikes/twoThresh/poplar_code.hpp
  custom_dynamic_sparse/custom_codelet_path.cpp
  custom_dynamic_sparse/string_util.cpp
)

target_include_directories(custom_dynamic_sparse
PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
PRIVATE
  .
)

target_link_libraries(custom_dynamic_sparse poprand popnn poplin popops popfloat poputil stdc++fs)