cmake_minimum_required(VERSION 3.10.2) # Default version in Ubuntu 18.04.
project("dynamic sparse custom IPU implementations")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" )
include(AddGPLibrary)
include(AddTool)

include(GNUInstallDirs)
# include(include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-DCUSTOM_CODELET_BASE_PATH=${CMAKE_BINARY_DIR}/lib/custom_dynamic_sparse")

set(POPC_EXECUTABLE popc)
set(POPC_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
# set(COMPILER_FLAGS -shared -fpic -Wl,-soname)
# set(COMPILER_LIBS -lcustom_dynamic_sparse -lpoplar -lpoputil -lpopnn -lpopops -lstdc++fs)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -g)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3)
  set(CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}")
  # set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif()


message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

add_subdirectory(lib)
add_subdirectory(custom_ops)
