# function that creates a GP executables for all targets
# TODO fucture could be build seperately for all targets

# this function requires the following global variables to exist (all of which
# are defined in the top level CMakeLists.txt):
#   - POPC_EXECUTABLE
#   - POPC_FLAGS

function(add_custom_ops)
  # $(CC) $(BASE_PATH_SELECTSPIKES)/custom_op.cpp -shared -fpic -Wl,-soname,$@ -o $@ $(BASE_PATH_SELECTSPIKES)/poplar_code.cpp $(LIBS)
  # add_executable(${name} ${ARGN})
  
  # find_library(CUSTOM_DYNAMIC_SPARSE_LIB NAME custom_dynamic_sparse REQUIRED PATHS "${CMAKE_BINARY_DIR}/lib")
  # message("CUSTOM_DYNAMIC_SPARSE_LIB: ${CUSTOM_DYNAMIC_SPARSE_LIB}")

  # # Find library's headers and add it as a search path.
  # # Provide the name of one header file you know should
  # # be present in mycustomlib's include dir.
  # find_path(MCL_HEADER_PATH select_spikes_twoThresh.hpp PATH_SUFFIXES ${CUSTOM_DYNAMIC_SPARSE_LIB})
  # message("MCL_HEADER_PATH: ${MCL_HEADER_PATH}")
  # # target_include_directories(myprogram PUBLIC ${MCL_HEADER_PATH})


  foreach(CPP_SOURCE ${ARGN})
    get_filename_component(FILEDIR ${CPP_SOURCE} DIRECTORY)
    get_filename_component(FILENAME ${CPP_SOURCE} NAME_WE)
    set(REL_SOURCE_FILE "${FILEDIR}/${FILENAME}")
    string(REPLACE "/" "_" CUSTOM_OP_NAME ${REL_SOURCE_FILE})

    # add_custom_target(${CUSTOM_OP_NAME} ALL DEPENDS ${CUSTOM_OP_NAME}
    #   SOURCES
    #     ${FULL_CPP_SOURCE_NAME}
    #   )
    add_library(${CUSTOM_OP_NAME} SHARED
      ${CPP_SOURCE}
    )

    # set(MYLIB ${CMAKE_BINARY_DIR}/lib/libcustom_dynamic_sparse.so)
    # message("MYLIB ${MYLIB}")

    # target_link_libraries(${CUSTOM_OP_NAME} ${MYLIB} poprand popnn poplin popops popfloat poputil stdc++fs)
    # target_link_libraries(${CUSTOM_OP_NAME} PUBLIC CUSTOM_DYNAMIC_SPARSE_LIB poprand popnn poplin popops popfloat poputil stdc++fs)
    target_link_libraries(${CUSTOM_OP_NAME} PUBLIC custom_dynamic_sparse poprand popnn poplin popops popfloat poputil stdc++fs)
    


    # TODO only include this when needed
    # possibly just 
    target_include_directories(${CUSTOM_OP_NAME}  
      PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/custom_dynamic_sparse>
        # $<BUILD_INTERFACE:${MCL_HEADER_PATH}>
      PRIVATE
        .
    )
    
    # target_link_libraries(${CUSTOM_OP_NAME} poprand popnn poplin popops popfloat poputil stdc++fs)
  endforeach()

  # target_link_libraries(${name} poprand popnn poplin popops popfloat poputil Boost::filesystem)

  # add_executable(TARGETS GXX_FLGAS "${name}" )
endfunction()