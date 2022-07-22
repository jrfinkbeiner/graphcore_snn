# function that creates a GP executables for all targets
# TODO fucture could be build seperately for all targets

# this function requires the following global variables to exist (all of which
# are defined in the top level CMakeLists.txt):
#   - POPC_EXECUTABLE
#   - POPC_FLAGS

function(add_gp_executables)
  # cmake_parse_arguments(CODELET "" "NAME" "CPP_SOURCES" ${ARGN})
  cmake_parse_arguments(CODELET "" "NAME" "CPP_SOURCES" ${ARGN})

  set(PARTIAL_OUTPUTS)

  # compile each C++ file in it's own gp file
  # in the furture compile separately for different targets
  foreach(CPP_SOURCE ${CODELET_CPP_SOURCES})
    get_filename_component(FILEDIR ${CPP_SOURCE} DIRECTORY)
    get_filename_component(FILENAME ${CPP_SOURCE} NAME_WE)
    set(FULL_GP_NAME "${CMAKE_CURRENT_BINARY_DIR}/${FILEDIR}/${FILENAME}.gp")

    # string(SHA1 MAGIC_NUMBER "${FULL_GP_NAME}")
    # set(TARGET_NAME ${MAGIC_NUMBER})
    # message("MAGIC_NUMBER: ${MAGIC_NUMBER}")

    string(REPLACE "/" "_" TARGET_NAME "${FULL_GP_NAME}")
    # set(TARGET_NAME "${CPP_SOURCE}")

    # message(FILEDIR)
    # message(${FILEDIR})
    # message(FILENAME)
    # message(${FILENAME})
    # message(FULL_GP_NAME)
    message("FULL_GP_NAME: ${FULL_GP_NAME}")

    set(FULL_CPP_SOURCE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/${CPP_SOURCE}")

    get_filename_component(GP_DIRNAME ${FULL_GP_NAME} DIRECTORY)
    # set(MKDIR_COMMAND "mkdir -p ${GP_DIRNAME} &&")


    # TODO better create at build time...
    file(MAKE_DIRECTORY ${GP_DIRNAME})
    # add_custom_target(build-time-make-directory ALL
    #   COMMAND ${CMAKE_COMMAND} -E make_directory ${GP_DIRNAME})

    set(COMMAND
    # ${CMAKE_COMMAND} -E env ${POPC_ENVIRONMENT}
    ${MKDIR_COMMAND}
    ${POPC_EXECUTABLE}
    ${POPC_FLAGS}
    )

    add_custom_command(
      OUTPUT
        ${FULL_GP_NAME}
      COMMAND
        ${COMMAND}
        -o ${FULL_GP_NAME}
        # --target ${TARGET}
        ${FULL_CPP_SOURCE_NAME}
      DEPENDS
        ${FULL_CPP_SOURCE_NAME}
    )

    # message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
    # message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
    # message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
    # message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

    add_custom_target(${TARGET_NAME} ALL DEPENDS ${FULL_GP_NAME}
      SOURCES
        ${FULL_CPP_SOURCE_NAME}
      )

    # install(FILES ${FULL_GP_NAME}
    #         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    #         # COMPONENT ${MAGIC_NUMBER})

  endforeach()

  # install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
  #         DESTINATION ${CMAKE_INSTALL_LIBDIR}
  #         COMPONENT ${CODELET_NAME})

endfunction()