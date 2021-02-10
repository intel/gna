#@copyright (C) 2020-2021 Intel Corporation
#SPDX-License-Identifier: LGPL-2.1-or-later

function (strip_symbols TARG_NAME)
  set(GNA_TARG_RELEASE_OUT_DIR ${GNA_TOOLS_RELEASE_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE})
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_custom_command(TARGET ${TARG_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        -DPDB_PUBLIC=${GNA_TARG_RELEASE_OUT_DIR}/${TARG_NAME}Public.pdb
        -DPDB_PATH=${GNA_TARG_RELEASE_OUT_DIR}
        -DFILE_NAME=${TARG_NAME}.pdb
        -P ${CMAKE_SOURCE_DIR}/build/cmake/pdb_public.cmake)
  endif()

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_BUILD_TYPE} STREQUAL "LNX_RELEASE")
      add_custom_command(TARGET ${TARG_NAME} POST_BUILD
        COMMAND cp $<TARGET_FILE:${TARG_NAME}> $<TARGET_FILE:${TARG_NAME}>.dbg
        COMMAND strip --only-keep-debug $<TARGET_FILE:${TARG_NAME}>.dbg
        COMMAND strip --strip-unneeded $<TARGET_FILE:${TARG_NAME}>)
    endif()
  endif()
endfunction(strip_symbols)

function (copy_pdb_windows TARG_NAME)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_custom_command(TARGET ${TARG_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PDB_FILE:gna-api>
        $<TARGET_FILE_DIR:${TARG_NAME}>)
  endif()
endfunction(copy_pdb_windows)

function (copy_gna_api DST_TARGET OUT_NEW_TARGET)
  set(NEW_TARGET "copy-gna-api-to-${DST_TARGET}")
  set(${OUT_NEW_TARGET} "${NEW_TARGET}" PARENT_SCOPE)
  add_custom_target(${NEW_TARGET} ALL
    COMMENT "Running target: ${NEW_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    $<TARGET_FILE_DIR:gna-api>
    $<TARGET_FILE_DIR:${DST_TARGET}>)
  set_target_properties(${NEW_TARGET}
    PROPERTIES
    FOLDER tools/${DST_TARGET})
endfunction(copy_gna_api)

function (set_gna_compile_options TARG_NAME)
  target_compile_options(${TARG_NAME}
    PRIVATE
    ${GNA_COMPILE_FLAGS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
    ${EXTRA_EXE_COMPILE_OPTIONS})
endfunction(set_gna_compile_options)

function (set_gna_compile_definitions TARG_NAME)
  target_compile_definitions(${TARG_NAME}
    PRIVATE
    ${GNA_COMPILE_DEFS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
    ${GNA_LIBRARY_VER})
endfunction(set_gna_compile_definitions)

function (set_gna_target_properties TARG_NAME)
  set_target_properties(${TARG_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_TOOLS_DEBUG_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE}
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_TOOLS_RELEASE_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE}
    OUTPUT_NAME ${TARG_NAME}
    FOLDER tools/${TARG_NAME})
endfunction(set_gna_target_properties)

macro(gna_add_shared_library_rc_properties TARGET_NAME TARGET_DESCRIPTION)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(gna_rc_files
      ${COMMON_DIR}/resource.h
      ${COMMON_DIR}/version.rc)
    set_source_files_properties(${gna_rc_files}PROPERTIES LANGUAGE RC)
    set_target_properties(${TARGET_NAME}
      PROPERTIES
      VS_USER_PROPS ${CMAKE_SOURCE_DIR}/build/common/version.props)
    target_sources(${TARGET_NAME} PRIVATE ${gna_rc_files})
    get_target_property(GNA_TARGET_OUTPUT_NAME ${TARGET_NAME} OUTPUT_NAME)
    if(NOT GNA_TARGET_OUTPUT_NAME)
      set(GNA_TARGET_OUTPUT_NAME ${TARGET_NAME})
    endif(NOT GNA_TARGET_OUTPUT_NAME)
    target_compile_definitions(${TARGET_NAME}
      PRIVATE
      -DGNA_TARGET_OUTPUT_NAME="${GNA_TARGET_OUTPUT_NAME}"
      -DGNA_TARGET_DESCRIPTION="${TARGET_DESCRIPTION}")
  endif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
endmacro()
