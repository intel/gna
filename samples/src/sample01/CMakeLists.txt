# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

cmake_minimum_required(VERSION 3.10)

add_executable(sample01
    sample01.cpp
)
target_link_libraries(sample01
    PRIVATE
    gna
)

target_include_directories(sample01
    PUBLIC
    .
    ${GNA_LIB_PATH}/include/
)

set_target_properties(sample01
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
  ARCHIVE_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
  RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}/sample01
)

add_custom_command(TARGET
    sample01 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:gna>
    $<TARGET_FILE_DIR:sample01>
)