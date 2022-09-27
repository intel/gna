# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

# minimum version of CMake
cmake_minimum_required(VERSION 3.10)
# global variables
set(PROJECT_NAME gna-lib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# project name
project(${PROJECT_NAME})

# global list of test dependencies
set(GNA_ALL_TESTS_REGISTERED_TARGETS "" CACHE INTERNAL "")

# main binary path
set(GNA_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

option(GNA_RUN_CPPCHECK "Run cppcheck analysis" OFF)
if(GNA_RUN_CPPCHECK)
  set(CMAKE_CXX_CPPCHECK cppcheck)
  set(CMAKE_C_CPPCHECK cppcheck)
endif()

option(GNA_RUN_IWYU "Run iwyu analysis" OFF)
if(GNA_RUN_IWYU)
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE iwyu)
  set(CMAKE_C_INCLUDE_WHAT_YOU_USE iwyu)
endif()

option(GNA_RUN_CLANG_TIDY "Run clang-tidy analysis" OFF)
if(GNA_RUN_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*,portability-*,clang-analyzer-*)
endif()

if(DEFINED ENV{GNA_LIBRARY_VERSION})
  set(GNA_LIBRARY_VER -DGNA_LIBRARY_VERSION_STRING=\"$ENV{GNA_LIBRARY_VERSION}\")
elseif(NOT DEFINED GNA_LIBRARY_VER)
  set(GNA_LIBRARY_VER -DGNA_LIBRARY_VERSION_STRING=\"3.0.0.0\")
endif()

# source path variables
set(APP_DIR ${CMAKE_SOURCE_DIR}/src)
set(COMMON_DIR ${APP_DIR}/common)
set(API_DIR ${APP_DIR}/gna-api)
set(SRC_DIR ${APP_DIR}/gna-lib)
set(API_IMPL_DIR ${APP_DIR}/gna-lib/gna-api)
set(KERNEL_DIR ${SRC_DIR}/kernels)
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)

# linking with libpthreads
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR
  ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  find_package(Threads)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(OS_PREFIX WIN)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(OS_PREFIX AOS)
else()
  set(OS_PREFIX LNX)
endif()

# setting default configuration
include(${CMAKE_SOURCE_DIR}/build/cmake/set_build_type.cmake)

# Visual Studio
if(MSVC)
  string(TOLOWER ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID} CMAKE_ARCHITECTURE)
elseif(NOT CMAKE_ARCHITECTURE)
  set(CMAKE_ARCHITECTURE "x64")
endif()

include(${CMAKE_SOURCE_DIR}/build/cmake/gna_compile_flags.cmake)

include(${CMAKE_SOURCE_DIR}/build/cmake/gna_cmake_tools.cmake)

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
  message(STATUS "Found ctags: ${CTAGS_PATH}")

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    execute_process(
      COMMAND
      bash -c "echo | gcc -Wp,-v -x c++ - -fsyntax-only 2>&1 | grep ^./usr | xargs"
      OUTPUT_VARIABLE CXX_STD_INCLUDE_DIRS)
  else()
    set(CXX_STD_INCLUDE_DIRS "")
  endif()
  add_custom_target(tags
    COMMAND
    bash -c "ctags -R --c++-kinds=+p --fields=+iaS --extra=+q ${CMAKE_SOURCE_DIR}/src ${CXX_STD_INCLUDE_DIRS}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

find_program(CSCOPE_PATH cscope)
if(CSCOPE_PATH)
  message(STATUS "Found cscope: ${CSCOPE_PATH}")

  add_custom_target(cscope
    COMMAND find -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" > cscope.files
    COMMAND cscope -Rqb -i cscope.files
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# HW C-package module
option(GNA_BUILD_WITH_HW_MODULE_ENABLED "" OFF)

set(HW_MODULE_PATH)
set(GNA_HW_LIB_INCLUDE_PATH)
set(GNA_HW_LIB_ENABLED )

if(${GNA_BUILD_WITH_HW_MODULE_ENABLED})
  set(HW_MODULE_PATH ${BUILD_DIR}/hw_module)
  set(HW_MODULE_NAME gnahw)

  set(GNA_HW_LIB_ENABLED ${GNA_HW_LIB_ENABLED} "-DGNA_HW_LIB_ENABLED=1")
  message("Enabling ${HW_MODULE_NAME} for GNA library")
  if(NOT EXISTS "${HW_MODULE_PATH}/CMakeLists.txt")

  message(FATAL_ERROR "HW submodule was not initialized properly.\n"
        "File <${HW_MODULE_PATH}/CMakeLists.txt> not found.\n"
        "Try using 'git submodule update --init --recursive' to load all submodules.")
  endif()
  add_subdirectory(${HW_MODULE_PATH})
  set_gna_compile_definitions(${HW_MODULE_NAME})

  set(GNA_ALL_TESTS_REGISTERED_TARGETS "${HW_MODULE_NAME};${GNA_ALL_TESTS_REGISTERED_TARGETS}" CACHE INTERNAL "")
else()
  message("HW Module 3.0 for GNA library will NOT be used.")
endif()

add_subdirectory(src/gna-lib/kernels)
add_subdirectory(src/gna-lib)
