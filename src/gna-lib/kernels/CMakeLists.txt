# @copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

# minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# global variables
set(PROJECT_NAME kernels)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

# SIMD compiler flags
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CXX_AVX2_FLAG "-march=core-avx2")
    set(CXX_AVX_FLAG "-mavx")
    set(CXX_SSE4_FLAG "-msse4.2")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(CXX_AVX2_FLAG "/arch:CORE-AVX2")
    set(CXX_AVX_FLAG "/arch:AVX")
    set(CXX_SSE4_FLAG "/arch:SSE4.2")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CXX_AVX2_FLAG "/arch:AVX2")
  set(CXX_AVX_FLAG "/arch:AVX")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CXX_AVX2_FLAG "-mavx2")
  set(CXX_AVX_FLAG "-mavx")
  set(CXX_SSE4_FLAG "-msse4.2")
endif()

set(KERNEL_DIR ${SRC_DIR}/kernels)

# --- XNN KERNELS --- #
set(xnn_kernel_sources
  isbmm8.cpp
  isbmm16.cpp
  pwl.cpp
  XnnKernel.cpp
  KernelArguments.cpp
  PoolingKernelArguments.cpp)

set(xnn_kernel_headers
  ${COMMON_DIR}/GnaException.h
  ${COMMON_DIR}/Macros.h
  convnet.h
  igemv.h
  igemv8.h
  igemv16.h
  KernelMacros.h
  pwl.h)

set(xnn_kernel_api
  ${KERNEL_DIR}/ConvolutionKernelArguments.h
  ${KERNEL_DIR}/KernelArguments.h
  ${KERNEL_DIR}/PoolingKernelArguments.h
  ${KERNEL_DIR}/XnnKernel.h)

set(xnn_generic_sources
  convnet_generic.cpp
  igemm16_generic.cpp
  igemm16_subset_generic.cpp
  igemm8_generic.cpp
  igemm8_subset_generic.cpp
  igemv16_generic.cpp
  igemv8_generic.cpp
  transpose16_generic.cpp)

set(xnn_generic_sat_sources
  convnet_generic-sat.cpp
  igemm16_generic-sat.cpp
  igemm16_subset_generic-sat.cpp
  igemm8_generic-sat.cpp
  igemm8_subset_generic-sat.cpp
  igemv16_generic-sat.cpp
  igemv8_generic-sat.cpp
  transpose16_generic.cpp)

set(xnn_sse4_sources
  convnet_sse4.cpp
  igemm16_sse4.cpp
  igemm16_subset_sse4.cpp
  igemm8_sse4.cpp
  igemm8_subset_sse4.cpp
  igemv16_sse4.cpp
  igemv8_sse4.cpp
  transpose16_sse4.cpp)

set(xnn_sse4_sat_sources
  convnet_sse4-sat.cpp
  igemm16_sse4-sat.cpp
  igemm16_subset_sse4-sat.cpp
  igemm8_sse4-sat.cpp
  igemm8_subset_sse4-sat.cpp
  igemv16_sse4-sat.cpp
  igemv8_sse4-sat.cpp
  transpose16_sse4.cpp)

set(xnn_avx1_sources
  convnet_avx1.cpp
  igemm16_avx1.cpp
  igemm16_subset_avx1.cpp
  igemm8_avx1.cpp
  igemm8_subset_avx1.cpp
  igemv16_avx1.cpp
  igemv8_avx1.cpp
  transpose16_avx1.cpp)

set(xnn_avx1_sat_sources
  convnet_avx1-sat.cpp
  igemm16_avx1-sat.cpp
  igemm16_subset_avx1-sat.cpp
  igemm8_avx1-sat.cpp
  igemm8_subset_avx1-sat.cpp
  igemv16_avx1-sat.cpp
  igemv8_avx1-sat.cpp
  transpose16_avx1.cpp)

set(xnn_avx2_sources
  convnet_avx2.cpp
  igemm16_avx2.cpp
  igemm16_subset_avx2.cpp
  igemm8_avx2.cpp
  igemm8_subset_avx2.cpp
  igemv16_avx2.cpp
  igemv8_avx2.cpp
  transpose16_avx2.cpp)

set(xnn_avx2_sat_sources
  convnet_avx2-sat.cpp
  igemm16_avx2-sat.cpp
  igemm16_subset_avx2-sat.cpp
  igemm8_avx2-sat.cpp
  igemm8_subset_avx2-sat.cpp
  igemv16_avx2-sat.cpp
  igemv8_avx2-sat.cpp
  transpose16_avx2.cpp)

macro(gna_add_xnn_kernel_library KERNEL_SUFIX EXTRA_DEFS EXTRA_OPTIONS)
  add_library(xnn_kernel_${KERNEL_SUFIX} STATIC ${xnn_kernel_sources})
  target_include_directories(xnn_kernel_${KERNEL_SUFIX}
    PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
  target_sources(xnn_kernel_${KERNEL_SUFIX}
    PRIVATE "${xnn_${KERNEL_SUFIX}_sources}" ${xnn_kernel_headers}
    PUBLIC ${xnn_kernel_api})
  target_compile_definitions(xnn_kernel_${KERNEL_SUFIX}
    PRIVATE
    ${GNA_COMPILE_DEFS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
    ${EXTRA_DEFS})
  target_compile_options(xnn_kernel_${KERNEL_SUFIX}
    PRIVATE
    ${GNA_COMPILE_FLAGS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
    ${EXTRA_OPTIONS})
endmacro()

gna_add_xnn_kernel_library(generic OPTGEN "")
gna_add_xnn_kernel_library(generic_sat OPTGEN_SAT "")
gna_add_xnn_kernel_library(sse4 OPTSSE4 "${CXX_SSE4_FLAG}")
gna_add_xnn_kernel_library(sse4_sat OPTSSE4_SAT "${CXX_SSE4_FLAG}")
gna_add_xnn_kernel_library(avx1 OPTAVX1 "${CXX_AVX_FLAG}")
gna_add_xnn_kernel_library(avx1_sat OPTAVX1_SAT "${CXX_AVX_FLAG}")
gna_add_xnn_kernel_library(avx2 OPTAVX2 "${CXX_AVX2_FLAG}")
gna_add_xnn_kernel_library(avx2_sat OPTAVX2_SAT "${CXX_AVX2_FLAG}")

set(xnn_kernel_libraries
  xnn_kernel_generic
  xnn_kernel_generic_sat
  xnn_kernel_sse4
  xnn_kernel_sse4_sat
  xnn_kernel_avx1
  xnn_kernel_avx1_sat
  xnn_kernel_avx2
  xnn_kernel_avx2_sat)

set_property(TARGET ${xnn_kernel_libraries} PROPERTY FOLDER library/kernels/xnn)

# ___ GMM KERNELS ___ #
set(gmm_kernel_sources
  gmm.cpp
  kernel-gmm.cpp
  KernelArguments.cpp)

set(gmm_kernel_headers
  ${COMMON_DIR}/Macros.h
  ${COMMON_DIR}/common.h
  ${COMMON_DIR}/GnaException.h
  KernelMacros.h
  kernel-gmm.h)

set(gmm_kernel_api
  ${KERNEL_DIR}/gmm.h
  ${KERNEL_DIR}/KernelArguments.h)

macro(gna_add_gmm_kernel_library KERNEL_SUFIX EXTRA_DEFS EXTRA_OPTIONS)
  add_library(gmm_kernel_${KERNEL_SUFIX} STATIC ${gmm_kernel_sources})
  target_include_directories(gmm_kernel_${KERNEL_SUFIX}
    PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
  target_sources(gmm_kernel_${KERNEL_SUFIX}
    PRIVATE ${gmm_kernel_headers}
    PUBLIC ${gmm_kernel_api})
  target_compile_definitions(gmm_kernel_${KERNEL_SUFIX}
    PRIVATE
    ${GNA_COMPILE_DEFS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
    ${EXTRA_DEFS})
  target_compile_options(gmm_kernel_${KERNEL_SUFIX}
    PRIVATE
    ${GNA_COMPILE_FLAGS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
    ${EXTRA_OPTIONS})
endmacro()

gna_add_gmm_kernel_library(generic OPTGEN "")
gna_add_gmm_kernel_library(sse4 OPTSSE4 "${CXX_SSE4_FLAG}")
gna_add_gmm_kernel_library(avx1 OPTAVX1 "${CXX_AVX_FLAG}")
gna_add_gmm_kernel_library(avx2 OPTAVX2 "${CXX_AVX2_FLAG}")

set(gmm_kernel_libraries
  gmm_kernel_generic
  gmm_kernel_sse4
  gmm_kernel_avx1
  gmm_kernel_avx2)

set_property(TARGET ${gmm_kernel_libraries} PROPERTY FOLDER library/kernels/gmm)

set(kernel_libraries
  ${gmm_kernel_libraries} ${xnn_kernel_libraries}
  PARENT_SCOPE)

set(kernel_api_headers
  ${KERNEL_DIR}/gmm.h
  ${KERNEL_DIR}/XnnKernel.h
  PARENT_SCOPE)

set_target_properties(xnn_kernel_generic xnn_kernel_generic_sat
  xnn_kernel_sse4 xnn_kernel_sse4_sat xnn_kernel_avx1 xnn_kernel_avx1_sat
  xnn_kernel_avx2 xnn_kernel_avx2_sat
  gmm_kernel_generic gmm_kernel_sse4 gmm_kernel_avx1 gmm_kernel_avx2
  PROPERTIES
  POSITION_INDEPENDENT_CODE True
  PREFIX "")
