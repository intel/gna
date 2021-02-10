# @copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

# minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# global variables
set(PROJECT_NAME gna-api)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(DEFINED ENV{GMM_VER_HW})
  set(GNA_VERSION_MAJOR $ENV{GMM_VER_HW})
elseif(NOT DEFINED GNA_VERSION_MAJOR)
  set(GNA_VERSION_MAJOR 2)
endif()

if(DEFINED ENV{GMM_VER_RELEASE})
  set(GNA_VERSION_MINOR $ENV{GMM_VER_RELEASE})
elseif(NOT DEFINED GNA_VERSION_MINOR)
  set(GNA_VERSION_MINOR 0)
endif()

if(DEFINED ENV{GMM_VER_FIX})
  set(GNA_VERSION_PATCH $ENV{GMM_VER_FIX})
elseif(NOT DEFINED GNA_VERSION_PATCH)
  set(GNA_VERSION_PATCH 0)
endif()

if(DEFINED ENV{GMM_VER_BUILD})
  set(GNA_VERSION_RELEASE $ENV{GMM_VER_BUILD})
elseif(NOT DEFINED GNA_VERSION_RELEASE)
  set(GNA_VERSION_RELEASE 0)
endif()

if(NOT DEFINED GNA_VERSION)
  set(GNA_VERSION ${GNA_VERSION_MAJOR}.${GNA_VERSION_MINOR}.${GNA_VERSION_PATCH}.${GNA_VERSION_RELEASE})
endif()

project(${PROJECT_NAME})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(API_EXTRA_LIBS ${CMAKE_DL_LIBS})
  set(API_EXTRA_DEFS "")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(API_EXTRA_LIBS "Cfgmgr32.lib")
  set(API_EXTRA_DEFS "-DINTEL_GNA_DLLEXPORT=1")
endif()

set(API_EXTRA_DEFS ${API_EXTRA_DEFS})

set(GNA_API_INCL_DIR ${GNA_BINARY_DIR}/gna-lib/include)
set(GNA_LIB_OUT_DIR ${GNA_BINARY_DIR}/gna-lib)
set(GNA_LIB_DEBUG_OUT_DIR ${GNA_LIB_OUT_DIR}/${OS_PREFIX}-DEBUG/${CMAKE_ARCHITECTURE})
set(GNA_LIB_RELEASE_OUT_DIR ${GNA_LIB_OUT_DIR}/${OS_PREFIX}-RELEASE/${CMAKE_ARCHITECTURE})

set(GNA_API_PDB_STRIPPED ${GNA_LIB_RELEASE_OUT_DIR}/gnaPublic.pdb)

set(gna_lib_sources ${gna_lib_sources}
  ${SRC_DIR}/AccelerationDetector.cpp
  ${SRC_DIR}/ActivationFunction.cpp
  ${SRC_DIR}/ActivationHelper.cpp
  ${SRC_DIR}/ActiveList.cpp
  ${SRC_DIR}/AffineFunctions.cpp
  ${SRC_DIR}/AffineLayerCapabilities.cpp
  ${SRC_DIR}/AffineLayers.cpp
  ${SRC_DIR}/AuxiliaryCapabilities.cpp
  ${SRC_DIR}/Bias.cpp
  ${SRC_DIR}/BufferMap.cpp
  ${SRC_DIR}/Capabilities.cpp
  ${SRC_DIR}/CompiledModel.cpp
  ${SRC_DIR}/Component.cpp
  ${SRC_DIR}/ConvolutionalFunctions.cpp
  ${SRC_DIR}/ConvolutionalFunctions2D.cpp
  ${SRC_DIR}/ConvolutionalLayer.cpp
  ${SRC_DIR}/ConvolutionalLayer2D.cpp
  ${SRC_DIR}/ConvolutionalLayer2DCapabilities.cpp
  ${SRC_DIR}/CopyLayer.cpp
  ${SRC_DIR}/DataMode.cpp
  ${SRC_DIR}/Device.cpp
  ${SRC_DIR}/DeviceLayerSupport.cpp
  ${SRC_DIR}/DeviceManager.cpp
  ${SRC_DIR}/DriverInterface.cpp
  ${SRC_DIR}/GmmLayer.cpp
  ${SRC_DIR}/GmmLayerCapabilities.cpp
  ${SRC_DIR}/HardwareCapabilities.cpp
  ${SRC_DIR}/HardwareLayer.cpp
  ${SRC_DIR}/HardwareModel.cpp
  ${SRC_DIR}/HardwareModelNoMMU.cpp
  ${SRC_DIR}/HardwareModelScorable.cpp
  ${SRC_DIR}/HardwareModelSue1.cpp
  ${SRC_DIR}/HardwareRequest.cpp
  ${SRC_DIR}/Layer.cpp
  ${SRC_DIR}/LayerCapabilities.cpp
  ${SRC_DIR}/LayerConfiguration.cpp
  ${SRC_DIR}/LayerDescriptor.cpp
  ${SRC_DIR}/LayerInput.cpp
  ${SRC_DIR}/LayerOutput.cpp
  ${SRC_DIR}/Layout.cpp
  ${SRC_DIR}/Logger.cpp
  ${SRC_DIR}/Memory.cpp
  ${SRC_DIR}/MemoryContainer.cpp
  ${SRC_DIR}/ModelDumper.cpp
  ${SRC_DIR}/ModelError.cpp
  ${SRC_DIR}/ModelExportConfig.cpp
  ${SRC_DIR}/ModelWrapper.cpp
  ${SRC_DIR}/OperationConfig.cpp
  ${SRC_DIR}/ParameterLimits.cpp
  ${SRC_DIR}/PoolingMode.cpp
  ${SRC_DIR}/PoolingFunctions.cpp
  ${SRC_DIR}/PoolingFunctions2D.cpp
  ${SRC_DIR}/ProfilerConfiguration.cpp
  ${SRC_DIR}/RecurrentFunction.cpp
  ${SRC_DIR}/RecurrentLayer.cpp
  ${SRC_DIR}/RequestBuilder.cpp
  ${SRC_DIR}/RequestConfiguration.cpp
  ${SRC_DIR}/Request.cpp
  ${SRC_DIR}/RequestHandler.cpp
  ${SRC_DIR}/Scalar.cpp
  ${SRC_DIR}/Shape.cpp
  ${SRC_DIR}/SoftwareModel.cpp
  ${SRC_DIR}/StringHelper.cpp
  ${SRC_DIR}/SubModel.cpp
  ${SRC_DIR}/Tensor.cpp
  ${SRC_DIR}/ThreadPool.cpp
  ${SRC_DIR}/Transform.cpp
  ${SRC_DIR}/TransformMap.cpp
  ${SRC_DIR}/TransposeLayer.cpp
  ${SRC_DIR}/Validator.cpp
  ${SRC_DIR}/Weight.cpp)

set(gna_driver_interface_src
  ${SRC_DIR}/LinuxDriverInterface.cpp
  ${SRC_DIR}/WindowsDriverInterface.cpp)

set(gna_api_impl_src_2
  ${API_IMPL_DIR}/gna2-common-impl.cpp
  ${API_IMPL_DIR}/gna2-device-impl.cpp
  ${API_IMPL_DIR}/gna2-inference-impl.cpp
  ${API_IMPL_DIR}/gna2-instrumentation-impl.cpp
  ${API_IMPL_DIR}/gna2-memory-impl.cpp
  ${API_IMPL_DIR}/gna2-model-impl.cpp
  ${API_IMPL_DIR}/gna2-model-export-impl.cpp
  ${API_IMPL_DIR}/gna2-capability-impl.cpp)

set(gna_lib_sources
  ${gna_lib_sources} ${gna_api_impl_src_2} ${gna_driver_interface_src})

set(gna_api_headers
  ${COMMON_DIR}/gna-api.h
  ${COMMON_DIR}/gna-api-types-xnn.h
  ${COMMON_DIR}/gna-api-types-gmm.h
  ${COMMON_DIR}/gna-api-status.h)

set(gna_lib_headers
  ${SRC_DIR}/AccelerationDetector.h
  ${SRC_DIR}/ActivationFunction.h
  ${SRC_DIR}/ActivationHelper.h
  ${SRC_DIR}/ActiveList.h
  ${SRC_DIR}/AffineFunctions.h
  ${SRC_DIR}/AffineLayerCapabilities.h
  ${SRC_DIR}/ApiWrapper.h
  ${SRC_DIR}/AuxiliaryCapabilities.h
  ${SRC_DIR}/BufferMap.h
  ${SRC_DIR}/Capabilities.h
  ${SRC_DIR}/CompiledModel.h
  ${SRC_DIR}/Component.h
  ${SRC_DIR}/ConvolutionalFunctions.h
  ${SRC_DIR}/ConvolutionalFunctions2D.h
  ${SRC_DIR}/ConvolutionalLayer.h
  ${SRC_DIR}/ConvolutionalLayer2D.h
  ${SRC_DIR}/ConvolutionalLayer2DCapabilities.h
  ${SRC_DIR}/CopyLayer.h
  ${SRC_DIR}/DataMode.h
  ${SRC_DIR}/Device.h
  ${SRC_DIR}/DeviceLayerSupport.h
  ${SRC_DIR}/DeviceManager.h
  ${SRC_DIR}/DriverInterface.h
  ${SRC_DIR}/Expect.h
  ${SRC_DIR}/GmmLayer.h
  ${SRC_DIR}/GmmLayerCapabilities.h
  ${SRC_DIR}/GnaTypes.h
  ${SRC_DIR}/HardwareCapabilities.h
  ${SRC_DIR}/HardwareLayer.h
  ${SRC_DIR}/HardwareModel.h
  ${SRC_DIR}/HardwareModelNoMMU.h
  ${SRC_DIR}/HardwareModelScorable.h
  ${SRC_DIR}/HardwareModelSue1.h
  ${SRC_DIR}/HardwareRequest.h
  ${SRC_DIR}/IScorable.h
  ${SRC_DIR}/LayerConfiguration.h
  ${SRC_DIR}/Layer.h
  ${SRC_DIR}/LayerCapabilities.h
  ${SRC_DIR}/LayerDescriptor.h
  ${SRC_DIR}/LayerInput.h
  ${SRC_DIR}/LayerOutput.h
  ${SRC_DIR}/Layout.h
  ${SRC_DIR}/Logger.h
  ${SRC_DIR}/Memory.h
  ${SRC_DIR}/MemoryContainer.h
  ${SRC_DIR}/ModelError.h
  ${SRC_DIR}/ModelExportConfig.h
  ${SRC_DIR}/ModelWrapper.h
  ${SRC_DIR}/OperationConfig.h
  ${SRC_DIR}/ParameterLimits.h
  ${SRC_DIR}/PoolingMode.h
  ${SRC_DIR}/PoolingFunctions.h
  ${SRC_DIR}/PoolingFunctions2D.h
  ${SRC_DIR}/ProfilerConfiguration.h
  ${SRC_DIR}/RecurrentFunction.h
  ${SRC_DIR}/RecurrentLayer.h
  ${SRC_DIR}/RequestBuilder.h
  ${SRC_DIR}/RequestConfiguration.h
  ${SRC_DIR}/Request.h
  ${SRC_DIR}/RequestHandler.h
  ${SRC_DIR}/Scalar.h
  ${SRC_DIR}/Shape.h
  ${SRC_DIR}/SoftwareModel.h
  ${SRC_DIR}/StringHelper.h
  ${SRC_DIR}/SubModel.h
  ${SRC_DIR}/Tensor.h
  ${SRC_DIR}/ThreadPool.h
  ${SRC_DIR}/ThresholdParameters.h
  ${SRC_DIR}/Transform.h
  ${SRC_DIR}/TransformMap.h
  ${SRC_DIR}/TransposeLayer.h
  ${SRC_DIR}/Validator.h
  ${SRC_DIR}/Weight.h)

set(gna_driver_interface_headers
  ${SRC_DIR}/LinuxDriverInterface.h
  ${SRC_DIR}/WindowsDriverInterface.h)

set(gna_hw_module_interface_sources
  ${SRC_DIR}/HwModuleInterface.hpp
  ${SRC_DIR}/WindowsHwModuleInterface.hpp
  ${SRC_DIR}/LinuxHwModuleInterface.hpp
  ${SRC_DIR}/HwModuleInterface.cpp
  ${SRC_DIR}/WindowsHwModuleInterface.cpp
  ${SRC_DIR}/LinuxHwModuleInterface.cpp
  )

set(gna_api_impl_headers_2
  ${API_IMPL_DIR}/gna2-impl.h
  ${API_IMPL_DIR}/gna2-capability-impl.h
  ${API_IMPL_DIR}/gna2-common-impl.h
  ${API_IMPL_DIR}/gna2-device-impl.h
  ${API_IMPL_DIR}/gna2-inference-impl.h
  ${API_IMPL_DIR}/gna2-instrumentation-impl.h
  ${API_IMPL_DIR}/gna2-memory-impl.h
  ${API_IMPL_DIR}/gna2-model-impl.h
  ${API_IMPL_DIR}/gna2-model-export-impl.h)

set(gna_lib_headers
  ${gna_api_headers} ${gna_lib_headers} ${gna_api_impl_headers_2} ${gna_driver_interface_headers})

set(gna_api_headers_2
  ${API_DIR}/gna2-api.h
  ${API_DIR}/gna2-common-api.h
  ${API_DIR}/gna2-device-api.h
  ${API_DIR}/gna2-inference-api.h
  ${API_DIR}/gna2-instrumentation-api.h
  ${API_DIR}/gna2-memory-api.h
  ${API_DIR}/gna2-model-api.h)

set(gna_api_headers_extra_2
  ${API_DIR}/gna2-model-export-api.h
  ${API_DIR}/gna2-model-suecreek-header.h
  ${API_DIR}/gna2-capability-api.h)

set(common_sources
  ${COMMON_DIR}/Address.h
  ${COMMON_DIR}/common.h
  ${COMMON_DIR}/GnaException.h
  ${COMMON_DIR}/Macros.h
  ${COMMON_DIR}/profiler.cpp
  ${COMMON_DIR}/profiler.h)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(windows_ddi
    ${COMMON_DIR}/GnaDrvApi.h
    ${COMMON_DIR}/GnaDrvApiWinDebug.h)
  list(APPEND common_sources ${windows_ddi})
endif()

set(api_headers ${gna_api_headers_2} ${gna_api_headers_extra_2})
source_group(Api2 FILES ${api_headers})

source_group("Api2 Source Files" FILES ${gna_api_impl_src_2})

add_library(gna-api SHARED)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(gna_lib_headers ${gna_lib_headers} ${COMMON_DIR}/gna.h)
endif()

target_include_directories(gna-api PRIVATE
  ${SRC_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR}
  $<BUILD_INTERFACE:${COMMON_DIR}>)

target_include_directories(gna-api PUBLIC
  $<INSTALL_INTERFACE:include/gna${GNA_VERSION_MAJOR}>)

target_sources(gna-api
  PRIVATE ${common_sources} ${gna_lib_sources} ${gna_lib_headers} ${kernel_api_headers}
  ${gna_hw_module_interface_sources})

target_sources(gna-api PUBLIC
  $<BUILD_INTERFACE:${api_headers}>)

set_target_properties(gna-api
  PROPERTIES
  FOLDER library
  VERSION ${GNA_VERSION}
  SOVERSION ${GNA_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  OUTPUT_NAME gna)
  gna_add_shared_library_rc_properties(gna-api "Main module")

target_compile_definitions(gna-api
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  ${API_EXTRA_DEFS}
  ${GNA_LIBRARY_VER}
  -DPROFILE -DPROFILE_DETAILED)

target_compile_options(gna-api
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>)

target_link_libraries(gna-api
  PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${API_EXTRA_LIBS} ${kernel_libraries})

add_dependencies(gna-api ${kernel_libraries})

file(MAKE_DIRECTORY ${GNA_API_INCL_DIR})
add_custom_command(TARGET gna-api POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${api_headers}
  ${GNA_API_INCL_DIR})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_custom_command(TARGET gna-api POST_BUILD
	COMMAND ${CMAKE_COMMAND}
	-DPDB_PUBLIC=${GNA_API_PDB_STRIPPED}
    -DPDB_PATH=${GNA_LIB_RELEASE_OUT_DIR}
    -DFILE_NAME=gna.pdb
    -P ${CMAKE_SOURCE_DIR}/build/cmake/pdb_public.cmake)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_custom_command(TARGET gna-api POST_BUILD
    COMMAND
    echo $(API_DLL) version: $(VERSION_QB_A) ($(VERSION_FILE_A)) > $(OutDir)$(API_DLL).ver)
endif()

#strip symbols on linux
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if(${CMAKE_BUILD_TYPE} STREQUAL "LNX_RELEASE")
    add_custom_command(TARGET gna-api POST_BUILD
      COMMAND cp $<TARGET_FILE:gna-api> $<TARGET_FILE:gna-api>.dbg
      COMMAND strip --only-keep-debug $<TARGET_FILE:gna-api>.dbg
      COMMAND strip --strip-unneeded $<TARGET_FILE:gna-api>)
  endif()
endif()

install(TARGETS gna-api EXPORT GnaTargets
  CONFIGURATIONS ${OS_PREFIX}_DEBUG ${OS_PREFIX}_RELEASE
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include)

install(FILES ${api_headers}
  DESTINATION include/gna${GNA_VERSION_MAJOR})

# package version
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/gna-config-version.cmake
  VERSION ${GNA_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/gna-config-version.cmake
  DESTINATION lib/cmake/gna${GNA_VERSION_MAJOR})

# build directory export
export(TARGETS gna-api
  FILE GnaTargetsConfig.cmake)

# system wide export
install(EXPORT GnaTargets
  FILE gna-config.cmake
  NAMESPACE Gna::
  DESTINATION lib/cmake/gna${GNA_VERSION_MAJOR})
