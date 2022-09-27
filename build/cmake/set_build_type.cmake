# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

get_property(IsMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT IsMultiConfig)
  if(NOT CMAKE_BUILD_TYPE
      OR (NOT ${CMAKE_BUILD_TYPE} STREQUAL "${OS_PREFIX}_DEBUG"
        AND NOT ${CMAKE_BUILD_TYPE} STREQUAL "${OS_PREFIX}_RELEASE"))
      set(CMAKE_BUILD_TYPE "${OS_PREFIX}_RELEASE" CACHE STRING "default build type" FORCE)
  endif()
else()
  # configuration types for multi-config generators
  set(CMAKE_CONFIGURATION_TYPES "${OS_PREFIX}_DEBUG;${OS_PREFIX}_RELEASE")
endif()

# postfix for debug libraries
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(CMAKE_${OS_PREFIX}_DEBUG_POSTFIX d)
endif()
