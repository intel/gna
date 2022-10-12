# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(GNA_COMPILE_FLAGS "${GNA_COMPILE_FLAGS} /Qm32")
  set(GNA_LINKER_FLAGS "${GNA_LINKER_FLAGS} /Qm32")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(GNA_COMPILE_FLAGS "${GNA_COMPILE_FLAGS} -L/usr/lib32 -m32")
endif()
