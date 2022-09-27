/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once
#include <chrono>
#include <ctime>

#include "gna2-instrumentation-api.h"

#if defined(_WIN32)
#if !defined(_MSC_VER)
#include <immintrin.h>
#else
#include <intrin.h>
#endif
#else
#include <mmintrin.h>
#endif // os

using chronoClock = std::chrono::high_resolution_clock;
using chronoUs = std::chrono::microseconds;
using chronoMs = std::chrono::milliseconds;

void getTsc(uint64_t * const result);

