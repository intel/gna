/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "common.h"
#include "KernelMacros.h"

#include <cstdint>
#include <immintrin.h>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define __forceinline inline
#endif

__forceinline void saturate(int64_t* const sum, uint32_t * const saturationCount)
{
    if (*sum > INT32_MAX)
    {
        *sum = INT32_MAX;
        (*saturationCount)++;
    }
    else if (*sum < INT32_MIN)
    {
        *sum = INT32_MIN;
        (*saturationCount)++;
    }
}

__forceinline void saturate_store_out(int64_t const * const sum, int32_t * const out, uint32_t * const saturationCount)
{
    if (*sum > INT32_MAX)
    {
        *out = INT32_MAX;
        (*saturationCount)++;
    }
    else if (*sum < INT32_MIN)
    {
        *out = INT32_MIN;
        (*saturationCount)++;
    }
    else
    {
        *out = (int32_t)*sum;
    }
}

