/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-inference-api.h"

#include <cstdint>

#if !defined(_MSC_VER)
#include <immintrin.h>
#else
#include <intrin.h>
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define __forceinline inline
#endif

/**
* Macros for decoration of function names build with different optimizations
*/
#define PASTER(x,y)     x ## y
#define EVALUATOR(x,y)  PASTER(x,y)
#define KERNEL(NAME)    EVALUATOR(NAME, KERNEL_SUFFIX)

/**
 * Definitions acceleration/optimization macros
 *
 * * OPT_LEVEL      - Build acceleration/optimization mode for numerical comparison
 * * KERNEL_SUFFIX  - suffix for decorating kernel names build for each optimization
 */

#if !defined(__LP64__) && !defined(_WIN64)
#define _mm_extract_epi64(a, i) ((((int64_t)_mm_extract_epi32(a,i*2+1)<<32)|_mm_extract_epi32(a,i*2)))
#define _mm256_extract_epi64(a, i) ((((int64_t)_mm256_extract_epi32(a,i*2+1)<<32)|_mm256_extract_epi32(a,i*2)))
#endif

#if     defined(OPTGEN)

#define OPT_LEVEL       0
#define KERNEL_SUFFIX   _generic
constexpr auto KernelAcceleration = Gna2AccelerationModeGeneric;

#elif   defined(OPTGEN_SAT)

#define OPT_LEVEL       1
#define KERNEL_SUFFIX   _generic_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeGeneric;

#elif   defined(OPTSSE4)

#define OPT_LEVEL       2
#define KERNEL_SUFFIX   _sse4
constexpr auto KernelAcceleration = Gna2AccelerationModeSse4x2;

#elif   defined(OPTSSE4_SAT)

#define OPT_LEVEL       3
#define KERNEL_SUFFIX   _sse4_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeSse4x2;
#include "common_sse4.hpp"

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi64(acc, _mm_add_epi64(
        _mm_cvtepi32_epi64(x),
        _mm_cvtepi32_epi64(_mm_srli_si128(x, 8))));
}
__forceinline int64_t vec_sum(__m128i x)
{
    return _mm_hsum_epi64(x);
}
__forceinline int64_t vec_sum32(__m128i x)
{
    return (int64_t)_mm_hsum_epi32(x);
}

#elif   defined(OPTAVX1)

#define OPT_LEVEL       4
#define KERNEL_SUFFIX   _avx1
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx1;

#elif   defined(OPTAVX1_SAT)

#define OPT_LEVEL       5
#define KERNEL_SUFFIX   _avx1_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx1;
#include "common_avx1.hpp"

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi64(acc, x);
}
__forceinline int64_t vec_sum(__m128i x)
{
    return _mm_extract_epi64(x, 0) + _mm_extract_epi64(x, 1);
}
__forceinline __m128i vec_madd16(__m256i x, __m256i y)
{
    __m128i m0 = _mm_madd_epi16(_mm256_castsi256_si128(x), _mm256_castsi256_si128(y));
    __m128i m1 = _mm_madd_epi16(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
    return _mm_add_epi64(
        _mm_add_epi64(_mm_cvtepi32_epi64(m0), _mm_cvtepi32_epi64(_mm_srli_si128(m0, 8))),
        _mm_add_epi64(_mm_cvtepi32_epi64(m1), _mm_cvtepi32_epi64(_mm_srli_si128(m1, 8))));
}
__forceinline int64_t vec_sum32(__m128i x)
{
    return (int64_t)_mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTAVX2)

#define OPT_LEVEL       6
#define KERNEL_SUFFIX   _avx2
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx2;

#elif   defined(OPTAVX2_SAT)

#define OPT_LEVEL       7
#define KERNEL_SUFFIX   _avx2_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx2;
#include "common_avx2.hpp"

__forceinline __m256i vec_accumulate(__m256i acc, __m256i x)
{
    return _mm256_add_epi64(acc, _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(x)),
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(x, 1))));
}
__forceinline int64_t vec_sum32(__m256i x)
{
    return (int64_t)_mm256_hsum_epi32(x);
}
__forceinline int64_t vec_sum(__m256i x)
{
    return _mm256_hsum_epi64(x);
}

#else

// Force compilation error to prevent build of unsupported acceleration mode
#error NO SUPPORTED ACCELERATION MODE DEFINED

#endif

#if OPT_LEVEL % 2 == 0
#define GNA_SAT 0
typedef int32_t gna_sum_t;
#else
#define GNA_SAT 1
typedef int64_t gna_sum_t;
#endif

#define SSE_16CAP 8

inline int32_t getBias(const void* ptr, uint32_t bytesPerElement, uint32_t idx = 0)
{
    switch (bytesPerElement)
    {
    case 1:
        return ((int8_t*)ptr)[idx];
    case 2:
        return ((int16_t*)ptr)[idx];
    case 4:
        return ((int32_t*)ptr)[idx];
    default:
        return 0;
    }
}
