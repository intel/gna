/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <stdint.h>

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

#define vec_accumulate KERNEL(vec_accumulate)
#define vec_sum KERNEL(vec_sum)
#define vec_sum32 KERNEL(vec_sum32)
#define vec_madd16 KERNEL(vec_madd16)

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

#pragma message("Building Generic kernel, level 0")
#define OPT_LEVEL       0
#define KERNEL_SUFFIX   _generic

#elif   defined(OPTGEN_SAT)

#pragma message("Building Generic kernel with saturation, level 1")
#define OPT_LEVEL       1
#define KERNEL_SUFFIX   _generic_sat

#elif   defined(OPTSSE4)

#pragma message("Building SSE4 kernel, level 2")
#define OPT_LEVEL       2
#define KERNEL_SUFFIX   _sse4
__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m128i x)
{
    return _mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTSSE4_SAT)

#pragma message("Building SSE4 kernel with saturation, level 3")
#define OPT_LEVEL       3
#define KERNEL_SUFFIX   _sse4_sat
__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi64(acc, _mm_add_epi64(
        _mm_cvtepi32_epi64(x),
        _mm_cvtepi32_epi64(_mm_srli_si128(x, 8))));
}
__forceinline int64_t vec_sum(__m128i x)
{
    return _mm_extract_epi64(x, 0) + _mm_extract_epi64(x, 1);
}
__forceinline int64_t vec_sum32(__m128i x)
{
    return (int64_t)_mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTAVX1)

#pragma message("Building AVX1 kernel, level 4")
#define OPT_LEVEL       4
#define KERNEL_SUFFIX   _avx1
__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m128i x)
{
    return _mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}
__forceinline __m128i vec_madd16(__m256i x, __m256i y)
{
    return _mm_add_epi32(
        _mm_madd_epi16(_mm256_castsi256_si128(x), _mm256_castsi256_si128(y)),
        _mm_madd_epi16(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1)));
}

#elif   defined(OPTAVX1_SAT)

#pragma message("Building AVX1 kernel with saturation, level 5")
#define OPT_LEVEL       5
#define KERNEL_SUFFIX   _avx1_sat
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

#pragma message("Building AVX2 kernel, level 6")
#define OPT_LEVEL       6
#define KERNEL_SUFFIX   _avx2
__forceinline __m256i vec_accumulate(__m256i acc, __m256i x)
{
    return _mm256_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m256i x)
{
    return _mm256_extract_epi32(x, 0) + _mm256_extract_epi32(x, 1) + _mm256_extract_epi32(x, 2) + _mm256_extract_epi32(x, 3)
         + _mm256_extract_epi32(x, 4) + _mm256_extract_epi32(x, 5) + _mm256_extract_epi32(x, 6) + _mm256_extract_epi32(x, 7);
}

#elif   defined(OPTAVX2_SAT)

#pragma message("Building AVX2 kernel with saturation, level 7")
#define OPT_LEVEL       7
#define KERNEL_SUFFIX   _avx2_sat
__forceinline __m256i vec_accumulate(__m256i acc, __m256i x)
{
    return _mm256_add_epi64(acc, _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(x)),
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(x, 1))));
}
__forceinline int64_t vec_sum32(__m256i x)
{
    return (int64_t)_mm256_extract_epi32(x, 0) + _mm256_extract_epi32(x, 1) + _mm256_extract_epi32(x, 2) + _mm256_extract_epi32(x, 3)
         + _mm256_extract_epi32(x, 4) + _mm256_extract_epi32(x, 5) + _mm256_extract_epi32(x, 6) + _mm256_extract_epi32(x, 7);
}
__forceinline int64_t vec_sum(__m256i x)
{
    return (int64_t)_mm256_extract_epi64(x, 0) + _mm256_extract_epi64(x, 1) + _mm256_extract_epi64(x, 2) + _mm256_extract_epi64(x, 3);
}

#else

// Force compilation error to prevent build of unsupported acceleration mode
#error NO SUPPORTED ACCELERATION MODE DEFINED

#endif

// SSE4_2
#if OPT_LEVEL == 2 || OPT_LEVEL == 3
typedef __m128i* mm_ptr;
#define VEC_16CAP 8
__forceinline __m128i vec_madd16(__m128i x, __m128i y)
{
    return _mm_madd_epi16(x, y);
}

__forceinline __m128i vec_lddqu(void *ptr)
{
    return _mm_lddqu_si128((__m128i*)ptr);
}
__forceinline __m128i vec_load(void *ptr)
{
    return _mm_load_si128((__m128i*)ptr);
}
#endif

// SSE4_2 & AVX1
#if OPT_LEVEL > 1 && OPT_LEVEL < 6
typedef __m128i mm_vector;
__forceinline __m128i vec_setzero()
{
    return _mm_setzero_si128();
}
#endif

// AVX1+
#if OPT_LEVEL > 3

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

typedef __m256i* mm_ptr;
#define VEC_16CAP 16

__forceinline __m256i vec_lddqu(void *ptr)
{
    return _mm256_lddqu_si256((__m256i*)ptr);
}
__forceinline __m256i vec_load(void *ptr)
{
    return _mm256_load_si256((__m256i*)ptr);
}
#endif

// AVX2+
#if OPT_LEVEL > 5
typedef __m256i mm_vector;
__forceinline __m256i vec_madd16(__m256i x, __m256i y)
{
    return _mm256_madd_epi16(x, y);
}
__forceinline __m256i vec_setzero()
{
    return _mm256_setzero_si256();
}
#endif

#if OPT_LEVEL % 2 == 0
#define GNA_SAT 0
typedef int32_t gna_sum_t;
#else
#define GNA_SAT 1
typedef int64_t gna_sum_t;
#endif

#define SSE_16CAP 8
