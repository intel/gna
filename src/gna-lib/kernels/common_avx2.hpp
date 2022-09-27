/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "saturate.h"

#include <cstdint>
#include <immintrin.h>
#include <limits>

#define VEC_16CAP 16
typedef __m256i *mm_ptr;
typedef __m256i mm_vector;
static __forceinline __m256i vec_lddqu(void *ptr)
{
    return _mm256_lddqu_si256((__m256i*)ptr);
}
static __forceinline __m256i vec_load(void *ptr)
{
    return _mm256_load_si256((__m256i*)ptr);
}
static __forceinline __m256i vec_madd16(__m256i x, __m256i y)
{
    return _mm256_madd_epi16(x, y);
}
static __forceinline __m256i vec_setzero()
{
    return _mm256_setzero_si256();
}
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

/** @brief Add 32b signed integers inside 256b register */
static inline int32_t _mm256_hsum_epi32(__m256i acc0)
{
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(acc0, 1), _mm256_castsi256_si128(acc0));
    __m128i sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    __m128i sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));

    return _mm_cvtsi128_si32(sum32);
}

/** @brief Add 64b signed integers inside 256b register */
static inline int64_t _mm256_hsum_epi64(__m256i acc0)
{
    __m128i sum128 = _mm_add_epi64(_mm256_extracti128_si256(acc0, 1), _mm256_castsi256_si128(acc0));
    __m128i sum64 = _mm_add_epi64(sum128, _mm_unpackhi_epi64(sum128, sum128));

    return _mm_cvtsi128_si64(sum64);
}

/** @brief Add 32b signed integers using saturation inside 256b register;
 *  set MSB of given 32b integer if saturation occured for given pair. */
static inline __m256i _mm256_adds_epi32(__m256i a, __m256i b, __m256i *setHighBitOnSat)
{
    static const __m256i satMax = _mm256_set1_epi32((std::numeric_limits<int32_t>::max)());
    __m256i sum = _mm256_add_epi32(a, b);
    __m256i ov = _mm256_andnot_si256(_mm256_xor_si256(b, a), _mm256_xor_si256(b, sum));
    a = _mm256_xor_si256(satMax, _mm256_srai_epi32(a, 32));
    *setHighBitOnSat = _mm256_or_si256(ov, *setHighBitOnSat);
    return _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(sum), _mm256_castsi256_ps(a), _mm256_castsi256_ps(ov)));
}

/** @brief Check if any MSB of 32b integers is set inside 256b register */
static inline bool _mm256_test_anyMSB_epi32(__m256i a)
{
    return 0 != _mm256_movemask_ps(_mm256_castsi256_ps(a));
}

/** @brief Check if any bit of 256b register is set */
static inline bool _mm256_test_any(__m256i a)
{
    return 0 == _mm256_testc_si256(a, _mm256_set1_epi64x(-1));
}

/** Add pairs of 32-bit integers from lower lane of 'a' to upper lane of 'a' and pack the signed 64-bit results */
static inline __m256i _mm256_sum_extend64(__m256i a)
{
    __m128i a_lo = _mm256_castsi256_si128(a);
    __m128i a_hi = _mm256_extracti128_si256(a, 1);

    __m256i a64_lo = _mm256_cvtepi32_epi64(a_lo);
    __m256i a64_hi = _mm256_cvtepi32_epi64(a_hi);

    return _mm256_add_epi64(a64_lo, a64_hi);
}

/** Saturate packed signed 64-bit integers to 32-bit values. Results remain 64-bit */
static inline __m256i _mm256_sat_epi64(__m256i a, uint32_t *saturationCounter)
{
    int64_t a64[4];

    _mm256_storeu_si256((__m256i *)a64, a);

    saturate(&a64[0], saturationCounter);
    saturate(&a64[1], saturationCounter);
    saturate(&a64[2], saturationCounter);
    saturate(&a64[3], saturationCounter);

    return _mm256_loadu_si256((__m256i *)a64);
}

/** Convert packed signed 64-bit integers from 'a' and 'b' to packed 32-bit integers using signed saturation */
static inline __m256i _mm256_packs_epi64(__m256i a, __m256i b, uint32_t *saturationCounter)
{
    int32_t dst[8];

    int64_t a64_1 = _mm256_extract_epi64(a, 0);
    int64_t a64_2 = _mm256_extract_epi64(a, 1);
    int64_t a64_3 = _mm256_extract_epi64(a, 2);
    int64_t a64_4 = _mm256_extract_epi64(a, 3);

    int64_t b64_1 = _mm256_extract_epi64(b, 0);
    int64_t b64_2 = _mm256_extract_epi64(b, 1);
    int64_t b64_3 = _mm256_extract_epi64(b, 2);
    int64_t b64_4 = _mm256_extract_epi64(b, 3);

    saturate(&a64_1, saturationCounter);
    saturate(&a64_2, saturationCounter);
    saturate(&a64_3, saturationCounter);
    saturate(&a64_4, saturationCounter);

    saturate(&b64_1, saturationCounter);
    saturate(&b64_2, saturationCounter);
    saturate(&b64_3, saturationCounter);
    saturate(&b64_4, saturationCounter);

    dst[0] = (int32_t)a64_1;
    dst[1] = (int32_t)a64_2;
    dst[2] = (int32_t)a64_3;
    dst[3] = (int32_t)a64_4;

    dst[4] = (int32_t)b64_1;
    dst[5] = (int32_t)b64_2;
    dst[6] = (int32_t)b64_3;
    dst[7] = (int32_t)b64_4;

    return _mm256_loadu_si256((__m256i *)dst);
}
