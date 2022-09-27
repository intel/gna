/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "saturate.h"

#include <cstdint>
#include <nmmintrin.h>
#include <limits>

#define VEC_16CAP 8
typedef __m128i *mm_ptr;
typedef __m128i mm_vector;
static __forceinline __m128i vec_setzero()
{
    return _mm_setzero_si128();
}
static __forceinline __m128i vec_madd16(__m128i x, __m128i y)
{
    return _mm_madd_epi16(x, y);
}
static __forceinline __m128i vec_lddqu(void *ptr)
{
    return _mm_lddqu_si128((__m128i*)ptr);
}
static __forceinline __m128i vec_load(void *ptr)
{
    return _mm_load_si128((__m128i*)ptr);
}

/** @brief Add 32b signed integers inside 128b register */
static inline int32_t _mm_hsum_epi32(__m128i sum128)
{
    __m128i sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    __m128i sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));

    return _mm_cvtsi128_si32(sum32);
}

/** @brief Add 64b signed integers inside 128b register */
static inline int64_t _mm_hsum_epi64(__m128i sum128)
{
    __m128i sum64 = _mm_add_epi64(sum128, _mm_unpackhi_epi64(sum128, sum128));

    return _mm_cvtsi128_si64(sum64);
}

/** @brief Add 32b signed integers using saturation inside 128b register;
 *  set MSB of given 32b integer if saturation occured for given pair. */
static inline __m128i _mm_adds_epi32(__m128i a, __m128i b, __m128i *setHighBitOnSat)
{
    static const __m128i satMax = _mm_set1_epi32((std::numeric_limits<int32_t>::max)());
    __m128i sum = _mm_add_epi32(a, b);
    __m128i ov = _mm_andnot_si128(_mm_xor_si128(b, a), _mm_xor_si128(b, sum));
    a = _mm_xor_si128(satMax, _mm_srai_epi32(a, 32));
    *setHighBitOnSat = _mm_or_si128(ov, *setHighBitOnSat);
    return _mm_castps_si128(
        _mm_blendv_ps(_mm_castsi128_ps(sum), _mm_castsi128_ps(a), _mm_castsi128_ps(ov)));
}

/** @brief Check if any MSB of 32b integers is set inside 128b register */
static inline bool _mm_test_anyMSB_epi32(__m128i a)
{
    return 0 != _mm_movemask_ps(_mm_castsi128_ps(a));
}

/** @brief Check if any bit of 128b register is set */
static inline bool _mm_test_any(__m128i a)
{
    return 0 == _mm_testc_si128(a, _mm_set1_epi64x(-1));
}

/** Add pairs of 32-bit integers from lower lane of 'a' to upper lane of 'a' and pack the signed 64-bit results */
static inline __m128i _mm_sum_extend64(__m128i a)
{
    __m128i a64_lo = _mm_cvtepi32_epi64(a);
    __m128i a64_hi = _mm_cvtepi32_epi64(_mm_bsrli_si128(a, 8));

    return _mm_add_epi64(a64_lo, a64_hi);
}

/** Saturate packed signed 64-bit integers to 32-bit values. Results remain 64-bit */
static inline __m128i _mm_sat_epi64(__m128i a, uint32_t *saturationCounter)
{
    int64_t a64[2];

    _mm_storeu_si128((__m128i *)a64, a);

    saturate(&a64[0], saturationCounter);
    saturate(&a64[1], saturationCounter);

    return _mm_loadu_si128((__m128i *)a64);
}

/** Convert packed signed 64-bit integers from 'a' and 'b' to packed 32-bit integers using signed saturation */
static inline __m128i _mm_packs_epi64(__m128i a, __m128i b, uint32_t *saturationCounter)
{
    int32_t dst[4];

    int64_t a64_1 = _mm_extract_epi64(a, 0);
    int64_t a64_2 = _mm_extract_epi64(a, 1);

    int64_t b64_1 = _mm_extract_epi64(b, 0);
    int64_t b64_2 = _mm_extract_epi64(b, 1);

    saturate(&a64_1, saturationCounter);
    saturate(&a64_2, saturationCounter);

    saturate(&b64_1, saturationCounter);
    saturate(&b64_2, saturationCounter);

    dst[0] = (int32_t)a64_1;
    dst[1] = (int32_t)a64_2;

    dst[2] = (int32_t)b64_1;
    dst[3] = (int32_t)b64_2;

    return _mm_loadu_si128((__m128i *)dst);
}
