/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "saturate.h"
#include <immintrin.h>

#define VEC_16CAP 16
typedef __m256i* mm_ptr;
typedef __m128i mm_vector;

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define __forceinline inline
#endif

static __forceinline __m128i vec_setzero()
{
    return _mm_setzero_si128();
}

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

static __forceinline __m256i vec_lddqu(void *ptr)
{
    return _mm256_lddqu_si256((__m256i*)ptr);
}
