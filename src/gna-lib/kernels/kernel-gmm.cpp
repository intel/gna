/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "kernel-gmm.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>

#if defined(_WIN32)
#pragma warning (disable: 592)
#endif

#ifdef INTEL64
#define CVT64_128(a) _mm_cvtsi64_si128(*(int64_t*)(a))
#else
#define CVT64_128(a) _mm_loadl_epi64((__m128i*)(a))
#endif

// Rather combining mixtures using log addition, this function selects
// the score of the single best scoring mixture as the representative.
// In unsigned version, score is negative of true score and best score
// is the smallest score.

#if OPT_LEVEL == 0 || OPT_LEVEL == 1 // NONE

void gmm_maxmix_8u8u_32u(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint32_t *consts = config->Gconst;
    uint32_t minScore = config->MaxScore;
    uint64_t sum64;
    uint32_t i, j;

    for (i = 0; i < config->MixtureCount; i++)
    {
        uint32_t Score32u = 0;

        for (j = 0; j < config->InputElementCount; j++)
        {
            int16_t Diff16s = static_cast<int16_t>(config->Input[j] - mean[j]);
            uint16_t SqrDiff16s = static_cast<uint16_t>(Diff16s * Diff16s);

            Score32u += static_cast<uint32_t>(SqrDiff16s * var[j]);
        }

        // sum may saturate depending on value of const
        sum64 = (uint64_t)Score32u + (uint64_t)*consts;
        Score32u = (sum64 > 0xffffffff) ? 0xffffffff : (uint32_t)sum64;

        minScore = (Score32u < minScore) ? Score32u : minScore;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    (*(config->Output)) = minScore;
}

void gmm_maxmix_8u16u_32u(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint16_t *var = config->Vars16;
    const uint32_t *consts = config->Gconst;
    uint32_t minScore = config->MaxScore;
    uint64_t sum64;
    uint32_t i, j;

    for (i = 0; i < config->MixtureCount; i++)
    {
        uint64_t Score64u = 0;
        uint32_t Score32u;

        for (j = 0; j < config->InputElementCount; j++)
        {
            int16_t Diff16s = static_cast<int16_t>(config->Input[j] - mean[j]);
            uint16_t SqrDiff16s = static_cast<uint16_t>(Diff16s * Diff16s);

            Score64u += static_cast<uint64_t>(SqrDiff16s * var[j]);
        }

        // sum may saturate depending on value of const
        sum64 = Score64u + (uint64_t)*consts;
        Score32u = (sum64 > 0xffffffff) ? 0xffffffff : (uint32_t)sum64;

        minScore = (Score32u < minScore) ? Score32u : minScore;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    (*(config->Output)) = minScore;
}

#endif //#if OPT_LEVEL == 0 || OPT_LEVEL == 1 // NONE

#if OPT_LEVEL > 1 // SSE4+

#if defined(_WIN32)
#pragma warning( disable : 700 )
#endif
void gmm_maxmix_8u8u_32u(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint32_t *consts = config->Gconst;
    uint64_t minScore = config->MaxScore;
    uint64_t sum64;
    uint32_t i, j;

    for (i = 0; i < config->MixtureCount; i++)
    {
        __m128i sum;

#if defined(__INTEL_COMPILER)
#pragma warning disable 592
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
        __m128i sum_1 = _mm_xor_si128(sum_1, sum_1);

        __m128i sum_2 = sum_1;

        for (j = 0; j < config->InputElementCount; j += 8)
        {
            __m128i load1 = CVT64_128(&config->Input[j]); // vector load 8x8-bit
            __m128i load2 = CVT64_128(&mean[j]); // vector load 8x8-bit
            __m128i load3 = CVT64_128(&var[j]); // vector load 8x8-bit
            __m128i zext1 = _mm_cvtepu8_epi16(load1); // convert to 8x16-bit
            __m128i zext2 = _mm_cvtepu8_epi16(load2); // convert to 8x16-bit
            __m128i zext3 = _mm_cvtepu8_epi16(load3); // convert to 8x16-bit
            __m128i diff16s = _mm_sub_epi16(zext1, zext2); // 8x16-bit subtract
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, zext3); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, zext3); // 8x16-bit mul (hi part)
            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
            sum_1 = _mm_add_epi32(sum_1, lower_prods); // 4x32-bit addition
            sum_2 = _mm_add_epi32(sum_2, upper_prods); // 4x32-bit addition
        }
        sum = _mm_add_epi32(sum_1, sum_2); // 4x32-bit addition
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xee)); // horizontal 32-bit add
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x55)); // horizontal 32-bit add

        // sum may saturate depending on value of const
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum) + (uint64_t)*consts;
        minScore = (sum64 < minScore) ? sum64 : minScore;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    *config->Output = (uint32_t) minScore;
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g1_sse4
void gmm_maxmix_8u8u_32u_g1(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input = config->Input;
    const uint32_t *consts = config->Gconst;
    uint32_t mixtureCount = config->MixtureCount;
    uint32_t i, j, k;
    uint64_t minScore;
    uint64_t sum64, sum64b;
    minScore = config->MaxScore;

    if ((mixtureCount & 1) == 1)
    {
        __m128i sum_1 = _mm_setzero_si128();

        __m128i load1 = CVT64_128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            __m128i load1_1 = _mm_cvtepu8_epi16(load1);
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);

            __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            load1 = CVT64_128((__m128i*)&input[j]);
            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

            sum_1 = _mm_add_epi32(sum_1, lower_prods);
            sum_1 = _mm_add_epi32(sum_1, upper_prods);
        }
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add

        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*consts;
        minScore = sum64 < minScore ? sum64 : minScore;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
        mixtureCount--;
    }

    for (i = 0; i < mixtureCount; i += 2)
    {
        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();

        __m128i load1 = CVT64_128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit
        __m128i load22 = CVT64_128((__m128i*)(mean + config->InputElementCount)); // vector load 8x8-bit
        __m128i load33 = CVT64_128((__m128i*)(var + config->InputElementCount)); // vector load 8x8-bit

        for (j = 8, k = 8 + config->InputElementCount; j <= config->InputElementCount; j += 8, k += 8)
        {
            __m128i load1_1 = _mm_cvtepu8_epi16(load1);
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);

            __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            load1 = CVT64_128((__m128i*)&input[j]);
            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit

            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
            __m128i load22_1 = _mm_cvtepu8_epi16(load22);
            __m128i load33_1 = _mm_cvtepu8_epi16(load33);

            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

            __m128i diff16s2 = _mm_sub_epi16(load1_1, load22_1); // convert to 8x16-bit
            __m128i sqrdiff16s2 = _mm_mullo_epi16(diff16s2, diff16s2); // 8x16-bit mul (hi zero)
            load22 = CVT64_128((__m128i*)&mean[k]); // vector load 8x8-bit
            load33 = CVT64_128((__m128i*)&var[k]); // vector load 8x8-bit

            __m128i prod_low2 = _mm_mullo_epi16(sqrdiff16s2, load33_1); // 8x16-bit mult (lo part)
            __m128i prod_high2 = _mm_mulhi_epu16(sqrdiff16s2, load33_1); // 8x16-bit mul (hi part)
            sum_1 = _mm_add_epi32(sum_1, lower_prods);
            sum_1 = _mm_add_epi32(sum_1, upper_prods);

            __m128i lower_prods2 = _mm_unpacklo_epi16(prod_low2, prod_high2); // lo 4x32-bit prd
            __m128i upper_prods2 = _mm_unpackhi_epi16(prod_low2, prod_high2); // hi 4x32-bit prd

            sum_2 = _mm_add_epi32(sum_2, lower_prods2);
            sum_2 = _mm_add_epi32(sum_2, upper_prods2);
        }

        __m128i horiz1 = _mm_hadd_epi32(sum_1, sum_2);
        horiz1 = _mm_hadd_epi32(horiz1, horiz1);

        sum64 = (uint64_t)_mm_cvtsi128_si32(horiz1) + (uint64_t)*consts;
        minScore = sum64 < minScore ? sum64 : minScore;

        sum64b = (uint64_t)_mm_extract_epi32(horiz1, 1) + (uint64_t)*(consts + 1);
        minScore = sum64b < minScore ? sum64b : minScore;

        mean += (config->InputElementCount << 1);
        var += (config->InputElementCount << 1);
        consts += 2;
    }

    *config->Output = (uint32_t) minScore;
}

#endif //#if OPT_LEVEL > 1 // SSE4+

#if (OPT_LEVEL > 1) && (OPT_LEVEL < 6) // SSE4/AVX1 only (same code, different compile options)

void gmm_maxmix_8u16u_32u(GmmConfig const * const config)
{
    const uint8_t  *mean = config->Means;
    const uint16_t *var = config->Vars16;
    const uint32_t *consts = config->Gconst;
    uint64_t minScore = config->MaxScore;
    uint32_t i, j;

    __m128i zero = _mm_setzero_si128();

    for (i = 0; i < config->MixtureCount; i++)
    {
        uint64_t Score64u;

        __m128i sum_lo = zero;
        __m128i sum_hi = zero;
        __m128i load1 = CVT64_128(config->Input); // vector load 8x8-bit
        __m128i load2 = CVT64_128(mean); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            __m128i zext1 = _mm_cvtepu8_epi16(load1); // convert to 8x16-bit
            __m128i zext2 = _mm_cvtepu8_epi16(load2); // convert to 8x16-bit

            __m128i load3 = _mm_loadu_si128((const __m128i*)var); // vector load 8x8-bit

            __m128i diff16s = _mm_sub_epi16(zext1, zext2); // 8x16-bit subtract
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)

            var += 8;
            mean += 8;

            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3); // 8x16-bit mul (hi part)

            load1 = CVT64_128(&config->Input[j]); // vector load 8x8-bit
            load2 = CVT64_128(mean); // vector load 8x8-bit

            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, zero); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, zero); // hi 4x32-bit prd

            sum_lo = _mm_add_epi32(sum_lo, lower_prods); // 4x32-bit addition
            sum_lo = _mm_add_epi32(sum_lo, upper_prods); // 4x32-bit addition
            sum_hi = _mm_adds_epu16(sum_hi, prod_high);
        }

        __m128i sum_hi_2 = _mm_shuffle_epi32(sum_hi, 0xee);
        __m128i sum_hi_1 = _mm_cvtepu16_epi32(sum_hi);
        __m128i sum_hi_3 = _mm_cvtepu16_epi32(sum_hi_2);

        sum_hi = _mm_add_epi32(sum_hi_1, sum_hi_3);
        sum_lo = _mm_add_epi32(sum_lo, _mm_shuffle_epi32(sum_lo, 0xee)); // horizontal 32-bit add
        sum_hi = _mm_add_epi32(sum_hi, _mm_shuffle_epi32(sum_hi, 0xee)); // horizontal 32-bit add
        sum_lo = _mm_add_epi32(sum_lo, _mm_shuffle_epi32(sum_lo, 0x55)); // horizontal 32-bit add
        sum_hi = _mm_add_epi32(sum_hi, _mm_shuffle_epi32(sum_hi, 0x55)); // horizontal 32-bit add

        Score64u = static_cast<uint64_t>(_mm_cvtsi128_si32(sum_hi)); // convert sum to 1x32-bit
        Score64u = (Score64u << 16) + static_cast<uint64_t>(_mm_cvtsi128_si32(sum_lo)) + *consts; // convert sum to 1x32-bit

        // sum may saturate depending on value of const

        minScore = (Score64u < minScore) ? Score64u : minScore;

        consts++;
    }

    (*(config->Output)) = (uint32_t) minScore;
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g2_sse4
void gmm_maxmix_8u8u_32u_g2(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t  gconst64;
    uint64_t  score[2];
    // init min scores
    uint64_t  minScore = config->MaxScore;
    __m128i minScores = CVT64_128((__m128i*) &minScore);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i load1 = _mm_load_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);

            __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
            __m128i load1_1 = _mm_cvtepu8_epi16(load1);
            load1_2 = _mm_cvtepu8_epi16(load1_2);

            __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
            __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
            __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
            load1 = _mm_load_si128((__m128i*)input);
            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
            __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

            sum_1 = _mm_add_epi32(sum_1, lower_prods);
            sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
            sum_1 = _mm_add_epi32(sum_1, upper_prods);
            sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
        }
        gconst64 = *consts;
        __m128i horiz1 = _mm_hadd_epi32(sum_1, sum_2);
        horiz1 = _mm_hadd_epi32(horiz1, horiz1);

        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));
        __m128i sum_64 = _mm_cvtepu32_epi64(horiz1);
        sum_64 = _mm_add_epi64(sum_64, gconst);
        __m128i cmpres = _mm_cmpgt_epi64(minScores, sum_64);
        __m128i res_min = _mm_andnot_si128(cmpres, minScores);
        __m128i res_sum = _mm_and_si128(cmpres, sum_64);
        minScores = _mm_or_si128(res_min, res_sum);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g3_sse4
void gmm_maxmix_8u8u_32u_g3(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t sum64;

    uint64_t gconst64;
    uint64_t score[2];
    uint64_t minScore2 = config->MaxScore;
    __m128i minScores = CVT64_128((__m128i*) &minScore2);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;
        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i load1_1 = _mm_loadu_si128((__m128i*)input);
        __m128i load1_3 = CVT64_128((__m128i*)(input + 16));
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 24;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);

            __m128i load1_2 = _mm_shuffle_epi32(load1_1, _MM_SHUFFLE(3, 2, 3, 2));
            load1_1 = _mm_cvtepu8_epi16(load1_1);
            load1_2 = _mm_cvtepu8_epi16(load1_2);
            load1_3 = _mm_cvtepu8_epi16(load1_3);

            __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
            __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
            __m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
            __m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)
            load1_1 = _mm_loadu_si128((__m128i*)input);
            load1_3 = CVT64_128((__m128i*)(input + 16));
            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
            __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
            __m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
            __m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)

            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
            __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
            __m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
            __m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

            sum_1 = _mm_add_epi32(sum_1, lower_prods);
            sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
            sum_3 = _mm_add_epi32(sum_3, lower_prods_3);
            sum_1 = _mm_add_epi32(sum_1, upper_prods);
            sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
            sum_3 = _mm_add_epi32(sum_3, upper_prods_3);
        }
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add

        gconst64 = *consts;
        sum_1 = _mm_hadd_epi32(sum_1, sum_2);
        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        sum_1 = _mm_hadd_epi32(sum_1, sum_1);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));
        sum_1 = _mm_cvtepu32_epi64(sum_1);
        sum_1 = _mm_add_epi64(sum_1, gconst);
        __m128i cmpres_1 = _mm_cmpgt_epi64(minScores, sum_1);
        __m128i res_min_1 = _mm_andnot_si128(cmpres_1, minScores);
        __m128i res_sum_1 = _mm_and_si128(cmpres_1, sum_1);
        minScores = _mm_or_si128(res_min_1, res_sum_1);

        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*consts;
        minScore2 = sum64 < minScore2 ? sum64 : minScore2;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    // store results
    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
    config->Output[2] = (uint32_t) minScore2;
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g4_sse4
void gmm_maxmix_8u8u_32u_g4(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t  gconst64;
    uint64_t  score[2];
    uint64_t  minScore = config->MaxScore;
    __m128i minScores = CVT64_128((__m128i*) &minScore);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));
    __m128i minScore2 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i sum_4 = _mm_setzero_si128();

        __m128i lower_prods;
        __m128i upper_prods;
        __m128i lower_prods_2;
        __m128i upper_prods_2;

        __m128i load1 = _mm_load_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
                load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_1 = _mm_add_epi32(sum_1, lower_prods);
                sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                sum_1 = _mm_add_epi32(sum_1, upper_prods);
                sum_2 = _mm_add_epi32(sum_2, upper_prods_2);

                lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_3 = _mm_add_epi32(sum_3, lower_prods);
                sum_4 = _mm_add_epi32(sum_4, lower_prods_2);
                sum_3 = _mm_add_epi32(sum_3, upper_prods);
                sum_4 = _mm_add_epi32(sum_4, upper_prods_2);
                load1 = _mm_load_si128((__m128i*)input);
            }
        }


        gconst64 = *consts;
        sum_1 = _mm_hadd_epi32(sum_1, sum_2);
        sum_2 = _mm_hadd_epi32(sum_3, sum_4);
        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        sum_1 = _mm_hadd_epi32(sum_1, sum_1);
        sum_2 = _mm_hadd_epi32(sum_2, sum_2);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));
        sum_1 = _mm_cvtepu32_epi64(sum_1);
        sum_2 = _mm_cvtepu32_epi64(sum_2);
        sum_1 = _mm_add_epi64(sum_1, gconst);
        sum_2 = _mm_add_epi64(sum_2, gconst);
        __m128i cmpres_1 = _mm_cmpgt_epi64(minScores, sum_1);
        __m128i cmpres_2 = _mm_cmpgt_epi64(minScore2, sum_2);
        __m128i res_min_1 = _mm_andnot_si128(cmpres_1, minScores);
        __m128i res_min_2 = _mm_andnot_si128(cmpres_2, minScore2);
        __m128i res_sum_1 = _mm_and_si128(cmpres_1, sum_1);
        __m128i res_sum_2 = _mm_and_si128(cmpres_2, sum_2);
        minScores = _mm_or_si128(res_min_1, res_sum_1);
        minScore2 = _mm_or_si128(res_min_2, res_sum_2);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    // store results
    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
    _mm_storeu_si128((__m128i*)score, minScore2);
    config->Output[2] = (uint32_t) score[0];
    config->Output[3] = (uint32_t) score[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g5_sse4
void gmm_maxmix_8u8u_32u_g5(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t sum64;
    uint64_t  gconst64;
    uint64_t  score[2];
    uint64_t  minScores3 = config->MaxScore;
    __m128i minScores = CVT64_128((__m128i*) &minScores3);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));
    __m128i minScore2 = minScores;


    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i sum_4 = _mm_setzero_si128();
        __m128i sum_5 = _mm_setzero_si128();
        __m128i load1 = _mm_loadu_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit
        __m128i load1_3;
        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i    load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
                load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_loadu_si128((__m128i*)input);
                load1_3 = CVT64_128((__m128i*)(input + 16));
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                __m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_1 = _mm_add_epi32(sum_1, lower_prods);
                sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
                sum_1 = _mm_add_epi32(sum_1, upper_prods);
                sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
            }
            input += 24;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);
                load1_3 = _mm_cvtepu8_epi16(load1_3);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)
                load1 = _mm_loadu_si128((__m128i*)input);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)

                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                __m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                __m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
                __m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

                sum_3 = _mm_add_epi32(sum_3, lower_prods);
                sum_4 = _mm_add_epi32(sum_4, lower_prods_2);
                sum_5 = _mm_add_epi32(sum_5, lower_prods_3);
                sum_3 = _mm_add_epi32(sum_3, upper_prods);
                sum_4 = _mm_add_epi32(sum_4, upper_prods_2);
                sum_5 = _mm_add_epi32(sum_5, upper_prods_3);
            }
        }
        gconst64 = *consts;
        sum_1 = _mm_hadd_epi32(sum_1, sum_2);
        sum_2 = _mm_hadd_epi32(sum_3, sum_4);
        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        sum_1 = _mm_hadd_epi32(sum_1, sum_1);
        sum_2 = _mm_hadd_epi32(sum_2, sum_2);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));
        sum_1 = _mm_cvtepu32_epi64(sum_1);
        sum_2 = _mm_cvtepu32_epi64(sum_2);
        sum_1 = _mm_add_epi64(sum_1, gconst);
        sum_2 = _mm_add_epi64(sum_2, gconst);
        __m128i cmpres_1 = _mm_cmpgt_epi64(minScores, sum_1);
        __m128i cmpres_2 = _mm_cmpgt_epi64(minScore2, sum_2);
        __m128i res_min_1 = _mm_andnot_si128(cmpres_1, minScores);
        __m128i res_min_2 = _mm_andnot_si128(cmpres_2, minScore2);
        __m128i res_sum_1 = _mm_and_si128(cmpres_1, sum_1);
        __m128i res_sum_2 = _mm_and_si128(cmpres_2, sum_2);
        minScores = _mm_or_si128(res_min_1, res_sum_1);
        minScore2 = _mm_or_si128(res_min_2, res_sum_2);
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add

        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*consts;
        minScores3 = sum64 < minScores3 ? sum64 : minScores3;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
    _mm_storeu_si128((__m128i*)score, minScore2);
    config->Output[2] = (uint32_t) score[0];
    config->Output[3] = (uint32_t) score[1];
    config->Output[4] = (uint32_t) minScores3;
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g6_sse4
void gmm_maxmix_8u8u_32u_g6(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t sum64;
    uint64_t minScores[6];

    minScores[0] = config->MaxScore;
    minScores[1] = config->MaxScore;
    minScores[2] = config->MaxScore;
    minScores[3] = config->MaxScore;
    minScores[4] = config->MaxScore;
    minScores[5] = config->MaxScore;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i sum_4 = _mm_setzero_si128();
        __m128i sum_5 = _mm_setzero_si128();
        __m128i sum_6 = _mm_setzero_si128();

        __m128i upper_prods;
        __m128i upper_prods_2;
        __m128i load1 = _mm_load_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
                load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_1 = _mm_add_epi32(sum_1, lower_prods);
                sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i    load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_1 = _mm_add_epi32(sum_1, upper_prods);
                sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_3 = _mm_add_epi32(sum_3, lower_prods);
                sum_4 = _mm_add_epi32(sum_4, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_3 = _mm_add_epi32(sum_3, upper_prods);
                sum_4 = _mm_add_epi32(sum_4, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_5 = _mm_add_epi32(sum_5, lower_prods);
                sum_6 = _mm_add_epi32(sum_6, lower_prods_2);
                sum_5 = _mm_add_epi32(sum_5, upper_prods);
                sum_6 = _mm_add_epi32(sum_6, upper_prods_2);
            }
        }
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
        sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0xee)); // horizontal 32-bit add
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add
        sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0x55)); // horizontal 32-bit add
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add
        sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0xee)); // horizontal 32-bit add
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add
        sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0x55)); // horizontal 32-bit add
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add
        sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0xee)); // horizontal 32-bit add
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add
        sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0x55)); // horizontal 32-bit add


        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*consts;
        minScores[0] = sum64 < minScores[0] ? sum64 : minScores[0];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_2) + (uint64_t)*consts;
        minScores[1] = sum64 < minScores[1] ? sum64 : minScores[1];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*consts;
        minScores[2] = sum64 < minScores[2] ? sum64 : minScores[2];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_4) + (uint64_t)*consts;
        minScores[3] = sum64 < minScores[3] ? sum64 : minScores[3];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*consts;
        minScores[4] = sum64 < minScores[4] ? sum64 : minScores[4];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_6) + (uint64_t)*consts;
        minScores[5] = sum64 < minScores[5] ? sum64 : minScores[5];

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    config->Output[0] = (uint32_t) minScores[0];
    config->Output[1] = (uint32_t) minScores[1];
    config->Output[2] = (uint32_t) minScores[2];
    config->Output[3] = (uint32_t) minScores[3];
    config->Output[4] = (uint32_t) minScores[4];
    config->Output[5] = (uint32_t) minScores[5];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g7_sse4
void gmm_maxmix_8u8u_32u_g7(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t sum64;
    uint64_t minScores[8];

    minScores[0] = config->MaxScore;
    minScores[1] = config->MaxScore;
    minScores[2] = config->MaxScore;
    minScores[3] = config->MaxScore;
    minScores[4] = config->MaxScore;
    minScores[5] = config->MaxScore;
    minScores[6] = config->MaxScore;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i sum_4 = _mm_setzero_si128();
        __m128i sum_5 = _mm_setzero_si128();
        __m128i sum_6 = _mm_setzero_si128();
        __m128i sum_7 = _mm_setzero_si128();
        __m128i upper_prods;
        __m128i upper_prods_2;
        __m128i load1 = _mm_loadu_si128((__m128i*)input);
        __m128i load1_3;
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
                load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_loadu_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_1 = _mm_add_epi32(sum_1, lower_prods);
                sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i    load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_1 = _mm_add_epi32(sum_1, upper_prods);
                sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_loadu_si128((__m128i*)input);
                load1_3 = CVT64_128((__m128i*)(input + 16)); // vector load 8x8-bit
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_3 = _mm_add_epi32(sum_3, lower_prods);
                sum_4 = _mm_add_epi32(sum_4, lower_prods_2);
            }
            input += 24;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);
                load1_3 = _mm_cvtepu8_epi16(load1_3);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)
                sum_3 = _mm_add_epi32(sum_3, upper_prods);
                sum_4 = _mm_add_epi32(sum_4, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)

                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                __m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
                __m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
                load1 = _mm_loadu_si128((__m128i*)input);

                sum_5 = _mm_add_epi32(sum_5, lower_prods);
                sum_6 = _mm_add_epi32(sum_6, lower_prods_2);
                sum_7 = _mm_add_epi32(sum_7, lower_prods_3);
                sum_5 = _mm_add_epi32(sum_5, upper_prods);
                sum_6 = _mm_add_epi32(sum_6, upper_prods_2);
                sum_7 = _mm_add_epi32(sum_7, upper_prods_3);
            }
        }
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
        sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0xee)); // horizontal 32-bit add
        sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add
        sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0x55)); // horizontal 32-bit add
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add
        sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0xee)); // horizontal 32-bit add
        sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add
        sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0x55)); // horizontal 32-bit add
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add
        sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0xee)); // horizontal 32-bit add
        sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add
        sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0x55)); // horizontal 32-bit add
        sum_7 = _mm_add_epi32(sum_7, _mm_shuffle_epi32(sum_7, 0xee)); // horizontal 32-bit add
        sum_7 = _mm_add_epi32(sum_7, _mm_shuffle_epi32(sum_7, 0x55)); // horizontal 32-bit add

        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*consts;
        minScores[0] = sum64 < minScores[0] ? sum64 : minScores[0];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_2) + (uint64_t)*consts;
        minScores[1] = sum64 < minScores[1] ? sum64 : minScores[1];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*consts;
        minScores[2] = sum64 < minScores[2] ? sum64 : minScores[2];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_4) + (uint64_t)*consts;
        minScores[3] = sum64 < minScores[3] ? sum64 : minScores[3];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*consts;
        minScores[4] = sum64 < minScores[4] ? sum64 : minScores[4];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_6) + (uint64_t)*consts;
        minScores[5] = sum64 < minScores[5] ? sum64 : minScores[5];
        sum64 = (uint64_t)_mm_cvtsi128_si32(sum_7) + (uint64_t)*consts;
        minScores[6] = sum64 < minScores[6] ? sum64 : minScores[6];

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    config->Output[0] = (uint32_t) minScores[0];
    config->Output[1] = (uint32_t) minScores[1];
    config->Output[2] = (uint32_t) minScores[2];
    config->Output[3] = (uint32_t) minScores[3];
    config->Output[4] = (uint32_t) minScores[4];
    config->Output[5] = (uint32_t) minScores[5];
    config->Output[6] = (uint32_t) minScores[6];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g8_sse4
void gmm_maxmix_8u8u_32u_g8(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t  gconst64;
    uint64_t  score[2];
    uint64_t  minScore = config->MaxScore;
    __m128i minScores = CVT64_128((__m128i*) &minScore);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));
    __m128i minScore2 = minScores;
    __m128i minScores3 = minScores;
    __m128i minScores4 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m128i sum_1 = _mm_setzero_si128();
        __m128i sum_2 = _mm_setzero_si128();
        __m128i sum_3 = _mm_setzero_si128();
        __m128i sum_4 = _mm_setzero_si128();
        __m128i sum_5 = _mm_setzero_si128();
        __m128i sum_6 = _mm_setzero_si128();
        __m128i sum_7 = _mm_setzero_si128();
        __m128i sum_8 = _mm_setzero_si128();

        __m128i upper_prods;
        __m128i upper_prods_2;
        __m128i load1 = _mm_load_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);
            __m128i load3_1 = _mm_cvtepu8_epi16(load3);
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
                load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_1 = _mm_add_epi32(sum_1, lower_prods);
                sum_2 = _mm_add_epi32(sum_2, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i    load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_1 = _mm_add_epi32(sum_1, upper_prods);
                sum_2 = _mm_add_epi32(sum_2, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_3 = _mm_add_epi32(sum_3, lower_prods);
                sum_4 = _mm_add_epi32(sum_4, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                sum_3 = _mm_add_epi32(sum_3, upper_prods);
                sum_4 = _mm_add_epi32(sum_4, upper_prods_2);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_5 = _mm_add_epi32(sum_5, lower_prods);
                sum_6 = _mm_add_epi32(sum_6, lower_prods_2);
            }
            input += 16;
            {
                __m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3, 2, 3, 2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1);
                load1_2 = _mm_cvtepu8_epi16(load1_2);

                __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
                __m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit
                __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
                __m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
                sum_5 = _mm_add_epi32(sum_5, upper_prods);
                sum_6 = _mm_add_epi32(sum_6, upper_prods_2);
                __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
                __m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
                __m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)
                load1 = _mm_load_si128((__m128i*)input);
                __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
                upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
                __m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
                upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
                sum_7 = _mm_add_epi32(sum_7, lower_prods);
                sum_8 = _mm_add_epi32(sum_8, lower_prods_2);
                sum_7 = _mm_add_epi32(sum_7, upper_prods);
                sum_8 = _mm_add_epi32(sum_8, upper_prods_2);
            }
        }
        gconst64 = *consts;
        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));
        sum_1 = _mm_hadd_epi32(sum_1, sum_2);
        sum_2 = _mm_hadd_epi32(sum_3, sum_4);
        sum_3 = _mm_hadd_epi32(sum_5, sum_6);
        sum_4 = _mm_hadd_epi32(sum_7, sum_8);
        sum_1 = _mm_hadd_epi32(sum_1, sum_2);
        sum_2 = _mm_shuffle_epi32(sum_1, _MM_SHUFFLE(3, 2, 3, 2));
        sum_3 = _mm_hadd_epi32(sum_3, sum_4);
        sum_4 = _mm_shuffle_epi32(sum_3, _MM_SHUFFLE(3, 2, 3, 2));
        sum_1 = _mm_cvtepu32_epi64(sum_1);
        sum_2 = _mm_cvtepu32_epi64(sum_2);
        sum_3 = _mm_cvtepu32_epi64(sum_3);
        sum_4 = _mm_cvtepu32_epi64(sum_4);
        sum_1 = _mm_add_epi64(sum_1, gconst);
        sum_2 = _mm_add_epi64(sum_2, gconst);
        sum_3 = _mm_add_epi64(sum_3, gconst);
        sum_4 = _mm_add_epi64(sum_4, gconst);

        __m128i cmpres_1 = _mm_cmpgt_epi64(minScores, sum_1);
        __m128i cmpres_2 = _mm_cmpgt_epi64(minScore2, sum_2);
        __m128i cmpres_3 = _mm_cmpgt_epi64(minScores3, sum_3);
        __m128i cmpres_4 = _mm_cmpgt_epi64(minScores4, sum_4);
        __m128i res_min_1 = _mm_andnot_si128(cmpres_1, minScores);
        __m128i res_min_2 = _mm_andnot_si128(cmpres_2, minScore2);
        __m128i res_min_3 = _mm_andnot_si128(cmpres_3, minScores3);
        __m128i res_min_4 = _mm_andnot_si128(cmpres_4, minScores4);
        __m128i res_sum_1 = _mm_and_si128(cmpres_1, sum_1);
        __m128i res_sum_2 = _mm_and_si128(cmpres_2, sum_2);
        __m128i res_sum_3 = _mm_and_si128(cmpres_3, sum_3);
        __m128i res_sum_4 = _mm_and_si128(cmpres_4, sum_4);
        minScores = _mm_or_si128(res_min_1, res_sum_1);
        minScore2 = _mm_or_si128(res_min_2, res_sum_2);
        minScores3 = _mm_or_si128(res_min_3, res_sum_3);
        minScores4 = _mm_or_si128(res_min_4, res_sum_4);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
    _mm_storeu_si128((__m128i*)score, minScore2);
    config->Output[2] = (uint32_t) score[0];
    config->Output[3] = (uint32_t) score[1];
    _mm_storeu_si128((__m128i*)score, minScores3);
    config->Output[4] = (uint32_t) score[0];
    config->Output[5] = (uint32_t) score[1];
    _mm_storeu_si128((__m128i*)score, minScores4);
    config->Output[6] = (uint32_t) score[0];
    config->Output[7] = (uint32_t) score[1];
}

#endif //#if (OPT_LEVEL > 1 && OPT_LEVEL < 6) // SSE4/AVX1 only

#if OPT_LEVEL > 5 // AVX2 +

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g2_avx2
void gmm_maxmix_8u8u_32u_g2(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t  i, j;
    uint64_t  gconst64;
    uint64_t  score[2];
    uint64_t  minScore = config->MaxScore;

    __m128i minScores = CVT64_128((__m128i*) &minScore);
    minScores = _mm_shuffle_epi32(minScores, _MM_SHUFFLE(1, 0, 1, 0));

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m256i sum_1 = _mm256_setzero_si256();

        __m128i load1 = _mm_loadu_si128((__m128i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit
        __m256i load256_1 = _mm256_cvtepu8_epi16(load1);
        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 16;
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s = _mm256_sub_epi16(load256_1, load256_2); // convert to 8x16-bit
            load1 = _mm_loadu_si128((__m128i*)input);
            __m256i sqrdiff16s = _mm256_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit
            __m256i prod_low = _mm256_mullo_epi16(sqrdiff16s, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high = _mm256_mulhi_epu16(sqrdiff16s, load256_3); // 8x16-bit mul (hi part)
            load256_1 = _mm256_cvtepu8_epi16(load1);
            __m256i lower_prods = _mm256_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m256i upper_prods = _mm256_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

            sum_1 = _mm256_add_epi32(sum_1, lower_prods);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods);
        }
        gconst64 = *consts;

        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f2 = _mm256_extractf128_si256(sum_1, 1);
        __m128i gconst = CVT64_128((__m128i*)&gconst64);
        gconst = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1, 0, 1, 0));

        __m128i sum_64 = _mm_hadd_epi32(sum_f1, sum_f2);
        sum_64 = _mm_hadd_epi32(sum_64, sum_64);
        sum_64 = _mm_cvtepu32_epi64(sum_64);
        sum_64 = _mm_add_epi64(sum_64, gconst);
        __m128i cmpres = _mm_cmpgt_epi64(minScores, sum_64);
        __m128i res_min = _mm_andnot_si128(cmpres, minScores);
        __m128i res_sum = _mm_and_si128(cmpres, sum_64);
        minScores = _mm_or_si128(res_min, res_sum);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    _mm_storeu_si128((__m128i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g3_avx2
void gmm_maxmix_8u8u_32u_g3(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];

    // init min scores
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;
        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 24;
            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);
        }

        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m256i sum_64 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        sum_64 = _mm256_add_epi64(sum_64, gconst);
        __m256i cmpres = _mm256_cmpgt_epi64(minScores, sum_64);
        __m256i res_min = _mm256_andnot_si256(cmpres, minScores);
        __m256i res_sum = _mm256_and_si256(cmpres, sum_64);
        minScores = _mm256_or_si256(res_min, res_sum);
        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }
    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g4_avx2
void gmm_maxmix_8u8u_32u_g4(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];

    // init min scores
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;
        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 32;
            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);
        }

        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m256i sum_64 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        sum_64 = _mm256_add_epi64(sum_64, gconst);
        __m256i cmpres = _mm256_cmpgt_epi64(minScores, sum_64);
        __m256i res_min = _mm256_andnot_si256(cmpres, minScores);
        __m256i res_sum = _mm256_and_si256(cmpres, sum_64);
        minScores = _mm256_or_si256(res_min, res_sum);
        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
    config->Output[3] = (uint32_t) score[3];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g5_avx2
void gmm_maxmix_8u8u_32u_g5(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];
    // init min scores
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));
    __m256i minScore2 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;
        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();
        __m256i sum_3 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 32;
            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_1_3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)input));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit
            __m256i diff16s_3 = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_3 = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)

            input += 8;

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_3 = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_3 = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
            __m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
            __m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_3 = _mm256_add_epi32(sum_3, lower_prods_3);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);
            sum_3 = _mm256_add_epi32(sum_3, upper_prods_3);
        }

        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        sum_3 = _mm256_hadd_epi32(sum_3, sum_3);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f3 = _mm256_castsi256_si128(sum_3);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        sum_3 = _mm256_permute4x64_epi64(sum_3, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f4 = _mm256_castsi256_si128(sum_3);
        __m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        __m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
        sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
        sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);
        __m256i cmpres_1 = _mm256_cmpgt_epi64(minScores, sum_64_1);
        __m256i cmpres_2 = _mm256_cmpgt_epi64(minScore2, sum_64_2);
        __m256i res_min_1 = _mm256_andnot_si256(cmpres_1, minScores);
        __m256i res_min_2 = _mm256_andnot_si256(cmpres_2, minScore2);
        __m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1);
        __m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2);
        minScores = _mm256_or_si256(res_min_1, res_sum_1);
        minScore2 = _mm256_or_si256(res_min_2, res_sum_2);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
    config->Output[3] = (uint32_t) score[3];
    config->Output[4] = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(minScore2)));
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g6_avx2
void gmm_maxmix_8u8u_32u_g6(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];
    // init min scores
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));
    __m256i minScore2 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;
        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();
        __m256i sum_3 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 32;
            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_1_3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)input));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit
            __m256i diff16s_3 = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_3 = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)

            input += 16;

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_3 = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_3 = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
            __m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
            __m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_3 = _mm256_add_epi32(sum_3, lower_prods_3);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);
            sum_3 = _mm256_add_epi32(sum_3, upper_prods_3);
        }

        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        sum_3 = _mm256_hadd_epi32(sum_3, sum_3);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f3 = _mm256_castsi256_si128(sum_3);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        sum_3 = _mm256_permute4x64_epi64(sum_3, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f4 = _mm256_castsi256_si128(sum_3);
        __m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        __m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
        sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
        sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);
        __m256i cmpres_1 = _mm256_cmpgt_epi64(minScores, sum_64_1);
        __m256i cmpres_2 = _mm256_cmpgt_epi64(minScore2, sum_64_2);
        __m256i res_min_1 = _mm256_andnot_si256(cmpres_1, minScores);
        __m256i res_min_2 = _mm256_andnot_si256(cmpres_2, minScore2);
        __m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1);
        __m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2);
        minScores = _mm256_or_si256(res_min_1, res_sum_1);
        minScore2 = _mm256_or_si256(res_min_2, res_sum_2);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
    config->Output[3] = (uint32_t) score[3];
    _mm256_storeu_si256((__m256i*)score, minScore2);
    config->Output[4] = (uint32_t) score[0];
    config->Output[5] = (uint32_t) score[2];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g7_avx2
void gmm_maxmix_8u8u_32u_g7(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));
    __m256i minScore2 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();
        __m256i sum_3 = _mm256_setzero_si256();
        __m256i sum_4 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 32;

            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)

            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit


            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);

            input += 24;

            __m256i load256_1_4 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_4 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_4));


            __m256i diff16s_3 = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit
            __m256i diff16s_4 = _mm256_sub_epi16(load256_1_4, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_3 = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_4 = _mm256_mullo_epi16(diff16s_4, diff16s_4); // 8x16-bit mul (hi zero)

            load1 = _mm256_loadu_si256((__m256i*)input);
            __m256i prod_low_3 = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_3 = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_4 = _mm256_mullo_epi16(sqrdiff16s_4, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_4 = _mm256_mulhi_epu16(sqrdiff16s_4, load256_3); // 8x16-bit mul (hi part)



            __m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
            __m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
            __m256i lower_prods_4 = _mm256_unpacklo_epi16(prod_low_4, prod_high_4); // lo 4x32-bit prd
            __m256i upper_prods_4 = _mm256_unpackhi_epi16(prod_low_4, prod_high_4); // hi 4x32-bit prd

            sum_3 = _mm256_add_epi32(sum_3, lower_prods_3);
            sum_4 = _mm256_add_epi32(sum_4, lower_prods_4);
            sum_3 = _mm256_add_epi32(sum_3, upper_prods_3);
            sum_4 = _mm256_add_epi32(sum_4, upper_prods_4);


        }
        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        sum_3 = _mm256_hadd_epi32(sum_3, sum_4);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f3 = _mm256_castsi256_si128(sum_3);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        sum_3 = _mm256_permute4x64_epi64(sum_3, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f4 = _mm256_castsi256_si128(sum_3);

        __m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
        __m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
        sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);

        __m256i cmpres_1 = _mm256_cmpgt_epi64(minScores, sum_64_1);
        __m256i cmpres_2 = _mm256_cmpgt_epi64(minScore2, sum_64_2);
        __m256i res_min_1 = _mm256_andnot_si256(cmpres_1, minScores);
        __m256i res_min_2 = _mm256_andnot_si256(cmpres_2, minScore2);
        __m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1);
        __m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2);
        minScores = _mm256_or_si256(res_min_1, res_sum_1);
        minScore2 = _mm256_or_si256(res_min_2, res_sum_2);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
    config->Output[3] = (uint32_t) score[3];
    _mm256_storeu_si256((__m256i*)score, minScore2);
    config->Output[4] = (uint32_t) score[0];
    config->Output[5] = (uint32_t) score[2];
    config->Output[6] = (uint32_t) score[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g8_avx2
void gmm_maxmix_8u8u_32u_g8(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint8_t *var = config->Vars;
    const uint8_t *input;
    const uint32_t *consts = config->Gconst;
    uint32_t i, j;
    uint64_t gconst64;
    uint64_t score[4];
    uint64_t  minScore = config->MaxScore;
    __m256i minScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &minScore));
    __m256i minScore2 = minScores;

    for (i = 0; i < config->MixtureCount; i++)
    {
        input = config->Input;

        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();
        __m256i sum_3 = _mm256_setzero_si256();
        __m256i sum_4 = _mm256_setzero_si256();

        __m256i load1 = _mm256_loadu_si256((__m256i*)input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = CVT64_128((__m128i*)var); // vector load 8x8-bit

        for (j = 8; j <= config->InputElementCount; j += 8)
        {
            input += 32;

            __m256i load256_1_2 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2));
            __m256i load256_2 = _mm256_cvtepu8_epi16(load2);
            __m256i load256_3 = _mm256_cvtepu8_epi16(load3);

            load256_2 = _mm256_permute4x64_epi64(load256_2, 0x44);
            load256_3 = _mm256_permute4x64_epi64(load256_3, 0x44);

            __m256i diff16s_1 = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit
            __m256i diff16s_2 = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_1 = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_2 = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)

            load1 = _mm256_loadu_si256((__m256i*)input);

            __m256i prod_low_1 = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_1 = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_2 = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_2 = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit
            load3 = CVT64_128((__m128i*)&var[j]); // vector load 8x8-bit


            __m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
            __m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
            __m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
            __m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

            sum_1 = _mm256_add_epi32(sum_1, lower_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, lower_prods_2);
            sum_1 = _mm256_add_epi32(sum_1, upper_prods_1);
            sum_2 = _mm256_add_epi32(sum_2, upper_prods_2);

            input += 32;

            __m256i load256_1_4 = _mm256_permute4x64_epi64(load1, 0xee);
            __m256i load256_1_3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1));
            load256_1_4 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_4));


            __m256i diff16s_3 = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit
            __m256i diff16s_4 = _mm256_sub_epi16(load256_1_4, load256_2); // convert to 8x16-bit

            __m256i sqrdiff16s_3 = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)
            __m256i sqrdiff16s_4 = _mm256_mullo_epi16(diff16s_4, diff16s_4); // 8x16-bit mul (hi zero)

            load1 = _mm256_loadu_si256((__m256i*)input);
            __m256i prod_low_3 = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_3 = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
            __m256i prod_low_4 = _mm256_mullo_epi16(sqrdiff16s_4, load256_3); // 8x16-bit mult (lo part)
            __m256i prod_high_4 = _mm256_mulhi_epu16(sqrdiff16s_4, load256_3); // 8x16-bit mul (hi part)



            __m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
            __m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
            __m256i lower_prods_4 = _mm256_unpacklo_epi16(prod_low_4, prod_high_4); // lo 4x32-bit prd
            __m256i upper_prods_4 = _mm256_unpackhi_epi16(prod_low_4, prod_high_4); // hi 4x32-bit prd

            sum_3 = _mm256_add_epi32(sum_3, lower_prods_3);
            sum_4 = _mm256_add_epi32(sum_4, lower_prods_4);
            sum_3 = _mm256_add_epi32(sum_3, upper_prods_3);
            sum_4 = _mm256_add_epi32(sum_4, upper_prods_4);


        }
        gconst64 = *consts;

        sum_1 = _mm256_hadd_epi32(sum_1, sum_2);
        sum_3 = _mm256_hadd_epi32(sum_3, sum_4);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f3 = _mm256_castsi256_si128(sum_3);
        sum_1 = _mm256_permute4x64_epi64(sum_1, 0xee);
        sum_3 = _mm256_permute4x64_epi64(sum_3, 0xee);
        __m256i gconst = _mm256_broadcastq_epi64(CVT64_128((__m128i*)&gconst64));
        __m128i sum_f2 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f4 = _mm256_castsi256_si128(sum_3);

        __m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
        sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
        __m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
        sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);

        __m256i cmpres_1 = _mm256_cmpgt_epi64(minScores, sum_64_1);
        __m256i cmpres_2 = _mm256_cmpgt_epi64(minScore2, sum_64_2);
        __m256i res_min_1 = _mm256_andnot_si256(cmpres_1, minScores);
        __m256i res_min_2 = _mm256_andnot_si256(cmpres_2, minScore2);
        __m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1);
        __m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2);
        minScores = _mm256_or_si256(res_min_1, res_sum_1);
        minScore2 = _mm256_or_si256(res_min_2, res_sum_2);

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    _mm256_storeu_si256((__m256i*)score, minScores);
    config->Output[0] = (uint32_t) score[0];
    config->Output[1] = (uint32_t) score[2];
    config->Output[2] = (uint32_t) score[1];
    config->Output[3] = (uint32_t) score[3];
    _mm256_storeu_si256((__m256i*)score, minScore2);
    config->Output[4] = (uint32_t) score[0];
    config->Output[5] = (uint32_t) score[2];
    config->Output[6] = (uint32_t) score[1];
    config->Output[7] = (uint32_t) score[3];
}

void gmm_maxmix_8u16u_32u(GmmConfig const * const config)
{
    const uint8_t *mean = config->Means;
    const uint16_t *var = config->Vars16;
    const uint32_t *consts = config->Gconst;
    uint64_t score[2];
    uint64_t sum64;
    uint32_t i, j;
    uint64_t minScore = config->MaxScore;

    for (i = 0; i < config->MixtureCount; i++)
    {
        __m256i sum_1 = _mm256_setzero_si256();
        __m256i sum_2 = _mm256_setzero_si256();
        __m128i load1 = CVT64_128((__m128i*)config->Input);
        __m128i load2 = CVT64_128((__m128i*)mean); // vector load 8x8-bit
        __m128i load3 = _mm_loadu_si128((__m128i*)var); // vector load 8x8-bit
        for (j = 8; j <= config->InputElementCount; j += 8)
        {

            __m128i load1_1 = _mm_cvtepu8_epi16(load1);
            __m128i load2_1 = _mm_cvtepu8_epi16(load2);


            __m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit
            __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
            load1 = CVT64_128((__m128i*)&config->Input[j]);
            load2 = CVT64_128((__m128i*)&mean[j]); // vector load 8x8-bit

            __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3); // 8x16-bit mult (lo part)
            __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3); // 8x16-bit mul (hi part)

            load3 = _mm_loadu_si128((__m128i*)&var[j]); // vector load 8x8-bit

            __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
            __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

            sum_1 = _mm256_add_epi64(sum_1, _mm256_cvtepu32_epi64(lower_prods));
            sum_2 = _mm256_add_epi64(sum_2, _mm256_cvtepu32_epi64(upper_prods));
        }


        sum_1 = _mm256_add_epi64(sum_1, sum_2);
        __m128i sum_f1 = _mm256_castsi256_si128(sum_1);
        __m128i sum_f2 = _mm256_extractf128_si256(sum_1, 1);
        sum_f1 = _mm_add_epi64(sum_f1, sum_f2);
        _mm_storeu_si128((__m128i*)score, sum_f1);
        sum64 = score[0] + score[1] + (uint64_t)*consts;

        minScore = sum64 < minScore ? sum64 : minScore;

        mean += config->InputElementCount;
        var += config->InputElementCount;
        consts++;
    }

    (*(config->Output)) = (uint32_t) minScore;
}

#endif //#if OPT_LEVEL > 5 // AVX2+
