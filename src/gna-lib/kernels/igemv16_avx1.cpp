/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <cstdint>
#include <immintrin.h>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t LDA = config->RequestConfig->Transform.outputElementCount + config->RequestConfig->Transform.inputElementCount;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t * feedback = config->RequestConfig->Transform.feedbackBuffer;

    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount - config->RequestConfig->Transform.inputElementCount % 16;
    int16_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount - config->RequestConfig->Transform.outputElementCount % 16;

    auto const * bias = reinterpret_cast<uint8_t const *>(config->RequestConfig->Transform.biasesSimple);
    auto const * const biasEnd = reinterpret_cast<uint8_t const *>(bias) +
        (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int16_t const * weight2 = weight + config->RequestConfig->Transform.inputElementCount;

    __m256i v0;
    __m128i s0;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m128i s5;

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        v0 = _mm256_lddqu_si256((__m256i*)input);

        s0 = _mm256_castsi256_si128(v0);
        s1 = _mm256_extractf128_si256(v0, 1);

        s2 = _mm_lddqu_si128((__m128i*)weight);
        s3 = _mm_lddqu_si128((__m128i*)(weight + 8));

        s5 = _mm_setzero_si128();

        while (input < inputEnd)
        {
            input += 16;
            weight += 16;

            s0 = _mm_madd_epi16(s0, s2);
            s1 = _mm_madd_epi16(s1, s3);

            s4 = _mm_add_epi32(s0, s1);
            s5 = _mm_add_epi32(s4, s5);

            v0 = _mm256_lddqu_si256((__m256i*)input);

            s0 = _mm256_castsi256_si128(v0);
            s1 = _mm256_extractf128_si256(v0, 1);

            s2 = _mm_lddqu_si128((__m128i*)weight);
            s3 = _mm_lddqu_si128((__m128i*)(weight + 8));
        }

        v0 = _mm256_lddqu_si256((__m256i*)feedback);
        s0 = _mm256_castsi256_si128(v0);
        s1 = _mm256_extractf128_si256(v0, 1);
        s2 = _mm_lddqu_si128((__m128i*)weight2);
        s3 = _mm_lddqu_si128((__m128i*)(weight2 + 8));

        while (feedback < feedbackEnd)
        {
            feedback += 16;
            weight2 += 16;

            s0 = _mm_madd_epi16(s0, s2);
            s1 = _mm_madd_epi16(s1, s3);

            s4 = _mm_add_epi32(s0, s1);
            s5 = _mm_add_epi32(s4, s5);

            v0 = _mm256_lddqu_si256((__m256i*)feedback);
            s0 = _mm256_castsi256_si128(v0);
            s1 = _mm256_extractf128_si256(v0, 1);
            s2 = _mm_lddqu_si128((__m128i*)weight2);
            s3 = _mm_lddqu_si128((__m128i*)(weight2 + 8));
        }

        *output = vec_sum(s5) + getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        while (input < inputEnd + config->RequestConfig->Transform.inputElementCount % 16)
        {
            *output += *input++ * *weight++;
        }

        while (feedback < feedbackEnd + config->RequestConfig->Transform.outputElementCount % 16)
        {
            *output += *feedback++ * *weight2++;
        }

        output++;

        weight += LDA - config->RequestConfig->Transform.inputElementCount;
        weight2 += LDA - config->RequestConfig->Transform.outputElementCount;
    }
}
