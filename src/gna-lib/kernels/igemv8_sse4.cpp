/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <cstdint>
#include <immintrin.h>

void RecurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t LDA = config->RequestConfig->Transform.outputElementCount + config->RequestConfig->Transform.inputElementCount;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t * feedback = config->RequestConfig->Transform.feedbackBuffer;

    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount - config->RequestConfig->Transform.inputElementCount % 8;
    int16_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount - config->RequestConfig->Transform.outputElementCount % 8;

    nn_bias_c const * bias = config->RequestConfig->Transform.biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->RequestConfig->Transform.outputElementCount;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    int8_t const * weight = config->RequestConfig->Transform.weights1B;
    int8_t const * weight2 = weight + config->RequestConfig->Transform.inputElementCount;

    __m128i v0;
    __m128i v1;
    __m128i v2;

    for (; bias < biasEnd; bias++)
    {
        v2 = _mm_setzero_si128();

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        v0 = _mm_lddqu_si128((__m128i*)input);
        v1 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));

        while (input < inputEnd)
        {
            input += 8;
            weight += 8;

            v1 = _mm_madd_epi16(v0, v1);
            v2 = _mm_add_epi32(v1, v2);

            v0 = _mm_lddqu_si128((__m128i*)input);
            v1 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
        }

        v0 = _mm_lddqu_si128((__m128i*)feedback);
        v1 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));

        while (feedback < feedbackEnd)
        {
            feedback += 8;
            weight2 += 8;

            v1 = _mm_madd_epi16(v0, v1);
            v2 = _mm_add_epi32(v1, v2);

            v0 = _mm_lddqu_si128((__m128i*)feedback);
            v1 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));
        }

        *output = vec_sum(v2);

        while (input < inputEnd + config->RequestConfig->Transform.inputElementCount % 8)
        {
            *output += *input++ * *weight++;
        }

        while (feedback < feedbackEnd + config->RequestConfig->Transform.outputElementCount % 8)
        {
            *output += *feedback++ * *weight2++;
        }

        *output = *output * bias->multiplier + bias->bias;
        output++;

        weight += LDA - config->RequestConfig->Transform.inputElementCount;
        weight2 += LDA - config->RequestConfig->Transform.outputElementCount;
    }
}
