/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <immintrin.h>
#include <cstdint>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t LDA = config->RequestConfig->Transform.outputElementCount + config->RequestConfig->Transform.inputElementCount;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t * feedback = config->RequestConfig->Transform.feedbackBuffer;

    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount - config->RequestConfig->Transform.inputElementCount % 16;
    int16_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount - config->RequestConfig->Transform.outputElementCount % 16;

    auto const * bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple);
    auto const * const biasEnd = bias + (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int16_t const * weight2 = weight + config->RequestConfig->Transform.inputElementCount;

    __m256i v0;
    __m256i v1;
    __m256i v2;

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        v2 = _mm256_setzero_si256();

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        v0 = _mm256_lddqu_si256((__m256i*)input);
        v1 = _mm256_lddqu_si256((__m256i*)weight);

        while (input < inputEnd)
        {
            input += 16;
            weight += 16;

            v1 = _mm256_madd_epi16(v0, v1);
            v2 = _mm256_add_epi32(v1, v2);

            v0 = _mm256_lddqu_si256((__m256i*)input);
            v1 = _mm256_lddqu_si256((__m256i*)weight);
        }

        v0 = _mm256_lddqu_si256((__m256i*)feedback);
        v1 = _mm256_lddqu_si256((__m256i*)weight2);

        while (feedback < feedbackEnd)
        {
            feedback += 16;
            weight2 += 16;

            v1 = _mm256_madd_epi16(v0, v1);
            v2 = _mm256_add_epi32(v1, v2);

            v0 = _mm256_lddqu_si256((__m256i*)feedback);
            v1 = _mm256_lddqu_si256((__m256i*)weight2);
        }

        *output = vec_sum(v2) + getBias(bias, config->RequestConfig->Transform.bytesPerBias);

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
