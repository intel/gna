/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv.h"
#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <immintrin.h>
#include <cstdint>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t kk;
    uint32_t j;
    uint32_t k;
    int64_t sum;

    int16_t const * input;
    int16_t * feedback;
    int16_t *feedbackEnd = config->RequestConfig->Transform.feedbackBuffer+config->RequestConfig->Transform.outputElementCount;

    auto const * bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple);
    auto const * const biasEnd = bias + (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    // simd inputs and weights
    __m256i in;
    __m256i w;

    // simd accumulators
    __m256i ma;
    __m256i acc;

    // simd intermediates
    __m256i inm0;
    __m256i inm1;
    __m256i inm2;

    uint32_t KK = config->RequestConfig->Transform.inputElementCount - config->RequestConfig->Transform.inputElementCount % VEC_16CAP;
    uint32_t part_sz = config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_sz = config->RequestConfig->Transform.inputElementCount % part_sz;
    uint32_t mpart_sz = config->RequestConfig->Transform.outputElementCount < part_sz - kpart_sz ? config->RequestConfig->Transform.outputElementCount
        : part_sz - kpart_sz;
    uint32_t mm = mpart_sz - mpart_sz % VEC_16CAP;
    uint32_t MM = config->RequestConfig->Transform.outputElementCount - (config->RequestConfig->Transform.outputElementCount - mpart_sz) % VEC_16CAP;

    uint32_t kparts = config->RequestConfig->Transform.inputElementCount / part_sz;
    uint32_t mparts = (config->RequestConfig->Transform.outputElementCount - mpart_sz) / part_sz;

    acc = _mm256_setzero_si256();

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;
        sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        // compute parts using AVX
        // if config->RequestConfig->Transform.inputElementCount has modulo 16 remainder, leave it
        for (j = 0; j < kparts + 1; j++)
        {
            in = _mm256_lddqu_si256((__m256i*)input);
            w = _mm256_lddqu_si256((__m256i*)weight);
            for (k = 0; k < part_sz && (j*part_sz + k < KK); k += VEC_16CAP)
            {
                input += VEC_16CAP;
                weight += VEC_16CAP;

                ma = _mm256_madd_epi16(in, w);
                inm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ma));
                inm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ma, 1));
                inm2 = _mm256_add_epi64(inm0, inm1);

                acc = _mm256_add_epi64(acc, inm2);

                in = _mm256_lddqu_si256((__m256i*)input);
                w = _mm256_lddqu_si256((__m256i*)weight);
            }

            // saturate if part size achieved
            if (k == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm256_setzero_si256();
                saturate_store_out(&sum, output, config->SaturationCount);
                sum = (int64_t)*output;
            }
        }

        // compute remainder
        for (k = KK; k < config->RequestConfig->Transform.inputElementCount; k++)
        {
            sum += *input++ * *weight++;
        }

        in = _mm256_lddqu_si256((__m256i*)feedback);
        w = _mm256_lddqu_si256((__m256i*)weight);

        // compute using AVX instructions until additions reach part size
        // or if loop reaches end of config->RequestConfig->Transform.outputElementCount (without the modulo 16 remainder)
        for (k = 0; k < mm; k += VEC_16CAP)
        {
            feedback += VEC_16CAP;
            weight += VEC_16CAP;

            ma = _mm256_madd_epi16(in, w);
            inm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ma));
            inm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ma, 1));
            inm2 = _mm256_add_epi64(inm0, inm1);

            acc = _mm256_add_epi64(acc, inm2);

            in = _mm256_lddqu_si256((__m256i*)feedback);
            w = _mm256_lddqu_si256((__m256i*)weight);
        }

        // if part size wasn't reached, but there is still config->RequestConfig->Transform.outputElementCount remainder
        for (; k < mpart_sz; k++)
        {
            sum += *feedback++ * *weight++;
        }

        sum += vec_sum(acc);
        acc = _mm256_setzero_si256();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        for (j = 0; j < mparts + 1; j++)
        {
            for (kk = 0; kk < part_sz && (j*part_sz + mpart_sz + kk < MM); kk += VEC_16CAP)
            {
                feedback += VEC_16CAP;
                weight += VEC_16CAP;

                ma = _mm256_madd_epi16(in, w);
                inm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ma));
                inm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ma, 1));
                inm2 = _mm256_add_epi64(inm0, inm1);

                acc = _mm256_add_epi64(acc, inm2);

                in = _mm256_lddqu_si256((__m256i*)feedback);
                w = _mm256_lddqu_si256((__m256i*)weight);
            }

            if (kk == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm256_setzero_si256();
                saturate_store_out(&sum, output, config->SaturationCount);
                sum = (int64_t)*output;
            }
        }

        // if there's remainder from mparts
        for (; feedback < feedbackEnd;)
        {
            sum += *feedback++ * *weight++;
        }

        sum += vec_sum(acc);
        acc = _mm256_setzero_si256();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        output++;
    }
}
