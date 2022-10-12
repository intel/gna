/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <immintrin.h>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t KK = config->RequestConfig.Transform.inputElementCount - config->RequestConfig.Transform.inputElementCount % VEC_16CAP;
    uint32_t part_sz = config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_sz = config->RequestConfig.Transform.inputElementCount % part_sz;
    uint32_t mpart_sz = config->RequestConfig.Transform.outputElementCount < part_sz - kpart_sz
        ? config->RequestConfig.Transform.outputElementCount
        : part_sz - kpart_sz;
    uint32_t mm = mpart_sz - mpart_sz % VEC_16CAP;
    uint32_t MM = config->RequestConfig.Transform.outputElementCount - (config->RequestConfig.Transform.outputElementCount - mpart_sz) % VEC_16CAP;
    uint32_t kparts = config->RequestConfig.Transform.inputElementCount / part_sz;
    uint32_t mparts = (config->RequestConfig.Transform.outputElementCount - mpart_sz) / part_sz;
    uint32_t kk;
    uint32_t j;
    uint32_t k;
    int64_t sum;

    int16_t const * input;
    int16_t * feedback;
    int16_t *feedbackEnd = config->RequestConfig.Transform.feedbackBuffer +
                            config->RequestConfig.Transform.outputElementCount;

    auto const *bias = (int8_t*)config->RequestConfig.Transform.biasesSimple;
    auto const * const biasEnd = bias + (config->RequestConfig.Transform.outputElementCount *
                                         config->RequestConfig.Transform.bytesPerBias);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Transform.output);
    int16_t const * weight = config->RequestConfig.Transform.weights2B;

    // simd input and weight
    __m128i in;
    __m128i w;

    // simd intermediates
    __m128i inm0;
    __m128i inm1;
    __m128i inm2;

    // simd accumulators
    __m128i ma;
    __m128i acc;

    acc = _mm_setzero_si128();

    for (; bias < biasEnd; bias += config->RequestConfig.Transform.bytesPerBias)
    {
        input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
        feedback = config->RequestConfig.Transform.feedbackBuffer;
        sum = getBias((void*)bias, config->RequestConfig.Transform.bytesPerBias);

        // compute parts using SSE
        // if config->RequestConfig.Transform.inputElementCount has modulo 16 remainder, leave it
        for (j = 0; j < kparts + 1; j++)
        {
            in = _mm_lddqu_si128((__m128i*)input);
            w = _mm_lddqu_si128((__m128i*)weight);
            for (k = 0; k < part_sz && (j*part_sz + k < KK); k += VEC_16CAP)
            {
                input += VEC_16CAP;
                weight += VEC_16CAP;

                ma = _mm_madd_epi16(in, w);
                inm0 = _mm_cvtepi32_epi64(ma);
                inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(ma, 8));
                inm2 = _mm_add_epi64(inm0, inm1);

                acc = _mm_add_epi64(acc, inm2);

                in = _mm_lddqu_si128((__m128i*)input);
                w = _mm_lddqu_si128((__m128i*)weight);
            }

            // saturate if part size achieved
            if (k == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm_setzero_si128();
                saturate_store_out(&sum, output, config->SaturationCount);
                sum = (int64_t)*output;
            }
        }

        // compute remainder
        for (k = KK; k < config->RequestConfig.Transform.inputElementCount; k++)
        {
            sum += *input++ * *weight++;
        }

        in = _mm_lddqu_si128((__m128i*)feedback);
        w = _mm_lddqu_si128((__m128i*)weight);

        // compute using SSE instructions until additions reach part size
        // or if loop reaches end of config->RequestConfig.Transform.outputElementCount (without the modulo 16 remainder)
        for (k = 0; k < mm; k += VEC_16CAP)
        {
            feedback += VEC_16CAP;
            weight += VEC_16CAP;

            in = _mm_madd_epi16(in, w);
            inm0 = _mm_cvtepi32_epi64(in);
            inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(in, 8));
            inm2 = _mm_add_epi64(inm0, inm1);

            acc = _mm_add_epi64(acc, inm2);

            in = _mm_lddqu_si128((__m128i*)feedback);
            w = _mm_lddqu_si128((__m128i*)weight);
        }

        // if part size wasn't reached, but there is still config->RequestConfig.Transform.outputElementCount remainder
        for (; k < mpart_sz; k++)
        {
            sum += *feedback++ * *weight++;
        }

        sum += vec_sum(acc);
        acc = _mm_setzero_si128();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        for (j = 0; j < mparts + 1; j++)
        {
            for (kk = 0; kk < part_sz && (j*part_sz + mpart_sz + kk < MM); kk += VEC_16CAP)
            {
                feedback += VEC_16CAP;
                weight += VEC_16CAP;

                in = _mm_madd_epi16(in, w);
                inm0 = _mm_cvtepi32_epi64(in);
                inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(in, 8));
                inm2 = _mm_add_epi64(inm0, inm1);

                acc = _mm_add_epi64(acc, inm2);

                in = _mm_lddqu_si128((__m128i*)feedback);
                w = _mm_lddqu_si128((__m128i*)weight);
            }

            if (kk == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm_setzero_si128();
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
        acc = _mm_setzero_si128();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        output++;
    }
}
