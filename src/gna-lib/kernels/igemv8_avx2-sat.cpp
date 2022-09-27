/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <immintrin.h>
#include <cstdint>

inline void VectorMadd(__m256i &acc, const int16_t **input, const int8_t **weight)
{
    auto in = _mm256_lddqu_si256((__m256i *)*input);
    auto w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)*weight));

    auto ma = _mm256_madd_epi16(in, w);
    auto inm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ma));
    auto inm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ma, 1));
    auto inm2 = _mm256_add_epi64(inm0, inm1);

    acc = _mm256_add_epi64(acc, inm2);

    *input += VEC_16CAP;
    *weight += VEC_16CAP;
}

void RecurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum;

    int16_t const *input;
    int16_t const *feedback;

    BiasCompound const * bias = config->RequestConfig.Transform.biasesCompound;
    BiasCompound const * const biasEnd = bias + config->RequestConfig.Transform.outputElementCount;
    int32_t *output = reinterpret_cast<int32_t *>(config->RequestConfig.Transform.output);
    int8_t const *weight = config->RequestConfig.Transform.weights1B;

    uint32_t kparts = config->RequestConfig.Transform.inputElementCount / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_rem = config->RequestConfig.Transform.inputElementCount % config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t middle_fill = config->BufferElementCount[0 + XNN_N_GROUP_MAX] - kpart_rem;
    uint32_t middle_part = (config->RequestConfig.Transform.outputElementCount < middle_fill) ? config->RequestConfig.Transform.outputElementCount : middle_fill;
    uint32_t mm = config->RequestConfig.Transform.outputElementCount - middle_part;
    uint32_t mparts = mm / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t mpart_rem = mm % config->BufferElementCount[0 + XNN_N_GROUP_MAX];

    uint32_t mpart_vec_rem = mpart_rem % VEC_16CAP;
    uint32_t kpart_vec_rem = kpart_rem % VEC_16CAP;
    uint32_t middle_part_vec_rem = middle_part % VEC_16CAP;

    __m256i acc = _mm256_setzero_si256();

    for (; bias < biasEnd; bias++)
    {
        input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
        feedback = config->RequestConfig.Transform.feedbackBuffer;
        sum = bias->Bias;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j+=VEC_16CAP)
            {
                VectorMadd(acc, &input, &weight);
            }

            sum += vec_sum(acc) * bias->Multiplier;
            acc = _mm256_setzero_si256();
            saturate_store_out(&sum, output, config->SaturationCount);
            sum = (int64_t)*output;
        }

        for (i = 0; i < kpart_rem - kpart_vec_rem; i+=VEC_16CAP)
        {
            VectorMadd(acc, &input, &weight);
        }

        for (i = 0; i < kpart_vec_rem; i++)
        {
            sum += *input++ * *weight++ * bias->Multiplier;
        }

        for (i = 0; i < middle_part - middle_part_vec_rem; i+=VEC_16CAP)
        {
            VectorMadd(acc, &feedback, &weight);
        }

        for (i = 0; i < middle_part_vec_rem; i++)
        {
            sum += *feedback++ * *weight++ * bias->Multiplier;
        }

        sum += vec_sum(acc) * bias->Multiplier;
        acc = _mm256_setzero_si256();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j += VEC_16CAP)
            {
                VectorMadd(acc, &feedback, &weight);
            }

            sum += vec_sum(acc) * bias->Multiplier;
            acc = _mm256_setzero_si256();
            saturate_store_out(&sum, output, config->SaturationCount);
            sum = (int64_t)*output;
        }

        for (i = 0; i < mpart_rem - mpart_vec_rem; i+=VEC_16CAP)
        {
            VectorMadd(acc, &feedback, &weight);
        }

        for (i = 0; i < mpart_vec_rem; i++)
        {
            sum += *feedback++ * *weight++ * bias->Multiplier;
        }

        sum += vec_sum(acc) * bias->Multiplier;
        acc = _mm256_setzero_si256();
        saturate_store_out(&sum, output, config->SaturationCount);
        sum = (int64_t)*output;

        output++;
    }
}
