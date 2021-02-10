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

#include <cstdint>
#include <cstring>
#include <immintrin.h>

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % VEC_16CAP; // config->RequestConfig->Transform.inputElementCount tail for manual processing
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT; // trimmed config->RequestConfig->Transform.inputElementCount for AVX2 processing
    uint32_t kpartial;
    uint32_t nKpartial;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    kpartial = (config->BufferElementCount[config->RequestConfig->Transform.inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->RequestConfig->Transform.inputVectorCount;
    nKpartial = config->RequestConfig->Transform.inputElementCount / kpartial;

    // simd inputs and weight
    __m256i in[8];
    __m256i w;

    // simd accumulators
    __m256i acc[8];

    // simd intermediate vectors
    __m256i imm0;
    __m256i imm1;
    __m256i imm2;
    __m256i imm3;
    __m256i imm4;
    __m256i imm5;
    __m256i imm6;
    __m256i imm7;
    __m256i imm8;
    __m256i imm9;

    // simd input pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));
    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    int16_t const * weight;
    int32_t * output;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    auto const * bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple);
    auto const * const biasEnd  = bias + (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    weight = config->RequestConfig->Transform.weights2B;

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        in_ptr0 = (__m256i*)config->RequestConfig->Inputs;
        *input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs) + KK;
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            *acc = _mm256_setzero_si256();
            *sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(sum, config->SaturationCount);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    *in = _mm256_load_si256(in_ptr0 + ix);
                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    *in = _mm256_madd_epi16(*in, w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(*in));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*in, 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    *acc = _mm256_add_epi64(*acc, imm2);
                }

                *sum += vec_sum(acc[0]);
                *acc = _mm256_setzero_si256();
            }

            *sum += vec_sum(acc[0]);

            for (j = 0; j < KT; j++, weight++)
            {
                *sum += (*input)[j] * *weight;
            }

            saturate_store_out(sum, output, config->SaturationCount);

            output++;
        }
        return;
    }

    if (config->RequestConfig->Transform.inputVectorCount == 8)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d7[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 7];
        }
        in_ptr7 = (__m256i*)config->Intermediate->d7;
        input[7] = config->Intermediate->d7 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        in_ptr6 = (__m256i*)config->Intermediate->d6;
        input[6] = config->Intermediate->d6 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        in_ptr5 = (__m256i*)config->Intermediate->d5;
        input[5] = config->Intermediate->d5 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        in_ptr4 = (__m256i*)config->Intermediate->d4;
        input[4] = config->Intermediate->d4 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        in_ptr3 = (__m256i*)config->Intermediate->d3;
        input[3] = config->Intermediate->d3 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        in_ptr2 = (__m256i*)config->Intermediate->d2;
        input[2] = config->Intermediate->d2 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        in_ptr1 = (__m256i*)config->Intermediate->d1;
        input[1] = config->Intermediate->d1 + KK;

        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        in_ptr0 = (__m256i*)config->Intermediate->d0;
        input[0] = config->Intermediate->d0 + KK;
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();

            sum[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            sum[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);

                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();

            sum[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            sum[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            sum[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->SaturationCount);

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);
            sum[3] += vec_sum(acc[3]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
                sum[3] += input[3][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->SaturationCount);
            saturate_store_out(&sum[3], &output[3], config->SaturationCount);

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();
            acc[3] = _mm256_setzero_si256();
            acc[4] = _mm256_setzero_si256();
            acc[5] = _mm256_setzero_si256();
            acc[6] = _mm256_setzero_si256();
            acc[7] = _mm256_setzero_si256();

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);
                    in[7] = _mm256_load_si256(in_ptr7 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);
                    in[7] = _mm256_madd_epi16(in[7], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[7]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[7], 1));
                    imm9 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                    acc[7] = _mm256_add_epi64(acc[7], imm9);
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], output++, config->SaturationCount);
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    const auto numberOfInputVectorElements = config->RequestConfig->Transform.inputElementCount;
    const auto numberOfInputVectors = config->RequestConfig->Transform.inputVectorCount;  // aka batching
    const auto numberOfBytesPerBias = config->RequestConfig->Transform.bytesPerBias;

    uint32_t KT = numberOfInputVectorElements % VEC_16CAP; // numberOfInputVectorElements tail for manual processing
    uint32_t KK = numberOfInputVectorElements - KT; // trimmed numberOfInputVectorElements for AVX2 processing
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;

    kpartial = (config->BufferElementCount[numberOfInputVectors - 1 + XNN_N_GROUP_MAX]) / numberOfInputVectors;
    nKpartial = numberOfInputVectorElements / kpartial;

    // simd inputs & weight
    __m256i in[8];
    __m256i w;

    // simd accumulators
    __m256i acc[8];

    // simd intermediate
    __m256i imm0;
    __m256i imm1;
    __m256i imm2;
    __m256i imm3;
    __m256i imm4;
    __m256i imm5;
    __m256i imm6;
    __m256i imm7;
    __m256i imm8;
    __m256i imm9;

    // simd input pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t const * weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias);

    auto biasStride = numberOfBytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount;
    auto const * const biasEnd = multiBias + biasStride * config->RequestConfig->Transform.outputElementCount;

    weight = config->RequestConfig->Transform.weights2B;
    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    if (1 == numberOfInputVectors)
    {
        in_ptr0 = (__m256i*)inputs;
        *input = inputs + KK;
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            *acc = _mm256_setzero_si256();
            *sum = getBias(multiBias, numberOfBytesPerBias);
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(sum, config->SaturationCount);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    *in = _mm256_load_si256(in_ptr0 + ix);
                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    *in = _mm256_madd_epi16(*in, w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(*in));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*in, 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    *acc = _mm256_add_epi64(*acc, imm2);
                }

                *sum += vec_sum(acc[0]);
                *acc = _mm256_setzero_si256();
            }

            *sum += vec_sum(acc[0]);

            for (j = 0; j < KT; j++, weight++)
            {
                *sum += (*input)[j] * *weight;
            }

            saturate_store_out(sum, output, config->SaturationCount);

            output++;
        }
        return;
    }

    if (numberOfInputVectors == 8)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d7[i] = inputs[i*numberOfInputVectors + 7];
        }
        in_ptr7 = (__m256i*)config->Intermediate->d7;
        input[7] = config->Intermediate->d7 + KK;
    }
    if (numberOfInputVectors >= 7)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d6[i] = inputs[i*numberOfInputVectors + 6];
        }
        in_ptr6 = (__m256i*)config->Intermediate->d6;
        input[6] = config->Intermediate->d6 + KK;
    }
    if (numberOfInputVectors >= 6)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d5[i] = inputs[i*numberOfInputVectors + 5];
        }
        in_ptr5 = (__m256i*)config->Intermediate->d5;
        input[5] = config->Intermediate->d5 + KK;
    }
    if (numberOfInputVectors >= 5)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d4[i] = inputs[i*numberOfInputVectors + 4];
        }
        in_ptr4 = (__m256i*)config->Intermediate->d4;
        input[4] = config->Intermediate->d4 + KK;
    }
    if (numberOfInputVectors >= 4)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d3[i] = inputs[i*numberOfInputVectors + 3];
        }
        in_ptr3 = (__m256i*)config->Intermediate->d3;
        input[3] = config->Intermediate->d3 + KK;
    }
    if (numberOfInputVectors >= 3)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d2[i] = inputs[i*numberOfInputVectors + 2];
        }
        in_ptr2 = (__m256i*)config->Intermediate->d2;
        input[2] = config->Intermediate->d2 + KK;
    }
    if (numberOfInputVectors >= 2)
    {
        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d1[i] = inputs[i*numberOfInputVectors + 1];
        }
        in_ptr1 = (__m256i*)config->Intermediate->d1;
        input[1] = config->Intermediate->d1 + KK;

        for (i = 0; i < numberOfInputVectorElements; i++)
        {
            config->Intermediate->d0[i] = inputs[i*numberOfInputVectors];
        }
        in_ptr0 = (__m256i*)config->Intermediate->d0;
        input[0] = config->Intermediate->d0 + KK;
    }

    if (2 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();

            sum[0] = getBias(multiBias, numberOfBytesPerBias);
            sum[1] = getBias(multiBias, numberOfBytesPerBias);

            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);

                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);

            output += numberOfInputVectors;
        }
    }

    if (3 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();

            sum[0] = getBias(multiBias, numberOfBytesPerBias);
            sum[1] = getBias(multiBias, numberOfBytesPerBias);
            sum[2] = getBias(multiBias, numberOfBytesPerBias);
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
            }
            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->SaturationCount);

            output += numberOfInputVectors;
        }
    }

    if (4 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < numberOfInputVectors; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(multiBias, numberOfBytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);
            sum[3] += vec_sum(acc[3]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
                sum[3] += input[3][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->SaturationCount);
            saturate_store_out(&sum[3], &output[3], config->SaturationCount);

            output += numberOfInputVectors;
        }
    }

    if (5 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < numberOfInputVectors; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(multiBias, numberOfBytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += numberOfInputVectors;
        }
    }

    if (6 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < numberOfInputVectors; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(multiBias, numberOfBytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += numberOfInputVectors;
        }
    }

    if (7 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < numberOfInputVectors; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = getBias(multiBias, numberOfBytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += numberOfInputVectors;
        }
    }

    if (8 == numberOfInputVectors)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();
            acc[3] = _mm256_setzero_si256();
            acc[4] = _mm256_setzero_si256();
            acc[5] = _mm256_setzero_si256();
            acc[6] = _mm256_setzero_si256();
            acc[7] = _mm256_setzero_si256();

            for (i = 0; i < numberOfInputVectors; i++)
            {
                sum[i] = getBias(multiBias, numberOfBytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->SaturationCount);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);
                    in[7] = _mm256_load_si256(in_ptr7 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);
                    in[7] = _mm256_madd_epi16(in[7], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[7]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[7], 1));
                    imm9 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                    acc[7] = _mm256_add_epi64(acc[7], imm9);
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < numberOfInputVectors; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < numberOfInputVectors; i++)
            {
                saturate_store_out(&sum[i], output++, config->SaturationCount);
            }
        }
    }
}
