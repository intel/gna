/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv.h"
#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

void AffineActiveListKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % VEC_16CAP; // config->RequestConfig->Transform.inputElementCount tail for manual processing
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT; // trimmed config->RequestConfig->Transform.inputElementCount for AVX2 processing
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    kpartial = (config->BufferElementCount[config->RequestConfig->Transform.inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->RequestConfig->Transform.inputVectorCount;
    nKpartial = config->RequestConfig->Transform.inputElementCount / kpartial;

    // simd input pointers
    __m256i *in_ptr = nullptr;
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    // simd inputs
    __m256i in0;
    __m256i in1;
    __m256i in2;
    __m256i in3;
    __m256i in4;
    __m256i in5;
    __m256i in6;
    __m256i in7;

    // simd accumulators
    __m256i acc0;
    __m256i acc1;
    __m256i acc2;
    __m256i acc3;
    __m256i acc4;
    __m256i acc5;
    __m256i acc6;
    __m256i acc7;

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;
    int64_t sum7;

    // simd accumulators
    __m256i acc[8];

    // simd inputs and weights
    __m256i in[8];
    __m256i w0;
    __m256i w1;
    __m256i w;

    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t const * input0;
    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        input0 = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs) + KK;
        in_ptr = (__m256i*)config->RequestConfig->Inputs;
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            ix = 0;
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            sum0 = bias->bias;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                acc0 = _mm256_add_epi32(acc0, acc1);
                sum0 += vec_sum32(acc0) * bias->multiplier;

                saturate_store_out(&sum0, output, config->SaturationCount);
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12288
                // 12288 / 16 = 768
                // 12888 / 256 = 48
                // so, max number of loops is 3
                acc_iters = niters / (VEC_16CAP * 256);
                rem_iters = niters % (VEC_16CAP * 256);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm256_setzero_si256();
                    acc1 = _mm256_setzero_si256();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix += 2)
                    {
                        in0 = _mm256_load_si256(in_ptr + ix);
                        in1 = _mm256_load_si256(in_ptr + ix + 1);

                        w0 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                        w1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight + VEC_16CAP)));

                        weight += 32;

                        // multiply and add - won't saturate
                        in0 = _mm256_madd_epi16(in0, w0);
                        in1 = _mm256_madd_epi16(in1, w1);

                        acc0 = _mm256_add_epi32(acc0, in0);
                        acc1 = _mm256_add_epi32(acc1, in1);

                        // load next vectors
                    }

                    acc0 = _mm256_add_epi32(acc0, acc1);
                    sum0 += vec_sum32(acc0) * bias->multiplier;
                }

                acc0 = _mm256_setzero_si256();
                acc1 = _mm256_setzero_si256();

                ix_end = ix + rem_iters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in0 = _mm256_load_si256(in_ptr + ix);
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));

                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm256_madd_epi16(in0, w);
                    acc0 = _mm256_add_epi32(acc0, in0);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                acc0 = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += input0[j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, output, config->SaturationCount);

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
        input[7] = config->Intermediate->d7 + KK;
        in_ptr7 = (__m256i*)config->Intermediate->d7;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        input[6] = config->Intermediate->d6 + KK;
        in_ptr6 = (__m256i*)config->Intermediate->d6;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        input[5] = config->Intermediate->d5 + KK;
        in_ptr5 = (__m256i*)config->Intermediate->d5;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        input[4] = config->Intermediate->d4 + KK;
        in_ptr4 = (__m256i*)config->Intermediate->d4;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        input[3] = config->Intermediate->d3 + KK;
        in_ptr3 = (__m256i*)config->Intermediate->d3;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        input[2] = config->Intermediate->d2 + KK;
        in_ptr2 = (__m256i*)config->Intermediate->d2;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        input[1] = config->Intermediate->d1 + KK;
        in_ptr1 = (__m256i*)config->Intermediate->d1;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        input[0] = config->Intermediate->d0 + KK;
        in_ptr0 = (__m256i*)config->Intermediate->d0;
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 6144
                // 6144 / 16 = 384
                // 6144 / 256 = 24
                // so, max number of loops is 1

                acc_iters = niters / (VEC_16CAP * 256);
                rem_iters = niters % (VEC_16CAP * 256);

                if (acc_iters == 1)
                {
                    acc[0] = _mm256_setzero_si256();
                    acc[1] = _mm256_setzero_si256();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                        weight += VEC_16CAP;

                        in[0] = _mm256_load_si256(in_ptr0 + ix);
                        in[1] = _mm256_load_si256(in_ptr1 + ix);

                        // multiply and add - won't saturate
                        in[0] = _mm256_madd_epi16(in[0], w);
                        in[1] = _mm256_madd_epi16(in[1], w);

                        acc[0] = _mm256_add_epi32(acc[0], in[0]);
                        acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    }

                    sum[0] += vec_sum32(acc[0]) * bias->multiplier;
                    sum[1] += vec_sum32(acc[1]) * bias->multiplier;
                }

                acc[0] = _mm256_setzero_si256();
                acc[1] = _mm256_setzero_si256();

                ix_end = ix + rem_iters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                }

                sum[0] += vec_sum32(acc[0]) * bias->multiplier;
                sum[1] += vec_sum32(acc[1]) * bias->multiplier;

                acc[0] = _mm256_setzero_si256();
                acc[1] = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += input[0][j] * *weight * bias->multiplier;
                sum[1] += input[1][j] * *weight * bias->multiplier;
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12096 / 3 = 4032
                // 4032 / 16 = 252
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                }

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                // kpartial = 12288 / 4 = 3072
                // 3072 / 16 = 192
                // accumulator will not saturate

                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                }

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12000 / 5 = 2400
                // 2400 / 16 = 150
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                }

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
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
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12288 / 6 = 2048
                // 2048 / 16 = 128
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                    acc[5] = _mm256_add_epi32(acc[5], in[5]);
                }

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
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
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate_store_out(&sum[i], &output[i], config->SaturationCount);
                    sum[i] = output[i];
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                // kpartial = 12288 / 7 = 1755
                // kpartial / 16 = 109
                // accumulator will not saturate

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

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                    acc[5] = _mm256_add_epi32(acc[5], in[5]);
                    acc[6] = _mm256_add_epi32(acc[6], in[6]);
                }

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
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
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;
            sum5 = bias->bias;
            sum6 = bias->bias;
            sum7 = bias->bias;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, config->SaturationCount);
                saturate(&sum1, config->SaturationCount);
                saturate(&sum2, config->SaturationCount);
                saturate(&sum3, config->SaturationCount);
                saturate(&sum4, config->SaturationCount);
                saturate(&sum5, config->SaturationCount);
                saturate(&sum6, config->SaturationCount);
                saturate(&sum7, config->SaturationCount);

                // kpartial = 12288 / 8 = 1536
                // 1536 / 16 = 96
                // accumulator will not saturate
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm256_load_si256(in_ptr0 + ix);
                    in1 = _mm256_load_si256(in_ptr1 + ix);
                    in2 = _mm256_load_si256(in_ptr2 + ix);
                    in3 = _mm256_load_si256(in_ptr3 + ix);
                    in4 = _mm256_load_si256(in_ptr4 + ix);
                    in5 = _mm256_load_si256(in_ptr5 + ix);
                    in6 = _mm256_load_si256(in_ptr6 + ix);
                    in7 = _mm256_load_si256(in_ptr7 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm256_madd_epi16(in0, w);
                    in1 = _mm256_madd_epi16(in1, w);
                    in2 = _mm256_madd_epi16(in2, w);
                    in3 = _mm256_madd_epi16(in3, w);
                    in4 = _mm256_madd_epi16(in4, w);
                    in5 = _mm256_madd_epi16(in5, w);
                    in6 = _mm256_madd_epi16(in6, w);
                    in7 = _mm256_madd_epi16(in7, w);

                    acc0 = _mm256_add_epi32(acc0, in0);
                    acc1 = _mm256_add_epi32(acc1, in1);
                    acc2 = _mm256_add_epi32(acc2, in2);
                    acc3 = _mm256_add_epi32(acc3, in3);
                    acc4 = _mm256_add_epi32(acc4, in4);
                    acc5 = _mm256_add_epi32(acc5, in5);
                    acc6 = _mm256_add_epi32(acc6, in6);
                    acc7 = _mm256_add_epi32(acc7, in7);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
                sum5 += vec_sum32(acc5) * bias->multiplier;
                sum6 += vec_sum32(acc6) * bias->multiplier;
                sum7 += vec_sum32(acc7) * bias->multiplier;

                acc0 = _mm256_setzero_si256();
                acc1 = _mm256_setzero_si256();
                acc2 = _mm256_setzero_si256();
                acc3 = _mm256_setzero_si256();
                acc4 = _mm256_setzero_si256();
                acc5 = _mm256_setzero_si256();
                acc6 = _mm256_setzero_si256();
                acc7 = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
                sum4 += input[4][j] * *weight * bias->multiplier;
                sum5 += input[5][j] * *weight * bias->multiplier;
                sum6 += input[6][j] * *weight * bias->multiplier;
                sum7 += input[7][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->SaturationCount);
            saturate_store_out(&sum4, &output[4], config->SaturationCount);
            saturate_store_out(&sum5, &output[5], config->SaturationCount);
            saturate_store_out(&sum6, &output[6], config->SaturationCount);
            saturate_store_out(&sum7, &output[7], config->SaturationCount);

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }
}
