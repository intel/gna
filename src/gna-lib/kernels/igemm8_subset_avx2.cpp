/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv.h"
#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

void AffineActiveListKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % VEC_16CAP;
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT;
    uint32_t ix_end = KK / VEC_16CAP;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int8_t const * weight;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    nn_bias_c const * bias;
    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;
    int16_t const *input_4 = nullptr;
    int16_t const *input_5 = nullptr;
    int16_t const *input_6 = nullptr;
    int16_t const *input_7 = nullptr;

    // simd input pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m256i in0;
    __m256i in1;
    __m256i in2;
    __m256i in3;
    __m256i in4;
    __m256i in5;
    __m256i in6;
    __m256i in7;
    __m256i v0;
    __m256i v1;
    __m256i v2;
    __m256i v3;
    __m256i v4;
    __m256i v5;
    __m256i v6;
    __m256i v7;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m256i w;

    // simd accumulators
    __m256i acc0;
    __m256i acc1;
    __m256i acc2;
    __m256i acc3;
    __m256i acc4;
    __m256i acc5;
    __m256i acc6;
    __m256i acc7;

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        input_0 = inputs + KK;
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            in_ptr0 = (__m256i*)config->RequestConfig->Inputs;
            acc0 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                acc0 = _mm256_add_epi32(acc0, in0);
            }

            *output = bias->bias + vec_sum(acc0) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                *output += input_0[j] * *weight * bias->multiplier;
            }

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
        input_7 = config->Intermediate->d7 + KK;
        in_ptr7 = (__m256i*)config->Intermediate->d7;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        input_6 = config->Intermediate->d6 + KK;
        in_ptr6 = (__m256i*)config->Intermediate->d6;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        input_5 = config->Intermediate->d5 + KK;
        in_ptr5 = (__m256i*)config->Intermediate->d5;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        input_4 = config->Intermediate->d4 + KK;
        in_ptr4 = (__m256i*)config->Intermediate->d4;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        input_3 = config->Intermediate->d3 + KK;
        in_ptr3 = (__m256i*)config->Intermediate->d3;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        input_2 = config->Intermediate->d2 + KK;
        in_ptr2 = (__m256i*)config->Intermediate->d2;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        input_1 = config->Intermediate->d1 + KK;
        in_ptr1 = (__m256i*)config->Intermediate->d1;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        input_0 = config->Intermediate->d0 + KK;
        in_ptr0 = (__m256i*)config->Intermediate->d0;
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;
            output[5] = bias->bias + vec_sum(acc5) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);
                in6 = _mm256_load_si256(in_ptr6 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);
                in6 = _mm256_madd_epi16(in6, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
                acc6 = _mm256_add_epi32(acc6, in6);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;
            output[5] = bias->bias + vec_sum(acc5) * bias->multiplier;
            output[6] = bias->bias + vec_sum(acc6) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
                output[6] += input_6[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (l = 0; l < al.count; l++)
        {
            i = al.indices[l];
            weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
            bias = config->RequestConfig->Transform.biasesCompound+i;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
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

            v0 = _mm256_set1_epi32(bias->multiplier);
            v1 = _mm256_set1_epi32(bias->bias);

            v2 = _mm256_hadd_epi32(acc0, acc1);
            v3 = _mm256_hadd_epi32(acc2, acc3);
            v4 = _mm256_hadd_epi32(v2, v3);

            v5 = _mm256_hadd_epi32(acc4, acc5);
            v6 = _mm256_hadd_epi32(acc6, acc7);
            v7 = _mm256_hadd_epi32(v5, v6);

            s1 = _mm256_castsi256_si128(v4);
            s2 = _mm256_extracti128_si256(v4, 1);

            s3 = _mm256_castsi256_si128(v7);
            s4 = _mm256_extracti128_si256(v7, 1);

            s1 = _mm_add_epi32(s1, s2);
            s2 = _mm_add_epi32(s3, s4);

            acc0 = _mm256_set_m128i(s2, s1);
            acc0 = _mm256_mullo_epi32(acc0, v0);
            acc0 = _mm256_add_epi32(acc0, v1);
            _mm256_store_si256((__m256i*)output, acc0);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
                output[6] += input_6[j] * *weight * bias->multiplier;
                output[7] += input_7[j] * *weight * bias->multiplier;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
        return;
    }
}
