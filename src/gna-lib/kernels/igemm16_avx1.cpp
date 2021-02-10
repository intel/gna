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

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % VEC_16CAP;
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT;
    uint32_t ix_end = KK / VEC_16CAP;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t k;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    // simd inputs
    __m256i v0;
    __m256i v1;
    __m256i v2;
    __m256i v3;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m128i s5;
    __m128i s6;

    // simd weights
    __m128i w;
    __m128i w0;
    __m128i w1;

    // simd input pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;

    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;

    int8_t const *bias;
    auto const * const biasEnd = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple) +
        (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i in7;

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        input_0 = inputs + KK;
        in_ptr0 = (__m256i*) inputs;
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc0 = _mm_add_epi32(acc0, in1);
            }

            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            for (j = 0; j < KT; j++, weight++)
            {
                *output += input_0[j] * *weight;
            }
            output++;
        }

        return;
    }

    in_ptr0 = (__m256i*)config->Intermediate->d0;
    in_ptr1 = (__m256i*)config->Intermediate->d1;
    in_ptr2 = (__m256i*)config->Intermediate->d2;
    in_ptr3 = (__m256i*)config->Intermediate->d3;

    input_0 = config->Intermediate->d0 + KK;
    input_1 = config->Intermediate->d1 + KK;
    input_2 = config->Intermediate->d2 + KK;
    input_3 = config->Intermediate->d3 + KK;

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 4:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
    case 3:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
    case 2:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);

                in2 = _mm256_extractf128_si256(v0, 1);
                in3 = _mm256_extractf128_si256(v1, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);

                in2 = _mm_madd_epi16(in2, w1);
                in3 = _mm_madd_epi16(in3, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);

                acc0 = _mm_add_epi32(acc0, in2);
                acc1 = _mm_add_epi32(acc1, in3);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);

                in3 = _mm256_extractf128_si256(v0, 1);
                in4 = _mm256_extractf128_si256(v1, 1);
                in5 = _mm256_extractf128_si256(v2, 1);

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);

                in3 = _mm_madd_epi16(in3, w1);
                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);

                acc0 = _mm_add_epi32(acc0, in3);
                acc1 = _mm_add_epi32(acc1, in4);
                acc2 = _mm_add_epi32(acc2, in5);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
                output[2] += input_2[j] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);
                in3 = _mm256_castsi256_si128(v3);

                in4 = _mm256_extractf128_si256(v0, 1);
                in5 = _mm256_extractf128_si256(v1, 1);
                in6 = _mm256_extractf128_si256(v2, 1);
                in7 = _mm256_extractf128_si256(v3, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));

                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);
                in3 = _mm_madd_epi16(in3, w0);

                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);
                in6 = _mm_madd_epi16(in6, w1);
                in7 = _mm_madd_epi16(in7, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);

                acc0 = _mm_add_epi32(acc0, in4);
                acc1 = _mm_add_epi32(acc1, in5);
                acc2 = _mm_add_epi32(acc2, in6);
                acc3 = _mm_add_epi32(acc3, in7);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
                output[2] += input_2[j] * *weight;
                output[3] += input_3[j] * *weight;
            }
            output += 4;
        }

        return;
    }

    KT = config->RequestConfig->Transform.inputElementCount % SSE_16CAP;
    KK = config->RequestConfig->Transform.inputElementCount - KT;
    ix_end = 2 * KK / VEC_16CAP;

    input_0 = config->Intermediate->d0 + 2 * KK;
    input_1 = config->Intermediate->d2 + 2 * KK;
    input_2 = config->Intermediate->d4 + 2 * KK;
    input_3 = config->Intermediate->d6 + 2 * KK;

    in_ptr0 = (__m256i*) config->Intermediate->d0;
    in_ptr1 = (__m256i*) config->Intermediate->d2;
    in_ptr2 = (__m256i*) config->Intermediate->d4;
    in_ptr3 = (__m256i*) config->Intermediate->d6;

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
            }
        }
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }

        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            input_2 = config->Intermediate->d4;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm_load_si128((__m128i*)input_2);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_2 += SSE_16CAP;
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
            }
        }

        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
            }
        }

        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }

        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            input_3 = config->Intermediate->d6;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm_load_si128((__m128i*)input_3);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_3 += SSE_16CAP;
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc6);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
                output[6] += input_3[j] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
                config->Intermediate->d6[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 6];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
                config->Intermediate->d6[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 7];
            }
        }

        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
            acc7 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm256_castsi256_si128(v3);
                in7 = _mm256_extractf128_si256(v3, 1);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);
                in7 = _mm_madd_epi16(in7, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
                acc7 = _mm_add_epi32(acc7, in7);
            }

            s1 = _mm_set1_epi32(getBias(bias, config->RequestConfig->Transform.bytesPerBias));

            s2 = _mm_hadd_epi32(acc0, acc1);
            s3 = _mm_hadd_epi32(acc2, acc3);
            s4 = _mm_hadd_epi32(s2, s3);
            s5 = _mm_add_epi32(s4, s1);

            s2 = _mm_hadd_epi32(acc4, acc5);
            s3 = _mm_hadd_epi32(acc6, acc7);
            s4 = _mm_hadd_epi32(s2, s3);
            s6 = _mm_add_epi32(s4, s1);

            v0 = _mm256_set_m128i(s6, s5);
            _mm256_stream_si256((__m256i*)output, v0);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
                output[6] += input_3[j] * *weight;
                output[7] += input_3[j + KT] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % VEC_16CAP;
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT;
    uint32_t ix_end = KK / VEC_16CAP;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t k;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int16_t const *input_0;
    int16_t const *input_1;
    int16_t const *input_2;
    int16_t const *input_3;

    int8_t const * multiBias;
    auto const * const biasEnd = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias) +
        (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.multiBiasVectorCount);
    auto biasStride = config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount;

    // simd inputs
    __m256i v0;
    __m256i v1;
    __m256i v2;
    __m256i v3;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m128i s5;
    __m128i s6;

    // simd weights
    __m128i w0;
    __m128i w1;
    __m128i w;

    // simd inputs pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i in7;

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        input_0 = inputs + KK;
        in_ptr0 = (__m256i*)inputs;
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc0 = _mm_add_epi32(acc0, in1);
            }

            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            for (j = 0; j < KT; j++, weight++)
            {
                *output += input_0[j] * *weight;
            }
            output++;
        }

        return;
    }

    in_ptr0 = (__m256i*)config->Intermediate->d0;
    in_ptr1 = (__m256i*)config->Intermediate->d1;
    in_ptr2 = (__m256i*)config->Intermediate->d2;
    in_ptr3 = (__m256i*)config->Intermediate->d3;

    input_0 = config->Intermediate->d0 + KK;
    input_1 = config->Intermediate->d1 + KK;
    input_2 = config->Intermediate->d2 + KK;
    input_3 = config->Intermediate->d3 + KK;

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 4:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
    case 3:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
    case 2:
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);

                in2 = _mm256_extractf128_si256(v0, 1);
                in3 = _mm256_extractf128_si256(v1, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);

                in2 = _mm_madd_epi16(in2, w1);
                in3 = _mm_madd_epi16(in3, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);

                acc0 = _mm_add_epi32(acc0, in2);
                acc1 = _mm_add_epi32(acc1, in3);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);

                in3 = _mm256_extractf128_si256(v0, 1);
                in4 = _mm256_extractf128_si256(v1, 1);
                in5 = _mm256_extractf128_si256(v2, 1);

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);

                in3 = _mm_madd_epi16(in3, w1);
                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);

                acc0 = _mm_add_epi32(acc0, in3);
                acc1 = _mm_add_epi32(acc1, in4);
                acc2 = _mm_add_epi32(acc2, in5);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
                output[2] += input_2[j] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);
                in3 = _mm256_castsi256_si128(v3);

                in4 = _mm256_extractf128_si256(v0, 1);
                in5 = _mm256_extractf128_si256(v1, 1);
                in6 = _mm256_extractf128_si256(v2, 1);
                in7 = _mm256_extractf128_si256(v3, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));

                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);
                in3 = _mm_madd_epi16(in3, w0);

                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);
                in6 = _mm_madd_epi16(in6, w1);
                in7 = _mm_madd_epi16(in7, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);

                acc0 = _mm_add_epi32(acc0, in4);
                acc1 = _mm_add_epi32(acc1, in5);
                acc2 = _mm_add_epi32(acc2, in6);
                acc3 = _mm_add_epi32(acc3, in7);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_1[j] * *weight;
                output[2] += input_2[j] * *weight;
                output[3] += input_3[j] * *weight;
            }
            output += 4;
        }

        return;
    }

    KT = config->RequestConfig->Transform.inputElementCount % SSE_16CAP;
    KK = config->RequestConfig->Transform.inputElementCount - KT;
    ix_end = 2 * KK / VEC_16CAP;

    input_0 = config->Intermediate->d0 + 2 * KK;
    input_1 = config->Intermediate->d2 + 2 * KK;
    input_2 = config->Intermediate->d4 + 2 * KK;
    input_3 = config->Intermediate->d6 + 2 * KK;

    in_ptr0 = (__m256i*) config->Intermediate->d0;
    in_ptr1 = (__m256i*) config->Intermediate->d2;
    in_ptr2 = (__m256i*) config->Intermediate->d4;
    in_ptr3 = (__m256i*) config->Intermediate->d6;

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
            }
        }
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }

        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            input_2 = config->Intermediate->d4;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm_load_si128((__m128i*)input_2);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_2 += SSE_16CAP;
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
            }
        }

        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
            }
        }

        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }

        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            input_3 = config->Intermediate->d6;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm_load_si128((__m128i*)input_3);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_3 += SSE_16CAP;
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc6);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
                output[6] += input_3[j] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (j = 0; j < 2 * config->RequestConfig->Transform.inputElementCount;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 2];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 4];
                config->Intermediate->d6[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 6];
            }
            for (k = i; k < i + 8 && k < config->RequestConfig->Transform.inputElementCount; k++, j++)
            {
                config->Intermediate->d0[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 1];
                config->Intermediate->d2[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 3];
                config->Intermediate->d4[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 5];
                config->Intermediate->d6[j] = inputs[k*config->RequestConfig->Transform.inputVectorCount + 7];
            }
        }

        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
            acc7 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm256_castsi256_si128(v3);
                in7 = _mm256_extractf128_si256(v3, 1);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);
                in7 = _mm_madd_epi16(in7, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
                acc7 = _mm_add_epi32(acc7, in7);
            }

            s1 = _mm_set1_epi32(getBias(multiBias, config->RequestConfig->Transform.bytesPerBias));

            s2 = _mm_hadd_epi32(acc0, acc1);
            s3 = _mm_hadd_epi32(acc2, acc3);
            s4 = _mm_hadd_epi32(s2, s3);
            s5 = _mm_add_epi32(s4, s1);

            s2 = _mm_hadd_epi32(acc4, acc5);
            s3 = _mm_hadd_epi32(acc6, acc7);
            s4 = _mm_hadd_epi32(s2, s3);
            s6 = _mm_add_epi32(s4, s1);

            v0 = _mm256_set_m128i(s6, s5);
            _mm256_stream_si256((__m256i*)output, v0);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight;
                output[1] += input_0[j + KT] * *weight;
                output[2] += input_1[j] * *weight;
                output[3] += input_1[j + KT] * *weight;
                output[4] += input_2[j] * *weight;
                output[5] += input_2[j + KT] * *weight;
                output[6] += input_3[j] * *weight;
                output[7] += input_3[j + KT] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }
}
