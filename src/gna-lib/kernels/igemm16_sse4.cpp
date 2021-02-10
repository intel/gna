/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <immintrin.h>

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % SSE_16CAP;
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;
    int16_t const *input_4 = nullptr;
    int16_t const *input_5 = nullptr;
    int16_t const *input_6 = nullptr;
    int16_t const *input_7 = nullptr;

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i w;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    int8_t const *bias;
    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    auto const * const biasEnd = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple) +
        (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        in_ptr0 = (__m128i*)inputs;
        input_0 = inputs + KK;
        ix_end = KK / SSE_16CAP;
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
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
        in_ptr7 = (__m128i*)config->Intermediate->d7;
        input_7 = config->Intermediate->d7 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->Intermediate->d6;
        input_6 = config->Intermediate->d6 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->Intermediate->d5;
        input_5 = config->Intermediate->d5 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->Intermediate->d4;
        input_4 = config->Intermediate->d4 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->Intermediate->d3;
        input_3 = config->Intermediate->d3 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->Intermediate->d2;
        input_2 = config->Intermediate->d2 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->Intermediate->d1;
        input_1 = config->Intermediate->d1 + KK;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->Intermediate->d0;
        input_0 = config->Intermediate->d0 + KK;
    }
    ix_end = KK / SSE_16CAP;

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);;
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);;

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);;
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);;
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);;


            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
            }

            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);

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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple); bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
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
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_load_si128(in_ptr0 + ix), w));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_load_si128(in_ptr1 + ix), w));
                acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_load_si128(in_ptr2 + ix), w));
                acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_load_si128(in_ptr3 + ix), w));
                acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_load_si128(in_ptr4 + ix), w));
                acc5 = _mm_add_epi32(acc5, _mm_madd_epi16(_mm_load_si128(in_ptr5 + ix), w));
                acc6 = _mm_add_epi32(acc6, _mm_madd_epi16(_mm_load_si128(in_ptr6 + ix), w));
                acc7 = _mm_add_epi32(acc7, _mm_madd_epi16(_mm_load_si128(in_ptr7 + ix), w));
            }


            output[0] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc6);
            output[7] = getBias(bias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc7);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
                output[7] += input_7[i] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t KT = config->RequestConfig->Transform.inputElementCount % SSE_16CAP;
    uint32_t KK = config->RequestConfig->Transform.inputElementCount - KT;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int8_t const * multiBias;
    int8_t const * const biasEnd = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias) +
        (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.multiBiasVectorCount);
    auto biasStride = config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;
    int16_t const *input_4 = nullptr;
    int16_t const *input_5 = nullptr;
    int16_t const *input_6 = nullptr;
    int16_t const *input_7 = nullptr;

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i w;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        in_ptr0 = (__m128i*)config->RequestConfig->Inputs;
        input_0 = inputs + KK;
        ix_end = KK / SSE_16CAP;
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
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
        in_ptr7 = (__m128i*)config->Intermediate->d7;
        input_7 = config->Intermediate->d7 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->Intermediate->d6;
        input_6 = config->Intermediate->d6 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->Intermediate->d5;
        input_5 = config->Intermediate->d5 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->Intermediate->d4;
        input_4 = config->Intermediate->d4 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->Intermediate->d3;
        input_3 = config->Intermediate->d3 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->Intermediate->d2;
        input_2 = config->Intermediate->d2 + KK;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->Intermediate->d1;
        input_1 = config->Intermediate->d1 + KK;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->Intermediate->d0;
        input_0 = config->Intermediate->d0 + KK;
    }
    ix_end = KK / SSE_16CAP;

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);

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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
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
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_load_si128(in_ptr0 + ix), w));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_load_si128(in_ptr1 + ix), w));
                acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_load_si128(in_ptr2 + ix), w));
                acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_load_si128(in_ptr3 + ix), w));
                acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_load_si128(in_ptr4 + ix), w));
                acc5 = _mm_add_epi32(acc5, _mm_madd_epi16(_mm_load_si128(in_ptr5 + ix), w));
                acc6 = _mm_add_epi32(acc6, _mm_madd_epi16(_mm_load_si128(in_ptr6 + ix), w));
                acc7 = _mm_add_epi32(acc7, _mm_madd_epi16(_mm_load_si128(in_ptr7 + ix), w));
            }

            output[0] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc6);
            output[7] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias) + vec_sum(acc7);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
                output[7] += input_7[i] * *weight;
            }
            output += config->RequestConfig->Transform.inputVectorCount;
        }

        return;
    }
}
