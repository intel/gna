/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "convnet.h"
#include "igemv.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>

void SumPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t *V)
{
    uint32_t k = 0;
    uint32_t index = 0;
    *V = 0;

    for (k = 0; k < PNE; k++)
    {
        index = (PSI + k) % PS;
        *V += P[index];
    }
}

void MaxPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t *V)
{
    uint32_t k = 0;
    uint32_t index = 0;
    *V = P[PSI % PS];

    for (k = 0; k < PNE; k++)
    {
        index = (PSI + k) % PS;
        if (P[index]>(*V))
        {
            *V = P[index];
        }
    }
}

void ConvolutionKernelImpl(ConvolutionConfig const * const filterConfig)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int16_t* const F = filterConfig->filters;
    const auto * const B = reinterpret_cast<uint8_t const *>(filterConfig->biases);
    int32_t * const O = filterConfig->convolutedOutputs;

    uint32_t i;
    uint32_t j;

    gna_sum_t sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;

    mm_ptr in1, in2, in3, in4, in5, in6, in7, in8, in_end, flt;
    int32_t *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    const uint8_t *bias;

    mm_vector f, v1, v2, v3, v4, v5, v6, v7, v8;

    mm_vector acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8,
            im1, im2, im3, im4, im5, im6, im7, im8;

    uint32_t FC_REM = FC % VEC_16CAP;
    uint32_t FC_VEC = FC - FC_REM;
    uint8_t gr = 8;
    uint32_t N_REM = num_filter_outputs % gr;
    uint32_t N_VEC = num_filter_outputs - N_REM;

    for (j = 0; j < N_VEC; j += gr)
    {
        in_end = (mm_ptr)(I + j * num_inputs_band_stride + FC_VEC);

        out1 = O + j * FN;
        out2 = out1 + FN;
        out3 = out2 + FN;
        out4 = out3 + FN;
        out5 = out4 + FN;
        out6 = out5 + FN;
        out7 = out6 + FN;
        out8 = out7 + FN;

        for (bias = B, i = 0; i < FN; i++)
        {
            flt = (mm_ptr)(F + i * FC);
            f = vec_lddqu(flt);

            in1 = (mm_ptr)(I + j * num_inputs_band_stride);
            in2 = (mm_ptr)(I + (j + 1) * num_inputs_band_stride);
            in3 = (mm_ptr)(I + (j + 2) * num_inputs_band_stride);
            in4 = (mm_ptr)(I + (j + 3) * num_inputs_band_stride);
            in5 = (mm_ptr)(I + (j + 4) * num_inputs_band_stride);
            in6 = (mm_ptr)(I + (j + 5) * num_inputs_band_stride);
            in7 = (mm_ptr)(I + (j + 6) * num_inputs_band_stride);
            in8 = (mm_ptr)(I + (j + 7) * num_inputs_band_stride);

            v1 = vec_lddqu(in1);
            v2 = vec_lddqu(in2);
            v3 = vec_lddqu(in3);
            v4 = vec_lddqu(in4);
            v5 = vec_lddqu(in5);
            v6 = vec_lddqu(in6);
            v7 = vec_lddqu(in7);
            v8 = vec_lddqu(in8);

            acc1 = vec_setzero();
            acc2 = vec_setzero();
            acc3 = vec_setzero();
            acc4 = vec_setzero();
            acc5 = vec_setzero();
            acc6 = vec_setzero();
            acc7 = vec_setzero();
            acc8 = vec_setzero();

            for (; in1 < in_end; )
            {
                in1++;
                in2++;
                in3++;
                in4++;
                in5++;
                in6++;
                in7++;
                in8++;
                flt++;

                im1 = vec_madd16(v1, f);
                im2 = vec_madd16(v2, f);
                im3 = vec_madd16(v3, f);
                im4 = vec_madd16(v4, f);
                im5 = vec_madd16(v5, f);
                im6 = vec_madd16(v6, f);
                im7 = vec_madd16(v7, f);
                im8 = vec_madd16(v8, f);

                acc1 = vec_accumulate(acc1, im1);
                acc2 = vec_accumulate(acc2, im2);
                acc3 = vec_accumulate(acc3, im3);
                acc4 = vec_accumulate(acc4, im4);
                acc5 = vec_accumulate(acc5, im5);
                acc6 = vec_accumulate(acc6, im6);
                acc7 = vec_accumulate(acc7, im7);
                acc8 = vec_accumulate(acc8, im8);

                v1 = vec_lddqu(in1);
                v2 = vec_lddqu(in2);
                v3 = vec_lddqu(in3);
                v4 = vec_lddqu(in4);
                v5 = vec_lddqu(in5);
                v6 = vec_lddqu(in6);
                v7 = vec_lddqu(in7);
                v8 = vec_lddqu(in8);
                f = vec_lddqu(flt);
            }

            sum1 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc1);
            sum2 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc2);
            sum3 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc3);
            sum4 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc4);
            sum5 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc5);
            sum6 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc6);
            sum7 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc7);
            sum8 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc8);

            bias += filterConfig->bytesPerBias;

// FC is mply by 8, for AVX load there might be a tail of 8

            *out1 = sum1;
            *out2 = sum2;
            *out3 = sum3;
            *out4 = sum4;
            *out5 = sum5;
            *out6 = sum6;
            *out7 = sum7;
            *out8 = sum8;

            out1++;
            out2++;
            out3++;
            out4++;
            out5++;
            out6++;
            out7++;
            out8++;
        }
    }

    for (j = N_VEC; j < num_filter_outputs; j++)
    {
        in_end = (mm_ptr)(I + j * num_inputs_band_stride + FC_VEC);

        out1 = O + j * FN;

        for (bias = B, i = 0; i < FN; i++)
        {
            in1 = (mm_ptr)(I + j * num_inputs_band_stride);
            flt = (mm_ptr)(F + i * FC);

            f = vec_lddqu(flt);
            v1 = vec_lddqu(in1);

            acc1 = vec_setzero();

            for (; in1 < in_end; )
            {
                in1++;
                flt++;

                im1 = vec_madd16(v1, f);
                acc1 = vec_accumulate(acc1, im1);

                f = vec_lddqu(flt);
                v1 = vec_lddqu(in1);
            }

            sum1 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc1);
            bias += filterConfig->bytesPerBias;

            *out1++ = sum1;
        }
    }
}

void ConvolutionPoolingKernelImpl(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int16_t* const F = filterConfig->filters;
    const nn_bias_s * const B = filterConfig->biases;
    int16_t * const O = filterConfig->pooledOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    const auto PT = poolConfig->Mode;
    const uint32_t PS = poolConfig->Size;
    const uint32_t PSTEP = poolConfig->Step;
    int64_t * const pool = poolConfig->Buffer;

    if (PS == 0)
    {
        return;
    }

    pwl->KERNEL(InitializeActivationFunctions)();

    void(*func_partial_pooling)(const uint32_t PS, const uint32_t pool_num_entries, const uint32_t pool_start_index, const int64_t *P, int64_t *V);

    if (PT == KernelPoolingModeSum)
    {
        func_partial_pooling = SumPartialPoolingFunction;
    }
    else
    {
        func_partial_pooling = MaxPartialPoolingFunction;
    }

    uint32_t pool_start_index = 0;
    uint32_t pool_end_index = 0;
    int32_t pool_num_entries = 0;
    uint32_t output_index = 0;
    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;
    uint32_t i;
    uint32_t j;
    int64_t value;
    uint32_t inc;
    gna_sum_t sum1, sum2, sum3, sum4, sum5, sum6;

    mm_ptr in1, in2, in3, in4, in5, in6, flt, in_end;
    mm_vector v1, v2, v3, v4, v5, v6, f;
    mm_vector im1, im2, im3, im4, im5, im6;
    mm_vector acc1, acc2, acc3, acc4, acc5, acc6;

    output_index = 0;
    pool_start_index = 0;
    pool_end_index = 0;
    pool_num_entries = 0;

    for (j = 0; j < num_filter_outputs; )
    {
        if (j >= output_index * PSTEP)
        {
            inc = (PS - static_cast<uint32_t>(pool_num_entries) < num_filter_outputs - j)
                ? PS - static_cast<uint32_t>(pool_num_entries)
                : num_filter_outputs - j;

            uint32_t FC_VEC = FC - FC % VEC_16CAP;
            in_end = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride + FC_VEC);

            // inc <1, 6>
            if (6 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    in5 = (mm_ptr)const_cast<int16_t*>(I + (j + 4) * num_inputs_band_stride);
                    in6 = (mm_ptr)const_cast<int16_t*>(I + (j + 5) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    v5 = vec_lddqu(in5);
                    v6 = vec_lddqu(in6);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();
                    acc5 = vec_setzero();
                    acc6 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];
                    sum5 = B[i];
                    sum6 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        in5++;
                        in6++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);
                        im5 = vec_madd16(v5, f);
                        im6 = vec_madd16(v6, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);
                        acc5 = vec_accumulate(acc5, im5);
                        acc6 = vec_accumulate(acc6, im6);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        v5 = vec_lddqu(in5);
                        v6 = vec_lddqu(in6);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);
                    sum5 += vec_sum(acc5);
                    sum6 += vec_sum(acc6);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+4)%PS] = sum5;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+5)%PS] = sum6;
                    pool_end_index = 0;
                }
            }
            if (5 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    in5 = (mm_ptr)const_cast<int16_t*>(I + (j + 4) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    v5 = vec_lddqu(in5);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();
                    acc5 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];
                    sum5 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        in5++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);
                        im5 = vec_madd16(v5, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);
                        acc5 = vec_accumulate(acc5, im5);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        v5 = vec_lddqu(in5);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);
                    sum5 += vec_sum(acc5);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+4)%PS] = sum5;
                }
            }
            if (4 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                }
            }
            if (3 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                }
            }
            if (2 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                }
            }
            if (1 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();

                    sum1 = B[i];


                    for (; in1 < in_end; )
                    {
                        in1++;
                        flt++;

                        im1 = vec_madd16(v1, f);

                        acc1 = vec_accumulate(acc1, im1);

                        v1 = vec_lddqu(in1);
                        f  = vec_lddqu(flt);
                    }

                    sum1 += vec_sum(acc1);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                }
            }
            pool_end_index += inc;
            pool_end_index %= PS;

            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
                    pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
                }

                pool_start_index = (pool_start_index + PSTEP) % PS;
                pool_num_entries -= PSTEP;
                if (pool_num_entries < 0)
                {
                    pool_start_index = 0;
                    pool_end_index = 0;
                    pool_num_entries = 0;
                }
                output_index++;
            }
        }
        else
        {
            j++;
        }
    }

    while (pool_num_entries > 0)
    {
        for (i = 0; i < FN; i++)
        {
            func_partial_pooling(PS, static_cast<uint32_t>(pool_num_entries), pool_start_index, pool + i * CNN_POOL_SIZE_MAX, &value);
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}
