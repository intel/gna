/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/


#include "common_avx2.hpp"
#include "convnet.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cmath>

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
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    uint32_t i;
    uint32_t j;

    gna_sum_t sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;

    mm_ptr in1, in2, in3, in4, in5, in6, in7, in8, in_end, flt;
    int32_t *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    const uint8_t * bias;

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
            if (FC_VEC < FC)
            {
                __m128i s1 = _mm256_castsi256_si128(v1);
                __m128i s2 = _mm256_castsi256_si128(v2);
                __m128i s3 = _mm256_castsi256_si128(v3);
                __m128i s4 = _mm256_castsi256_si128(v4);
                __m128i s5 = _mm256_castsi256_si128(v5);
                __m128i s6 = _mm256_castsi256_si128(v6);
                __m128i s7 = _mm256_castsi256_si128(v7);
                __m128i s8 = _mm256_castsi256_si128(v8);
                __m128i sf = _mm256_castsi256_si128(f);

                s1 = _mm_madd_epi16(s1, sf);
                s2 = _mm_madd_epi16(s2, sf);
                s3 = _mm_madd_epi16(s3, sf);
                s4 = _mm_madd_epi16(s4, sf);
                s5 = _mm_madd_epi16(s5, sf);
                s6 = _mm_madd_epi16(s6, sf);
                s7 = _mm_madd_epi16(s7, sf);
                s8 = _mm_madd_epi16(s8, sf);

                sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
                sum6 += _mm_extract_epi32(s6, 0) + _mm_extract_epi32(s6, 1) + _mm_extract_epi32(s6, 2) + _mm_extract_epi32(s6, 3);
                sum7 += _mm_extract_epi32(s7, 0) + _mm_extract_epi32(s7, 1) + _mm_extract_epi32(s7, 2) + _mm_extract_epi32(s7, 3);
                sum8 += _mm_extract_epi32(s8, 0) + _mm_extract_epi32(s8, 1) + _mm_extract_epi32(s8, 2) + _mm_extract_epi32(s8, 3);
            }

            saturate_store_out(&sum1, out1, saturationCount);
            saturate_store_out(&sum2, out2, saturationCount);
            saturate_store_out(&sum3, out3, saturationCount);
            saturate_store_out(&sum4, out4, saturationCount);
            saturate_store_out(&sum5, out5, saturationCount);
            saturate_store_out(&sum6, out6, saturationCount);
            saturate_store_out(&sum7, out7, saturationCount);
            saturate_store_out(&sum8, out8, saturationCount);

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
            if (FC_VEC < FC)
            {
                __m128i s1 = _mm256_castsi256_si128(v1);
                __m128i sf = _mm256_castsi256_si128(f);

                s1 = _mm_madd_epi16(s1, sf);
                sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
            }

            saturate_store_out(&sum1, out1++, saturationCount);
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
    const BiasRegular * const B = filterConfig->biases;
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, s5, s6, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);
                        s5 = _mm256_castsi256_si128(v5);
                        s6 = _mm256_castsi256_si128(v6);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);
                        s5 = _mm_madd_epi16(s5, sf);
                        s6 = _mm_madd_epi16(s6, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                        sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
                        sum6 += _mm_extract_epi32(s6, 0) + _mm_extract_epi32(s6, 1) + _mm_extract_epi32(s6, 2) + _mm_extract_epi32(s6, 3);
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, s5, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);
                        s5 = _mm256_castsi256_si128(v5);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);
                        s5 = _mm_madd_epi16(s5, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                        sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
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

                    if (FC_VEC < FC)
                    {
                        __m128i s1, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);

                        s1 = _mm_madd_epi16(s1, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
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
                    gna_saturate_cast(value, *saturationCount);
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
            gna_saturate_cast(value, *saturationCount);
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

using m256i_x2 = decltype(std::make_pair(_mm256_setzero_si256(), _mm256_setzero_si256()));
/* Multiply and add sizeof(__m256i) elements, supports 8 and 16bit types
 * possibly zero elements that are past data bound (if apply_mask is set),
 * mask should have one or two elements (the latter only in case of "2b2b").
 *
 * Steps:
 * extend both input and filter from epi8 to epi16 for multiplication (when used with 8bit types)
 * multiply them
 * (elements zeroed previously in input now yield zeros, so would not bother later accumulation)
 * (use _mm256_madd_epi16() instead of _mm256_mullo_epi16(), so we get one level of horizontal sum
 * for free, it also expands data to 32bit, as would be needed in accumulation anyway).
 *
 * Returns as two registers to decrease dependency chain between (outer) loop steps/
 *
 * Compilation note:
 * versions of GCC prior to 7 and ICC prior to 19 do not support constexpr if.
 * However marking appropiate conditions as constexpr if do not result in different code,
 * at least when comparing between ICC 18 and ICC 19 on Linux.
 */
template <typename filter_t, typename input_t>
static inline m256i_x2 madd_32_elems(filter_t *F, input_t *I, __m256i *mask, bool apply_mask)
{
    __m256i filter_0 = _mm256_loadu_si256((const __m256i *)F);
    __m256i input_0 = _mm256_loadu_si256((const __m256i *)I);
    __m256i filter_1, input_1;
    if (sizeof(filter_t) == 2) {
        filter_1 = _mm256_loadu_si256((const __m256i *)F + 1);
    }
    if (sizeof(input_t) == 2) {
        input_1 = _mm256_loadu_si256((const __m256i *)I + 1);
    }
    if (apply_mask) {
        if (sizeof(filter_t) <= sizeof(input_t)) {
            filter_0 = _mm256_and_si256(filter_0, mask[0]);
            if (sizeof(filter_t) == 2) {
                filter_1 = _mm256_and_si256(filter_1, mask[1]);
            }
        }
        else {
            input_0 = _mm256_and_si256(input_0, mask[0]);
        }
    }
    if (sizeof(filter_t) == 1) {
        filter_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(filter_0, 1));
        filter_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(filter_0));
    }
    if (sizeof(input_t) == 1) {
        input_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(input_0, 1));
        input_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(input_0));
    }
    __m256i m0 = _mm256_madd_epi16(input_0, filter_0);
    __m256i m1 = _mm256_madd_epi16(input_1, filter_1);
    return { m0, m1 };
}

template<typename T, size_t N>
static constexpr std::array<T, N> initByHalves(T firstHalfVal, T secondHalfVal) {
	std::array<T, N> a = {};
	size_t i = 0;
	for (; i < N/2; ++i) {
		a[i] = firstHalfVal;
	}
	for (; i < N; ++i) {
		a[i] = secondHalfVal;
	}
	return a;
}

template <typename filter_t, typename input_t>
static void cnn2d(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    static_assert(sizeof(filter_t) <= 2 && sizeof(input_t) <= 2,
                  "filter_t and input_t must be int8_t or int16_t");
    const auto & conf = config->RequestConfig;
    const input_t *const I = (input_t *)conf.Inputs;
    const filter_t *const F = (filter_t *)conf.Transform.FilterData;
    int32_t *O = (int32_t *)conf.Outputs;

    uint32_t inputDepth = conf.Transform.InputDepth;
    uint32_t inputHeight = conf.Transform.InputHeight;
    uint32_t inputWidth = conf.Transform.InputWidth;

    uint32_t numFilters = conf.Transform.NumberOfFilters;
    uint32_t filterHeight = conf.Transform.FilterHeight;
    uint32_t filterWidth = conf.Transform.FilterWidth;
    constexpr uint32_t sizeof_filter_t = sizeof(filter_t);
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth * sizeof_filter_t);
    uint32_t filterPadding = (Gna2RoundUp(memForFilter, 16) - memForFilter) / sizeof_filter_t;

    uint32_t padHeight = conf.Transform.ZeroPaddingHeight;
    uint32_t padWidth = conf.Transform.ZeroPaddingWidth;
    uint32_t strideHeight = conf.Transform.StrideHeight;
    uint32_t strideWidth = conf.Transform.StrideWidth;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    auto biasMode = conf.Transform.BiasMode;
    auto biasPrecission = conf.Transform.BiasDataMode;
    const void *biasData = conf.Transform.BiasData;

    // algo moves by the same number of elements per one step, irrelevant of input/filter width
    // (so it moves by twice as many bytes in 2B case vs 1B case)
    constexpr const uint32_t elems = sizeof(__m256i) / sizeof(int16_t);
    constexpr const uint32_t step = elems * 2; // how many elems are processed per loop step
    constexpr const bool is_2b2b = sizeof(filter_t) == 2 && sizeof(input_t) == 2;
    using mask_t = typename std::conditional<is_2b2b, int16_t, int8_t>::type;
    const auto maskArray = initByHalves<mask_t, step*2>(-1, 0);
    const auto mask = maskArray.data();

    for (uint32_t OD = 0; OD < numFilters; OD++) {
        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OH = 0; OH < outHeight; OH++) {
            for (uint32_t OW = 0; OW < outWidth; OW++) {

                int64_t outVal;
                if (biasMode == KernelBiasModePerFilter) {
                    outVal = getBias(biasData, biasPrecission, OD);
                }
                else if (biasMode == KernelBiasModeDisabled) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, biasPrecission,
                                     numFilters * outWidth * OH + numFilters * OW + OD);
                }

                /* Thanks to the fact that data is packed, we could iterate over W and Z dimensions via one loop.
                 * This observation improves performance a lot for case when W*Z is big, but one of dims is small. */
                uint32_t fIdxH = 0, inIdxH = 0;
                if (OH * strideHeight < padHeight) {
                    fIdxH = padHeight - OH * strideHeight;
                }
                else {
                    inIdxH = OH * strideHeight - padHeight;
                }
                uint32_t boundH = (std::min)(filterHeight, inputHeight + padHeight - OH * strideHeight);
                uint32_t steps = boundH - fIdxH;
                inIdxH *= inputDepth * inputWidth;
                fIdxH *= inputDepth * filterWidth;
                uint32_t fIdxW = 0, inIdxW = 0;
                if (OW * strideWidth < padWidth) {
                    fIdxW = padWidth - OW * strideWidth;
                }
                else {
                    inIdxW = OW * strideWidth - padWidth;
                }
                const uint32_t span = (std::min)(inputWidth - inIdxW, filterWidth - fIdxW);
                const uint32_t stepsPerWxZ = span * inputDepth;
                inIdxW *= inputDepth;
                fIdxW *= inputDepth;
                uint32_t idxI = inIdxH + inIdxW, idxF = fIdxN + fIdxH + fIdxW;
                const uint32_t stepsRounded = Gna2RoundUp(stepsPerWxZ, step);
                uint32_t stepIH = inputDepth * inputWidth - stepsRounded;
                uint32_t stepFH = inputDepth * filterWidth - stepsRounded;
                __m256i acc_0 = _mm256_setzero_si256();
                __m256i acc_1 = _mm256_setzero_si256();
                __m256i acc_2 = _mm256_setzero_si256();
                __m256i acc_3 = _mm256_setzero_si256();
                __m256i masked[2];
                masked[0] = _mm256_loadu_si256((const __m256i *)(mask + step - stepsPerWxZ % step));
                if (is_2b2b) {
                    masked[1] = _mm256_loadu_si256((const __m256i *)(mask + step + elems - stepsPerWxZ % step));
                }
                for (; steps--; idxF += stepFH, idxI += stepIH) {
                    for (uint32_t i = 0; i < stepsPerWxZ; i += step, idxF += step, idxI += step) {
                        auto m = madd_32_elems(F + idxF, I + idxI, masked, (i + step > stepsPerWxZ));
                        auto m0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m.first));
                        auto m1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m.first, 1));
                        auto m2 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m.second));
                        auto m3 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m.second, 1));
                        acc_0 = _mm256_add_epi64(acc_0, m0);
                        acc_1 = _mm256_add_epi64(acc_1, m1);
                        acc_2 = _mm256_add_epi64(acc_2, m2);
                        acc_3 = _mm256_add_epi64(acc_3, m3);
                    }
                }
                acc_0 = _mm256_add_epi64(acc_0, acc_1);
                acc_2 = _mm256_add_epi64(acc_2, acc_3);
                acc_0 = _mm256_add_epi64(acc_0, acc_2);
                outVal += _mm256_hsum_epi64(acc_0);
                gna_saturate_cast(outVal, *config->SaturationCount);
                O[numFilters * outWidth * OH + numFilters * OW + OD] = (int32_t)outVal;
            }
        }
    }
}

void Convolution2DKernelImpl1B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    return cnn2d<int8_t, int8_t>(config);
}

void Convolution2DKernelImpl1B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    return cnn2d<int8_t, int16_t>(config);
}

void Convolution2DKernelImpl2B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    return cnn2d<int16_t, int8_t>(config);
}

void Convolution2DKernelImpl2B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    return cnn2d<int16_t, int16_t>(config);
}

template <typename data_t>
static void poolMax2d(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    const auto & conf = config->RequestConfig;
    data_t *I = (data_t *)conf.Inputs;
    data_t *O = (data_t *)conf.Outputs;

    uint32_t inputW = conf.Transform.InputWidth;
    uint32_t inputH = conf.Transform.InputHeight;
    uint32_t numFilters = conf.Transform.InputDepth;

    uint32_t poolStrideH = conf.Transform.StrideHeight;
    uint32_t poolStrideW = conf.Transform.StrideWidth;
    uint32_t windowHeight = conf.Transform.WindowHeight;
    uint32_t windowWidth = conf.Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const bool is1B = (sizeof(data_t) == 1);
    constexpr const bool is2B = (sizeof(data_t) == 2);
    constexpr const bool is4B = (sizeof(data_t) == 4);
    constexpr const uint32_t elems = sizeof(__m256i) / sizeof(data_t);
    constexpr data_t minLimit = (std::numeric_limits<data_t>::min)();
    if (is1B) {
        memset(O, minLimit, poolOutH * poolOutW * numFilters);
    }
    else {
        std::fill(O, O + poolOutH * poolOutW * numFilters, minLimit);
    }

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;

            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }
            for (uint32_t OH = 0; OH < limH; OH++) {
                uint32_t winIdxH = numFilters * inputW * OH;

                uint32_t limW = windowWidth;
                if (POW * poolStrideW > inputW) {
                    limW = 0;
                }
                else if (inputW - POW * poolStrideW < limW) {
                    limW = inputW - POW * poolStrideW;
                }
                for (uint32_t OW = 0; OW < limW; OW++) {
                    uint32_t winIdxW = numFilters * OW;

                    uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;
                    uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                    uint32_t offset = 0;
                    constexpr const uint32_t step = elems;
                    for (; offset + step <= numFilters; offset += step) {
                        __m256i in = _mm256_loadu_si256((const __m256i *)(I + inBaseIdx + offset));
                        __m256i cur = _mm256_loadu_si256((const __m256i *)(O + outBaseIdx + offset));
                        __m256i mx;
                        if (is1B) {
                            mx = _mm256_max_epi8(cur, in);
                        }
                        if (is2B) {
                            mx = _mm256_max_epi16(cur, in);
                        }
                        if (is4B) {
                            mx = _mm256_max_epi32(cur, in);
                        }
                        _mm256_storeu_si256((__m256i *)(O + outBaseIdx + offset), mx);
                    }
                    if (offset < numFilters) {
                        data_t unevenPartI[step] = {};
                        data_t unevenPartO[step] = {};
                        memcpy(unevenPartI, I + inBaseIdx  + offset, (numFilters - offset) * sizeof(data_t));
                        memcpy(unevenPartO, O + outBaseIdx + offset, (numFilters - offset) * sizeof(data_t));
                        __m256i  in = _mm256_loadu_si256((const __m256i *)unevenPartI);
                        __m256i cur = _mm256_loadu_si256((const __m256i *)unevenPartO);
                        __m256i mx;
                        if (is1B) {
                            mx = _mm256_max_epi8(cur, in);
                        }
                        if (is2B) {
                            mx = _mm256_max_epi16(cur, in);
                        }
                        if (is4B) {
                            mx = _mm256_max_epi32(cur, in);
                        }
                        _mm256_storeu_si256((__m256i *)unevenPartO, mx);
                        memcpy(O + outBaseIdx + offset, unevenPartO, (numFilters - offset) * sizeof(data_t));
                    }
                }
            }
        }
    }
}

template <typename data_t>
static void poolSum2d(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    const auto & conf = config->RequestConfig;
    data_t *I = (data_t *)conf.Inputs;
    data_t *O = (data_t *)conf.Outputs;

    uint32_t inputW = conf.Transform.InputWidth;
    uint32_t inputH = conf.Transform.InputHeight;
    uint32_t numFilters = conf.Transform.InputDepth;

    uint32_t poolStrideH = conf.Transform.StrideHeight;
    uint32_t poolStrideW = conf.Transform.StrideWidth;
    uint32_t windowHeight = conf.Transform.WindowHeight;
    uint32_t windowWidth = conf.Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const bool is1B = (sizeof(data_t) == 1);
    constexpr const bool is2B = (sizeof(data_t) == 2);
    /* note: 4B variant has specialized version, so it's handling was removed here */
    constexpr const uint32_t elems = sizeof(__m256i) / sizeof(data_t);
    constexpr const uint32_t step = elems * 1;  // how many elems are processed per loop step
    __m256i satCntHighBit0 = _mm256_setzero_si256();
    __m256i satCntHighBit1 = _mm256_setzero_si256();
    __m256i satCntAllBits0 = _mm256_setzero_si256();
    __m256i satCntAllBits1 = _mm256_setzero_si256();

    /* Following variable is used in 1B variant, to avoid bumping of SaturationCount when it is not necessary.
     * Calculations are still performed saturation-aware, it is only the counter that is not touched.
     * This approach yields about 14% improvement over naive version (if-less, with all instructions of "if" path).
     *
     * Similar trick makes no sense in 2B variant, where satCnt is calculated with single vectorized "or".
     */
    uint32_t needsSatCnt = 1;

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;

            uint32_t limW = windowWidth;
            if (POW * poolStrideW > inputW) {
                limW = 0;
            }
            else if (inputW - POW * poolStrideW < limW) {
                limW = inputW - POW * poolStrideW;
            }
            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }

            // usage of following variable speeds up 2B case by ~25% on perftest's data
            const bool couldEverOV = (limW * limH * numFilters > 0x10000);
            uint32_t couldOV = 0;
            uint32_t offset = 0;
            for (; offset + step <= numFilters; offset += step) {
                __m256i cur0 = _mm256_setzero_si256();
                __m256i cur1 = _mm256_setzero_si256();
                for (uint32_t OW = 0; OW < limW; OW++) {
                    uint32_t winIdxW = numFilters * OW;
                    for (uint32_t OH = 0; OH < limH; OH++) {
                        uint32_t winIdxH = numFilters * inputW * OH;

                        uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                        __m256i in0 = _mm256_loadu_si256((const __m256i *)(I + inBaseIdx + offset));
                        if (is1B) {
                            __m256i in01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(in0, 1));
                            __m256i in00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(in0));
                            if ((couldOV += needsSatCnt) > 0x100) {
                                __m256i noSat0 = _mm256_add_epi16(cur0, in00);
                                __m256i noSat1 = _mm256_add_epi16(cur1, in01);
                                cur0 = _mm256_adds_epi16(cur0, in00);
                                cur1 = _mm256_adds_epi16(cur1, in01);
                                __m256i x0 = _mm256_xor_si256(noSat0, cur0);
                                __m256i x1 = _mm256_xor_si256(noSat1, cur1);
                                satCntAllBits0 = _mm256_or_si256(satCntAllBits0, x0);
                                satCntAllBits1 = _mm256_or_si256(satCntAllBits1, x1);
                            }
                            else {
                                cur0 = _mm256_adds_epi16(cur0, in00);
                                cur1 = _mm256_adds_epi16(cur1, in01);
                            }
                        }
                        if (is2B) {
                            __m256i in01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(in0, 1));
                            __m256i in00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(in0));
                            if (couldEverOV && ++couldOV > 0x10000) {
                                cur0 = _mm256_adds_epi32(cur0, in00, &satCntHighBit0);
                                cur1 = _mm256_adds_epi32(cur1, in01, &satCntHighBit1);
                            }
                            else {
                                cur0 = _mm256_add_epi32(cur0, in00);
                                cur1 = _mm256_add_epi32(cur1, in01);
                            }
                        }
                    }
                }
                __m256i ret0, ret00, ret01;
                if (is1B) {
                    ret0 = _mm256_packs_epi16(cur0, cur1);
                    ret0 = _mm256_permute4x64_epi64(ret0, 0xd8);
                    if (needsSatCnt) {
                        ret01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ret0, 1));
                        ret00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(ret0));
                    }
                }
                if (is2B) {
                    ret0 = _mm256_packs_epi32(cur0, cur1);
                    ret0 = _mm256_permute4x64_epi64(ret0, 0xd8);
                    ret01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ret0, 1));
                    ret00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ret0));
                }
                if (needsSatCnt) {
                    __m256i x1 = _mm256_xor_si256(ret01, cur1);
                    __m256i x0 = _mm256_xor_si256(ret00, cur0);
                    satCntAllBits0 = _mm256_or_si256(satCntAllBits0, x0);
                    satCntAllBits1 = _mm256_or_si256(satCntAllBits1, x1);
                    if (is1B) {
                        if (_mm256_test_any(satCntAllBits0) || _mm256_test_any(satCntAllBits1)) {
                            needsSatCnt = 0;
                        }
                    }
                }
                _mm256_storeu_si256((__m256i *)(O + outBaseIdx + offset), ret0);
            }
            if (offset < numFilters) {
                __m256i cur0 = _mm256_setzero_si256();
                __m256i cur1 = _mm256_setzero_si256();
                for (uint32_t OW = 0; OW < limW; OW++) {
                    uint32_t winIdxW = numFilters * OW;
                    for (uint32_t OH = 0; OH < limH; OH++) {
                        uint32_t winIdxH = numFilters * inputW * OH;

                        uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                        data_t unevenPartI[step] = {};
                        memcpy(unevenPartI, I + inBaseIdx  + offset, (numFilters - offset) * sizeof(data_t));
                        __m256i in0 = _mm256_loadu_si256((const __m256i *) unevenPartI);
                        if (is1B) {
                            __m256i in01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(in0, 1));
                            __m256i in00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(in0));
                            if ((couldOV += needsSatCnt) > 0x100) {
                                __m256i noSat0 = _mm256_add_epi16(cur0, in00);
                                __m256i noSat1 = _mm256_add_epi16(cur1, in01);
                                cur0 = _mm256_adds_epi16(cur0, in00);
                                cur1 = _mm256_adds_epi16(cur1, in01);
                                __m256i x0 = _mm256_xor_si256(noSat0, cur0);
                                __m256i x1 = _mm256_xor_si256(noSat1, cur1);
                                satCntAllBits0 = _mm256_or_si256(satCntAllBits0, x0);
                                satCntAllBits1 = _mm256_or_si256(satCntAllBits1, x1);
                            }
                            else {
                                cur0 = _mm256_adds_epi16(cur0, in00);
                                cur1 = _mm256_adds_epi16(cur1, in01);
                            }
                        }
                        if (is2B) {
                            __m256i in01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(in0, 1));
                            __m256i in00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(in0));
                            if (couldEverOV && ++couldOV > 0x10000) {
                                cur0 = _mm256_adds_epi32(cur0, in00, &satCntHighBit0);
                                cur1 = _mm256_adds_epi32(cur1, in01, &satCntHighBit1);
                            }
                            else {
                                cur0 = _mm256_add_epi32(cur0, in00);
                                cur1 = _mm256_add_epi32(cur1, in01);
                            }
                        }
                    }
                }
                data_t unevenPartO[step] = {};
                __m256i ret0, ret00, ret01;
                if (is1B) {
                    ret0 = _mm256_packs_epi16(cur0, cur1);
                    ret0 = _mm256_permute4x64_epi64(ret0, 0xd8);
                    if (needsSatCnt) {
                        ret01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ret0, 1));
                        ret00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(ret0));
                    }
                }
                if (is2B) {
                    ret0 = _mm256_packs_epi32(cur0, cur1);
                    ret0 = _mm256_permute4x64_epi64(ret0, 0xd8);
                    ret01 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ret0, 1));
                    ret00 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ret0));
                }
                if (needsSatCnt) {
                    __m256i x1 = _mm256_xor_si256(ret01, cur1);
                    __m256i x0 = _mm256_xor_si256(ret00, cur0);
                    satCntAllBits0 = _mm256_or_si256(satCntAllBits0, x0);
                    satCntAllBits1 = _mm256_or_si256(satCntAllBits1, x1);
                    if (is1B) {
                        if (_mm256_test_any(satCntAllBits0) || _mm256_test_any(satCntAllBits1)) {
                            needsSatCnt = 0;
                        }
                    }
                }
                _mm256_storeu_si256((__m256i *) unevenPartO, ret0);
                memcpy(O + outBaseIdx + offset, unevenPartO, (numFilters - offset) * sizeof(data_t));
            }
        }
    }
    if (!is1B) {
        *config->SaturationCount += _mm256_test_anyMSB_epi32(satCntHighBit0);
    }
    if (is2B) {
        *config->SaturationCount += _mm256_test_anyMSB_epi32(satCntHighBit1);
    }
    *config->SaturationCount += _mm256_test_any(satCntAllBits0);
    *config->SaturationCount += _mm256_test_any(satCntAllBits1);
}

/* This is a specialized version of pooling Sum.
 * It is 1.8x times faster than generic version because it uses different algorithm.
 *
 * Main difference is loop order,
 * what means that we compute each output value via many LOAD/ADDS/STORE cycles.
 * Thanks to the fact above we could traverse input in more favorable way memory-wise. */
template <>
void poolSum2d<int32_t>(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    const auto & conf = config->RequestConfig;
    using data_t = int32_t;
    data_t *I = (data_t *)conf.Inputs;
    data_t *O = (data_t *)conf.Outputs;

    uint32_t inputW = conf.Transform.InputWidth;
    uint32_t inputH = conf.Transform.InputHeight;
    uint32_t numFilters = conf.Transform.InputDepth;

    uint32_t poolStrideH = conf.Transform.StrideHeight;
    uint32_t poolStrideW = conf.Transform.StrideWidth;
    uint32_t windowHeight = conf.Transform.WindowHeight;
    uint32_t windowWidth = conf.Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const uint32_t elems = sizeof(__m256i) / sizeof(data_t);
    constexpr const uint32_t step = elems * 1;  // how many elems are processed per loop step
    memset(O, 0, poolOutH * poolOutW * numFilters * sizeof(data_t));  // we are calculating out via many parial sums
    __m256i satCnt = _mm256_setzero_si256();

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;

            uint32_t limW = windowWidth;
            if (POW * poolStrideW > inputW) {
                limW = 0;
            }
            else if (inputW - POW * poolStrideW < limW) {
                limW = inputW - POW * poolStrideW;
            }
            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }

            for (uint32_t OW = 0; OW < limW; OW++) {
                uint32_t winIdxW = numFilters * OW;
                for (uint32_t OH = 0; OH < limH; OH++) {
                    uint32_t winIdxH = numFilters * inputW * OH;

                    uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                    uint32_t offset = 0;
                    for (; offset + step <= numFilters; offset += step) {
                        __m256i in = _mm256_loadu_si256((const __m256i *)(I + inBaseIdx + offset));
                        __m256i cur = _mm256_loadu_si256((const __m256i *)(O + outBaseIdx + offset));
                        __m256i ret = _mm256_adds_epi32(cur, in, &satCnt);
                        _mm256_storeu_si256((__m256i *)(O + outBaseIdx + offset), ret);
                    }
                    if (offset < numFilters) {
                        data_t unevenPartI[step] = {};
                        data_t unevenPartO[step] = {};
                        memcpy(unevenPartI, I + inBaseIdx  + offset, (numFilters - offset) * sizeof(data_t));
                        memcpy(unevenPartO, O + outBaseIdx + offset, (numFilters - offset) * sizeof(data_t));
                        __m256i  in = _mm256_loadu_si256((const __m256i *)unevenPartI);
                        __m256i cur = _mm256_loadu_si256((const __m256i *)unevenPartO);
                        __m256i ret = _mm256_adds_epi32(cur, in, &satCnt);
                        _mm256_storeu_si256((__m256i *)unevenPartO, ret);
                        memcpy(O + outBaseIdx + offset, unevenPartO, (numFilters - offset) * sizeof(data_t));
                    }
                }
            }
        }
    }
    *config->SaturationCount += _mm256_test_anyMSB_epi32(satCnt);
}

template <typename data_t>
static inline void pool2d(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    auto mode = config->RequestConfig.Transform.Mode;
    if (mode == KernelPoolingModeMax) {
        poolMax2d<data_t>(config);
    }
    else if (mode == KernelPoolingModeSum) {
        poolSum2d<data_t>(config);
    }
}

void Pooling2DKernelImpl1B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    pool2d<int8_t>(config);
}

void Pooling2DKernelImpl2B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    pool2d<int16_t>(config);
}

void Pooling2DKernelImpl4B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    pool2d<int32_t>(config);
}
