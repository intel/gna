/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"
#include "Macros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

static void transposeM2(int16_t *input, int16_t *output, uint32_t N);
static void transposeM3(int16_t *input, int16_t *output, uint32_t N);
static void transposeM4(int16_t *input, int16_t *output, uint32_t N);
static void transposeM5(int16_t *input, int16_t *output, uint32_t N);
static void transposeM6(int16_t *input, int16_t *output, uint32_t N);
static void transposeM7(int16_t *input, int16_t *output, uint32_t N);
static void transposeM8(int16_t *input, int16_t *output, uint32_t N);

static void transposeN2(int16_t *input, int16_t *output, uint32_t M);
static void transposeN3(int16_t *input, int16_t *output, uint32_t M);
static void transposeN4(int16_t *input, int16_t *output, uint32_t M);
static void transposeN5(int16_t *input, int16_t *output, uint32_t M);
static void transposeN6(int16_t *input, int16_t *output, uint32_t M);
static void transposeN7(int16_t *input, int16_t *output, uint32_t M);
static void transposeN8(int16_t *input, int16_t *output, uint32_t M);

void TransposeKernelImpl(TransposeConfig const * const transposeConfig)
{
    uint32_t M = transposeConfig->rowCount;
    uint32_t N = transposeConfig->columnCount;
    const int16_t * const I = transposeConfig->input;
    int16_t * const O = transposeConfig->output;

    // input matrix is a vector - copy
    if (M == 1 || N == 1)
    {
        memmove_s(O, M * N * sizeof(int16_t), I, M * N * sizeof(int16_t));
        return;
    }

    // very small matrix - generic tranpose
    if (M * N < SSE_16CAP * SSE_16CAP)
    {
        for (uint32_t i = 0; i < M; i++)
        {
            for (uint32_t j = 0; j < N; j++)
            {
                O[j * M + i] = I[i * N + j];
            }
        }

        return;
    }

    int16_t *in0 = const_cast<int16_t*>(I);

    // INTERLEAVE
    // MAX M is 8, MAX N is UINT16_MAX
    if (N >= VEC_16CAP)
    {
        switch(M)
        {
            case 2:
                transposeM2(in0, O, N);
                break;
            case 3:
                transposeM3(in0, O, N);
                break;
            case 4:
                transposeM4(in0, O, N);
                break;
            case 5:
                transposeM5(in0, O, N);
                break;
            case 6:
                transposeM6(in0, O, N);
                break;
            case 7:
                transposeM7(in0, O, N);
                break;
            case 8:
                transposeM8(in0, O, N);
                break;
            default:
                break;
        }
    }

    // DEINTERLEAVE
    // MAX N is 8, MAX M is UINT16_MAX
    switch(N)
    {
        case 2:
            transposeN2(in0, O, M);
            break;
        case 3:
            transposeN3(in0, O, M);
            break;
        case 4:
            transposeN4(in0, O, M);
            break;
        case 5:
            transposeN5(in0, O, M);
            break;
        case 6:
            transposeN6(in0, O, M);
            break;
        case 7:
            transposeN7(in0, O, M);
            break;
        case 8:
            transposeN8(in0, O, M);
            break;
    }
}

void transposeM2(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 2;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *out0 = output;
    int16_t *out1 = out0 + SSE_16CAP;

    __m128i a, b, ab_lo, ab_hi;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);

        _mm_stream_si128((__m128i*) out0, ab_lo);
        _mm_stream_si128((__m128i*) out1, ab_hi);

        out0 += M * SSE_16CAP;
        out1 += M * SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
    }

    out1 = out0 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;

        out0 += M;
        out1 += M;
    }
}

void transposeM3(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 3;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *out0 = output;
    int16_t *out1 = out0 + SSE_16CAP;
    int16_t *out2 = out1 + SSE_16CAP;

    __m128i a, b, c, ab_lo, ab_hi, ab1, ab2, ab3, c1, c2, c3, ab2a, ab2b,
            mask1, mask2a, mask2b, mask3, cmask1, cmask2, cmask3, mix1, mix2, mix3;

    mask1 = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 0, 1, 8, 9, 10, 11);
    mask2a = _mm_setr_epi8(0, 1, 12, 13, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    mask2b = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 4, 5);
    mask3 = _mm_setr_epi8(6, 7, 0, 1, 8, 9, 10, 11, 0, 1, 12, 13, 14, 15, 0, 1);
    cmask1 = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1);
    cmask2 = _mm_setr_epi8(4, 5, 0, 1, 0, 1, 6, 7, 0, 1, 0, 1, 8, 9, 0, 1);
    cmask3 = _mm_setr_epi8(0, 1, 10, 11, 0, 1, 0, 1, 12, 13, 0, 1, 0, 1, 14, 15);

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);

        ab1 = _mm_shuffle_epi8(ab_lo, mask1);
        ab3 = _mm_shuffle_epi8(ab_hi, mask3);

        ab2a = _mm_shuffle_epi8(ab_lo, mask2a);
        ab2b = _mm_shuffle_epi8(ab_hi, mask2b);
        ab2 = _mm_blend_epi16(ab2a, ab2b, 240);

        c1 = _mm_shuffle_epi8(c, cmask1);
        c2 = _mm_shuffle_epi8(c, cmask2);
        c3 = _mm_shuffle_epi8(c, cmask3);

        mix1 = _mm_blend_epi16(ab1, c1, 36);
        mix2 = _mm_blend_epi16(ab2, c2, 73);
        mix3 = _mm_blend_epi16(ab3, c3, 146);

        _mm_stream_si128((__m128i*) out0, mix1);
        _mm_stream_si128((__m128i*) out1, mix2);
        _mm_stream_si128((__m128i*) out2, mix3);

        out0 += M * SSE_16CAP;
        out1 += M * SSE_16CAP;
        out2 += M * SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
    }

    out1 = out0 + 1;
    out2 = out1 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;

        out0 += M;
        out1 += M;
        out2 += M;
    }
}

void transposeM4(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 4;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;

    int16_t *out0 = output;
    int16_t *out1 = out0 + SSE_16CAP;
    int16_t *out2 = out1 + SSE_16CAP;
    int16_t *out3 = out2 + SSE_16CAP;

    __m128i a, b, c, d, ab_lo, ab_hi, cd_lo, cd_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;
        in3 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        _mm_stream_si128((__m128i*) out0, abcd_lo);
        _mm_stream_si128((__m128i*) out1, abcd_lohi);
        _mm_stream_si128((__m128i*) out2, abcd_hi);
        _mm_stream_si128((__m128i*) out3, abcd_hilo);

        out0 += M * SSE_16CAP;
        out1 += M * SSE_16CAP;
        out2 += M * SSE_16CAP;
        out3 += M * SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
    }

    out1 = out0 + 1;
    out2 = out1 + 1;
    out3 = out2 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;
        *out3 = *in3++;

        out0 += M;
        out1 += M;
        out2 += M;
        out3 += M;
    }
}

void transposeM5(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 5;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;
    int16_t *in4 = in3 + N;

    int16_t *out0 = output;

    __m128i a, b, c, d, e, ab_lo, ab_hi, cd_lo, cd_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);
    e = _mm_lddqu_si128((__m128i*)in4);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;
        in3 += SSE_16CAP;
        in4 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        _mm_storel_epi64((__m128i*)out0, abcd_lo);
        *(out0 + 4) = (int16_t)_mm_extract_epi16(e, 0);

        _mm_storel_epi64((__m128i*)(out0 + 5), _mm_srli_si128(abcd_lo, 8));
        *(out0 + 9) = (int16_t)_mm_extract_epi16(e, 1);

        _mm_storel_epi64((__m128i*)(out0 + 10), abcd_lohi);
        *(out0 + 14) = (int16_t)_mm_extract_epi16(e, 2);

        _mm_storel_epi64((__m128i*)(out0 + 15), _mm_srli_si128(abcd_lohi, 8));
        *(out0 + 19) = (int16_t)_mm_extract_epi16(e, 3);

        _mm_storel_epi64((__m128i*)(out0 + 20), abcd_hi);
        *(out0 + 24) = (int16_t)_mm_extract_epi16(e, 4);

        _mm_storel_epi64((__m128i*)(out0 + 25), _mm_srli_si128(abcd_hi, 8));
        *(out0 + 29) = (int16_t)_mm_extract_epi16(e, 5);

        _mm_storel_epi64((__m128i*)(out0 + 30), abcd_hilo);
        *(out0 + 34) = (int16_t)_mm_extract_epi16(e, 6);

        _mm_storel_epi64((__m128i*)(out0 + 35), _mm_srli_si128(abcd_hilo, 8));
        *(out0 + 39) = (int16_t)_mm_extract_epi16(e, 7);

        out0 += 40;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
    }

    int16_t *out1 = out0 + 1;
    int16_t *out2 = out1 + 1;
    int16_t *out3 = out2 + 1;
    int16_t *out4 = out3 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;
        *out3 = *in3++;
        *out4 = *in4++;

        out0 += M;
        out1 += M;
        out2 += M;
        out3 += M;
        out4 += M;
    }
}

void transposeM6(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 6;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;
    int16_t *in4 = in3 + N;
    int16_t *in5 = in4 + N;

    int16_t *out0 = output;

    __m128i a, b, c, d, e, f, ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);
    e = _mm_lddqu_si128((__m128i*)in4);
    f = _mm_lddqu_si128((__m128i*)in5);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;
        in3 += SSE_16CAP;
        in4 += SSE_16CAP;
        in5 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);
        ef_lo = _mm_unpacklo_epi16(e, f);
        ef_hi = _mm_unpackhi_epi16(e, f);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        _mm_storel_epi64((__m128i*)out0, abcd_lo);
        *(int32_t*)(out0 + 4) = _mm_extract_epi32(ef_lo, 0);

        _mm_storel_epi64((__m128i*)(out0 + 6), _mm_srli_si128(abcd_lo, 8));
        *(int32_t*)(out0 + 10) = _mm_extract_epi32(ef_lo, 1);

        _mm_storel_epi64((__m128i*)(out0 + 12), abcd_lohi);
        *(int32_t*)(out0 + 16) = _mm_extract_epi32(ef_lo, 2);

        _mm_storel_epi64((__m128i*)(out0 + 18), _mm_srli_si128(abcd_lohi, 8));
        *(int32_t*)(out0 + 22) = _mm_extract_epi32(ef_lo, 3);

        _mm_storel_epi64((__m128i*)(out0 + 24), abcd_hi);
        *(int32_t*)(out0 + 28) = _mm_extract_epi32(ef_hi, 0);

        _mm_storel_epi64((__m128i*)(out0 + 30), _mm_srli_si128(abcd_hi, 8));
        *(int32_t*)(out0 + 34) = _mm_extract_epi32(ef_hi, 1);

        _mm_storel_epi64((__m128i*)(out0 + 36), abcd_hilo);
        *(int32_t*)(out0 + 40) = _mm_extract_epi32(ef_hi, 2);

        _mm_storel_epi64((__m128i*)(out0 + 42), _mm_srli_si128(abcd_hilo, 8));
        *(int32_t*)(out0 + 46) = _mm_extract_epi32(ef_hi, 3);

        out0 += 48;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
    }

    int16_t *out1 = out0 + 1;
    int16_t *out2 = out1 + 1;
    int16_t *out3 = out2 + 1;
    int16_t *out4 = out3 + 1;
    int16_t *out5 = out4 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;
        *out3 = *in3++;
        *out4 = *in4++;
        *out5 = *in5++;

        out0 += M;
        out1 += M;
        out2 += M;
        out3 += M;
        out4 += M;
        out5 += M;
    }
}

void transposeM7(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 7;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;
    int16_t *in4 = in3 + N;
    int16_t *in5 = in4 + N;
    int16_t *in6 = in5 + N;

    int16_t *out0 = output;

    __m128i a, b, c, d, e, f, g, ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi,
            abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);
    e = _mm_lddqu_si128((__m128i*)in4);
    f = _mm_lddqu_si128((__m128i*)in5);
    g = _mm_lddqu_si128((__m128i*)in6);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;
        in3 += SSE_16CAP;
        in4 += SSE_16CAP;
        in5 += SSE_16CAP;
        in6 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);
        ef_lo = _mm_unpacklo_epi16(e, f);
        ef_hi = _mm_unpackhi_epi16(e, f);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        _mm_storel_epi64((__m128i*)out0, abcd_lo);
        *(int32_t*)(out0 + 4) = _mm_extract_epi32(ef_lo, 0);
        *(out0 + 6) = (int16_t)_mm_extract_epi16(g, 0);

        _mm_storel_epi64((__m128i*)(out0 + 7), _mm_srli_si128(abcd_lo, 8));
        *(int32_t*)(out0 + 11) = _mm_extract_epi32(ef_lo, 1);
        *(out0 + 13) = (int16_t)_mm_extract_epi16(g, 1);

        _mm_storel_epi64((__m128i*)(out0 + 14), abcd_lohi);
        *(int32_t*)(out0 + 18) = _mm_extract_epi32(ef_lo, 2);
        *(out0 + 20) = (int16_t)_mm_extract_epi16(g, 2);

        _mm_storel_epi64((__m128i*)(out0 + 21), _mm_srli_si128(abcd_lohi, 8));
        *(int32_t*)(out0 + 25) = _mm_extract_epi32(ef_lo, 3);
        *(out0 + 27) = (int16_t)_mm_extract_epi16(g, 3);

        _mm_storel_epi64((__m128i*)(out0 + 28), abcd_hi);
        *(int32_t*)(out0 + 32) = _mm_extract_epi32(ef_hi, 0);
        *(out0 + 34) = (int16_t)_mm_extract_epi16(g, 4);

        _mm_storel_epi64((__m128i*)(out0 + 35), _mm_srli_si128(abcd_hi, 8));
        *(int32_t*)(out0 + 39) = _mm_extract_epi32(ef_hi, 1);
        *(out0 + 41) = (int16_t)_mm_extract_epi16(g, 5);

        _mm_storel_epi64((__m128i*)(out0 + 42), abcd_hilo);
        *(int32_t*)(out0 + 46) = _mm_extract_epi32(ef_hi, 2);
        *(out0 + 48) = (int16_t)_mm_extract_epi16(g, 6);

        _mm_storel_epi64((__m128i*)(out0 + 49), _mm_srli_si128(abcd_hilo, 8));
        *(int32_t*)(out0 + 53) = _mm_extract_epi32(ef_hi, 3);
        *(out0 + 55) = (int16_t)_mm_extract_epi16(g, 7);

        out0 += 56;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
    }

    int16_t *out1 = out0 + 1;
    int16_t *out2 = out1 + 1;
    int16_t *out3 = out2 + 1;
    int16_t *out4 = out3 + 1;
    int16_t *out5 = out4 + 1;
    int16_t *out6 = out5 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;
        *out3 = *in3++;
        *out4 = *in4++;
        *out5 = *in5++;
        *out6 = *in6++;

        out0 += M;
        out1 += M;
        out2 += M;
        out3 += M;
        out4 += M;
        out5 += M;
        out6 += M;
    }
}

void transposeM8(int16_t *input, int16_t *output, uint32_t N)
{
    uint32_t M = 8;
    uint32_t N_VEC = N - N % VEC_16CAP;
    int16_t *input_end = input + N_VEC;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;
    int16_t *in4 = in3 + N;
    int16_t *in5 = in4 + N;
    int16_t *in6 = in5 + N;
    int16_t *in7 = in6 + N;

    int16_t *out0 = output;
    int16_t *out1 = out0 + SSE_16CAP;
    int16_t *out2 = out1 + SSE_16CAP;
    int16_t *out3 = out2 + SSE_16CAP;
    int16_t *out4 = out3 + SSE_16CAP;
    int16_t *out5 = out4 + SSE_16CAP;
    int16_t *out6 = out5 + SSE_16CAP;
    int16_t *out7 = out6 + SSE_16CAP;

    __m128i a, b, c, d, e, f, g, h;
    __m128i ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi, gh_lo, gh_hi;
    __m128i abcd_lo, abcd_lohi, abcd_hi, abcd_hilo;
    __m128i efgh_lo, efgh_lohi, efgh_hi, efgh_hilo;
    __m128i pack1, pack2, pack3, pack4, pack5, pack6, pack7, pack8;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);
    e = _mm_lddqu_si128((__m128i*)in4);
    f = _mm_lddqu_si128((__m128i*)in5);
    g = _mm_lddqu_si128((__m128i*)in6);
    h = _mm_lddqu_si128((__m128i*)in7);

    for (; input < input_end;)
    {
        input += SSE_16CAP;
        in1 += SSE_16CAP;
        in2 += SSE_16CAP;
        in3 += SSE_16CAP;
        in4 += SSE_16CAP;
        in5 += SSE_16CAP;
        in6 += SSE_16CAP;
        in7 += SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);
        ef_lo = _mm_unpacklo_epi16(e, f);
        ef_hi = _mm_unpackhi_epi16(e, f);
        gh_lo = _mm_unpacklo_epi16(g, h);
        gh_hi = _mm_unpackhi_epi16(g, h);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        efgh_lo = _mm_unpacklo_epi32(ef_lo, gh_lo);
        efgh_lohi = _mm_unpackhi_epi32(ef_lo, gh_lo);
        efgh_hi = _mm_unpacklo_epi32(ef_hi, gh_hi);
        efgh_hilo = _mm_unpackhi_epi32(ef_hi, gh_hi);

        pack1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
        pack2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
        pack3 = _mm_unpacklo_epi64(abcd_lohi, efgh_lohi);
        pack4 = _mm_unpackhi_epi64(abcd_lohi, efgh_lohi);

        pack5 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
        pack6 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);
        pack7 = _mm_unpacklo_epi64(abcd_hilo, efgh_hilo);
        pack8 = _mm_unpackhi_epi64(abcd_hilo, efgh_hilo);

        _mm_stream_si128((__m128i*) out0, pack1);
        _mm_stream_si128((__m128i*) out1, pack2);
        _mm_stream_si128((__m128i*) out2, pack3);
        _mm_stream_si128((__m128i*) out3, pack4);
        _mm_stream_si128((__m128i*) out4, pack5);
        _mm_stream_si128((__m128i*) out5, pack6);
        _mm_stream_si128((__m128i*) out6, pack7);
        _mm_stream_si128((__m128i*) out7, pack8);

        out0 += M * SSE_16CAP;
        out1 += M * SSE_16CAP;
        out2 += M * SSE_16CAP;
        out3 += M * SSE_16CAP;
        out4 += M * SSE_16CAP;
        out5 += M * SSE_16CAP;
        out6 += M * SSE_16CAP;
        out7 += M * SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
        h = _mm_lddqu_si128((__m128i*)in7);
    }

    out1 = out0 + 1;
    out2 = out1 + 1;
    out3 = out2 + 1;
    out4 = out3 + 1;
    out5 = out4 + 1;
    out6 = out5 + 1;
    out7 = out6 + 1;
    for (uint32_t i = N_VEC; i < N; i++)
    {
        *out0 = *input++;
        *out1 = *in1++;
        *out2 = *in2++;
        *out3 = *in3++;
        *out4 = *in4++;
        *out5 = *in5++;
        *out6 = *in6++;
        *out7 = *in7++;

        out0 += M;
        out1 += M;
        out2 += M;
        out3 += M;
        out4 += M;
        out5 += M;
        out6 += M;
        out7 += M;
    }
}


void transposeN2(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 2;
    int16_t *in1 = input + SSE_16CAP;
    uint32_t end = M - M % SSE_16CAP;
    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    uint32_t i;

    __m128i ad, eh, ah1, ah2, mask;

    ad = _mm_lddqu_si128((__m128i*)input);
    eh = _mm_lddqu_si128((__m128i*)in1);
    mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

    for (i = 0; i < end; i += SSE_16CAP)
    {
        input += N * SSE_16CAP;
        in1 += N * SSE_16CAP;

        ad = _mm_shuffle_epi8(ad, mask);
        eh = _mm_shuffle_epi8(eh, mask);

        ah1 = _mm_unpacklo_epi64(ad, eh);
        ah2 = _mm_unpackhi_epi64(ad, eh);

        _mm_storeu_si128((__m128i*) out0, ah1);
        _mm_storeu_si128((__m128i*) out1, ah2);

        out0 += SSE_16CAP;
        out1 += SSE_16CAP;

        ad = _mm_lddqu_si128((__m128i*)input);
        eh = _mm_lddqu_si128((__m128i*)in1);
    }
    for (; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }
}

void transposeN3(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 3;
    uint32_t M_VEC = M - M % VEC_16CAP;
    int16_t *in1 = input + SSE_16CAP;
    int16_t *in2 = in1 + SSE_16CAP;

    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    int16_t *out2 = out1 + M;

    uint32_t i;

    __m128i mix1, mix2, mix3, t1, t2, t3, mask1, mask2, mask3;
    mask1 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    mask2 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    mask3 = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

    mix1 = _mm_lddqu_si128((__m128i*)input);
    mix2 = _mm_lddqu_si128((__m128i*)in1);
    mix3 = _mm_lddqu_si128((__m128i*)in2);

    for (i = 0; i < M_VEC; i += SSE_16CAP)
    {
        input += N * SSE_16CAP;
        in1 += N * SSE_16CAP;
        in2 += N * SSE_16CAP;

        t1 = _mm_blend_epi16(mix1, mix2, 146);
        t1 = _mm_blend_epi16(t1, mix3, 36);
        t1 = _mm_shuffle_epi8(t1, mask1);

        t2 = _mm_blend_epi16(mix1, mix2, 36);
        t2 = _mm_blend_epi16(t2, mix3, 73);
        t2 = _mm_shuffle_epi8(t2, mask2);

        t3 = _mm_blend_epi16(mix1, mix2, 73);
        t3 = _mm_blend_epi16(t3, mix3, 146);
        t3 = _mm_shuffle_epi8(t3, mask3);

        _mm_storeu_si128((__m128i*) out0, t1);
        _mm_storeu_si128((__m128i*) out1, t2);
        _mm_storeu_si128((__m128i*) out2, t3);

        out0 += SSE_16CAP;
        out1 += SSE_16CAP;
        out2 += SSE_16CAP;

        mix1 = _mm_lddqu_si128((__m128i*)input);
        mix2 = _mm_lddqu_si128((__m128i*)in1);
        mix3 = _mm_lddqu_si128((__m128i*)in2);
    }
    for (; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }
}

void transposeN4(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 4;
    uint32_t end = M - M % SSE_16CAP;
    int16_t *in1 = input + SSE_16CAP;
    int16_t *in2 = in1 + SSE_16CAP;
    int16_t *in3 = in2 + SSE_16CAP;

    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    int16_t *out2 = out1 + M;
    int16_t *out3 = out2 + M;

    uint32_t i;

    __m128i ab, cd, ef, gh, ac, bd, eg, fh;
    __m128i abcd_lo, abcd_hi, efgh_lo, efgh_hi;
    __m128i mask = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
    __m128i full1, full2, full3, full4;

    ab = _mm_lddqu_si128((__m128i*)input);
    cd = _mm_lddqu_si128((__m128i*)in1);
    ef = _mm_lddqu_si128((__m128i*)in2);
    gh = _mm_lddqu_si128((__m128i*)in3);

    for (i = 0; i < end; i += SSE_16CAP)
    {
        input += N * SSE_16CAP;
        in1 += N * SSE_16CAP;
        in2 += N * SSE_16CAP;
        in3 += N * SSE_16CAP;

        ac = _mm_unpacklo_epi16(ab, cd);
        bd = _mm_unpackhi_epi16(ab, cd);
        eg = _mm_unpacklo_epi16(ef, gh);
        fh = _mm_unpackhi_epi16(ef, gh);

        abcd_lo = _mm_unpacklo_epi32(ac, bd);
        abcd_hi = _mm_unpackhi_epi32(ac, bd);

        efgh_lo = _mm_unpacklo_epi32(eg, fh);
        efgh_hi = _mm_unpackhi_epi32(eg, fh);

        full1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
        full2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
        full3 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
        full4 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);

        full1 = _mm_shuffle_epi8(full1, mask);
        full2 = _mm_shuffle_epi8(full2, mask);
        full3 = _mm_shuffle_epi8(full3, mask);
        full4 = _mm_shuffle_epi8(full4, mask);

        _mm_storeu_si128((__m128i*) out0, full1);
        _mm_storeu_si128((__m128i*) out1, full2);
        _mm_storeu_si128((__m128i*) out2, full3);
        _mm_storeu_si128((__m128i*) out3, full4);

        out0 += SSE_16CAP;
        out1 += SSE_16CAP;
        out2 += SSE_16CAP;
        out3 += SSE_16CAP;

        ab = _mm_lddqu_si128((__m128i*)input);
        cd = _mm_lddqu_si128((__m128i*)in1);
        ef = _mm_lddqu_si128((__m128i*)in2);
        gh = _mm_lddqu_si128((__m128i*)in3);
    }
    for (; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }
}

void transposeN5(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 5;
    uint32_t end = M - M % SSE_16CAP;
    int16_t *in1 = input + SSE_16CAP;
    int16_t *in2 = in1 + SSE_16CAP;
    int16_t *in3 = in2 + SSE_16CAP;
    int16_t *in4 = in3 + SSE_16CAP;

    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    int16_t *out2 = out1 + M;
    int16_t *out3 = out2 + M;
    int16_t *out4 = out3 + M;

    uint32_t i;

    __m128i a, b, c, d, e;
    __m128i v1, v2, v3, v4, v5;
    __m128i mask1, mask2, mask3, mask4, mask5;

    mask1 = _mm_setr_epi8(0, 1, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7);
    mask2 = _mm_setr_epi8(2, 3, 12, 13, 6, 7, 0, 1, 10, 11, 4, 5, 14, 15, 8, 9);
    mask3 = _mm_setr_epi8(4, 5, 14, 15, 8, 9, 2, 3, 12, 13, 6, 7, 0, 1, 10, 11);
    mask4 = _mm_setr_epi8(6, 7, 0, 1, 10, 11, 4, 5, 14, 15, 8, 9, 2, 3, 12, 13);
    mask5 = _mm_setr_epi8(8, 9, 2, 3, 12, 13, 6, 7, 0, 1, 10, 11, 4, 5, 14, 15);

    a = _mm_load_si128((__m128i*)input);
    b = _mm_load_si128((__m128i*)in1);
    c = _mm_load_si128((__m128i*)in2);
    d = _mm_load_si128((__m128i*)in3);
    e = _mm_load_si128((__m128i*)in4);

    for (i = 0; i < end; i += SSE_16CAP)
    {
        input += N * SSE_16CAP;
        in1 += N * SSE_16CAP;
        in2 += N * SSE_16CAP;
        in3 += N * SSE_16CAP;
        in4 += N * SSE_16CAP;

        v1 = _mm_blend_epi16(a, b, 132);
        v1 = _mm_blend_epi16(v1, c, 16);
        v1 = _mm_blend_epi16(v1, d, 66);
        v1 = _mm_blend_epi16(v1, e, 8);
        v1 = _mm_shuffle_epi8(v1, mask1);

        v2 = _mm_blend_epi16(a, b, 8);
        v2 = _mm_blend_epi16(v2, c, 33);
        v2 = _mm_blend_epi16(v2, d, 132);
        v2 = _mm_blend_epi16(v2, e, 16);
        v2 = _mm_shuffle_epi8(v2, mask2);

        v3 = _mm_blend_epi16(a, b, 16);
        v3 = _mm_blend_epi16(v3, c, 66);
        v3 = _mm_blend_epi16(v3, d, 8);
        v3 = _mm_blend_epi16(v3, e, 33);
        v3 = _mm_shuffle_epi8(v3, mask3);

        v4 = _mm_blend_epi16(a, b, 33);
        v4 = _mm_blend_epi16(v4, c, 132);
        v4 = _mm_blend_epi16(v4, d, 16);
        v4 = _mm_blend_epi16(v4, e, 66);
        v4 = _mm_shuffle_epi8(v4, mask4);

        v5 = _mm_blend_epi16(a, b, 66);
        v5 = _mm_blend_epi16(v5, c, 8);
        v5 = _mm_blend_epi16(v5, d, 33);
        v5 = _mm_blend_epi16(v5, e, 132);
        v5 = _mm_shuffle_epi8(v5, mask5);

        _mm_storeu_si128((__m128i*) out0, v1);
        _mm_storeu_si128((__m128i*) out1, v2);
        _mm_storeu_si128((__m128i*) out2, v3);
        _mm_storeu_si128((__m128i*) out3, v4);
        _mm_storeu_si128((__m128i*) out4, v5);

        out0 += SSE_16CAP;
        out1 += SSE_16CAP;
        out2 += SSE_16CAP;
        out3 += SSE_16CAP;
        out4 += SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
    }

    for (; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }

}

void transposeN6(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 6;
    uint32_t end = M - M % 4;
    int16_t *in1 = input + SSE_16CAP;
    int16_t *in2 = in1 + SSE_16CAP;
    int16_t *input_end = input + end * N;

    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    int16_t *out2 = out1 + M;
    int16_t *out3 = out2 + M;
    int16_t *out4 = out3 + M;
    int16_t *out5 = out4 + M;

    __m128i ab, bc, cd;
    __m128i mix1, mix2, mix3;
    __m128i mask1, mask2, mask3;

    ab = _mm_lddqu_si128((__m128i*)input);
    bc = _mm_lddqu_si128((__m128i*)in1);
    cd = _mm_lddqu_si128((__m128i*)in2);

    mask1 = _mm_setr_epi8(0, 1, 12, 13, 8, 9, 4, 5, 2, 3, 14, 15, 10, 11, 6, 7);
    mask2 = _mm_setr_epi8(4, 5, 0, 1, 12, 13, 8, 9, 6, 7, 2, 3, 14, 15, 10, 11);
    mask3 = _mm_setr_epi8(8, 9, 4, 5, 0, 1, 12, 13, 10, 11, 6, 7, 2, 3, 14, 15);

    for (; input < input_end;)
    {
        input += N * 4;
        in1 += N * 4;
        in2 += N * 4;

        mix1 = _mm_blend_epi16(ab, bc, 48);
        mix1 = _mm_blend_epi16(mix1, cd, 12);
        mix1 = _mm_shuffle_epi8(mix1, mask1);

        mix2 = _mm_blend_epi16(ab, bc, 195);
        mix2 = _mm_blend_epi16(mix2, cd, 48);
        mix2 = _mm_shuffle_epi8(mix2, mask2);

        mix3 = _mm_blend_epi16(ab, bc, 12);
        mix3 = _mm_blend_epi16(mix3, cd, 195);
        mix3 = _mm_shuffle_epi8(mix3, mask3);

        _mm_storel_epi64((__m128i*)out0, mix1);
        _mm_storel_epi64((__m128i*)out1, _mm_srli_si128(mix1, 8));
        _mm_storel_epi64((__m128i*)out2, mix2);
        _mm_storel_epi64((__m128i*)out3, _mm_srli_si128(mix2, 8));
        _mm_storel_epi64((__m128i*)out4, mix3);
        _mm_storel_epi64((__m128i*)out5, _mm_srli_si128(mix3, 8));

        out0 += 4;
        out1 += 4;
        out2 += 4;
        out3 += 4;
        out4 += 4;
        out5 += 4;

        ab = _mm_lddqu_si128((__m128i*)input);
        bc = _mm_lddqu_si128((__m128i*)in1);
        cd = _mm_lddqu_si128((__m128i*)in2);
    }

    for (uint32_t i = end; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }
}

void transposeN7(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 7;
    uint32_t end = M - M % SSE_16CAP;
    int16_t *in1 = input + N;
    int16_t *in2 = in1 + N;
    int16_t *in3 = in2 + N;
    int16_t *in4 = in3 + N;
    int16_t *in5 = in4 + N;
    int16_t *in6 = in5 + N;
    int16_t *in7 = in6 + N;
    int16_t *input_end = input + end * N;

    int16_t *out0 = output;
    int16_t *out1 = out0 + M;
    int16_t *out2 = out1 + M;
    int16_t *out3 = out2 + M;
    int16_t *out4 = out3 + M;
    int16_t *out5 = out4 + M;
    int16_t *out6 = out5 + M;

    __m128i a, b, c, d, e, f, g, h;
    __m128i ab_lo, ab_hi, cd_lo, cd_hi, ef_lo, ef_hi, gh_lo, gh_hi;
    __m128i abcd_lo, abcd_hi, abcd_lohi, abcd_hilo, efgh_lo, efgh_lohi, efgh_hi, efgh_hilo;
    __m128i full1, full2, full3, full4, full5, full6, full7;

    a = _mm_lddqu_si128((__m128i*)input);
    b = _mm_lddqu_si128((__m128i*)in1);
    c = _mm_lddqu_si128((__m128i*)in2);
    d = _mm_lddqu_si128((__m128i*)in3);
    e = _mm_lddqu_si128((__m128i*)in4);
    f = _mm_lddqu_si128((__m128i*)in5);
    g = _mm_lddqu_si128((__m128i*)in6);
    h = _mm_lddqu_si128((__m128i*)in7);

    for (; input < input_end;)
    {
        input += N * SSE_16CAP;
        in1 += N * SSE_16CAP;
        in2 += N * SSE_16CAP;
        in3 += N * SSE_16CAP;
        in4 += N * SSE_16CAP;
        in5 += N * SSE_16CAP;
        in6 += N * SSE_16CAP;
        in7 += N * SSE_16CAP;

        ab_lo = _mm_unpacklo_epi16(a, b);
        ab_hi = _mm_unpackhi_epi16(a, b);
        cd_lo = _mm_unpacklo_epi16(c, d);
        cd_hi = _mm_unpackhi_epi16(c, d);
        ef_lo = _mm_unpacklo_epi16(e, f);
        ef_hi = _mm_unpackhi_epi16(e, f);
        gh_lo = _mm_unpacklo_epi16(g, h);
        gh_hi = _mm_unpackhi_epi16(g, h);

        abcd_lo = _mm_unpacklo_epi32(ab_lo, cd_lo);
        abcd_lohi = _mm_unpackhi_epi32(ab_lo, cd_lo);
        abcd_hi = _mm_unpacklo_epi32(ab_hi, cd_hi);
        abcd_hilo = _mm_unpackhi_epi32(ab_hi, cd_hi);

        efgh_lo = _mm_unpacklo_epi32(ef_lo, gh_lo);
        efgh_lohi = _mm_unpackhi_epi32(ef_lo, gh_lo);
        efgh_hi = _mm_unpacklo_epi32(ef_hi, gh_hi);
        efgh_hilo = _mm_unpackhi_epi32(ef_hi, gh_hi);

        full1 = _mm_unpacklo_epi64(abcd_lo, efgh_lo);
        full2 = _mm_unpackhi_epi64(abcd_lo, efgh_lo);
        full3 = _mm_unpacklo_epi64(abcd_lohi, efgh_lohi);
        full4 = _mm_unpackhi_epi64(abcd_lohi, efgh_lohi);
        full5 = _mm_unpacklo_epi64(abcd_hi, efgh_hi);
        full6 = _mm_unpackhi_epi64(abcd_hi, efgh_hi);
        full7 = _mm_unpacklo_epi64(abcd_hilo, efgh_hilo);

        _mm_storeu_si128((__m128i*)out0, full1);
        _mm_storeu_si128((__m128i*)out1, full2);
        _mm_storeu_si128((__m128i*)out2, full3);
        _mm_storeu_si128((__m128i*)out3, full4);
        _mm_storeu_si128((__m128i*)out4, full5);
        _mm_storeu_si128((__m128i*)out5, full6);
        _mm_storeu_si128((__m128i*)out6, full7);

        out0 += SSE_16CAP;
        out1 += SSE_16CAP;
        out2 += SSE_16CAP;
        out3 += SSE_16CAP;
        out4 += SSE_16CAP;
        out5 += SSE_16CAP;
        out6 += SSE_16CAP;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
        h = _mm_lddqu_si128((__m128i*)in7);
    }
    for (uint32_t i = end; i < M; i++)
    {
        out0 = output + i;
        for (uint32_t j = 0; j < N; j++)
        {
            *out0 = *input++;
            out0 += M;
        }
    }
}

void transposeN8(int16_t *input, int16_t *output, uint32_t M)
{
    uint32_t N = 8;
    uint32_t M_VEC = M - M % VEC_16CAP;
    uint32_t N_VEC = N - N % VEC_16CAP;
    __m128i a, b, c, d, e, f, g, h;
    __m128i ab_lo, cd_lo, ef_lo, gh_lo;
    __m128i ab_hi, cd_hi, ef_hi, gh_hi;
    __m128i abcd12, abcd34, abcd56, abcd78;
    __m128i efgh12, efgh34, efgh56, efgh78;
    __m128i abcdefgh1, abcdefgh2, abcdefgh3, abcdefgh4;
    __m128i abcdefgh5, abcdefgh6, abcdefgh7, abcdefgh8;

    int16_t *in1;
    int16_t *in2;
    int16_t *in3;
    int16_t *in4;
    int16_t *in5;
    int16_t *in6;
    int16_t *in7;
    int16_t *input_end = input + M_VEC * N;

    int16_t *out0 = output;
    int16_t *out1;
    int16_t *out2;
    int16_t *out3;
    int16_t *out4;
    int16_t *out5;
    int16_t *out6;
    int16_t *out7;

    for (; input < input_end;)
    {
        in1 = input + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        in5 = in4 + N;
        in6 = in5 + N;
        in7 = in6 + N;

        a = _mm_lddqu_si128((__m128i*)input);
        b = _mm_lddqu_si128((__m128i*)in1);
        c = _mm_lddqu_si128((__m128i*)in2);
        d = _mm_lddqu_si128((__m128i*)in3);
        e = _mm_lddqu_si128((__m128i*)in4);
        f = _mm_lddqu_si128((__m128i*)in5);
        g = _mm_lddqu_si128((__m128i*)in6);
        h = _mm_lddqu_si128((__m128i*)in7);

        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;
        out5 = out4 + M;
        out6 = out5 + M;
        out7 = out6 + M;

        int16_t *col_end = input + N_VEC;
        for (; input < col_end;)
        {
            input += SSE_16CAP;
            in1 += SSE_16CAP;
            in2 += SSE_16CAP;
            in3 += SSE_16CAP;
            in4 += SSE_16CAP;
            in5 += SSE_16CAP;
            in6 += SSE_16CAP;
            in7 += SSE_16CAP;

            ab_lo = _mm_unpacklo_epi16(a, b);
            cd_lo = _mm_unpacklo_epi16(c, d);
            ef_lo = _mm_unpacklo_epi16(e, f);
            gh_lo = _mm_unpacklo_epi16(g, h);

            ab_hi = _mm_unpackhi_epi16(a, b);
            cd_hi = _mm_unpackhi_epi16(c, d);
            ef_hi = _mm_unpackhi_epi16(e, f);
            gh_hi = _mm_unpackhi_epi16(g, h);

            abcd12 = _mm_unpacklo_epi32(ab_lo, cd_lo);
            abcd34 = _mm_unpackhi_epi32(ab_lo, cd_lo);
            abcd56 = _mm_unpacklo_epi32(ab_hi, cd_hi);
            abcd78 = _mm_unpackhi_epi32(ab_hi, cd_hi);

            efgh12 = _mm_unpacklo_epi32(ef_lo, gh_lo);
            efgh34 = _mm_unpackhi_epi32(ef_lo, gh_lo);
            efgh56 = _mm_unpacklo_epi32(ef_hi, gh_hi);
            efgh78 = _mm_unpackhi_epi32(ef_hi, gh_hi);

            abcdefgh1 = _mm_unpacklo_epi64(abcd12, efgh12);
            abcdefgh2 = _mm_unpackhi_epi64(abcd12, efgh12);
            abcdefgh3 = _mm_unpacklo_epi64(abcd34, efgh34);
            abcdefgh4 = _mm_unpackhi_epi64(abcd34, efgh34);
            abcdefgh5 = _mm_unpacklo_epi64(abcd56, efgh56);
            abcdefgh6 = _mm_unpackhi_epi64(abcd56, efgh56);
            abcdefgh7 = _mm_unpacklo_epi64(abcd78, efgh78);
            abcdefgh8 = _mm_unpackhi_epi64(abcd78, efgh78);

            _mm_storeu_si128((__m128i*)out0, abcdefgh1);
            _mm_storeu_si128((__m128i*)out1, abcdefgh2);
            _mm_storeu_si128((__m128i*)out2, abcdefgh3);
            _mm_storeu_si128((__m128i*)out3, abcdefgh4);
            _mm_storeu_si128((__m128i*)out4, abcdefgh5);
            _mm_storeu_si128((__m128i*)out5, abcdefgh6);
            _mm_storeu_si128((__m128i*)out6, abcdefgh7);
            _mm_storeu_si128((__m128i*)out7, abcdefgh8);

            out0 += M * 8;
            out1 += M * 8;
            out2 += M * 8;
            out3 += M * 8;
            out4 += M * 8;
            out5 += M * 8;
            out6 += M * 8;
            out7 += M * 8;

            a = _mm_lddqu_si128((__m128i*)input);
            b = _mm_lddqu_si128((__m128i*)in1);
            c = _mm_lddqu_si128((__m128i*)in2);
            d = _mm_lddqu_si128((__m128i*)in3);
            e = _mm_lddqu_si128((__m128i*)in4);
            f = _mm_lddqu_si128((__m128i*)in5);
            g = _mm_lddqu_si128((__m128i*)in6);
            h = _mm_lddqu_si128((__m128i*)in7);
        }

        input = input - N_VEC + SSE_16CAP * N;
        out0 = out0 - N_VEC * M + SSE_16CAP;
    }

    for (uint32_t i = M_VEC; i < M; i++)
    {
        for (uint32_t j = 0; j < N_VEC; j++)
        {
            output[j * M + i] = input[i * N + j];
        }

    }

    for (uint32_t i = N_VEC; i < N; i++)
    {
        for (uint32_t j = 0; j < M; j++)
        {
            output[i * M + j] = input[j * N + i];
        }
    }
}

