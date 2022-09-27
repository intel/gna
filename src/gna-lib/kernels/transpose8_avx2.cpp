/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Macros.h"
#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

static void transposeM2(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM3(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM4(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM5(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM6(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM7(const int8_t *input, int8_t *output, uint32_t N);
static void transposeM8(const int8_t *input, int8_t *output, uint32_t N);

static void transposeN2(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN3(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN4(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN5(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN6(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN7(const int8_t *input, int8_t *output, uint32_t M);
static void transposeN8(const int8_t *input, int8_t *output, uint32_t M);

/** @brief Single iteration size for transposeN* functions */
#define N_ITERATION_SIZE (16)

/** @brief Single iteration size for transposeM* functions */
#define M_ITERATION_SIZE (32)

/** @brief Helper to compute imm8 register for permutation */
#define PERMUTE(_A, _B, _C, _D) (((_A) << 0) | ((_B) << 2) | ((_C) << 4) | ((_D) << 6))

/** @brief Helper for most commonly used permutation */
#define PERMUTE_0213 PERMUTE(0, 2, 1, 3)

/** @brief Helper to compute imm8 register for 32-bit mask */
#define MASK_32(_A, _B, _C, _D) \
    (((_A * 3) << 0) | ((_B * 3) << 2) | ((_C * 3) << 4) | ((_D * 3) << 6))

/** @brief Helper to compute imm8 register for 16-bit mask */
#define MASK_16(_A, _B, _C, _D, _E, _F, _G, _H) \
    ((_A << 0) | (_B << 1) | (_C << 2) | (_D << 3) | ((_E << 4) | (_F << 5) | (_G << 6) | (_H << 7)))

/** @brief Main function
 *
 *  Transposition is implemented for:
 *  - N [1,8] M[16, 2^16 - 16, %16]
 *  - M [1,8] N[16, 2^16 - 16, %16]
 */
void TransposeKernelImpl1B(TransposeConfig const *const transposeConfig)
{
    uint32_t M = transposeConfig->rowCount;
    uint32_t N = transposeConfig->columnCount;

    const int8_t *const I = reinterpret_cast<const int8_t *>(transposeConfig->input);
    int8_t *const O = reinterpret_cast<int8_t *>(transposeConfig->output);

    // input matrix is a vector - copy
    if (M == 1 || N == 1)
    {
        memmove_s(O, M * N * sizeof(int8_t), I, M * N * sizeof(int8_t));
        return;
    }

    // very small matrix - generic tranpose
    // or N, M between 8 and 16 for large matrices - not implemented
    if (M * N < N_ITERATION_SIZE * N_ITERATION_SIZE || (N > 8 && N < 16) || (M > 8 && M < 16))
    {
        uint32_t i;
        uint32_t j;
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < N; j++)
            {
                O[j * M + i] = I[i * N + j];
            }
        }

        return;
    }

    // INTERLEAVE
    // MAX M is 8, MAX N is UINT16_MAX
    if (N >= 8)
    {
        switch (M)
        {
        case 2:
            transposeM2(I, O, N);
            break;
        case 3:
            transposeM3(I, O, N);
            break;
        case 4:
            transposeM4(I, O, N);
            break;
        case 5:
            transposeM5(I, O, N);
            break;
        case 6:
            transposeM6(I, O, N);
            break;
        case 7:
            transposeM7(I, O, N);
            break;
        case 8:
            transposeM8(I, O, N);
            break;
        default:
            break;
        }
    }

    // DEINTERLEAVE
    // MAX N is 8, MAX M is UINT16_MAX
    switch (N)
    {
    case 2:
        transposeN2(I, O, M);
        break;
    case 3:
        transposeN3(I, O, M);
        break;
    case 4:
        transposeN4(I, O, M);
        break;
    case 5:
        transposeN5(I, O, M);
        break;
    case 6:
        transposeN6(I, O, M);
        break;
    case 7:
        transposeN7(I, O, M);
        break;
    case 8:
        transposeN8(I, O, M);
        break;
    }
}

/** @brief Helper to copy end of the data for transposeM* functions
 *
 *  Since transposeM* functions deal with data in iterations of size M_ITERATION_SIZE,
 *  we have to manually copy the remaining M * (N % M_ITERATION_SIZE) bytes
 *
 *  @param M Column count
 *  @param N Row count
 *  @param input Pointer to the first byte to copy from
 *  @param output Pointer to the first byte to write to
 */
inline void CopyTailM(uint32_t M, uint32_t N, const int8_t *input, int8_t *output)
{
    for (uint32_t i = 0; i < N % 32; ++i)
    {
        for (uint32_t j = 0; j < M; ++j)
        {
            output[i * M + j] = input[j * N + i];
        }
    }
}

void transposeM2(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 2;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;

    int8_t *out0;
    int8_t *out1;

    __m256i data0;
    __m256i data1;

    __m256i unpack0;
    __m256i unpack1;

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        out0 = output;
        out1 = out0 + 32;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);

        _mm256_storeu_si256((__m256i *)out0, unpack0);
        _mm256_storeu_si256((__m256i *)out1, unpack1);
    }

    CopyTailM(M, N, input, output);
}

void transposeM3(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 3;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3 = _mm256_setzero_si256();

    __m256i unpack0;
    __m256i unpack1;
    __m256i unpack2;
    __m256i unpack3;

    __m128i store0;
    __m128i store1;
    __m128i store2;
    __m128i store3;
    __m128i store4;
    __m128i store5;
    __m128i store6;
    __m128i store7;

    // NOTE: We have interleaved elements from rows 1 2 3 and 4th element is "dummy", as in: 123_123_123_123_ ...
    //       We change it to: 123123123123____ ____123123123123 so we can perform 128b write and 96b write
    __m256i shuffle_together = _mm256_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, //
                                                -1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14);

    /** @brief Mask used to skip first 4 bytes of 2nd write */
    __m128i store_mask = _mm_setr_epi32(0, -1, -1, -1);

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;

        // NOTE: Our outputs are 24 bytes apart, instead of 32, because we write 4x(12 + 12) bytes
        //       This still fills 3x32 bytes, which is our target for each iteration
        out0 = output;
        out1 = out0 + 24;
        out2 = out1 + 24;
        out3 = out2 + 24;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);

        data0 = _mm256_shuffle_epi8(unpack0, shuffle_together);
        data1 = _mm256_shuffle_epi8(unpack1, shuffle_together);
        data2 = _mm256_shuffle_epi8(unpack2, shuffle_together);
        data3 = _mm256_shuffle_epi8(unpack3, shuffle_together);

        store0 = _mm256_castsi256_si128(data0);
        store1 = _mm256_castsi256_si128(data2);
        store2 = _mm256_castsi256_si128(data1);
        store3 = _mm256_castsi256_si128(data3);

        store4 = _mm256_extracti128_si256(data0, 1);
        store5 = _mm256_extracti128_si256(data2, 1);
        store6 = _mm256_extracti128_si256(data1, 1);
        store7 = _mm256_extracti128_si256(data3, 1);

        _mm_storeu_si128((__m128i *)(out0), store0);
        _mm_storeu_si128((__m128i *)(out1), store1);
        _mm_storeu_si128((__m128i *)(out2), store2);
        _mm_storeu_si128((__m128i *)(out3), store3);

        _mm_maskstore_epi32((int *)(out0 + 8), store_mask, store4);
        _mm_maskstore_epi32((int *)(out1 + 8), store_mask, store5);
        _mm_maskstore_epi32((int *)(out2 + 8), store_mask, store6);
        _mm_maskstore_epi32((int *)(out3 + 8), store_mask, store7);
    }

    CopyTailM(M, N, input, output);
}

void transposeM4(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 4;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;
    const int8_t *in3;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3;

    __m256i unpack0;
    __m256i unpack1;
    __m256i unpack2;
    __m256i unpack3;

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;

        out0 = output;
        out1 = out0 + 32;
        out2 = out1 + 32;
        out3 = out2 + 32;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);
        data3 = _mm256_lddqu_si256((__m256i *)in3);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(data2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(data3, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);

        _mm256_storeu_si256((__m256i *)out0, unpack0);
        _mm256_storeu_si256((__m256i *)out1, unpack1);
        _mm256_storeu_si256((__m256i *)out2, unpack2);
        _mm256_storeu_si256((__m256i *)out3, unpack3);
    }

    CopyTailM(M, N, input, output);
}

void transposeM5(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 5;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;
    const int8_t *in3;
    const int8_t *in4;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;
    int8_t *out7;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3;
    __m256i data4;
    __m256i data5 = _mm256_setzero_si256();
    __m256i data6 = _mm256_setzero_si256();
    __m256i data7 = _mm256_setzero_si256();

    __m256i unpack0, unpack1, unpack2, unpack3, unpack4, unpack5, unpack6, unpack7;

    __m128i store0, store1, store2, store3, store4, store5, store6, store7;
    __m128i store8, store9, storeA, storeB, storeC, storeD, storeE, storeF;

    // NOTE: We have interleaved elements from rows 1 2 3 4 5 and rest of the elements is "dummy",
    //       as in: 12345___12345___12345___12345___ ...
    //       We change it to: 1234512345______ 34512345__12____ so we can perform 128b write and 64b
    //       write after blending so that "12" jumps into first 128b
    __m256i shuffle_together = _mm256_setr_epi8(0, 1, 2, 3, 4, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1, //
                                                2, 3, 4, 8, 9, 10, 11, 12, -1, -1, 0, 1, -1, -1, -1, -1);

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;

        // NOTE: Our outputs are 20 bytes apart, instead of 32, because we write 8x(12 + 8) bytes
        //       This still fills 5x32 bytes, which is our target for each iteration
        out0 = output;
        out1 = out0 + 20;
        out2 = out1 + 20;
        out3 = out2 + 20;
        out4 = out3 + 20;
        out5 = out4 + 20;
        out6 = out5 + 20;
        out7 = out6 + 20;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);
        data3 = _mm256_lddqu_si256((__m256i *)in3);
        data4 = _mm256_lddqu_si256((__m256i *)in4);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(data2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(data3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(data4, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);
        unpack4 = _mm256_unpacklo_epi8(data4, data5);
        unpack5 = _mm256_unpackhi_epi8(data4, data5);
        unpack6 = _mm256_unpacklo_epi8(data6, data7);
        unpack7 = _mm256_unpackhi_epi8(data6, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);
        unpack4 = _mm256_unpacklo_epi16(data4, data6);
        unpack5 = _mm256_unpackhi_epi16(data4, data6);
        unpack6 = _mm256_unpacklo_epi16(data5, data7);
        unpack7 = _mm256_unpackhi_epi16(data5, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi32(data0, data4);
        unpack1 = _mm256_unpackhi_epi32(data0, data4);
        unpack2 = _mm256_unpacklo_epi32(data1, data5);
        unpack3 = _mm256_unpackhi_epi32(data1, data5);
        unpack4 = _mm256_unpacklo_epi32(data2, data6);
        unpack5 = _mm256_unpackhi_epi32(data2, data6);
        unpack6 = _mm256_unpacklo_epi32(data3, data7);
        unpack7 = _mm256_unpackhi_epi32(data3, data7);

        data0 = _mm256_shuffle_epi8(unpack0, shuffle_together);
        data1 = _mm256_shuffle_epi8(unpack1, shuffle_together);
        data2 = _mm256_shuffle_epi8(unpack2, shuffle_together);
        data3 = _mm256_shuffle_epi8(unpack3, shuffle_together);
        data4 = _mm256_shuffle_epi8(unpack4, shuffle_together);
        data5 = _mm256_shuffle_epi8(unpack5, shuffle_together);
        data6 = _mm256_shuffle_epi8(unpack6, shuffle_together);
        data7 = _mm256_shuffle_epi8(unpack7, shuffle_together);

        store0 = _mm256_castsi256_si128(data0);
        store1 = _mm256_castsi256_si128(data1);
        store2 = _mm256_castsi256_si128(data2);
        store3 = _mm256_castsi256_si128(data3);
        store4 = _mm256_castsi256_si128(data4);
        store5 = _mm256_castsi256_si128(data5);
        store6 = _mm256_castsi256_si128(data6);
        store7 = _mm256_castsi256_si128(data7);

        store8 = _mm256_extracti128_si256(data0, 1);
        store9 = _mm256_extracti128_si256(data1, 1);
        storeA = _mm256_extracti128_si256(data2, 1);
        storeB = _mm256_extracti128_si256(data3, 1);
        storeC = _mm256_extracti128_si256(data4, 1);
        storeD = _mm256_extracti128_si256(data5, 1);
        storeE = _mm256_extracti128_si256(data6, 1);
        storeF = _mm256_extracti128_si256(data7, 1);

        // Now blend the missing 2 bytes into store0-4
        store0 = _mm_blend_epi16(store0, store8, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store1 = _mm_blend_epi16(store1, store9, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store2 = _mm_blend_epi16(store2, storeA, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store3 = _mm_blend_epi16(store3, storeB, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store4 = _mm_blend_epi16(store4, storeC, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store5 = _mm_blend_epi16(store5, storeD, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store6 = _mm_blend_epi16(store6, storeE, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));
        store7 = _mm_blend_epi16(store7, storeF, MASK_16(0, 0, 0, 0, 0, 1, 1, 1));

        _mm_storeu_si128((__m128i *)(out0), store0);
        _mm_storeu_si128((__m128i *)(out1), store1);
        _mm_storeu_si128((__m128i *)(out2), store2);
        _mm_storeu_si128((__m128i *)(out3), store3);
        _mm_storeu_si128((__m128i *)(out4), store4);
        _mm_storeu_si128((__m128i *)(out5), store5);
        _mm_storeu_si128((__m128i *)(out6), store6);
        _mm_storeu_si128((__m128i *)(out7), store7);

        _mm_storel_epi64((__m128i *)(out0 + 12), store8);
        _mm_storel_epi64((__m128i *)(out1 + 12), store9);
        _mm_storel_epi64((__m128i *)(out2 + 12), storeA);
        _mm_storel_epi64((__m128i *)(out3 + 12), storeB);
        _mm_storel_epi64((__m128i *)(out4 + 12), storeC);
        _mm_storel_epi64((__m128i *)(out5 + 12), storeD);
        _mm_storel_epi64((__m128i *)(out6 + 12), storeE);
        _mm_storel_epi64((__m128i *)(out7 + 12), storeF);
    }

    CopyTailM(M, N, input, output);
}

void transposeM6(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 6;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;
    const int8_t *in3;
    const int8_t *in4;
    const int8_t *in5;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;
    int8_t *out7;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3;
    __m256i data4;
    __m256i data5;
    __m256i data6 = _mm256_setzero_si256();
    __m256i data7 = _mm256_setzero_si256();

    __m256i unpack0, unpack1, unpack2, unpack3, unpack4, unpack5, unpack6, unpack7;

    __m128i store0, store1, store2, store3, store4, store5, store6, store7;
    __m128i store8, store9, storeA, storeB, storeC, storeD, storeE, storeF;

    // NOTE: We have interleaved elements from rows 1 2 3 4 5 6 and rest of the elements is "dummy",
    //       as in: 123456__123456__123456__123456__
    //       We change it to: 123456123456____ 56123456____1234 so we can perform 128b write and 64b
    //       write after blending so that "1234" jumps into first 128b
    __m256i shuffle_together = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, //
                                                4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, 0, 1, 2, 3);

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        in5 = in4 + N;

        // NOTE: Our outputs are 24 bytes apart, instead of 32, because we write 8x(16 + 8) bytes
        //       This still fills 6x32 bytes, which is our target for each iteration
        out0 = output;
        out1 = out0 + 24;
        out2 = out1 + 24;
        out3 = out2 + 24;
        out4 = out3 + 24;
        out5 = out4 + 24;
        out6 = out5 + 24;
        out7 = out6 + 24;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);
        data3 = _mm256_lddqu_si256((__m256i *)in3);
        data4 = _mm256_lddqu_si256((__m256i *)in4);
        data5 = _mm256_lddqu_si256((__m256i *)in5);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(data2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(data3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(data4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(data5, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);
        unpack4 = _mm256_unpacklo_epi8(data4, data5);
        unpack5 = _mm256_unpackhi_epi8(data4, data5);
        unpack6 = _mm256_unpacklo_epi8(data6, data7);
        unpack7 = _mm256_unpackhi_epi8(data6, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);
        unpack4 = _mm256_unpacklo_epi16(data4, data6);
        unpack5 = _mm256_unpackhi_epi16(data4, data6);
        unpack6 = _mm256_unpacklo_epi16(data5, data7);
        unpack7 = _mm256_unpackhi_epi16(data5, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi32(data0, data4);
        unpack1 = _mm256_unpackhi_epi32(data0, data4);
        unpack2 = _mm256_unpacklo_epi32(data1, data5);
        unpack3 = _mm256_unpackhi_epi32(data1, data5);
        unpack4 = _mm256_unpacklo_epi32(data2, data6);
        unpack5 = _mm256_unpackhi_epi32(data2, data6);
        unpack6 = _mm256_unpacklo_epi32(data3, data7);
        unpack7 = _mm256_unpackhi_epi32(data3, data7);

        data0 = _mm256_shuffle_epi8(unpack0, shuffle_together);
        data1 = _mm256_shuffle_epi8(unpack1, shuffle_together);
        data2 = _mm256_shuffle_epi8(unpack2, shuffle_together);
        data3 = _mm256_shuffle_epi8(unpack3, shuffle_together);
        data4 = _mm256_shuffle_epi8(unpack4, shuffle_together);
        data5 = _mm256_shuffle_epi8(unpack5, shuffle_together);
        data6 = _mm256_shuffle_epi8(unpack6, shuffle_together);
        data7 = _mm256_shuffle_epi8(unpack7, shuffle_together);

        store0 = _mm256_castsi256_si128(data0);
        store1 = _mm256_castsi256_si128(data1);
        store2 = _mm256_castsi256_si128(data2);
        store3 = _mm256_castsi256_si128(data3);
        store4 = _mm256_castsi256_si128(data4);
        store5 = _mm256_castsi256_si128(data5);
        store6 = _mm256_castsi256_si128(data6);
        store7 = _mm256_castsi256_si128(data7);

        store8 = _mm256_extracti128_si256(data0, 1);
        store9 = _mm256_extracti128_si256(data1, 1);
        storeA = _mm256_extracti128_si256(data2, 1);
        storeB = _mm256_extracti128_si256(data3, 1);
        storeC = _mm256_extracti128_si256(data4, 1);
        storeD = _mm256_extracti128_si256(data5, 1);
        storeE = _mm256_extracti128_si256(data6, 1);
        storeF = _mm256_extracti128_si256(data7, 1);

        // Now blend the missing 2 bytes into store0-4
        store0 = _mm_blend_epi16(store0, store8, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store1 = _mm_blend_epi16(store1, store9, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store2 = _mm_blend_epi16(store2, storeA, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store3 = _mm_blend_epi16(store3, storeB, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store4 = _mm_blend_epi16(store4, storeC, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store5 = _mm_blend_epi16(store5, storeD, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store6 = _mm_blend_epi16(store6, storeE, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));
        store7 = _mm_blend_epi16(store7, storeF, MASK_16(0, 0, 0, 0, 0, 0, 1, 1));

        _mm_storeu_si128((__m128i *)(out0), store0);
        _mm_storeu_si128((__m128i *)(out1), store1);
        _mm_storeu_si128((__m128i *)(out2), store2);
        _mm_storeu_si128((__m128i *)(out3), store3);
        _mm_storeu_si128((__m128i *)(out4), store4);
        _mm_storeu_si128((__m128i *)(out5), store5);
        _mm_storeu_si128((__m128i *)(out6), store6);
        _mm_storeu_si128((__m128i *)(out7), store7);

        _mm_storel_epi64((__m128i *)(out0 + 16), store8);
        _mm_storel_epi64((__m128i *)(out1 + 16), store9);
        _mm_storel_epi64((__m128i *)(out2 + 16), storeA);
        _mm_storel_epi64((__m128i *)(out3 + 16), storeB);
        _mm_storel_epi64((__m128i *)(out4 + 16), storeC);
        _mm_storel_epi64((__m128i *)(out5 + 16), storeD);
        _mm_storel_epi64((__m128i *)(out6 + 16), storeE);
        _mm_storel_epi64((__m128i *)(out7 + 16), storeF);
    }

    CopyTailM(M, N, input, output);
}

void transposeM7(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 7;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;
    const int8_t *in3;
    const int8_t *in4;
    const int8_t *in5;
    const int8_t *in6;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;
    int8_t *out7;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3;
    __m256i data4;
    __m256i data5;
    __m256i data6;
    __m256i data7 = _mm256_setzero_si256();

    __m256i unpack0, unpack1, unpack2, unpack3, unpack4, unpack5, unpack6, unpack7;

    __m128i store0, store1, store2, store3, store4, store5, store6, store7;
    __m128i store8, store9, storeA, storeB, storeC, storeD, storeE, storeF;

    // NOTE: We have interleaved elements from rows 1 2 3 4 5 6 7 and last element is "dummy",
    //       as in: 1234567_1234567_1234567_1234567_
    //       We change it to: 12345671234567__ 345671234567__12 so we can perform 128b write and
    //       112b write after blending so that "12" jumps into first 128b
    __m256i shuffle_together = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, -1, -1, //
                                                2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, -1, -1, 0, 1);

    /** @brief Mask used to skip last empty 4 bytes of 2nd write for each register */
    __m128i store_mask = _mm_setr_epi32(-1, -1, -1, 0);

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        in5 = in4 + N;
        in6 = in5 + N;

        // NOTE: Our outputs are 28 bytes apart, instead of 32, because we write 8x(16 + 12) bytes
        //       This still fills 7x32 bytes, which is our target for each iteration
        out0 = output;
        out1 = out0 + 28;
        out2 = out1 + 28;
        out3 = out2 + 28;
        out4 = out3 + 28;
        out5 = out4 + 28;
        out6 = out5 + 28;
        out7 = out6 + 28;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);
        data3 = _mm256_lddqu_si256((__m256i *)in3);
        data4 = _mm256_lddqu_si256((__m256i *)in4);
        data5 = _mm256_lddqu_si256((__m256i *)in5);
        data6 = _mm256_lddqu_si256((__m256i *)in6);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(data2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(data3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(data4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(data5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(data6, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);
        unpack4 = _mm256_unpacklo_epi8(data4, data5);
        unpack5 = _mm256_unpackhi_epi8(data4, data5);
        unpack6 = _mm256_unpacklo_epi8(data6, data7);
        unpack7 = _mm256_unpackhi_epi8(data6, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);
        unpack4 = _mm256_unpacklo_epi16(data4, data6);
        unpack5 = _mm256_unpackhi_epi16(data4, data6);
        unpack6 = _mm256_unpacklo_epi16(data5, data7);
        unpack7 = _mm256_unpackhi_epi16(data5, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi32(data0, data4);
        unpack1 = _mm256_unpackhi_epi32(data0, data4);
        unpack2 = _mm256_unpacklo_epi32(data1, data5);
        unpack3 = _mm256_unpackhi_epi32(data1, data5);
        unpack4 = _mm256_unpacklo_epi32(data2, data6);
        unpack5 = _mm256_unpackhi_epi32(data2, data6);
        unpack6 = _mm256_unpacklo_epi32(data3, data7);
        unpack7 = _mm256_unpackhi_epi32(data3, data7);

        data0 = _mm256_shuffle_epi8(unpack0, shuffle_together);
        data1 = _mm256_shuffle_epi8(unpack1, shuffle_together);
        data2 = _mm256_shuffle_epi8(unpack2, shuffle_together);
        data3 = _mm256_shuffle_epi8(unpack3, shuffle_together);
        data4 = _mm256_shuffle_epi8(unpack4, shuffle_together);
        data5 = _mm256_shuffle_epi8(unpack5, shuffle_together);
        data6 = _mm256_shuffle_epi8(unpack6, shuffle_together);
        data7 = _mm256_shuffle_epi8(unpack7, shuffle_together);

        store0 = _mm256_castsi256_si128(data0);
        store1 = _mm256_castsi256_si128(data1);
        store2 = _mm256_castsi256_si128(data2);
        store3 = _mm256_castsi256_si128(data3);
        store4 = _mm256_castsi256_si128(data4);
        store5 = _mm256_castsi256_si128(data5);
        store6 = _mm256_castsi256_si128(data6);
        store7 = _mm256_castsi256_si128(data7);

        store8 = _mm256_extracti128_si256(data0, 1);
        store9 = _mm256_extracti128_si256(data1, 1);
        storeA = _mm256_extracti128_si256(data2, 1);
        storeB = _mm256_extracti128_si256(data3, 1);
        storeC = _mm256_extracti128_si256(data4, 1);
        storeD = _mm256_extracti128_si256(data5, 1);
        storeE = _mm256_extracti128_si256(data6, 1);
        storeF = _mm256_extracti128_si256(data7, 1);

        // Now blend the missing 2 bytes into store0-4
        store0 = _mm_blend_epi16(store0, store8, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store1 = _mm_blend_epi16(store1, store9, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store2 = _mm_blend_epi16(store2, storeA, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store3 = _mm_blend_epi16(store3, storeB, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store4 = _mm_blend_epi16(store4, storeC, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store5 = _mm_blend_epi16(store5, storeD, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store6 = _mm_blend_epi16(store6, storeE, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));
        store7 = _mm_blend_epi16(store7, storeF, MASK_16(0, 0, 0, 0, 0, 0, 0, 1));

        _mm_storeu_si128((__m128i *)(out0), store0);
        _mm_storeu_si128((__m128i *)(out1), store1);
        _mm_storeu_si128((__m128i *)(out2), store2);
        _mm_storeu_si128((__m128i *)(out3), store3);
        _mm_storeu_si128((__m128i *)(out4), store4);
        _mm_storeu_si128((__m128i *)(out5), store5);
        _mm_storeu_si128((__m128i *)(out6), store6);
        _mm_storeu_si128((__m128i *)(out7), store7);

        _mm_maskstore_epi32((int *)(out0 + 16), store_mask, store8);
        _mm_maskstore_epi32((int *)(out1 + 16), store_mask, store9);
        _mm_maskstore_epi32((int *)(out2 + 16), store_mask, storeA);
        _mm_maskstore_epi32((int *)(out3 + 16), store_mask, storeB);
        _mm_maskstore_epi32((int *)(out4 + 16), store_mask, storeC);
        _mm_maskstore_epi32((int *)(out5 + 16), store_mask, storeD);
        _mm_maskstore_epi32((int *)(out6 + 16), store_mask, storeE);
        _mm_maskstore_epi32((int *)(out7 + 16), store_mask, storeF);
    }

    CopyTailM(M, N, input, output);
}

void transposeM8(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M = 8;
    const uint32_t IT = N / M_ITERATION_SIZE;

    const int8_t *in0;
    const int8_t *in1;
    const int8_t *in2;
    const int8_t *in3;
    const int8_t *in4;
    const int8_t *in5;
    const int8_t *in6;
    const int8_t *in7;

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;
    int8_t *out7;

    __m256i data0;
    __m256i data1;
    __m256i data2;
    __m256i data3;
    __m256i data4;
    __m256i data5;
    __m256i data6;
    __m256i data7;

    __m256i unpack0;
    __m256i unpack1;
    __m256i unpack2;
    __m256i unpack3;
    __m256i unpack4;
    __m256i unpack5;
    __m256i unpack6;
    __m256i unpack7;

    for (uint32_t i = 0; i < IT; ++i)
    {

        in0 = input;
        in1 = in0 + N;
        in2 = in1 + N;
        in3 = in2 + N;
        in4 = in3 + N;
        in5 = in4 + N;
        in6 = in5 + N;
        in7 = in6 + N;

        out0 = output;
        out1 = out0 + 32;
        out2 = out1 + 32;
        out3 = out2 + 32;
        out4 = out3 + 32;
        out5 = out4 + 32;
        out6 = out5 + 32;
        out7 = out6 + 32;

        input += 32;
        output += 32 * M;

        data0 = _mm256_lddqu_si256((__m256i *)in0);
        data1 = _mm256_lddqu_si256((__m256i *)in1);
        data2 = _mm256_lddqu_si256((__m256i *)in2);
        data3 = _mm256_lddqu_si256((__m256i *)in3);
        data4 = _mm256_lddqu_si256((__m256i *)in4);
        data5 = _mm256_lddqu_si256((__m256i *)in5);
        data6 = _mm256_lddqu_si256((__m256i *)in6);
        data7 = _mm256_lddqu_si256((__m256i *)in7);

        data0 = _mm256_permute4x64_epi64(data0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(data1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(data2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(data3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(data4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(data5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(data6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(data7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi8(data0, data1);
        unpack1 = _mm256_unpackhi_epi8(data0, data1);
        unpack2 = _mm256_unpacklo_epi8(data2, data3);
        unpack3 = _mm256_unpackhi_epi8(data2, data3);
        unpack4 = _mm256_unpacklo_epi8(data4, data5);
        unpack5 = _mm256_unpackhi_epi8(data4, data5);
        unpack6 = _mm256_unpacklo_epi8(data6, data7);
        unpack7 = _mm256_unpackhi_epi8(data6, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi16(data0, data2);
        unpack1 = _mm256_unpackhi_epi16(data0, data2);
        unpack2 = _mm256_unpacklo_epi16(data1, data3);
        unpack3 = _mm256_unpackhi_epi16(data1, data3);
        unpack4 = _mm256_unpacklo_epi16(data4, data6);
        unpack5 = _mm256_unpackhi_epi16(data4, data6);
        unpack6 = _mm256_unpacklo_epi16(data5, data7);
        unpack7 = _mm256_unpackhi_epi16(data5, data7);

        data0 = _mm256_permute4x64_epi64(unpack0, PERMUTE_0213);
        data1 = _mm256_permute4x64_epi64(unpack1, PERMUTE_0213);
        data2 = _mm256_permute4x64_epi64(unpack2, PERMUTE_0213);
        data3 = _mm256_permute4x64_epi64(unpack3, PERMUTE_0213);
        data4 = _mm256_permute4x64_epi64(unpack4, PERMUTE_0213);
        data5 = _mm256_permute4x64_epi64(unpack5, PERMUTE_0213);
        data6 = _mm256_permute4x64_epi64(unpack6, PERMUTE_0213);
        data7 = _mm256_permute4x64_epi64(unpack7, PERMUTE_0213);

        unpack0 = _mm256_unpacklo_epi32(data0, data4);
        unpack1 = _mm256_unpackhi_epi32(data0, data4);
        unpack2 = _mm256_unpacklo_epi32(data1, data5);
        unpack3 = _mm256_unpackhi_epi32(data1, data5);
        unpack4 = _mm256_unpacklo_epi32(data2, data6);
        unpack5 = _mm256_unpackhi_epi32(data2, data6);
        unpack6 = _mm256_unpacklo_epi32(data3, data7);
        unpack7 = _mm256_unpackhi_epi32(data3, data7);

        _mm256_storeu_si256((__m256i *)out0, unpack0);
        _mm256_storeu_si256((__m256i *)out1, unpack1);
        _mm256_storeu_si256((__m256i *)out2, unpack2);
        _mm256_storeu_si256((__m256i *)out3, unpack3);
        _mm256_storeu_si256((__m256i *)out4, unpack4);
        _mm256_storeu_si256((__m256i *)out5, unpack5);
        _mm256_storeu_si256((__m256i *)out6, unpack6);
        _mm256_storeu_si256((__m256i *)out7, unpack7);
    }

    CopyTailM(M, N, input, output);
}

void transposeN2(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 2;
    uint32_t M_VEC = M - M % N_ITERATION_SIZE;

    __m128i ah1;
    __m128i ah2;
    __m256i ah;
    __m256i pick_row_elements = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                                 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    int8_t *out0;
    int8_t *out1;

    ah = _mm256_load_si256((__m256i *)input);

    uint32_t i;

    for (i = 0; i < M_VEC; i += N_ITERATION_SIZE)
    {
        input += N_ITERATION_SIZE * N;

        ah = _mm256_shuffle_epi8(ah, pick_row_elements);
        ah = _mm256_permute4x64_epi64(ah, PERMUTE_0213);

        ah1 = _mm256_castsi256_si128(ah);
        ah2 = _mm256_extracti128_si256(ah, 1);

        out0 = output + i;
        out1 = out0 + M;

        _mm_storeu_si128((__m128i *)out0, ah1);
        _mm_storeu_si128((__m128i *)out1, ah2);

        ah = _mm256_lddqu_si256((__m256i *)input);
    }
}

/** @brief Transpose - variant for 3 columns
 *
 * Shuffle is 128-bit lane limited. If we read data sequentially, we won't be able to fill a lane with data from single row.
 * Instead, we skip 2 bytes after every 32-bit gather. We do this for each of the rows.
 * This way we read 2N samples but each row needs single shuffle and permute.
 *
 * We've checked that for largest input this solution is about 20% faster than one with sequential read
 * and about 10% faster than mixed mode in which we read 1.5N samples and compose row 2 out of data read for row 1 and row 3.
 *
 * Not to mention, this solution is simpler and easier to read.
 */
void transposeN3(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 3;
    uint32_t M_VEC = M - M % N_ITERATION_SIZE;

    __m256i ah1_, ah2_, ah3_;
    __m128i ah1, ah2, ah3;

    __m256i gather_n3 = _mm256_setr_epi32(0, 6, 12, 18, 24, 30, 36, 42);

    // NOTE: We don't care about upper 64 bits of each lane (these are interleaved data from other 2 columns)
    __m256i pick_row_elements = _mm256_setr_epi8(0, 3, 4, 7, 8, 11, 12, 15, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 3, 4, 7, 8, 11, 12, 15, 0, 0, 0, 0, 0, 0, 0, 0);

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;

    ah1_ = _mm256_i32gather_epi32((int32_t *)input, gather_n3, 1);
    ah2_ = _mm256_i32gather_epi32((int32_t *)(input + 1), gather_n3, 1);
    ah3_ = _mm256_i32gather_epi32((int32_t *)(input + 2), gather_n3, 1);

    uint32_t i;

    for (i = 0; i < M_VEC; i += N_ITERATION_SIZE)
    {
        input += N_ITERATION_SIZE * N;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;

        ah1_ = _mm256_shuffle_epi8(ah1_, pick_row_elements);
        ah2_ = _mm256_shuffle_epi8(ah2_, pick_row_elements);
        ah3_ = _mm256_shuffle_epi8(ah3_, pick_row_elements);

        ah1_ = _mm256_permute4x64_epi64(ah1_, PERMUTE_0213);
        ah2_ = _mm256_permute4x64_epi64(ah2_, PERMUTE_0213);
        ah3_ = _mm256_permute4x64_epi64(ah3_, PERMUTE_0213);

        ah1 = _mm256_castsi256_si128(ah1_);
        ah2 = _mm256_castsi256_si128(ah2_);
        ah3 = _mm256_castsi256_si128(ah3_);

        _mm_storeu_si128((__m128i *)out0, ah1);
        _mm_storeu_si128((__m128i *)out1, ah2);
        _mm_storeu_si128((__m128i *)out2, ah3);

        ah1_ = _mm256_i32gather_epi32((int32_t *)input, gather_n3, 1);
        ah2_ = _mm256_i32gather_epi32((int32_t *)(input + 1), gather_n3, 1);
        ah3_ = _mm256_i32gather_epi32((int32_t *)(input + 2), gather_n3, 1);
    }
}

void transposeN4(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 4;
    uint32_t M_VEC = M - M % N_ITERATION_SIZE;

    __m256i lo1234_, hi1234_, ah13_, ah24_;

    // Shuffle every 4th element into correct quarter
    __m256i shuffle_into_quarters = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                                     0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    // Swap 32b elements in LO register into correct order
    __m256i permute_lo = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    // Swap 32b elements in HI register into correct order
    __m256i permute_hi = _mm256_setr_epi32(1, 5, 0, 4, 3, 7, 2, 6);

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;

    lo1234_ = _mm256_load_si256((__m256i *)input);
    hi1234_ = _mm256_load_si256((__m256i *)(input + 2 * N_ITERATION_SIZE));

    __m128i ah1, ah2, ah3, ah4;

    uint32_t i;

    for (i = 0; i < M_VEC; i += N_ITERATION_SIZE)
    {
        input += N_ITERATION_SIZE * N;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;

        lo1234_ = _mm256_shuffle_epi8(lo1234_, shuffle_into_quarters);
        hi1234_ = _mm256_shuffle_epi8(hi1234_, shuffle_into_quarters);

        lo1234_ = _mm256_permutevar8x32_epi32(lo1234_, permute_lo);
        hi1234_ = _mm256_permutevar8x32_epi32(hi1234_, permute_hi);

        // At this point: lo1234_ has first  64b of each row in order 1, 2, 3, 4
        //                hi1234_ has second 64b of each row in order 2, 1, 4, 3

        ah13_ = _mm256_blend_epi32(lo1234_, hi1234_, MASK_32(0, 1, 0, 1));
        ah24_ = _mm256_blend_epi32(lo1234_, hi1234_, MASK_32(1, 0, 1, 0));
        ah24_ = _mm256_permute4x64_epi64(ah24_, PERMUTE(1, 0, 3, 2));

        ah1 = _mm256_castsi256_si128(ah13_);
        ah2 = _mm256_castsi256_si128(ah24_);
        ah3 = _mm256_extracti128_si256(ah13_, 1);
        ah4 = _mm256_extracti128_si256(ah24_, 1);

        _mm_storeu_si128((__m128i *)out0, ah1);
        _mm_storeu_si128((__m128i *)out1, ah2);
        _mm_storeu_si128((__m128i *)out2, ah3);
        _mm_storeu_si128((__m128i *)out3, ah4);

        lo1234_ = _mm256_lddqu_si256((__m256i *)input);
        hi1234_ = _mm256_lddqu_si256((__m256i *)(input + 2 * N_ITERATION_SIZE));
    }
}

void transposeN5(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 5;

    // NOTE: This algorithm has non-standard iteration size
    const uint32_t SINGLE_ITERATION = 8;

    uint32_t M_VEC = M - M % SINGLE_ITERATION;

    __m256i ad1234, ad5___;
    __m128i a12, a34, a5;

    __m256i gather_n5 = _mm256_setr_epi32(0, 5, 10, 15, 20, 25, 30, 35);

    __m256i pick_row_elements = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                                 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    __m256i permute32 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;

    ad1234 = _mm256_i32gather_epi32((int32_t *)input, gather_n5, 1);
    ad5___ = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n5, 1);

    uint32_t i;

    for (i = 0; i < M_VEC; i += SINGLE_ITERATION)
    {
        input += N * SINGLE_ITERATION;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;

        ad1234 = _mm256_shuffle_epi8(ad1234, pick_row_elements);
        ad1234 = _mm256_permutevar8x32_epi32(ad1234, permute32);

        ad5___ = _mm256_shuffle_epi8(ad5___, pick_row_elements);
        ad5___ = _mm256_permutevar8x32_epi32(ad5___, permute32);

        a12 = _mm256_castsi256_si128(ad1234);
        a34 = _mm256_extracti128_si256(ad1234, 1);
        a5 = _mm256_castsi256_si128(ad5___);

        _mm_storel_epi64((__m128i *)out0, a12);
        _mm_storel_epi64((__m128i *)out1, _mm_srli_si128(a12, 8));
        _mm_storel_epi64((__m128i *)out2, a34);
        _mm_storel_epi64((__m128i *)out3, _mm_srli_si128(a34, 8));
        _mm_storel_epi64((__m128i *)out4, a5);

        ad1234 = _mm256_i32gather_epi32((int32_t *)input, gather_n5, 1);
        ad5___ = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n5, 1);
    }
}

/** @brief Transpose - variant for 6 columns
 *
 * This algorithm creates 3 buffers by reading 4B, then skipping 8B
 * This way every 2nd element of rows 1, 3 and 5 is already in correct buffer
 * which removes one blend per 2 rows.
 */
void transposeN6(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 6;
    uint32_t M_VEC = M - M % N_ITERATION_SIZE;

    __m256i mix1;
    __m256i mix2;
    __m256i mix3;
    __m256i a12_;
    __m256i a34_;
    __m256i a56_;

    // We skip 8 bytes after each 32b gather
    __m256i gather_n4 = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const int gather_scale = 4;

    // Split bytes so that every 2nd byte is in first 64b of a lane
    __m256i split_every_2nd_byte = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, //
                                                    0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    // Swap bytes in every pair
    __m256i swap_bytes = _mm256_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, //
                                          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    // Mask used to interleave bytes from two registers
    __m256i interleave = _mm256_setr_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, //
                                          0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);

    mix1 = _mm256_i32gather_epi32((int32_t *)input, gather_n4, gather_scale);
    mix2 = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n4, gather_scale);
    mix3 = _mm256_i32gather_epi32((int32_t *)(input + 8), gather_n4, gather_scale);

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;

    uint32_t i;

    for (i = 0; i < M_VEC; i += N_ITERATION_SIZE)
    {
        input += N_ITERATION_SIZE * N;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;
        out5 = out4 + M;

        mix1 = _mm256_shuffle_epi8(mix1, split_every_2nd_byte);
        mix2 = _mm256_shuffle_epi8(mix2, split_every_2nd_byte);
        mix3 = _mm256_shuffle_epi8(mix3, split_every_2nd_byte);

        a12_ = _mm256_blendv_epi8(mix1, mix2, interleave);
        a34_ = _mm256_blendv_epi8(mix3, mix1, interleave);
        a56_ = _mm256_blendv_epi8(mix2, mix3, interleave);

        a12_ = _mm256_permute4x64_epi64(a12_, PERMUTE_0213);
        a34_ = _mm256_permute4x64_epi64(a34_, PERMUTE_0213);
        a56_ = _mm256_permute4x64_epi64(a56_, PERMUTE_0213);

        // NOTE: mix3 starts from 2nd elements of rows 3 and 4,
        //       so a34_ is the only register that needs its bytes swapped in pairs
        a34_ = _mm256_shuffle_epi8(a34_, swap_bytes);

        _mm_storeu_si128((__m128i *)out0, _mm256_castsi256_si128(a12_));
        _mm_storeu_si128((__m128i *)out1, _mm256_extracti128_si256(a12_, 1));
        _mm_storeu_si128((__m128i *)out2, _mm256_castsi256_si128(a34_));
        _mm_storeu_si128((__m128i *)out3, _mm256_extracti128_si256(a34_, 1));
        _mm_storeu_si128((__m128i *)out4, _mm256_castsi256_si128(a56_));
        _mm_storeu_si128((__m128i *)out5, _mm256_extracti128_si256(a56_, 1));

        mix1 = _mm256_i32gather_epi32((int32_t *)input, gather_n4, gather_scale);
        mix2 = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n4, gather_scale);
        mix3 = _mm256_i32gather_epi32((int32_t *)(input + 8), gather_n4, gather_scale);
    }
}

void transposeN7(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 7;

    // NOTE: This algorithm has non-standard iteration size
    const uint32_t SINGLE_ITERATION = 8;

    uint32_t M_VEC = M - M % SINGLE_ITERATION;

    __m256i ad1234, ad567_;
    __m128i a12, a34, a56, a7;

    __m256i gather_n7 = _mm256_setr_epi32(0, 7, 14, 21, 28, 35, 42, 49);

    __m256i pick_row_elements = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                                 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    __m256i permute32 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;

    ad1234 = _mm256_i32gather_epi32((int32_t *)input, gather_n7, 1);
    ad567_ = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n7, 1);

    uint32_t i;

    for (i = 0; i < M_VEC; i += SINGLE_ITERATION)
    {
        input += N * SINGLE_ITERATION;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;
        out5 = out4 + M;
        out6 = out5 + M;

        ad1234 = _mm256_shuffle_epi8(ad1234, pick_row_elements);
        ad1234 = _mm256_permutevar8x32_epi32(ad1234, permute32);

        ad567_ = _mm256_shuffle_epi8(ad567_, pick_row_elements);
        ad567_ = _mm256_permutevar8x32_epi32(ad567_, permute32);

        a12 = _mm256_castsi256_si128(ad1234);
        a34 = _mm256_extracti128_si256(ad1234, 1);
        a56 = _mm256_castsi256_si128(ad567_);
        a7 = _mm256_extracti128_si256(ad567_, 1);

        _mm_storel_epi64((__m128i *)out0, a12);
        _mm_storel_epi64((__m128i *)out1, _mm_srli_si128(a12, 8));
        _mm_storel_epi64((__m128i *)out2, a34);
        _mm_storel_epi64((__m128i *)out3, _mm_srli_si128(a34, 8));
        _mm_storel_epi64((__m128i *)out4, a56);
        _mm_storel_epi64((__m128i *)out5, _mm_srli_si128(a56, 8));
        _mm_storel_epi64((__m128i *)out6, a7);

        ad1234 = _mm256_i32gather_epi32((int32_t *)input, gather_n7, 1);
        ad567_ = _mm256_i32gather_epi32((int32_t *)(input + 4), gather_n7, 1);
    }
}

void transposeN8(const int8_t *input, int8_t *output, uint32_t M)
{
    uint32_t N = 8;
    uint32_t M_VEC = M - M % N_ITERATION_SIZE;

    __m256i ab;
    __m256i cd;
    __m256i ef;
    __m256i gh;
    __m256i abcd_lo;
    __m256i abcd_hi;
    __m256i efgh_lo;
    __m256i efgh_hi;
    __m256i full1;
    __m256i full2;
    __m256i full3;
    __m256i full4;

    __m256i mask1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    __m256i mask2 = _mm256_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, //
                                     0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);

    __m256i mask3 = _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, //
                                     0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    int8_t *out0;
    int8_t *out1;
    int8_t *out2;
    int8_t *out3;
    int8_t *out4;
    int8_t *out5;
    int8_t *out6;
    int8_t *out7;

    ab = _mm256_load_si256((__m256i *)input);
    cd = _mm256_load_si256((__m256i *)(input + 2 * N_ITERATION_SIZE));
    ef = _mm256_load_si256((__m256i *)(input + 4 * N_ITERATION_SIZE));
    gh = _mm256_load_si256((__m256i *)(input + 6 * N_ITERATION_SIZE));

    uint32_t i;

    for (i = 0; i < M_VEC; i += N_ITERATION_SIZE)
    {
        input += N_ITERATION_SIZE * N;
        out0 = output + i;
        out1 = out0 + M;
        out2 = out1 + M;
        out3 = out2 + M;
        out4 = out3 + M;
        out5 = out4 + M;
        out6 = out5 + M;
        out7 = out6 + M;

        ab = _mm256_permutevar8x32_epi32(ab, mask1);
        cd = _mm256_permutevar8x32_epi32(cd, mask1);
        ef = _mm256_permutevar8x32_epi32(ef, mask1);
        gh = _mm256_permutevar8x32_epi32(gh, mask1);

        ab = _mm256_shuffle_epi8(ab, mask2);
        cd = _mm256_shuffle_epi8(cd, mask2);
        ef = _mm256_shuffle_epi8(ef, mask2);
        gh = _mm256_shuffle_epi8(gh, mask2);

        ab = _mm256_shuffle_epi8(ab, mask3);
        cd = _mm256_shuffle_epi8(cd, mask3);
        ef = _mm256_shuffle_epi8(ef, mask3);
        gh = _mm256_shuffle_epi8(gh, mask3);

        abcd_lo = _mm256_unpacklo_epi32(ab, cd);
        abcd_hi = _mm256_unpackhi_epi32(ab, cd);
        efgh_lo = _mm256_unpacklo_epi32(ef, gh);
        efgh_hi = _mm256_unpackhi_epi32(ef, gh);

        full1 = _mm256_unpacklo_epi64(abcd_lo, efgh_lo);
        full2 = _mm256_unpackhi_epi64(abcd_lo, efgh_lo);
        full3 = _mm256_unpacklo_epi64(abcd_hi, efgh_hi);
        full4 = _mm256_unpackhi_epi64(abcd_hi, efgh_hi);

        _mm_storeu_si128((__m128i *)out0, _mm256_castsi256_si128(full1));
        _mm_storeu_si128((__m128i *)out1, _mm256_castsi256_si128(full2));
        _mm_storeu_si128((__m128i *)out2, _mm256_castsi256_si128(full3));
        _mm_storeu_si128((__m128i *)out3, _mm256_castsi256_si128(full4));
        _mm_storeu_si128((__m128i *)out4, _mm256_extracti128_si256(full1, 1));
        _mm_storeu_si128((__m128i *)out5, _mm256_extracti128_si256(full2, 1));
        _mm_storeu_si128((__m128i *)out6, _mm256_extracti128_si256(full3, 1));
        _mm_storeu_si128((__m128i *)out7, _mm256_extracti128_si256(full4, 1));

        ab = _mm256_load_si256((__m256i *)input);
        cd = _mm256_load_si256((__m256i *)(input + 2 * N_ITERATION_SIZE));
        ef = _mm256_load_si256((__m256i *)(input + 4 * N_ITERATION_SIZE));
        gh = _mm256_load_si256((__m256i *)(input + 6 * N_ITERATION_SIZE));
    }
}
