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
#include <nmmintrin.h>

/** Single iteration step */
#define IT_STEP (16)

/** Generic transpose for N[2,8] and M > 0, M % 16 == 0 */
template <size_t M>
static void transposeM(const int8_t *input, int8_t *output, uint32_t N);

/** Generic transpose for N[2,8] and N > 0, N % 16 == 0 */
template <size_t N>
static void transposeN(const int8_t *input, int8_t *output, uint32_t M);

/** @brief Main function
 *
 *  Transposition is implemented for:
 *  - N [1,8] M[16, 2^16 - 16], where M % 16 == 0
 *  - M [1,8] N[16, 2^16 - 16], where N % 16 == 0
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

    // INTERLEAVE
    // MAX M is 8, MAX N is UINT16_MAX
    if (M <= 8)
    {
        switch (M)
        {
        case 2:
            transposeM<2>(I, O, N);
            break;
        case 3:
            transposeM<3>(I, O, N);
            break;
        case 4:
            transposeM<4>(I, O, N);
            break;
        case 5:
            transposeM<5>(I, O, N);
            break;
        case 6:
            transposeM<6>(I, O, N);
            break;
        case 7:
            transposeM<7>(I, O, N);
            break;
        case 8:
            transposeM<8>(I, O, N);
            break;
        default:
            break;
        }
    }
    // DEINTERLEAVE
    // MAX N is 8, MAX M is UINT16_MAX
    else
    {
        switch (N)
        {
        case 2:
            transposeN<2>(I, O, M);
            break;
        case 3:
            transposeN<3>(I, O, M);
            break;
        case 4:
            transposeN<4>(I, O, M);
            break;
        case 5:
            transposeN<5>(I, O, M);
            break;
        case 6:
            transposeN<6>(I, O, M);
            break;
        case 7:
            transposeN<7>(I, O, M);
            break;
        case 8:
            transposeN<8>(I, O, M);
            break;
        default:
            break;
        }
    }
}

template <size_t M>
void transposeM(const int8_t *input, int8_t *output, uint32_t N)
{
    const uint32_t M_SIZE = (M > 4) ? 8 : ((M > 2) ? 4 : 2); // M rounded up to nearest power of 2
    const uint32_t IT_COUNT = N / IT_STEP;

    const int8_t *in[M_SIZE];
    int8_t *out[M_SIZE];

    __m128i data[M_SIZE];
    __m128i unpack[M_SIZE];
    __m128i shuffle_together;

    // NOTE: For rows 3 and 5-7 we have interleaved elements from rows and some zeroes at the end
    //       namely: 123_123_123_123_, 12345___12345___, 123456__123456__, 1234567_1234567_
    //       We group them so we can perform single write
    switch (M)
    {
    case 3:
        shuffle_together = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
        break;
    case 5:
        shuffle_together = _mm_setr_epi8(0, 1, 2, 3, 4, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1);
        break;
    case 6:
        shuffle_together = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1);
        break;
    case 7:
        shuffle_together = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, -1, -1);
        break;
    default:
        shuffle_together = _mm_setzero_si128();
        break;
    }

    // Registers without data serve as a padding for unpacking
    for (uint32_t i = M; i < M_SIZE; ++i)
    {
        data[i] = _mm_setzero_si128();
    }

    for (uint32_t i = 0; i < IT_COUNT; ++i)
    {
        in[0] = input;
        out[0] = output;

        input += IT_STEP;
        output += IT_STEP * M;

        for (uint32_t m = 1; m < M; ++m)
        {
            in[m] = in[m - 1] + N;
        }

        for (uint32_t m = 1; m < M_SIZE; ++m)
        {
            out[m] = out[m - 1] + IT_STEP * M / M_SIZE;
        }

        for (uint32_t m = 0; m < M; ++m)
        {
            data[m] = _mm_lddqu_si128((__m128i *)in[m]);
        }

        for (uint32_t m = 0; m < M_SIZE; m += 2)
        {
            unpack[m + 0] = _mm_unpacklo_epi8(data[m], data[m + 1]);
            unpack[m + 1] = _mm_unpackhi_epi8(data[m], data[m + 1]);
        }

        if (M_SIZE > 2)
        {
            for (uint32_t m = 0; m < M_SIZE; ++m)
            {
                data[m] = unpack[m];
            }

            for (uint32_t m = 0; m < M_SIZE; m += 4)
            {
                unpack[m + 0] = _mm_unpacklo_epi16(data[m + 0], data[m + 2]);
                unpack[m + 1] = _mm_unpackhi_epi16(data[m + 0], data[m + 2]);
                unpack[m + 2] = _mm_unpacklo_epi16(data[m + 1], data[m + 3]);
                unpack[m + 3] = _mm_unpackhi_epi16(data[m + 1], data[m + 3]);
            }

            if (M_SIZE > 4)
            {
                for (uint32_t m = 0; m < M_SIZE; ++m)
                {
                    data[m] = unpack[m];
                }

                for (uint32_t m = 0; m < M_SIZE / 2; m += 1)
                {
                    unpack[2 * m + 0] = _mm_unpacklo_epi32(data[m + 0], data[m + 4]);
                    unpack[2 * m + 1] = _mm_unpackhi_epi32(data[m + 0], data[m + 4]);
                }
            }
        }

        if (M != M_SIZE)
        {
            for (uint32_t m = 0; m < M_SIZE; ++m)
            {
                unpack[m] = _mm_shuffle_epi8(unpack[m], shuffle_together);
            }

            // NOTE: Our writes store less than 16B of data. We don't want our last write to go out of bounds
            //       We tried:
            //       1) _mm_maskmoveu_si128 - this made overall transposition 3x slower!
            //       2) Write 64b then maskmove_si64 - similar result!
            //       3) Write many times (e.g. 64b + 32b + 16b for 14B write) - similar result!
            //
            // Fortunately if we append last few bytes of previous register to front of the last
            // register and do 128b write, we get only 5% slowdown!

            // NOTE: This switch-case is only for compiler backward-compatibility.
            //       Old gcc cannot easily determine what is actually compiler-time const...
            switch (M)
            {
            case 3: /* fallthrough */
            case 6:
                out[M_SIZE - 1] -= 4;
                unpack[M_SIZE - 1] = _mm_or_si128(_mm_bslli_si128(unpack[M_SIZE - 1], 4), _mm_bsrli_si128(unpack[M_SIZE - 2], 8));
                break;
            case 5:
                out[M_SIZE - 1] -= 6;
                unpack[M_SIZE - 1] = _mm_or_si128(_mm_bslli_si128(unpack[M_SIZE - 1], 6), _mm_bsrli_si128(unpack[M_SIZE - 2], 4));
                break;
            case 7:
                out[M_SIZE - 1] -= 2;
                unpack[M_SIZE - 1] = _mm_or_si128(_mm_bslli_si128(unpack[M_SIZE - 1], 2), _mm_bsrli_si128(unpack[M_SIZE - 2], 12));
                break;
            default:
                break;
            }
        }

        for (uint32_t m = 0; m < M_SIZE; ++m)
        {
            _mm_storeu_si128((__m128i *)out[m], unpack[m]);
        }
    }
}

template <size_t N>
void transposeN(const int8_t *input, int8_t *output, uint32_t M)
{
    __m128i in[N];
    __m128i shuffled[N];
    __m128i shuffle_mask[N * N];
    int8_t shuffle_data[N * N][16];

    memset(shuffle_data, -1, sizeof(shuffle_data));

    for (size_t i = 0; i < N * 16; ++i)
    {
        shuffle_data[i % N + i / 16 * N][i / N] = static_cast<int8_t>(i % 16);
    }

    for (size_t i = 0; i < N * N; ++i)
    {
        shuffle_mask[i] = _mm_loadu_si128((const __m128i *)shuffle_data[i]);
    }

    for (size_t i = 0; i < M; i += 16)
    {
        for (size_t j = 0; j < N; ++j)
        {
            in[j] = _mm_loadu_si128(j + (const __m128i *)input);
            shuffled[j] = _mm_shuffle_epi8(in[0], shuffle_mask[j]);
        }

        for (size_t j = N; j < N * N; j += N)
        {
            shuffled[(0 + j) % N] = _mm_or_si128(shuffled[(0 + j) % N], _mm_shuffle_epi8(in[(0 + j) / N], shuffle_mask[0 + j]));
            shuffled[(1 + j) % N] = _mm_or_si128(shuffled[(1 + j) % N], _mm_shuffle_epi8(in[(1 + j) / N], shuffle_mask[1 + j]));

            if (N >= 3)
            {
                shuffled[(2 + j) % N] = _mm_or_si128(shuffled[(2 + j) % N], _mm_shuffle_epi8(in[(2 + j) / N], shuffle_mask[2 + j]));
            }
            if (N >= 4)
            {
                shuffled[(3 + j) % N] = _mm_or_si128(shuffled[(3 + j) % N], _mm_shuffle_epi8(in[(3 + j) / N], shuffle_mask[3 + j]));
            }
            if (N >= 5)
            {
                shuffled[(4 + j) % N] = _mm_or_si128(shuffled[(4 + j) % N], _mm_shuffle_epi8(in[(4 + j) / N], shuffle_mask[4 + j]));
            }
            if (N >= 6)
            {
                shuffled[(5 + j) % N] = _mm_or_si128(shuffled[(5 + j) % N], _mm_shuffle_epi8(in[(5 + j) / N], shuffle_mask[5 + j]));
            }
            if (N >= 7)
            {
                shuffled[(6 + j) % N] = _mm_or_si128(shuffled[(6 + j) % N], _mm_shuffle_epi8(in[(6 + j) / N], shuffle_mask[6 + j]));
            }
            if (N == 8)
            {
                shuffled[(7 + j) % N] = _mm_or_si128(shuffled[(7 + j) % N], _mm_shuffle_epi8(in[(7 + j) / N], shuffle_mask[7 + j]));
            }
        }

        for (size_t j = 0; j < N; ++j)
        {
            _mm_storeu_si128((__m128i *)(output + M * j), shuffled[j]);
        }

        input += 16 * N;
        output += 16;
    }
}
