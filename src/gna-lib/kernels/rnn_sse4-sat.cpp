/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "common_sse4.hpp"
#include "igemv8.h"
#include "igemv16.h"
#include "KernelArguments.h"
#include <cstdint>

/** Iteration step. Output must be divisible by this value
 *
 *  NOTE: For values other than multiplicities of 16 unaligned loads have to be used
 */
#define IT_STEP (16)

/** RNN kernel - generic implementation
 *
 * Uses 64-bit intermediate sums.
 *
 * This implementation can be used for 1B weights but 32b variant is much faster.
 *
 * NOTE: Due to HW limitations computations must be split based on buffer size
 *       and intermediate saturation is added
 *
 * @param BPW Bytes per weight (1 or 2)
 * @param config Forwarded execution configuration
 */
template <int BPW>
void RNN1B(ExecutionKernelConfig<RecurrentConfig> const * const config);

/** RNN kernel implementation for 1B input 1B weight
 *
 * ASSUMPTIONS:
 *   Input  is Kx1 where K [16, 2^16 - 16], K % 16 == 0
 *   Output is Mx1 where M [16, 2^15], M % 16 == 0
 *   Buffer size % 16 == 0
 *   Biases can be 1B, 2B or 4B
 *   Feedback is 1B
 *
 * Note that output limit is determined by maximum model size, which is 256MB.
 * Since for RNN there are M * (K + M) weights, maximum output is in fact less than 12k
 */
void RecurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    RNN1B<1>(config);
}

/** RNN kernel implementation for 1B input 2B weight
 *
 * ASSUMPTIONS:
 *   Input  is Kx1 where K [16, 2^16 - 16], K % 16 == 0
 *   Output is Mx1 where M [16, 2^15], M % 16 == 0
 *   Buffer size % 16 == 0
 *   Biases can be 1B, 2B or 4B
 *   Feedback is 1B
 *
 * Note that output limit is determined by maximum model size, which is 256MB.
 * Since for RNN there are M * (K + M) weights, maximum output is in fact less than 12k
 */
void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    RNN1B<2>(config);
}

namespace
{
    /** Input data needed for Madd16 functions */
    struct Madd16Input
    {
        Madd16Input(const int8_t *firstArray, const int8_t *secondArray, uint32_t firstLimit, uint32_t stepSize)
            : first{firstArray},
              second{secondArray},
              limit{firstLimit},
              step{stepSize}
        {
        }

        const int8_t *first;
        const int8_t *second;
        uint32_t limit;
        uint32_t step;
    };

    /** Output data passed between functions. Depending on the width of multiplied elements, these data
     *  are either 32-bit or 64-bit. Conversion is done only when necessary.
     */
    struct RnnOutput
    {
        RnnOutput(uint32_t *saturationCount)
            : saturationCounter{saturationCount},
              is64{false}
        {
        }

        /** Extends data to 64-bit. Does nothing if data is already 64-bit */
        void ExtendTo64()
        {
            if (!is64)
            {
                out64_1 = _mm_cvtepi32_epi64(out32_1);
                out64_2 = _mm_cvtepi32_epi64(_mm_bsrli_si128(out32_1, 8));
                out64_3 = _mm_cvtepi32_epi64(out32_2);
                out64_4 = _mm_cvtepi32_epi64(_mm_bsrli_si128(out32_2, 8));
                out64_5 = _mm_cvtepi32_epi64(out32_3);
                out64_6 = _mm_cvtepi32_epi64(_mm_bsrli_si128(out32_3, 8));
                out64_7 = _mm_cvtepi32_epi64(out32_4);
                out64_8 = _mm_cvtepi32_epi64(_mm_bsrli_si128(out32_4, 8));
                is64 = true;
            }
        }

        /** Saturate 64-bit data to 32-bit. Result remains 64-bit. Does nothing if data is 32-bit */
        void Saturate64()
        {
            if (is64)
            {
                out64_1 = _mm_sat_epi64(out64_1, saturationCounter);
                out64_2 = _mm_sat_epi64(out64_2, saturationCounter);
                out64_3 = _mm_sat_epi64(out64_3, saturationCounter);
                out64_4 = _mm_sat_epi64(out64_4, saturationCounter);
                out64_5 = _mm_sat_epi64(out64_5, saturationCounter);
                out64_6 = _mm_sat_epi64(out64_6, saturationCounter);
                out64_7 = _mm_sat_epi64(out64_7, saturationCounter);
                out64_8 = _mm_sat_epi64(out64_8, saturationCounter);
            }
        }

        /** Packs 64-bit data to 32-bit with saturation. Does nothing if data is already 32-bit */
        void SaturatePack32()
        {
            if (is64)
            {
                out32_1 = _mm_packs_epi64(out64_1, out64_2, saturationCounter);
                out32_2 = _mm_packs_epi64(out64_3, out64_4, saturationCounter);
                out32_3 = _mm_packs_epi64(out64_5, out64_6, saturationCounter);
                out32_4 = _mm_packs_epi64(out64_7, out64_8, saturationCounter);
                is64 = false;
            }
        }

        __m128i out32_1;
        __m128i out32_2;
        __m128i out32_3;
        __m128i out32_4;

        __m128i out64_1;
        __m128i out64_2;
        __m128i out64_3;
        __m128i out64_4;
        __m128i out64_5;
        __m128i out64_6;
        __m128i out64_7;
        __m128i out64_8;

        uint32_t *saturationCounter;

    private:
        bool is64;
    };
}

/** Load bias to output
 *
 * @param biases Pointer to biases
 * @param BPB Bytes per bias (1, 2 or 4)
 * @param[out] out1 First output register
 * @param[out] out2 Second output register
 */
static inline void GetBias(int8_t *biases, uint32_t BPB, RnnOutput &out)
{
    __m128i bias8;
    __m128i bias16_1;
    __m128i bias16_2;

    switch (BPB)
    {
    case 1:
        bias8 = _mm_load_si128((__m128i *)biases);
        out.out32_1 = _mm_cvtepi8_epi32(bias8);
        out.out32_2 = _mm_cvtepi8_epi32(_mm_bsrli_si128(bias8, 4));
        out.out32_3 = _mm_cvtepi8_epi32(_mm_bsrli_si128(bias8, 8));
        out.out32_4 = _mm_cvtepi8_epi32(_mm_bsrli_si128(bias8, 12));
        break;
    case 2:
        bias16_1 = _mm_load_si128((__m128i *)biases);
        bias16_2 = _mm_load_si128((__m128i *)(biases + sizeof(__m128i)));
        out.out32_1 = _mm_cvtepi16_epi32(bias16_1);
        out.out32_2 = _mm_cvtepi16_epi32(_mm_bsrli_si128(bias16_1, 8));
        out.out32_3 = _mm_cvtepi16_epi32(bias16_2);
        out.out32_4 = _mm_cvtepi16_epi32(_mm_bsrli_si128(bias16_2, 8));
        break;
    case 4:
        out.out32_1 = _mm_load_si128((__m128i *)biases);
        out.out32_2 = _mm_load_si128((__m128i *)(biases + sizeof(__m128i)));
        out.out32_3 = _mm_load_si128((__m128i *)(biases + 2 * sizeof(__m128i)));
        out.out32_4 = _mm_load_si128((__m128i *)(biases + 3 * sizeof(__m128i)));
        break;
    default:
        break;
    }
}

/** Madd16_64b - vectorized multiply and add for 16x64b outputs
 *
 * Adds sum of 'first' * 'second' to output ensuring no saturation occurs.
 * Returns 16x64b outputs packed into eight registers.
 *
 * Intermediate sums are stored as 32b and extended to 64b only when saturation would have
 * occured, which for 2B * 2B multiplication is every iteration and for 1B * 2B or 2B * 1B
 * every 256th iteration.
 *
 * This implementation can also cover 1B * 1B but 32b variant is much faster.
 *
 * ASSUMPTIONS:
 *   Arrays 'first' and 'second' have size modulo 16
 *   Array 'first' is bounded by 'limit'
 *
 * @param BPF Bytes per first array element (1 or 2)
 * @param BPS Bytes per second array element (1 or 2)
 * @param[in] in Array pointers, first array limit and step size
 * @param[out] out Output registers
 */
template <int BPF, int BPS>
void Madd16_64b(const Madd16Input &in, RnnOutput &out)
{
    static const size_t HSUM_WIDTH = sizeof(__m128i) / sizeof(int64_t);
    static const size_t HSUM_COUNT = IT_STEP / HSUM_WIDTH;

    // NOTE: Sum of int8_t * int16_t can overflow during 512th sum.
    //       We use madd which halves this limit.
    static const uint32_t PARTIAL_SUM_LIMIT = (BPF == 2 && BPS == 2) ? 1 : 255;

    __m128i first8;
    __m128i first16_1;
    __m128i first16_2;

    __m128i second8;
    __m128i second16_1;
    __m128i second16_2;

    __m128i mul_1;
    __m128i mul_2;

    __m128i sum32_1[IT_STEP];
    __m128i sum32_2[IT_STEP];
    __m128i sum64_1;
    __m128i sum64_2;

    __m128i sum_1[IT_STEP];
    __m128i sum_2[IT_STEP];

    int64_t hsum64_1[IT_STEP];
    int64_t hsum64_2[IT_STEP];

    __m128i hsum_1[HSUM_COUNT];
    __m128i hsum_2[HSUM_COUNT];

    int64_t partial_sum_counter = 0;

    const int8_t *first;
    const int8_t *second = in.second;

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        sum_1[k] = _mm_setzero_si128();
        sum_2[k] = _mm_setzero_si128();
        sum32_1[k] = _mm_setzero_si128();
        sum32_2[k] = _mm_setzero_si128();
    }

    for (uint32_t j = 0; j < in.limit / IT_STEP; ++j)
    {
        first = in.first + j * IT_STEP * BPF;

        if (BPS == 1)
        {
            second8 = _mm_load_si128((__m128i *)second);
            second16_1 = _mm_cvtepi8_epi16(second8);
            second16_2 = _mm_cvtepi8_epi16(_mm_bsrli_si128(second8, 8));
        }
        else
        {
            second16_1 = _mm_load_si128((__m128i *)second);
            second16_2 = _mm_load_si128((__m128i *)(second + sizeof(__m128i)));
        }

        for (uint32_t k = 0; k < IT_STEP; ++k)
        {
            if (BPF == 1)
            {
                first8 = _mm_load_si128((__m128i *)first);
                first16_1 = _mm_cvtepi8_epi16(first8);
                first16_2 = _mm_cvtepi8_epi16(_mm_bsrli_si128(first8, 8));
            }
            else
            {
                first16_1 = _mm_load_si128((__m128i *)first);
                first16_2 = _mm_load_si128((__m128i *)(first + sizeof(__m128i)));
            }

            mul_1 = _mm_madd_epi16(first16_1, second16_1);
            mul_2 = _mm_madd_epi16(first16_2, second16_2);

            sum32_1[k] = _mm_add_epi32(sum32_1[k], mul_1);
            sum32_2[k] = _mm_add_epi32(sum32_2[k], mul_2);

            first += in.step * BPF;
        }

        if (++partial_sum_counter >= PARTIAL_SUM_LIMIT)
        {
            for (uint32_t k = 0; k < IT_STEP; ++k)
            {
                sum64_1 = _mm_sum_extend64(sum32_1[k]);
                sum64_2 = _mm_sum_extend64(sum32_2[k]);
                sum_1[k] = _mm_add_epi64(sum_1[k], sum64_1);
                sum_2[k] = _mm_add_epi64(sum_2[k], sum64_2);

                sum32_1[k] = _mm_setzero_si128();
                sum32_2[k] = _mm_setzero_si128();
                partial_sum_counter = 0;
            }
        }

        second += IT_STEP * BPS;
    }

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        if (partial_sum_counter > 0)
        {
            sum64_1 = _mm_sum_extend64(sum32_1[k]);
            sum64_2 = _mm_sum_extend64(sum32_2[k]);
            sum_1[k] = _mm_add_epi64(sum_1[k], sum64_1);
            sum_2[k] = _mm_add_epi64(sum_2[k], sum64_2);
        }

        hsum64_1[k] = _mm_hsum_epi64(sum_1[k]);
        hsum64_2[k] = _mm_hsum_epi64(sum_2[k]);
    }

    for (uint32_t k = 0; k < HSUM_COUNT; ++k)
    {
        hsum_1[k] = _mm_loadu_si128((__m128i *)(hsum64_1 + k * HSUM_WIDTH));
        hsum_2[k] = _mm_loadu_si128((__m128i *)(hsum64_2 + k * HSUM_WIDTH));

        hsum_1[k] = _mm_add_epi64(hsum_1[k], hsum_2[k]);
    }

    out.out64_1 = _mm_add_epi64(out.out64_1, hsum_1[0]);
    out.out64_2 = _mm_add_epi64(out.out64_2, hsum_1[1]);
    out.out64_3 = _mm_add_epi64(out.out64_3, hsum_1[2]);
    out.out64_4 = _mm_add_epi64(out.out64_4, hsum_1[3]);
    out.out64_5 = _mm_add_epi64(out.out64_5, hsum_1[4]);
    out.out64_6 = _mm_add_epi64(out.out64_6, hsum_1[5]);
    out.out64_7 = _mm_add_epi64(out.out64_7, hsum_1[6]);
    out.out64_8 = _mm_add_epi64(out.out64_8, hsum_1[7]);
}

/** Madd16_32b - 32-bit version of Madd16_64b for 1B * 1B multiplication
 *
 * Sums are stored as 32b.
 *
 * NOTE: Unlike Madd16_64b, since output is 32b this function saturates output.
 *       Intermediate calculations will not saturate under previous assumptions
 *       but saturation can occur due to inital value of 'out'.
 *
 * See @ref Madd16_64b for other details.
 */
void Madd16_32b(const Madd16Input &in, RnnOutput &out)
{
    static const size_t HSUM_WIDTH = sizeof(__m128i) / sizeof(int32_t);
    static const size_t HSUM_COUNT = IT_STEP / HSUM_WIDTH;

    __m128i first8;
    __m128i first16_1;
    __m128i first16_2;

    __m128i second8;
    __m128i second16_1;
    __m128i second16_2;

    __m128i mul_1;
    __m128i mul_2;

    __m128i sum_1[IT_STEP];
    __m128i sum_2[IT_STEP];

    int32_t hsum32_1[IT_STEP];
    int32_t hsum32_2[IT_STEP];

    __m128i hsum_1[HSUM_COUNT];
    __m128i hsum_2[HSUM_COUNT];

    __m128i satAcc = _mm_setzero_si128();

    const int8_t *first;
    const int8_t *second = in.second;

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        sum_1[k] = _mm_setzero_si128();
        sum_2[k] = _mm_setzero_si128();
    }

    for (uint32_t j = 0; j < in.limit / IT_STEP; ++j)
    {
        first = in.first + j * IT_STEP;

        second8 = _mm_loadu_si128((__m128i *)second);
        second16_1 = _mm_cvtepi8_epi16(second8);
        second16_2 = _mm_cvtepi8_epi16(_mm_bsrli_si128(second8, 8));

        for (uint32_t k = 0; k < IT_STEP; ++k)
        {
            first8 = _mm_loadu_si128((__m128i *)first);
            first16_1 = _mm_cvtepi8_epi16(first8);
            first16_2 = _mm_cvtepi8_epi16(_mm_bsrli_si128(first8, 8));

            mul_1 = _mm_madd_epi16(first16_1, second16_1);
            mul_2 = _mm_madd_epi16(first16_2, second16_2);

            sum_1[k] = _mm_add_epi32(sum_1[k], mul_1);
            sum_2[k] = _mm_add_epi32(sum_2[k], mul_2);

            first += in.step;
        }

        second += IT_STEP;
    }

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        hsum32_1[k] = _mm_hsum_epi32(sum_1[k]);
        hsum32_2[k] = _mm_hsum_epi32(sum_2[k]);
    }

    for (uint32_t k = 0; k < HSUM_COUNT; ++k)
    {
        hsum_1[k] = _mm_loadu_si128((__m128i *)(hsum32_1 + k * HSUM_WIDTH));
        hsum_2[k] = _mm_loadu_si128((__m128i *)(hsum32_2 + k * HSUM_WIDTH));

        hsum_1[k] = _mm_add_epi32(hsum_1[k], hsum_2[k]);
    }

    out.out32_1 = _mm_adds_epi32(out.out32_1, hsum_1[0], &satAcc);
    out.out32_2 = _mm_adds_epi32(out.out32_2, hsum_1[1], &satAcc);
    out.out32_3 = _mm_adds_epi32(out.out32_3, hsum_1[2], &satAcc);
    out.out32_4 = _mm_adds_epi32(out.out32_4, hsum_1[3], &satAcc);

    *out.saturationCounter += _mm_test_anyMSB_epi32(satAcc);
}

template <int BPW>
void RNN1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    const uint32_t BUF = config->BufferElementCount[0];
    const uint32_t K = config->RequestConfig.Transform.inputElementCount;
    const uint32_t M = config->RequestConfig.Transform.outputElementCount;
    const uint32_t BPB = config->RequestConfig.Transform.bytesPerBias;

    const int8_t *inputs = (int8_t *)config->RequestConfig.Inputs;
    const int8_t *weights = config->RequestConfig.Transform.weights1B;

    int8_t *biases = (int8_t *)config->RequestConfig.Transform.biasesSimple;
    int8_t *feedbacks = (int8_t *)config->RequestConfig.Transform.feedbackBuffer;
    int32_t *outputs_1 = config->RequestConfig.Transform.output;
    int32_t *outputs_2 = outputs_1 + sizeof(__m128i) / sizeof(int32_t);
    int32_t *outputs_3 = outputs_2 + sizeof(__m128i) / sizeof(int32_t);
    int32_t *outputs_4 = outputs_3 + sizeof(__m128i) / sizeof(int32_t);

    RnnOutput out(config->SaturationCount);

    uint32_t segment1count = K / BUF;
    uint32_t segment2size1 = K % BUF;
    uint32_t segment2size2 = (M < BUF - segment2size1) ? M : BUF - segment2size1;
    uint32_t segment3count = (M - segment2size2) / BUF;
    uint32_t segment4size = (M - segment2size2) % BUF;

    size_t stepSize = IT_STEP * (K + M) * BPW;

    for (uint32_t i = 0; i < M / IT_STEP; ++i)
    {
        Madd16Input madd16input(weights + i * stepSize, inputs, 0, K + M);

        GetBias(biases, BPB, out);

        // Segment 1: full input buffers. If K < BUF this segment is skipped
        if (segment1count > 0)
        {
            uint32_t count = segment1count;

            madd16input.limit = BUF;

            do
            {
                if (BPW == 1)
                {
                    Madd16_32b(madd16input, out);
                }
                else
                {
                    out.ExtendTo64();
                    Madd16_64b<2, 1>(madd16input, out);
                    out.Saturate64();
                }

                madd16input.first += BUF * BPW;
                madd16input.second += BUF;
            } while (--count > 0);
        }

        out.ExtendTo64();

        // Segment 2: reminder of inputs and part of feedbacks up to BUF size
        //            with no saturation between these two parts
        madd16input.limit = segment2size1;

        // NOTE: We cannot optimize these sums for 1B weight because
        //       this would add an intermediate saturation
        Madd16_64b<BPW, 1>(madd16input, out);

        madd16input.first += segment2size1 * BPW;
        madd16input.second = feedbacks;
        madd16input.limit = segment2size2;

        Madd16_64b<BPW, 1>(madd16input, out);

        if (BPW == 1)
        {
            out.SaturatePack32();
        }
        else
        {
            out.Saturate64();
        }

        // Segment 3: full feedback buffers. If (M - segment2size2) < BUF this segment is skipped
        if (segment3count)
        {
            uint32_t count = segment3count;
            madd16input.first += madd16input.limit * BPW;
            madd16input.second += madd16input.limit;
            madd16input.limit = BUF;

            do
            {
                if (BPW == 1)
                {
                    Madd16_32b(madd16input, out);
                }
                else
                {
                    Madd16_64b<2, 1>(madd16input, out);
                    out.Saturate64();
                }

                madd16input.first += BUF * BPW;
                madd16input.second += BUF;
            } while (--count > 0);
        }

        // Segment 4: reminder of feedbacks. If M < (BUF - segment2size1) this segment is skipped
        if (segment4size > 0)
        {
            madd16input.limit = segment4size;

            if (BPW == 1)
            {
                Madd16_32b(madd16input, out);
            }
            else
            {
                Madd16_64b<2, 1>(madd16input, out);
            }
        }

        if (BPW != 1)
        {
            out.SaturatePack32();
        }

        _mm_storeu_si128((__m128i *)outputs_1, out.out32_1);
        _mm_storeu_si128((__m128i *)outputs_2, out.out32_2);
        _mm_storeu_si128((__m128i *)outputs_3, out.out32_3);
        _mm_storeu_si128((__m128i *)outputs_4, out.out32_4);

        outputs_1 += IT_STEP;
        outputs_2 += IT_STEP;
        outputs_3 += IT_STEP;
        outputs_4 += IT_STEP;

        biases += IT_STEP * BPB;
    }
}
