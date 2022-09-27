/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "common_avx2.hpp"
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
static void RNN1B(ExecutionKernelConfig<RecurrentConfig> const * const config);

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
                out64_1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(out32_1));
                out64_2 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(out32_1, 1));
                out64_3 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(out32_2));
                out64_4 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(out32_2, 1));

                is64 = true;
            }
        }

        /** Saturate 64-bit data to 32-bit. Result remains 64-bit. Does nothing if data is 32-bit */
        void Saturate64()
        {
            if (is64)
            {
                out64_1 = _mm256_sat_epi64(out64_1, saturationCounter);
                out64_2 = _mm256_sat_epi64(out64_2, saturationCounter);
                out64_3 = _mm256_sat_epi64(out64_3, saturationCounter);
                out64_4 = _mm256_sat_epi64(out64_4, saturationCounter);
            }
        }

        /** Packs 64-bit data to 32-bit with saturation. Does nothing if data is already 32-bit */
        void SaturatePack32()
        {
            if (is64)
            {
                out32_1 = _mm256_packs_epi64(out64_1, out64_2, saturationCounter);
                out32_2 = _mm256_packs_epi64(out64_3, out64_4, saturationCounter);
                is64 = false;
            }
        }

        __m256i out32_1;
        __m256i out32_2;

        __m256i out64_1;
        __m256i out64_2;
        __m256i out64_3;
        __m256i out64_4;

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
static inline void GetBias(int8_t *biases, uint32_t BPB, __m256i &out1, __m256i &out2)
{
    __m128i bias8;
    __m256i bias16;

    switch (BPB)
    {
    case 1:
        bias8 = _mm_load_si128((__m128i *)biases);
        out1 = _mm256_cvtepi8_epi32(bias8);
        out2 = _mm256_cvtepi8_epi32(_mm_srli_si128(bias8, 8));
        break;
    case 2:
        bias16 = _mm256_load_si256((__m256i *)biases);
        out1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(bias16));
        out2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(bias16, 1));
        break;
    case 4:
        out1 = _mm256_load_si256((__m256i *)biases);
        out2 = _mm256_load_si256((__m256i *)(biases + sizeof(__m256i)));
        break;
    default:
        break;
    }
}

/** Madd16_64b - vectorized multiply and add for 16x64b outputs
 *
 * Adds sum of 'first' * 'second' to output ensuring no saturation occurs.
 * Returns 16x64b outputs packed into four registers.
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
    // NOTE: Sum of int8_t * int16_t can overflow during 512th sum.
    //       We use madd which halves this limit.
    static const uint32_t PARTIAL_SUM_LIMIT = (BPF == 2 && BPS == 2) ? 1 : 255;

    __m256i first16;
    __m256i second16;

    __m256i mul;

    __m256i sum32[IT_STEP];
    __m256i sum64;

    __m256i sum[IT_STEP];

    __m256i hsum64_1;
    __m256i hsum64_2;
    __m256i hsum64_3;
    __m256i hsum64_4;

    int64_t hsum64[IT_STEP];

    int64_t partial_sum_counter = 0;

    const int8_t *first;
    const int8_t *second = in.second;

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        sum[k] = _mm256_setzero_si256();
        sum32[k] = _mm256_setzero_si256();
    }

    for (uint32_t j = 0; j < in.limit / IT_STEP; ++j)
    {
        first = in.first + j * IT_STEP * BPF;

        if (BPS == 1)
        {
            second16 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)second));
        }
        else
        {
            second16 = _mm256_loadu_si256((__m256i *)second);
        }

        for (uint32_t k = 0; k < IT_STEP; ++k)
        {
            if (BPF == 1)
            {
                first16 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)first));
            }
            else
            {
                first16 = _mm256_loadu_si256((__m256i *)first);
            }

            mul = _mm256_madd_epi16(first16, second16);
            sum32[k] = _mm256_add_epi32(sum32[k], mul);

            first += in.step * BPF;
        }

        if (++partial_sum_counter >= PARTIAL_SUM_LIMIT)
        {
            for (uint32_t k = 0; k < IT_STEP; ++k)
            {
                sum64 = _mm256_sum_extend64(sum32[k]);
                sum[k] = _mm256_add_epi64(sum[k], sum64);
                sum32[k] = _mm256_setzero_si256();
                partial_sum_counter = 0;
            }
        }

        second += IT_STEP * BPS;
    }

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        if (partial_sum_counter > 0)
        {
            sum64 = _mm256_sum_extend64(sum32[k]);
            sum[k] = _mm256_add_epi64(sum[k], sum64);
        }
        hsum64[k] = _mm256_hsum_epi64(sum[k]);
    }

    hsum64_1 = _mm256_loadu_si256((__m256i *)hsum64);
    hsum64_2 = _mm256_loadu_si256((__m256i *)(hsum64 + sizeof(__m256i) / sizeof(int64_t)));
    hsum64_3 = _mm256_loadu_si256((__m256i *)(hsum64 + 2 * sizeof(__m256i) / sizeof(int64_t)));
    hsum64_4 = _mm256_loadu_si256((__m256i *)(hsum64 + 3 * sizeof(__m256i) / sizeof(int64_t)));

    out.out64_1 = _mm256_add_epi64(out.out64_1, hsum64_1);
    out.out64_2 = _mm256_add_epi64(out.out64_2, hsum64_2);
    out.out64_3 = _mm256_add_epi64(out.out64_3, hsum64_3);
    out.out64_4 = _mm256_add_epi64(out.out64_4, hsum64_4);
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
    __m128i first8;
    __m256i first16;
    __m128i second8;
    __m256i second16;
    __m256i mul;
    __m256i sum[IT_STEP];

    __m256i hsum_lo;
    __m256i hsum_hi;
    int32_t hsum[IT_STEP];

    __m256i satAcc = _mm256_setzero_si256();

    const int8_t *first;
    const int8_t *second = in.second;

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        sum[k] = _mm256_setzero_si256();
    }

    for (uint32_t j = 0; j < in.limit / IT_STEP; ++j)
    {
        first = in.first + j * IT_STEP;

        second8 = _mm_load_si128((__m128i *)second);
        second16 = _mm256_cvtepi8_epi16(second8);

        for (uint32_t k = 0; k < IT_STEP; ++k)
        {
            first8 = _mm_load_si128((__m128i *)first);
            first16 = _mm256_cvtepi8_epi16(first8);

            mul = _mm256_madd_epi16(first16, second16);

            sum[k] = _mm256_add_epi32(sum[k], mul);

            first += in.step;
        }

        second += IT_STEP;
    }

    for (uint32_t k = 0; k < IT_STEP; ++k)
    {
        hsum[k] = _mm256_hsum_epi32(sum[k]);
    }

    hsum_lo = _mm256_loadu_si256((__m256i *)hsum);
    hsum_hi = _mm256_loadu_si256((__m256i *)(hsum + sizeof(__m256i) / sizeof(int32_t)));

    out.out32_1 = _mm256_adds_epi32(out.out32_1, hsum_lo, &satAcc);
    out.out32_2 = _mm256_adds_epi32(out.out32_2, hsum_hi, &satAcc);

    *out.saturationCounter += _mm256_test_anyMSB_epi32(satAcc);
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
    int32_t *outputs_lo = config->RequestConfig.Transform.output;
    int32_t *outputs_hi = outputs_lo + sizeof(__m256i) / sizeof(int32_t);

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

        GetBias(biases, BPB, out.out32_1, out.out32_2);

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

        _mm256_storeu_si256((__m256i *)outputs_lo, out.out32_1);
        _mm256_storeu_si256((__m256i *)outputs_hi, out.out32_2);

        outputs_lo += IT_STEP;
        outputs_hi += IT_STEP;

        biases += IT_STEP * BPB;
    }
}
