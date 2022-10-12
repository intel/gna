/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv8.h"
#include "igemv16.h"
#include "common.hpp"
#include "common_avx2.hpp"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

/** Transpose input and select template for calculation
 *  @param BPW Bytes per weight
 *  @param IndexSourceType Index source type
 *  @param config Execution config
 *  @param idx Index source. Must derive from @ref IndexSource. Use @ref SequenceIndexSource for simple bias and multibias and @ref ActiveListIndexSource for Active List
 *  @param biases Pointer to either simple bias or multibias
 *  @param bias_vector Index of vector used in multibias. For simple bias it must be set to 1
 */
template <size_t BPW, typename IndexSourceType>
static void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const * const config, IndexSourceType & idx, void * biases, uint32_t bias_vector = 1);

/** Affine kernel implementation for 1B input 1B weight
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    SequenceIndexSource idx(config->RequestConfig.Transform.outputElementCount);
    void *simple_bias = (void *)config->RequestConfig.Transform.biasesSimple;

    TransposeAndRun<1>(config, idx, simple_bias);
}

/** Affine kernel implementation for 1B input 1B weight, multibias
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    SequenceIndexSource idx(config->RequestConfig.Transform.outputElementCount);
    void *multi_bias = (void *)config->RequestConfig.Transform.multiBias;
    uint32_t bias_vector = config->RequestConfig.Transform.multiBiasVectorCount;

    TransposeAndRun<1>(config, idx, multi_bias, bias_vector);
}

/** Affine kernel implementation for 1B input 1B weight, active list
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    void *simple_bias = (void *)config->RequestConfig.Transform.biasesSimple;
    ActiveListIndexSource idx(al);

    TransposeAndRun<1>(config, idx, simple_bias);
}

/** Affine kernel implementation for 1B input 2B weight
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    SequenceIndexSource idx(config->RequestConfig.Transform.outputElementCount);
    void *simple_bias = (void *)config->RequestConfig.Transform.biasesSimple;

    TransposeAndRun<2>(config, idx, simple_bias);
}

/** Affine kernel implementation for 1B input 2B weight, multibias
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    SequenceIndexSource idx(config->RequestConfig.Transform.outputElementCount);
    void *multi_bias = (void *)config->RequestConfig.Transform.multiBias;
    uint32_t bias_vector = config->RequestConfig.Transform.multiBiasVectorCount;

    TransposeAndRun<2>(config, idx, multi_bias, bias_vector);
}

/** Affine kernel implementation for 1B input 2B weight, active list
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    void *simple_bias = (void *)config->RequestConfig.Transform.biasesSimple;
    ActiveListIndexSource idx(al);

    TransposeAndRun<2>(config, idx, simple_bias);
}

/** @brief Add 8x32b partial sum to 4x64 total sum
 *
 *  Partial sum is set to zero
 *
 *  @param[in,out] sum 4x64 sum
 *  @param[in,out] partial 8x32 partial sum
 */
inline void PartialSum(__m256i &sum, __m256i &partial)
{
    __m128i partial_lo32 = _mm256_castsi256_si128(partial);
    __m128i partial_hi32 = _mm256_extracti128_si256(partial, 1);

    __m256i partial_lo = _mm256_cvtepi32_epi64(partial_lo32);
    __m256i partial_hi = _mm256_cvtepi32_epi64(partial_hi32);

    sum = _mm256_add_epi64(sum, partial_lo);
    sum = _mm256_add_epi64(sum, partial_hi);

    partial = _mm256_setzero_si256();
}

/** Generic implementation for simple bias, multibias and active list
 *
 * This implementation expects input to be already transposed.
 *
 * @param BPW Bytes per weight
 * @param N Input vector size
 * @param IndexSourceType Index source type
 * @param config Execution config
 * @param idx Index source. Must derive from @ref IndexSource. Use @ref SequenceIndexSource for simple bias and multibias and @ref ActiveListIndexSource for Active List
 * @param biases Pointer to either simple bias or multibias
 * @param bias_vector Index of vector used in multibias. For simple bias it must be set to 1
 */
template <size_t BPW, size_t N, class IndexSourceType>
static void Affine(ExecutionKernelConfig<AffineConfig> const * const config, IndexSourceType & idx, void * biases, const uint32_t bias_vector)
{
    static_assert(std::is_base_of<IndexSource, IndexSourceType>::value, "Index source type must derive from IndexSource");

    static const uint32_t IT_STEP = 16;

    // NOTE: Sum of int8_t * int16_t can overflow during 512th sum.
    //       We use madd which halves this limit.
    static const uint32_t PARTIAL_SUM_LIMIT = 255;

    // NOTE: For compatibility with HW we limit the amount of unsaturated sums based on the buffer size
    const uint32_t unsaturated_sum_limit = (uint32_t)((config->BufferElementCount[N - 1]) / N / IT_STEP);

    const uint32_t K = config->RequestConfig.Transform.inputElementCount;
    const uint32_t KK = K / IT_STEP;

    int8_t const *const weights = config->RequestConfig.Transform.weights1B;

    int8_t const *inputs[N] = {(const int8_t *)config->Intermediate->d0};
    int32_t *outputs[N] = {reinterpret_cast<int32_t *>(config->RequestConfig.Outputs)};

    size_t inputOffset = 0;
    size_t weightOffset = 0;

    __m128i weight8;
    __m256i weight16;

    __m128i input8[N];
    __m256i input16[N];
    __m256i sum[N];
    __m256i sum_partial[N];
    __m256i mul[N];

    int64_t final_sum[N];

    uint32_t partial_sum_counter = 0;
    uint32_t unsaturated_sum_counter = 0;

    for (uint32_t n = 1; n < N; ++n)
    {
        inputs[n] = inputs[n - 1] + K;
        outputs[n] = outputs[n - 1] + 1;
    }

    while (idx.HasNext())
    {
        uint32_t i = idx.Next();

        const int32_t bias = getBias(biases, config->RequestConfig.Transform.bytesPerBias, i * bias_vector);

        for (uint32_t n = 0; n < N; ++n)
        {
            sum[n] = _mm256_setzero_si256();
            sum_partial[n] = _mm256_setzero_si256();
            final_sum[n] = bias;
        }

        unsaturated_sum_counter = 0;

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = (inputOffset + i * K) * BPW;

            if (BPW == 1)
            {
                weight8 = _mm_load_si128((__m128i *)(weights + weightOffset));
                weight16 = _mm256_cvtepi8_epi16(weight8);
            }
            else
            {
                weight16 = _mm256_loadu_si256((__m256i *)(weights + weightOffset));
            }

            for (uint32_t n = 0; n < N; ++n)
            {

                input8[n] = _mm_load_si128((__m128i *)(inputs[n] + inputOffset));
                input16[n] = _mm256_cvtepi8_epi16(input8[n]);
                mul[n] = _mm256_madd_epi16(input16[n], weight16);

                if (BPW == 1)
                {
                    sum[n] = _mm256_add_epi32(sum[n], mul[n]);
                }
                else
                {
                    sum_partial[n] = _mm256_add_epi32(sum_partial[n], mul[n]);
                }
            }

            // NOTE: Partial sum is only needed for 2B weight
            if (BPW != 1 && ++partial_sum_counter >= PARTIAL_SUM_LIMIT)
            {
                for (uint32_t n = 0; n < N; ++n)
                {
                    PartialSum(sum[n], sum_partial[n]);
                }

                partial_sum_counter = 0;
            }

            // NOTE: This part is only for HW compatibility
            if (++unsaturated_sum_counter >= unsaturated_sum_limit)
            {
                // NOTE: Partial sum is only needed for 2B weight
                if (BPW != 1 && partial_sum_counter > 0)
                {
                    for (uint32_t n = 0; n < N; ++n)
                    {
                        PartialSum(sum[n], sum_partial[n]);
                    }

                    partial_sum_counter = 0;
                }

                for (uint32_t n = 0; n < N; ++n)
                {
                    final_sum[n] += (BPW == 1) ? _mm256_hsum_epi32(sum[n]) : _mm256_hsum_epi64(sum[n]);

                    saturate(&final_sum[n], config->SaturationCount);
                    sum[n] = _mm256_setzero_si256();
                }

                unsaturated_sum_counter = 0;
            }
        }

        // NOTE: Partial sum is only needed for 2B weight
        if (BPW != 1 && partial_sum_counter > 0)
        {
            for (uint32_t n = 0; n < N; ++n)
            {
                PartialSum(sum[n], sum_partial[n]);
            }
            partial_sum_counter = 0;
        }

        for (uint32_t n = 0; n < N; ++n)
        {
            final_sum[n] += (BPW == 1) ? _mm256_hsum_epi32(sum[n]) : _mm256_hsum_epi64(sum[n]);

            saturate_store_out(&final_sum[n], outputs[n], config->SaturationCount);
            outputs[n] += N;
        }
    }
}

template <size_t BPW, typename IndexSourceType>
void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const * const config, IndexSourceType & idx, void * biases, uint32_t bias_vector)
{
    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    switch (config->RequestConfig.Transform.inputVectorCount)
    {
    case 1:
        Affine<BPW, 1>(config, idx, biases, bias_vector);
        break;
    case 2:
        Affine<BPW, 2>(config, idx, biases, bias_vector);
        break;
    case 3:
        Affine<BPW, 3>(config, idx, biases, bias_vector);
        break;
    case 4:
        Affine<BPW, 4>(config, idx, biases, bias_vector);
        break;
    case 5:
        Affine<BPW, 5>(config, idx, biases, bias_vector);
        break;
    case 6:
        Affine<BPW, 6>(config, idx, biases, bias_vector);
        break;
    case 7:
        Affine<BPW, 7>(config, idx, biases, bias_vector);
        break;
    case 8:
        Affine<BPW, 8>(config, idx, biases, bias_vector);
        break;
    default:
        break;
    }
}
