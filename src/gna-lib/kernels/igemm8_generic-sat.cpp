/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv8.h"
#include "igemv16.h"

#include "KernelArguments.h"


#include <cstdint>

void AffineKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t kk;
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    int64_t sum;
    int64_t acc;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = config->RequestConfig.Transform.biasesCompound[i].Bias;
            for (kk = 0; kk < nKpartial + 1; kk++) {
                niters = kpartial < inputElementCount - kk * kpartial
                    ? kpartial : inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; k++)
                {
                    input = config->Intermediate->d0 + j * inputElementCount + kk * kpartial + k * 512;
                    weight = config->RequestConfig.Transform.weights1B + i * inputElementCount + kk * kpartial + k * 512;
                    for (l = 0; l < 512; l++)
                    {
                        acc += weight[l] * input[l];
                    }
                    sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;
                    acc = 0;
                }

                input = config->Intermediate->d0 + j * inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->RequestConfig.Transform.weights1B + i * inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; k++)
                {
                    acc += weight[k] * input[k];
                }
                // conversion to signed int needed - multiplier is unsigned, and temporary result would biasEnd also unsigned
                sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j];
            }
        }
    }
}

void AffineKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t niters, acc_iters, rem_iters;
    uint32_t i, j, k, l;
    int64_t sum;
    int64_t acc;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = config->RequestConfig.Transform.biasesCompound[i].Bias;
            for (kk = 0; kk < nKpartial + 1; kk++) {
                niters = kpartial < inputElementCount - kk * kpartial ? kpartial : inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; k++)
                {
                    input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + k * 512;
                    weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + k * 512;
                    for (l = 0; l < 512; l++)
                    {
                        acc += weight[l] * input[l];
                    }
                    sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;
                    acc = 0;
                }

                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; k++)
                {
                    acc += weight[k] * input[k];
                }
                // conversion to signed int needed - multiplier is unsigned, and temporary result would biasEnd also unsigned
                sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j];
            }
        }
    }
}

void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    int8_t const * input;
    int8_t const * weight;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    kpartial = (config->BufferElementCount[inputVectorCount - 1]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk * kpartial;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineMultiBiasKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    const uint32_t kpartial = config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX] / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    int64_t sum;
    int64_t acc;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    for (i = 0; i < outputElementCount; ++i)
    {
        for (j = 0; j < inputVectorCount; ++j)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; ++kk) {
                niters = kpartial < inputElementCount - kk * kpartial ? kpartial : inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; ++k)
                {
                    input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + k * 512;
                    weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + k * 512;
                    for (l = 0; l < 512; ++l)
                    {
                        acc += weight[l] * input[l];
                    }
                    sum += acc * config->RequestConfig.Transform.weightScaleFactors[i].Multiplier;
                    acc = 0;
                }

                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; ++k)
                {
                    acc += weight[k] * input[k];
                }
                // conversion to signed int needed - multiplier is unsigned, and temporary result would biasEnd also unsigned
                sum += acc * config->RequestConfig.Transform.weightScaleFactors[i].Multiplier;
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = output[i*inputVectorCount + j];
            }
        }
    }
}

void AffineMultiBiasKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    uint32_t niters, acc_iters, rem_iters;
    uint32_t i, j, k, l;
    int64_t sum;
    int64_t acc;
    uint32_t kk;
    const uint32_t kpartial = config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX] / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    for (i = 0; i < outputElementCount; ++i)
    {
        for (j = 0; j < inputVectorCount; ++j)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; ++kk) {
                niters = kpartial < inputElementCount - kk * kpartial ? kpartial : inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; ++k)
                {
                    input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + k * 512;
                    weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + k * 512;
                    for (l = 0; l < 512; ++l)
                    {
                        acc += weight[l] * input[l];
                    }
                    sum += acc * config->RequestConfig.Transform.weightScaleFactors[i].Multiplier;
                    acc = 0;
                }

                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; ++k)
                {
                    acc += weight[k] * input[k];
                }
                // conversion to signed int needed - multiplier is unsigned, and temporary result would biasEnd also unsigned
                sum += acc * config->RequestConfig.Transform.weightScaleFactors[i].Multiplier;
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = output[i*inputVectorCount + j];
            }
        }
    }
}

void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;

    const uint32_t kpartial = (config->BufferElementCount[inputVectorCount - 1]) / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    int8_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk*kpartial;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk*kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++)
                {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = output[i*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

