/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv16.h"

#include "KernelArguments.h"

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    int16_t const * weight;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial;
                weight = weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    int16_t const * weight;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial;
                weight = weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    int16_t const * weight;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int8_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    kpartial = (config->BufferElementCount[inputVectorCount - 1]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk * kpartial;
                weight = weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[i*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[i*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    const uint32_t kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * weight;
    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = config->Intermediate->d0 + j*inputElementCount + kk*kpartial;
                weight = weights2B + i*inputElementCount + kk*kpartial;
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

void AffineMultiBiasKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    int64_t sum = 0;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    const uint32_t kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * weight;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = config->Intermediate->d0 + j*inputElementCount + kk*kpartial;
                weight = weights2B + i*inputElementCount + kk*kpartial;
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

void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;
    auto input = reinterpret_cast<int8_t const *>(config->RequestConfig.Inputs);

    auto weights2B = config->RequestConfig.Transform.weights2B;

    auto outputElementCount = config->RequestConfig.Transform.outputElementCount;
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    const uint32_t kpartial = (config->BufferElementCount[inputVectorCount - 1]) / inputVectorCount;
    const uint32_t nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    int16_t const * weight;
    int64_t sum = 0;
    for (i = 0; i < outputElementCount; i++)
    {
        for (j = 0; j < inputVectorCount; j++)
        {
            sum = getBias(config->RequestConfig.Transform.multiBias, config->RequestConfig.Transform.bytesPerBias, i*config->RequestConfig.Transform.multiBiasVectorCount);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk*kpartial;
                weight = weights2B + i*inputElementCount + kk*kpartial;
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

