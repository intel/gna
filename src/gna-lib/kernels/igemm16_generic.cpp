/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"

#include <cstdint>

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount *
                                           config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j;
    uint32_t k;
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount *
                                           config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j;
    uint32_t k;
    int8_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias
        + (config->RequestConfig->Transform.outputElementCount
                * config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = (int8_t*)config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = (int8_t*)config->RequestConfig->Transform.multiBias;
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias);
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = (int8_t*)config->RequestConfig->Transform.multiBias;
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = (int8_t*)config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}
