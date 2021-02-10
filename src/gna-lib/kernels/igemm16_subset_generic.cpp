/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"

#include "common.h"

#include <cstdint>

void AffineActiveListKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i, j, k, l;
    int16_t const * input;
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];

        input = config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights2B + i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            auto bias = getBias((void*)config->RequestConfig->Transform.biasesSimple, config->RequestConfig->Transform.bytesPerBias, i);
            output[l*config->RequestConfig->Transform.inputVectorCount + j] = static_cast<int32_t>(bias);

            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}
void AffineActiveListKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i, j, k, l;
    int16_t const * input;
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];

        input = config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights2B + i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            auto bias = getBias((void*)config->RequestConfig->Transform.biasesSimple, config->RequestConfig->Transform.bytesPerBias, i);
            output[l*config->RequestConfig->Transform.inputVectorCount + j] = static_cast<int32_t>(bias);

            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}

void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i, j, k, l;
    int8_t const * input;
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];

        input = (int8_t*)config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights2B + i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            auto bias = getBias((void*)config->RequestConfig->Transform.biasesSimple, config->RequestConfig->Transform.bytesPerBias, i);
            output[l*config->RequestConfig->Transform.inputVectorCount + j] = static_cast<int32_t>(bias);

            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}
