/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "igemv16.h"
#include "saturate.h"

#include "KernelArguments.h"

#include <cstdint>

void DiagonalKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig.Transform.weights2B;
    auto const * input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig.Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig.Transform.bytesPerBias, i)
                + (weight[i] * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig.Transform.weights2B;
    auto const * input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig.Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig.Transform.bytesPerBias, i)
                + (weight[i] * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig.Transform.weights2B;
    auto const * input = reinterpret_cast<int8_t const *>(config->RequestConfig.Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig.Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig.Transform.bytesPerBias, i)
                + (weight[i] * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}
