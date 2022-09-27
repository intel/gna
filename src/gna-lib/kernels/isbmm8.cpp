/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "igemv8.h"
#include "saturate.h"

#include "KernelArguments.h"

#include <cstdint>

void DiagonalKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum;
    int64_t weightValue;
    int8_t const * weight = config->RequestConfig.Transform.weights1B;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    BiasCompound const * bias = config->RequestConfig.Transform.biasesCompound;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        weightValue = bias[i].Multiplier * weight[i];
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum = bias[i].Bias + (weightValue * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);
            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i, j;
    int64_t sum;
    int64_t weightValue;
    int8_t const * weight = config->RequestConfig.Transform.weights1B;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig.Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    BiasCompound const * bias = config->RequestConfig.Transform.biasesCompound;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        weightValue = bias[i].Multiplier * weight[i];
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum =  bias[i].Bias + (weightValue * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);
            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int8_t const * weight = config->RequestConfig.Transform.weights1B;
    int8_t const * input = (int8_t*)config->RequestConfig.Inputs;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig.Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig.Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++)
        {
            sum =  getBias(bias, config->RequestConfig.Transform.bytesPerBias, i)
                +  (weight[i] * (int64_t) input[i * config->RequestConfig.Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig.Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}
