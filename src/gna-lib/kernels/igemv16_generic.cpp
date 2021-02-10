/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv16.h"

#include "KernelArguments.h"

#include <cstdint>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount;
    int16_t * feedback = config->RequestConfig->Transform.feedbackBuffer;
    int16_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);

    for (; bias < biasEnd; bias+=config->RequestConfig->Transform.bytesPerBias, output++)
    {
        *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        for (; input < inputEnd;)
        {
            *output += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
        {
            *output += *feedback++ * *weight++;
        }
    }
}

void RecurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount;
    int8_t * feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;
    int8_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerOutput;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias, output++)
    {
        *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;

        for (; input < inputEnd;)
        {
            *output += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                *output += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                *output += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }
    }
}

void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias;
    int8_t const * input = (int8_t*)config->RequestConfig->Inputs;
    int8_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount;
    int8_t * feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;
    int8_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerOutput;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias, output++)
    {
        *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = (int8_t*)config->RequestConfig->Inputs;
        feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;

        for (; input < inputEnd;)
        {
            *output += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                *output += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                *output += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }
    }
}
