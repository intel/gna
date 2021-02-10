/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "igemv.h"
#include "igemv16.h"

#include "KernelArguments.h"

#include "gna-api-types-xnn.h"

#include <cstdint>

void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;

    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias);
    int16_t const * input;
    int16_t * feedback;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    uint32_t kparts = config->RequestConfig->Transform.inputElementCount / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_rem = config->RequestConfig->Transform.inputElementCount % config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t middle_fill = config->BufferElementCount[0 + XNN_N_GROUP_MAX] - kpart_rem;
    uint32_t middle_part = (config->RequestConfig->Transform.outputElementCount < middle_fill) ? config->RequestConfig->Transform.outputElementCount : middle_fill;
    uint32_t mm = config->RequestConfig->Transform.outputElementCount - middle_part;
    uint32_t mparts = mm / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t mpart_rem = mm % config->BufferElementCount[0 + XNN_N_GROUP_MAX];

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *feedback++ * *weight++;
            }

            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        output++;
    }
}

void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;

    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias);
    int8_t const * input;
    int8_t * feedback;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    uint32_t kparts = config->RequestConfig->Transform.inputElementCount / config->BufferElementCount[0];
    uint32_t kpart_rem = config->RequestConfig->Transform.inputElementCount % config->BufferElementCount[0];
    uint32_t middle_fill = config->BufferElementCount[0] - kpart_rem;
    uint32_t middle_part = (config->RequestConfig->Transform.outputElementCount < middle_fill) ? config->RequestConfig->Transform.outputElementCount : middle_fill;
    uint32_t mm = config->RequestConfig->Transform.outputElementCount - middle_part;
    uint32_t mparts = mm / config->BufferElementCount[0];
    uint32_t mpart_rem = mm % config->BufferElementCount[0];

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = (int8_t*)config->RequestConfig->Inputs;
        feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0]; j++)
            {
                if (config->RequestConfig->Transform.bytesPerOutput == 1)
                {
                    sum += *feedback++ * *weight++;
                }
                else if (config->RequestConfig->Transform.bytesPerOutput == 2)
                {
                    sum += *(int16_t*)feedback * *weight++;
                    feedback += 2;
                }
            }

            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        output++;
    }
}

void RecurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;

    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias);
    int16_t const * input;
    int8_t * feedback;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    uint32_t kparts = config->RequestConfig->Transform.inputElementCount / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_rem = config->RequestConfig->Transform.inputElementCount % config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t middle_fill = config->BufferElementCount[0 + XNN_N_GROUP_MAX] - kpart_rem;
    uint32_t middle_part = (config->RequestConfig->Transform.outputElementCount < middle_fill) ? config->RequestConfig->Transform.outputElementCount : middle_fill;
    uint32_t mm = config->RequestConfig->Transform.outputElementCount - middle_part;
    uint32_t mparts = mm / config->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t mpart_rem = mm % config->BufferElementCount[0 + XNN_N_GROUP_MAX];

    for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
    {
        sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = (int8_t*)config->RequestConfig->Transform.feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                if (config->RequestConfig->Transform.bytesPerOutput == 1)
                {
                    sum += *feedback++ * *weight++;
                }
                else if (config->RequestConfig->Transform.bytesPerOutput == 2)
                {
                    sum += *(int16_t*)feedback * *weight++;
                    feedback += 2;
                }
            }

            saturate_store_out(&sum, output, config->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            if (config->RequestConfig->Transform.bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->RequestConfig->Transform.bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->SaturationCount);
        output++;
    }
}

