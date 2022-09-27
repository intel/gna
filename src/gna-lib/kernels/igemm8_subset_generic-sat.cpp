/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv8.h"
#include "igemv16.h"

#include "KernelArguments.h"

#include <cstdint>

void AffineActiveListKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t m;
    int64_t sum;
    int64_t acc;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < inputVectorCount; j++) {
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
                    for (m = 0; m < 512; m++)
                    {
                        acc += weight[m] * input[m];
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
                sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;

                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j];
            }
        }
    }
}

void AffineActiveListKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t m;
    int64_t sum;
    int64_t acc;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < inputVectorCount; j++) {
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
                    for (m = 0; m < 512; m++)
                    {
                        acc += weight[m] * input[m];
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
                sum += acc * config->RequestConfig.Transform.biasesCompound[i].Multiplier;

                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j];
            }
        }
    }
}

void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    int8_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    int64_t sum = 0;
    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < inputVectorCount; j++) {

            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk * kpartial;
                weight = config->RequestConfig.Transform.weights1B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}
