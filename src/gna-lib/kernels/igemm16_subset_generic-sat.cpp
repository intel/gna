/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "saturate.h"
#include "igemv16.h"

#include "KernelArguments.h"


#include <cstdint>

void AffineActiveListKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t i;
    uint32_t j;
    uint32_t kk;
    uint32_t k;
    uint32_t l;

    auto inputVectorCount = config->RequestConfig.Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig.Transform.inputElementCount;

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    int64_t sum = 0;
    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < inputVectorCount; j++) {

            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial;
                weight = config->RequestConfig.Transform.weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineActiveListKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
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

    kpartial = (config->BufferElementCount[inputVectorCount - 1 + XNN_N_GROUP_MAX]) / inputVectorCount;
    nKpartial = inputElementCount / kpartial;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    int64_t sum = 0;
    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < inputVectorCount; j++) {

            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->Intermediate->d0 + j*inputElementCount + kk * kpartial;
                weight = config->RequestConfig.Transform.weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
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
    int16_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig.Outputs);

    int64_t sum = 0;
    for (l = 0; l < al.count; l++) {
        i = al.indices[l];
        for (j = 0; j < config->RequestConfig.Transform.inputVectorCount; j++) {
            sum = getBias(config->RequestConfig.Transform.biasesSimple, config->RequestConfig.Transform.bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->Intermediate->d0) + j*inputElementCount + kk * kpartial;
                weight = config->RequestConfig.Transform.weights2B + i*inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &output[l*inputVectorCount + j], config->SaturationCount);
                sum = (int64_t)output[l*inputVectorCount + j]; // load the temp sum
            }
        }
    }
}
