/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "XnnKernel.h"

#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"
#include "Macros.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace GNA
{

#define activationKernelImpl KERNEL(activationKernelImpl)
#define recurrentKernelImpl1B KERNEL(recurrentKernelImpl1B)
#define recurrentKernelImpl2B KERNEL(recurrentKernelImpl2B)
#define copyKernelImpl KERNEL(copyKernelImpl)
#define copyKernelImpl1B KERNEL(copyKernelImpl1B)
#define copyKernelImpl2B KERNEL(copyKernelImpl2B)
#define InitializeActivationFunctions KERNEL(InitializeActivationFunctions)

#if OPT_LEVEL < 2
#define recurrentKernelImpl1B1B KERNEL(recurrentKernelImpl1B1B)
#define recurrentKernelImpl1B2B KERNEL(recurrentKernelImpl1B2B)
#define recurrentKernelImpl2B1B KERNEL(recurrentKernelImpl2B1B)
#define recurrentKernelImpl2B2B KERNEL(recurrentKernelImpl2B2B)
#endif

void activationKernelImpl(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    config->RequestConfig->Transform.Kernel->InitializeActivationFunctions();
    config->RequestConfig->Transform.Kernel->ActivateAll(config);
}

void recurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

#if OPT_LEVEL < 2
void recurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B1B(config);
        config->RequestConfig->Inputs += inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}
void recurrentKernelImpl1B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B2B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B1B(config);
        config->RequestConfig->Inputs += inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B2B(config);
        config->RequestConfig->Inputs += inputElementCount * 2;
        if(config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

#endif
void copyKernelImpl(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int16_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            config->output + (config->outputColumnCount * row),
            bytesToCopy,
            config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

void copyKernelImpl1B(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int8_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            (int8_t*)config->output + (config->outputColumnCount * row),
            bytesToCopy,
            (int8_t*)config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

void copyKernelImpl2B(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int16_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            config->output + (config->outputColumnCount * row),
            bytesToCopy,
            config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}
#if OPT_LEVEL >=2
static void CodeCaveMitigationFakeKernel()
{
    throw std::logic_error("Call to not defined GNA kernel found!");
}
#endif

XnnKernel KERNEL(xnnKernel) =
{
    AffineKernelImpl1B,
    AffineKernelImpl2B,

    AffineActiveListKernelImpl1B,
    AffineActiveListKernelImpl2B,

    AffineMultiBiasKernelImpl1B,
    AffineMultiBiasKernelImpl2B,

    DiagonalKernelImpl1B,
    DiagonalKernelImpl2B,

    recurrentKernelImpl1B,
    recurrentKernelImpl2B,

    ConvolutionKernelImpl,
    ConvolutionPoolingKernelImpl,

    activationKernelImpl,
    TransposeKernelImpl,
    copyKernelImpl,

#if OPT_LEVEL < 2

    AffineKernelImpl1B1B,
    AffineKernelImpl2B1B,
    AffineKernelImpl1B2B,
    AffineKernelImpl2B2B,
    AffineActiveListKernelImpl1B1B,
    AffineActiveListKernelImpl2B1B,
    AffineActiveListKernelImpl1B2B,
    AffineActiveListKernelImpl2B2B,
    AffineMultiBiasKernelImpl1B1B,
    AffineMultiBiasKernelImpl2B1B,
    AffineMultiBiasKernelImpl1B2B,
    AffineMultiBiasKernelImpl2B2B,
    DiagonalKernelImpl1B1B,
    DiagonalKernelImpl2B1B,
    DiagonalKernelImpl1B2B,
    DiagonalKernelImpl2B2B,
    recurrentKernelImpl1B1B,
    recurrentKernelImpl2B1B,
    recurrentKernelImpl1B2B,
    recurrentKernelImpl2B2B,
    ConvolutionKernelImpl1B,
    ConvolutionPoolingKernelImpl1B,
    ConvolutionKernelImpl2B,
    ConvolutionPoolingKernelImpl2B,
    TransposeKernelImpl1B,
    TransposeKernelImpl2B,
    copyKernelImpl1B,
    copyKernelImpl2B,

    Convolution2DKernelImpl1B1B,
    Convolution2DKernelImpl1B2B,
    Convolution2DKernelImpl2B1B,
    Convolution2DKernelImpl2B2B,

    Pooling2DKernelImpl1B,
    Pooling2DKernelImpl2B,
    Pooling2DKernelImpl4B
#else
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineActiveListKernel)CodeCaveMitigationFakeKernel,
    (AffineActiveListKernel)CodeCaveMitigationFakeKernel,
    (AffineActiveListKernel)CodeCaveMitigationFakeKernel,
    (AffineActiveListKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (AffineKernel)CodeCaveMitigationFakeKernel,
    (RecurrentKernel)CodeCaveMitigationFakeKernel,
    (RecurrentKernel)CodeCaveMitigationFakeKernel,
    (RecurrentKernel)CodeCaveMitigationFakeKernel,
    (RecurrentKernel)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel)CodeCaveMitigationFakeKernel,
    (ConvolutionPoolingKernel)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel)CodeCaveMitigationFakeKernel,
    (ConvolutionPoolingKernel)CodeCaveMitigationFakeKernel,
    (TransposeKernel)CodeCaveMitigationFakeKernel,
    (TransposeKernel)CodeCaveMitigationFakeKernel,
    (CopyKernel)CodeCaveMitigationFakeKernel,
    (CopyKernel)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel2D)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel2D)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel2D)CodeCaveMitigationFakeKernel,
    (ConvolutionKernel2D)CodeCaveMitigationFakeKernel,
    (PoolingKernel2D)CodeCaveMitigationFakeKernel,
    (PoolingKernel2D)CodeCaveMitigationFakeKernel,
    (PoolingKernel2D)CodeCaveMitigationFakeKernel
#endif
};

}
