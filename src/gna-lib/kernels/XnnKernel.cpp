/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
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
#include <stdexcept>

static GNA::VoidKernel GetXnnKernelHelper(GNA::KernelType type);

namespace GNA
{

#define activationKernelImpl KERNEL(activationKernelImpl)
#define recurrentKernelImpl1B KERNEL(recurrentKernelImpl1B)
#define recurrentKernelImpl2B KERNEL(recurrentKernelImpl2B)
#define copyKernelImpl KERNEL(copyKernelImpl)
#define copyKernelImpl1B KERNEL(copyKernelImpl1B)
#define copyKernelImpl2B KERNEL(copyKernelImpl2B)
#define InitializeActivationFunctions KERNEL(InitializeActivationFunctions)

#if OPT_LEVEL < 2 || OPT_LEVEL == 3 || OPT_LEVEL == 7
#define recurrentKernelImpl1B1B KERNEL(recurrentKernelImpl1B1B)
#define recurrentKernelImpl2B1B KERNEL(recurrentKernelImpl2B1B)
#endif
#if OPT_LEVEL < 2
#define recurrentKernelImpl1B2B KERNEL(recurrentKernelImpl1B2B)
#define recurrentKernelImpl2B2B KERNEL(recurrentKernelImpl2B2B)
#endif

void activationKernelImpl(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    config->RequestConfig.Transform.Kernel->InitializeActivationFunctions();
    config->RequestConfig.Transform.Kernel->ActivateAll(config);
}

void recurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B(config);
        config->RequestConfig.Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io.Inputs = io.Inputs + activation.ElementCount * 4;
        io.Outputs = io.Outputs +
            activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
}

void recurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B(config);
        config->RequestConfig.Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io.Inputs += activation.ElementCount * 4;
        io.Outputs += activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
}

#if OPT_LEVEL < 2 || OPT_LEVEL == 3 || OPT_LEVEL == 7
void recurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B1B(config);
        config->RequestConfig.Inputs += inputElementCount;
        if (config->RequestConfig.Transform.bytesPerOutput == 1)
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
        io.Inputs += activation.ElementCount * 4;
        io.Outputs += activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
}

void recurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B1B(config);
        config->RequestConfig.Inputs += inputElementCount;
        if (config->RequestConfig.Transform.bytesPerOutput == 1)
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
        io.Inputs = io.Inputs + activation.ElementCount * 4;
        io.Outputs = io.Outputs +
            activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
}
#endif

#if OPT_LEVEL < 2
void recurrentKernelImpl1B2B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B2B(config);
        config->RequestConfig.Inputs += 2 * inputElementCount;
        if (config->RequestConfig.Transform.bytesPerOutput == 1)
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
        io.Inputs = io.Inputs + activation.ElementCount * 4;
        io.Outputs = io.Outputs +
            activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
}

void recurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> * const config)
{
    auto & runConfig = config->RequestConfig.Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        runConfig.activation, *config};
    auto & activation = activationCfg.RequestConfig.Transform;
    auto & io = activationCfg.RequestConfig;
    io.Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io.Outputs = config->RequestConfig.Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig.Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B2B(config);
        config->RequestConfig.Inputs += inputElementCount * 2;
        if(config->RequestConfig.Transform.bytesPerOutput == 1)
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
        io.Inputs = io.Inputs + activation.ElementCount * 4;
        io.Outputs = io.Outputs +
            activation.ElementCount * config->RequestConfig.Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig.Inputs = inputs;
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

/* All possible options are defined below.
 * Enabled options are defined as `1', disabled are defined as nothing
 * Only one could be enabled simultaneously.
 */
#if defined(OPTGEN)
    #define OPT_GEN 1
#else
    #define OPT_GEN
#endif
#if defined(OPTGEN_SAT)
	#define OPT_GEN_SAT 1
#else
	#define OPT_GEN_SAT
#endif
#if defined(OPTSSE4)
	#define OPT_SSE4 1
#else
	#define OPT_SSE4
#endif
#if defined(OPTSSE4_SAT)
	#define OPT_SSE4_SAT 1
#else
	#define OPT_SSE4_SAT
#endif
#if defined(OPTAVX1)
	#define OPT_AVX1 1
#else
	#define OPT_AVX1
#endif
#if defined(OPTAVX1_SAT)
	#define OPT_AVX1_SAT 1
#else
	#define OPT_AVX1_SAT
#endif
#if defined(OPTAVX2)
	#define OPT_AVX2 1
#else
	#define OPT_AVX2
#endif
#if defined(OPTAVX2_SAT)
	#define OPT_AVX2_SAT 1
#else
	#define OPT_AVX2_SAT
#endif
#define OPT_ANY 1
#define OPT_GEN_OR_SAT OPT_GEN OPT_GEN_SAT
#define OPT_AVX2_OR_AVX2_SAT OPT_AVX2 OPT_AVX2_SAT
#define OPT_SSE4_OR_SSE4_SAT OPT_SSE4 OPT_SSE4_SAT

#define VERBATIM(...) __VA_ARGS__
#define GetKernel(name, opts)    VERBATIM(IF(opts, 0))(ToUnifiedKernel(name), ToUnifiedKernel(CodeCaveMitigationFakeKernel<>))
#define IF(cond, ...)            IFHELPER ## cond
#define IFHELPER1(code, notused) code
#define IFHELPER(notused, code)  code

template<int possiblyUnused = 1>
static void CodeCaveMitigationFakeKernel()
{
    throw std::logic_error("Call to not defined GNA kernel found!");
}
template<typename KernelFunctionType>
VoidKernel ToUnifiedKernel(KernelFunctionType kernel)
{
    return reinterpret_cast<VoidKernel>(kernel);
}

template<>
VoidKernel GetXnnKernel<KernelAcceleration>(KernelType type)
{
    return GetXnnKernelHelper(type);
}

} // namespace GNA

#ifdef __INTEL_COMPILER
    #pragma intel optimization_level 0
#elif defined(__GNUC__) && !defined(__clang__)
    #pragma GCC optimize ("O0")
#endif

static GNA::VoidKernel GetXnnKernelHelper(GNA::KernelType type)
{
    using namespace GNA;
    static const VoidKernel Kernels[]=
    {
        GetKernel(AffineKernelImpl1B, OPT_ANY),
        GetKernel(AffineKernelImpl2B, OPT_ANY),

        GetKernel(AffineActiveListKernelImpl1B, OPT_ANY),
        GetKernel(AffineActiveListKernelImpl2B, OPT_ANY),

        GetKernel(AffineMultiBiasKernelImpl1B, OPT_ANY),
        GetKernel(AffineMultiBiasKernelImpl2B, OPT_ANY),

        GetKernel(DiagonalKernelImpl1B, OPT_ANY),
        GetKernel(DiagonalKernelImpl2B, OPT_ANY),

        GetKernel(recurrentKernelImpl1B, OPT_ANY),
        GetKernel(recurrentKernelImpl2B, OPT_ANY),

        GetKernel(TransposeKernelImpl1B, OPT_GEN_OR_SAT OPT_AVX2_OR_AVX2_SAT OPT_SSE4_OR_SSE4_SAT),
        GetKernel(TransposeKernelImpl2B, OPT_ANY),

        GetKernel(ConvolutionKernelImpl, OPT_ANY),
        GetKernel(ConvolutionPoolingKernelImpl, OPT_ANY),

        GetKernel(activationKernelImpl, OPT_ANY),
        GetKernel(copyKernelImpl, OPT_ANY),

        GetKernel(AffineKernelImpl1B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineKernelImpl2B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineKernelImpl1B2B, OPT_GEN_OR_SAT),
        GetKernel(AffineKernelImpl2B2B, OPT_GEN_OR_SAT),
        GetKernel(AffineActiveListKernelImpl1B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineActiveListKernelImpl2B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineActiveListKernelImpl1B2B, OPT_GEN_OR_SAT),
        GetKernel(AffineActiveListKernelImpl2B2B, OPT_GEN_OR_SAT),
        GetKernel(AffineMultiBiasKernelImpl1B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineMultiBiasKernelImpl2B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(AffineMultiBiasKernelImpl1B2B, OPT_GEN_OR_SAT),
        GetKernel(AffineMultiBiasKernelImpl2B2B, OPT_GEN_OR_SAT),
        GetKernel(DiagonalKernelImpl1B1B, OPT_GEN_OR_SAT),
        GetKernel(DiagonalKernelImpl2B1B, OPT_GEN_OR_SAT),
        GetKernel(DiagonalKernelImpl1B2B, OPT_GEN_OR_SAT),
        GetKernel(DiagonalKernelImpl2B2B, OPT_GEN_OR_SAT),
        GetKernel(recurrentKernelImpl1B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(recurrentKernelImpl2B1B, OPT_GEN_OR_SAT OPT_AVX2_SAT OPT_SSE4_SAT),
        GetKernel(recurrentKernelImpl1B2B, OPT_GEN_OR_SAT),
        GetKernel(recurrentKernelImpl2B2B, OPT_GEN_OR_SAT),
        GetKernel(ConvolutionKernelImpl1B, OPT_GEN_OR_SAT),
        GetKernel(ConvolutionPoolingKernelImpl1B, OPT_GEN_OR_SAT),
        GetKernel(ConvolutionKernelImpl2B, OPT_GEN_OR_SAT),
        GetKernel(ConvolutionPoolingKernelImpl2B, OPT_GEN_OR_SAT),
        GetKernel(copyKernelImpl1B, OPT_GEN_OR_SAT),
        GetKernel(copyKernelImpl2B, OPT_GEN_OR_SAT),

        GetKernel(Convolution2DKernelImpl1B1B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
        GetKernel(Convolution2DKernelImpl1B2B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
        GetKernel(Convolution2DKernelImpl2B1B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
        GetKernel(Convolution2DKernelImpl2B2B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),

        GetKernel(Pooling2DKernelImpl1B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
        GetKernel(Pooling2DKernelImpl2B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
        GetKernel(Pooling2DKernelImpl4B, OPT_GEN_OR_SAT OPT_SSE4_SAT OPT_AVX2_SAT),
    };
    return Kernels[type];
}

/* note:
 * be aware that optimization level is decreased for code placed here;
 * if you want to add more code, consider placing it before GetXnnKernelHelper() */
