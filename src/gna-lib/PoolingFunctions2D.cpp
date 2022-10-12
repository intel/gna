/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "PoolingFunctions2D.h"

#include "OperationConfig.h"
#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2D.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "PoolingMode.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"

#include "gna2-memory-impl.h"

#include <map>
#include <memory>
#include <utility>

using namespace GNA;

std::unique_ptr<PoolingFunction2D> PoolingFunction2D::Create(
    const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    return create(config, operation);
}

std::unique_ptr<PoolingFunction2D> PoolingFunction2D::create(
    const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    auto poolingMode = ToKernelPoolingMode(operation.Mode);

    if (KernelPoolingModeNone != poolingMode)
    {
        auto stride = ConvolutionalLayer2D::CreateComponentFromParameter(operation.PoolingStride,
            config.validator, PoolingStrideParamIndex);
        auto window = ConvolutionalLayer2D::CreateComponentFromParameter(operation.PoolingWindow,
            config.validator, PoolingWindowParamIndex);

        return std::make_unique<PoolingFunction2D>(
            BaseTransformConfig<PoolingKernel2D>{config,
            AccelerationDetector::GetKernelMap<PoolingKernel2D>(KERNEL_POOLING_2D, KernelMode{ config.input->Mode })},
            poolingMode, std::move(window), std::move(stride));
    }
    return std::unique_ptr<PoolingFunction2D>(nullptr);
}

// unreachable code warning suppression
#if defined(_WIN32)
#pragma warning(disable : 702)
#endif
PoolingFunction2D::PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
    const KernelPoolingMode mode, std::unique_ptr<const Component> window,
    std::unique_ptr<const Component> stride) :
    Transform{ PoolingTransform2D, &config.kernels, config.input },
    Mode{ mode },
    Window{ std::move(window) },
    Stride{ std::move(stride) }
{
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, PoolingModeParamIndex };
    ModelErrorHelper::ExpectInSet(Mode,  { KernelPoolingModeMax, KernelPoolingModeSum }, ctx);
    ModelErrorHelper::ExpectInRange(Window->at('W'), Input->at('W'));
    ModelErrorHelper::ExpectInRange(Window->at('H'), Input->at('H'));
    ModelErrorHelper::ExpectInRange(Stride->at('W'), Window->at('W'));
    ModelErrorHelper::ExpectInRange(Stride->at('H'), Window->at('H'));

    Shape outputDims;
    outputDims[GNA_DIM_N] = Input->Dimensions.at(GNA_DIM_N);
    outputDims[GNA_DIM_D] = Input->Dimensions.at(GNA_DIM_D);
    outputDims.LayoutOrder = Input->Dimensions.LayoutOrder;

    for (const auto& iter : Stride->Dimensions)
    {
        auto const dim = iter.first;
        outputDims[dim] = 1;
        outputDims[dim] += GnaCeilDiv(Input->Dimensions.at(dim) - Window->Dimensions.at(dim),
            iter.second);
    }

    Output = std::make_unique<OutputTensor>(outputDims, Input->Mode, config.outputBuffer,
        config.validator, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex));

    const auto output = Output->Dimensions;
    output.ExpectFits(Input->Dimensions);

    PoolingConfig2D kernelPoolingConfiguration{
        Input->at(GNA_DIM_W),
        Input->at(GNA_DIM_H),
        Input->at(GNA_DIM_D),
        Mode,
        Stride->at(GNA_DIM_W),
        Stride->at(GNA_DIM_H),
        Window->at(GNA_DIM_W),
        Window->at(GNA_DIM_H) };

    hiddenConfig = std::make_unique<KernelConfig<PoolingConfig2D>>(kernelPoolingConfiguration,
        BaseConfig{ Input->Buffer, Output->Buffer });
}
