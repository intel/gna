/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "PoolingFunctions2D.h"

#include "OperationConfig.h"
#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "ParameterLimits.h"
#include "PoolingMode.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const SetLimits<KernelPoolingMode> PoolingFunction2D::modeLimits =
{
    { KernelPoolingModeMax, KernelPoolingModeSum }, Gna2StatusCnnErrorPoolType
};

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
    auto poolingMode = operation.Mode;

    if (Gna2PoolingModeDisabled != poolingMode)
    {
        auto stride = OperationConfig::CreateCnnComponent(operation.PoolingStride,
            config.validator, ConvolutionalLayer2DCapabilities::GetParameters(PoolingStrideParamIndex));
        auto window = OperationConfig::CreateCnnComponent(operation.PoolingWindow,
            config.validator, ConvolutionalLayer2DCapabilities::GetParameters(PoolingWindowParamIndex));

        return std::make_unique<PoolingFunction2D>(
            BaseTransformConfig<PoolingKernel2D>{config,
            AccelerationDetector::GetKernelMap<PoolingKernel2D>(KERNEL_POOLING_2D, { config.input->Mode.Value })},
            poolingMode, std::move(window), std::move(stride));
    }
    return std::unique_ptr<PoolingFunction2D>(nullptr);
}

// unreachable code warning suppression
#if defined(_WIN32)
#pragma warning(disable : 702)
#endif
PoolingFunction2D::PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
    const PoolingMode mode, std::unique_ptr<const Component> window,
    std::unique_ptr<const Component> stride) :
    Transform{ PoolingTransform2D, &config.kernels, config.input },
    Mode{ mode },
    Window{ std::move(window) },
    Stride{ std::move(stride) }
{
    Expect::InSet(Mode, modeLimits);
    Expect::InRange(Window->at(GNA_DIM_W), Input->at(GNA_DIM_W),
        Gna2StatusCnnErrorPoolSize);

    if (INTEL_CONVOLUTIONAL_1D == Window->GetEffectiveOperationType() &&
        INTEL_CONVOLUTIONAL_1D == Stride->GetEffectiveOperationType())
    {
        is1D = true;
        /*Expect::InRange(Stride->at(GNA_DIM_W), Window->at(GNA_DIM_W),
            Gna2StatusCnnErrorPoolStride);*/
    }

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

    Output = std::make_unique<Tensor>(outputDims, Input->Mode, config.outputBuffer,
        Validator{ config.validator, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex) });

    const auto output = Output->Dimensions;
    output.ExpectFits(Input->Dimensions);

    const gna_3d_dimensions input = Input->Dimensions;
    const gna_3d_dimensions poolingStride = Stride->Dimensions;
    const gna_3d_dimensions poolingWindow = Window->Dimensions;

    PoolingConfig2D kernelPoolingConfiguration{ input.width, input.height, input.depth,
        Mode, poolingStride.width, poolingStride.height,
        poolingWindow.width, poolingWindow.height };

    hiddenConfig = std::make_unique<KernelConfig<PoolingConfig2D>>(kernelPoolingConfiguration,
        BaseConfig{ Input->Buffer, Output->Buffer });
}
