/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "PoolingFunctions.h"

#include "AccelerationDetector.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "GnaException.h"
#include "ModelWrapper.h"
#include "Validator.h"

#include <utility>
#include "PoolingMode.h"

namespace GNA
{
struct PwlCached;
}

using namespace GNA;
using CnnCaps = GNA::ConvolutionalLayer2DCapabilities;

const std::map<const nn_operation, const ShapeLimits> PoolingFunction::windowLimits =
{
    {INTEL_CONVOLUTIONAL,
        {{GNA_DIM_W, {CnnCaps::PoolingWindowSizeMin, CnnCaps::PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolSize}}}
    },
};

const std::map<const nn_operation, const ShapeLimits> PoolingFunction::strideLimits =
{
    {INTEL_CONVOLUTIONAL,
        {{GNA_DIM_W, {CnnCaps::PoolingWindowSizeMin, CnnCaps::PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolStride}}}
    },
};

void PoolingFunction::ExpectValid(Gna2Operation const & apiOperation)
{
    auto const hasPoolingWindow = ModelWrapper::HasParameter(apiOperation, PoolingWindowParamIndex);
    auto const hasPoolingStride = ModelWrapper::HasParameter(apiOperation, PoolingStrideParamIndex);

    if (hasPoolingWindow || hasPoolingStride)
    {
        ModelWrapper::ExpectParameterAvailable(apiOperation, PoolingModeParamIndex);
        ModelWrapper::ExpectParameterAvailable(apiOperation, PoolingWindowParamIndex);
        ModelWrapper::ExpectParameterAvailable(apiOperation, PoolingStrideParamIndex);
    }
}

std::unique_ptr<const PoolingFunction> PoolingFunction::Create(Gna2Operation const & apiOperation,
    const Shape & inputDimensions, const LayerValidator& validatorIn, const DataMode & inputMode)
{
    Expect::Equal(INTEL_CONVOLUTIONAL, validatorIn.Operation, Gna2StatusXnnErrorLyrOperation);
    ExpectValid(apiOperation);

    const auto poolingModeApi = ModelWrapper::GetOptionalParameter<Gna2PoolingMode>(apiOperation, PoolingModeParamIndex,
        Gna2PoolingModeDisabled);
    const auto poolingMode = ToKernelPoolingMode(poolingModeApi);

    if (KernelPoolingModeMax == poolingMode || KernelPoolingModeSum == poolingMode)
    {
        const auto apiStride = ModelWrapper::GetParameter<Gna2Shape>(
            apiOperation, PoolingStrideParamIndex);
        const auto strideShape = Shape::Create(apiStride, GNA_TENSOR_W);

        const auto apiWindow = ModelWrapper::GetParameter<Gna2Shape>(
            apiOperation, PoolingWindowParamIndex);
        const auto windowShape = Shape::Create(apiWindow, GNA_TENSOR_W);

        return std::make_unique<const PoolingFunction>(validatorIn.Operation, inputDimensions, windowShape,
            strideShape, poolingMode,
            AccelerationDetector::GetKernelMap<ConvolutionPoolingKernel>(KERNEL_POOLING, inputMode));
    }
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, PoolingModeParamIndex };
    ModelErrorHelper::ExpectInSet(poolingMode, { KernelPoolingModeNone }, ctx);
    return std::unique_ptr<const PoolingFunction>(nullptr);
}

PoolingFunction::PoolingFunction(nn_operation const operation, const Shape& inputDimensions,
    const Shape& window, const Shape& stride,
    const KernelPoolingMode mode, const KernelMap<ConvolutionPoolingKernel>& kernelsIn) :
    Mode{ mode },
    Window{ window, PoolingWindowParamIndex, true },
    Stride{ stride, PoolingStrideParamIndex, true },
    kernels{ kernelsIn },
    hiddenConfig{ std::make_unique<PoolingConfig>(Mode, Window.at(GNA_DIM_W), Stride.at(GNA_DIM_W)) }
{
    Expect::InSet(Mode, { KernelPoolingModeMax, KernelPoolingModeSum }, Gna2StatusCnnErrorPoolType);

    Stride.ExpectShapeIsValid(strideLimits.at(operation));
    Window.ExpectShapeIsValid(windowLimits.at(operation));

    OutputsPerFilterCount = 1;
    OutputDimensions[GNA_DIM_D] = inputDimensions.at(GNA_DIM_D);
    for (const auto& dim : Stride.Dimensions)
    {
        if (GNA_DIM_D != dim.first)
        {
            OutputDimensions[dim.first] =  ((inputDimensions.at(dim.first) - 1) / dim.second + 1);
            OutputsPerFilterCount *= OutputDimensions[dim.first];
            Expect::InRange(OutputDimensions[dim.first], 1u, inputDimensions.at(dim.first), Gna2StatusCnnErrorPoolSize);
        }
    }
}

void PoolingFunction::Compute(const ConvolutionConfig * convolutionConfig, AccelerationMode accel, int64_t * poolScratchPad,
    const PwlCached * pwl) const
{
    auto poolConfig = PoolingConfig{ hiddenConfig.get(), poolScratchPad };
    kernels.at(accel)(convolutionConfig, &poolConfig, pwl);
}
