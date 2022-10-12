/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ActivationFunction.h"

#include "AffineLayers.h"
#include "ActivationHelper.h"
#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GnaException.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Shape.h"
#include "Validator.h"

#include "gna2-common-api.h"
#include "gna2-device-api.h"

#include <algorithm>
#include <memory>

using namespace GNA;

static const auto pwlLimit = std::make_shared<TensorLimits>(
    TensorLimits{
        {GNA_TENSOR_H},    // W - #inputs, H - #outputs
        {{GNA_DIM_H, { 2, ActivationFunction::ActivationFunctionSegmentCountMax, 1, Gna2StatusXnnErrorPwlSegments}}},
        {{ Gna2DataTypePwlSegment }, Gna2StatusXnnErrorOutputBytes}});

static const auto pwlLimit3_0 = std::make_shared<TensorLimits>(
    TensorLimits{
        {GNA_TENSOR_H},    // W - #inputs, H - #outputs
        {{GNA_DIM_H, { 1, ActivationFunction::ActivationFunctionSegmentCountMax, 1, Gna2StatusXnnErrorPwlSegments}}},
        {{ Gna2DataTypePwlSegment }, Gna2StatusXnnErrorOutputBytes}});

const FullCapabilitiesMap ActivationFunction::capabilities =
{
    {INTEL_AFFINE,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }},
    {INTEL_AFFINE_DIAGONAL,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }},
    {INTEL_AFFINE_MULTIBIAS,{
        {Gna2DeviceGeneration2_0, pwlLimit},
    }},
    {INTEL_CONVOLUTIONAL,{
        {Gna2DeviceGeneration1_0, pwlLimit},
    }},
    {INTEL_CONVOLUTIONAL_1D,{
        {Gna2DeviceGeneration3_0, pwlLimit3_0}
    }},
    {INTEL_CONVOLUTIONAL_2D,{
        {Gna2DeviceGeneration3_0, pwlLimit3_0}
    }},
    {INTEL_RECURRENT,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }}
};

const FullCapabilitiesMap ActivationFunction::outputCapabilities = LayerCapabilities::MakeFullCaps<OutputOperandIndex>();

std::unique_ptr<ActivationFunction> ActivationFunction::Create(const TransformFactoryConfig& config)
{
    if (config.IsActivationNotSupported())
    {
        return std::unique_ptr<ActivationFunction>(nullptr);
    }
    const auto mandatory = config.HasMandatoryActivation();

    const Gna2Tensor activation = config.GetActivation();

    if (mandatory || activation.Mode != Gna2TensorModeDisabled)
    {
        try
        {
            ActivationHelper::ExpectProper(activation);
            auto pwlFunction = std::make_unique<PwlTensor>(
                Shape(GNA_TENSOR_H, activation.Shape.Dimensions[0]),
                Gna2DataTypePwlSegment, activation.Data, Validator{ config.validator, capabilities, false });
            return std::make_unique<ActivationFunction>(
                BaseTransformConfig<ActivationKernel>{config,
                AccelerationDetector::GetKernelMap<ActivationKernel>(KERNEL_PWL)}, config.outputMode,
                std::move(pwlFunction));
        }
        catch (GnaException&)
        {
            GnaModelErrorException::DispatchAndFill(PwlOperandIndex);
        }
    }
    return std::unique_ptr<ActivationFunction>(nullptr);
}

void ActivationFunction::UpdateActiveOutputCount(
    std::unique_ptr<BaseConfig> configs[TransformOperationCount], uint32_t outputCount) const
{
    auto config = GetConfig(configs);
    config->Transform.ElementCount = outputCount;
}


std::unique_ptr<PwlCached const> ActivationFunction::createPwlCached(uint32_t elementSize,
    PwlSegment const * segmentsIn, uint32_t segmentCountIn)
{
    if (nullptr == segmentsIn)
    {
        segmentsIn = reinterpret_cast<PwlSegment const *>(AffineBaseLayer::GetGlobal2MBScratchpad());
    }
    try
    {
        return std::make_unique<PwlCached const>(elementSize, segmentsIn, segmentCountIn);
    }
    catch (const std::runtime_error&)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
}

ActivationFunction::ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
    const DataMode& mode, std::unique_ptr<Tensor> pwl) :
    Transform{ ActivationTransform, &config.kernels, config.input },
    Segments{ std::move(pwl) },
    Pwl{ createPwlCached(config.outputMode.Size, Segments->Buffer, Segments->Count) }
{
    Output = std::make_unique<OutputTensor>(config.input->Dimensions, mode, config.outputBuffer,
        config.validator, outputCapabilities);

    hiddenConfig = std::make_unique<KernelConfig<ActivationConfig>>(
        ActivationConfig{ Output->Count, Pwl.get() }, BaseConfig{ Input->Buffer, config.outputBuffer });
}

Tensor const & ActivationFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case 2:
    {
        return GetOperandIfExistOrThrow(Segments);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}

void ActivationFunction::ValidateActiveList(ActiveList const& activeList) const
{
    Expect::InRange(activeList.IndicesCount,
        1u, Output->at(GNA_DIM_H), Gna2StatusActiveListIndicesInvalid);
}
