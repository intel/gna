/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ConvolutionalFunctions.h"

#include "AccelerationDetector.h"
#include "Capabilities.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "ModelWrapper.h"
#include "Tensor.h"
#include "Validator.h"


#include <memory>
#include <utility>
#include "ConvolutionalLayer2DCapabilities.h"

using namespace GNA;
using CnnCaps = GNA::ConvolutionalLayer2DCapabilities;

FiltersTensor::FiltersTensor(const Shape& dimensions, const DataMode & dataMode, void * buffer,
    const LayerValidator& validatorIn)
try :
    WeightTensor{ dimensions, dataMode, buffer, validatorIn },
    Count{ Dimensions.at(GNA_DIM_N) },
    CoefficientCount{ Dimensions.at(GNA_DIM_W) }
{
    // validate buffer size with padding
    if (Gna2TensorModeDisabled != Mode.Mode)
    {
        const auto kernelMemorySize = HardwareLayerCnn2D::GetKernelMemorySize(this);
        const auto caps = static_cast<const TensorLimits *>(validator->Capabilities);
        validator->ValidateBufferIfSet(Buffer, kernelMemorySize * Count, caps->GetAddressAlign());
    }
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(FilterOperandIndex);
}

std::unique_ptr<const FiltersTensor> FiltersTensor::Create(const Gna2Tensor& filtersTensor, const LayerValidator& validatorIn)
{
    auto const buildCommand = [&]()
    {
        return std::make_unique<const FiltersTensor>(
            Shape::Create(filtersTensor.Shape, GNA_TENSOR_NHWD),
            GetDataMode(filtersTensor),
            filtersTensor.Data,
            validatorIn);
    };
    return ModelErrorHelper::ExecuteForModelItem(buildCommand, FilterOperandIndex);
}

const FullCapabilitiesMap ConvolutionFunction::strideLimits
{
    {INTEL_CONVOLUTIONAL, {
        { Gna2DeviceGeneration1_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_W},
            { { GNA_DIM_W, { 1, CnnCaps::Filter1DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}}}))}
    }},
};

ConvolutionFunction::ConvolutionFunction(const KernelMap<ConvolutionKernel>& kernelsIn,
    const Tensor* input, const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases, std::unique_ptr<const Component> stride) :
    Biases{ move(biases) },
    Filters{ move(filters) },
    Stride{ move(stride) },
    kernels{ kernelsIn }
{
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    Output[GNA_DIM_D] = Filters->at(GNA_DIM_N);
    OutputsPerFilterCount = 1;
    for (const auto& dim : Stride->Dimensions)
    {

        Expect::InRange(dim.second, 1u, Filters->at(dim.first), Gna2StatusXnnErrorLyrCfg);
        Output[dim.first] =
            (input->Dimensions.at(dim.first) - Filters->at(dim.first)) / dim.second + 1;
        OutputsPerFilterCount *= Output[dim.first];
    }

    for (const auto& dim : Filters->Dimensions)
    {
        if (GNA_DIM_N != dim.first)
        {
            Expect::True(dim.second <= input->Dimensions.at(dim.first), Gna2StatusXnnErrorLyrCfg);
        }
    }

    hiddenConfig = std::make_unique<ConvolutionConfig>(Stride->at(GNA_DIM_W), OutputsPerFilterCount,
        Filters->at(GNA_DIM_N), Filters->at(GNA_DIM_W),
        input->Buffer, Filters->Buffer, Biases->Buffer, output->Buffer, Biases->Mode.Size, Filters->Mode.Size);
}

std::unique_ptr<const ConvolutionConfig> ConvolutionFunction::GetRequestConfig(const BaseAddress& inputs, const BaseAddress& outputs) const
{
    return std::make_unique<const ConvolutionConfig>(hiddenConfig.get(), inputs, outputs);
}

void ConvolutionFunction::ComputeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{hiddenConfig.get(), execution};

    kernels.at(accel)(&convConfig);
}

void ConvolutionFunction::Compute(const ConvolutionConfig* const config, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ config, execution };

    kernels.at(accel)(&convConfig);
}

std::unique_ptr<const ConvolutionFunction> ConvolutionFunction::finalizeCreation(
    const Tensor* input, const Tensor* output, std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const Component> stride, std::unique_ptr<const BiasTensor> biases)
{
    return std::make_unique<const ConvolutionFunction>(
        AccelerationDetector::GetKernelMap<ConvolutionKernel>(
            static_cast<kernel_op>(INTEL_CONVOLUTIONAL), KernelMode{ input->Mode }),
        input, output, std::move(filters), std::move(biases), std::move(stride));
}

std::unique_ptr<const FiltersTensor> ConvolutionFunction::createFilters(const Gna2Operation & apiOperation,
    const LayerValidator& validatorIn)
try
{
    const auto& apiFilters = ModelWrapper::GetEnabledOperand(apiOperation, FilterOperandIndex);
    return std::make_unique<const FiltersTensor>(Shape::Create(apiFilters.Shape, GNA_TENSOR_NW),
        Tensor::GetDataMode(apiFilters), apiFilters.Data, validatorIn);
}
catch(GnaException&)
{
    GnaModelErrorException::DispatchAndFill(FilterOperandIndex);
    throw;
}

std::unique_ptr<const Component> ConvolutionFunction::createStride(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
{
    auto const command = [&]()
    {
        const auto& strideShape = ModelWrapper::GetParameter<Gna2Shape>(apiOperation, ParameterIndexConvolutionStride);
        const Shape stride{ GNA_TENSOR_W, strideShape.Dimensions[0] };
        return std::make_unique<const Component>(stride, Validator{ validatorIn, strideLimits }, true, ConvolutionStrideParamIndex);
    };
    return ModelErrorHelper::ExecuteForModelItem(command, Gna2DisabledU32, ConvolutionStrideParamIndex);
}

std::unique_ptr<const BiasTensor> ConvolutionFunction::createBiases(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
try
{
    const auto& apiBias = ModelWrapper::GetEnabledOperand(apiOperation, BiasOperandIndex);
    return std::make_unique<const BiasTensor>(Shape::Create(apiBias.Shape, GNA_TENSOR_N),
        0, Tensor::GetDataMode(apiBias), apiBias.Data, validatorIn);
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(BiasOperandIndex);
    throw;
}

void ConvolutionFunction::expectValid(const Gna2Operation& apiOperation)
{
    const auto& apiInput = ModelWrapper::GetEnabledOperand(apiOperation, InputOperandIndex);
    auto const biasModeIndex = ModelWrapper::GetOperationInfo(apiOperation.Type, ParameterIndexBiasMode);
    const auto biasMode = ModelWrapper::GetOptionalParameter<Gna2BiasMode>(apiOperation, biasModeIndex, Gna2BiasModeDefault);
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, biasModeIndex };
    ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault }, ctx);

    const auto featureCount = ModelWrapper::ShapeGetNumberOfElements(&apiInput.Shape);
    Expect::True(featureCount >= CnnCaps::Filter1DElementsMin, Gna2StatusXnnErrorLyrCfg);
}
