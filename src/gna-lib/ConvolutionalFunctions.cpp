/**
 @copyright (C) 2018-2021 Intel Corporation
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

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <memory>
#include <utility>

using namespace GNA;

FiltersTensor::FiltersTensor(const Shape& dimensions, const DataMode & dataMode, void * buffer,
    const LayerValidator& validatorIn) :
    WeightTensor{ dimensions, dataMode, buffer, validatorIn },
    Count{ Dimensions.at(GNA_DIM_N) },
    CoefficientCount{ Dimensions.at(GNA_DIM_W) }
{
    // validate buffer size with padding
    if (GNA_DATA_DISABLED != Mode)
    {
        const auto kernelMemorySize = HardwareLayerCnn2D::GetKernelMemorySize(
            validator->HwCapabilities.GetDeviceVersion(), this);
        const auto caps = static_cast<const TensorLimits *>(validator->Capabilities);
        validator->ValidateBufferIfSet(Buffer, kernelMemorySize * Count, caps->Align);
    }

    if (INTEL_CONVOLUTIONAL_1D == validator->Operation &&
        Gna2DataTypeInt16 == Mode.Type)
    {
        Expect::InRange(at(GNA_DIM_W),
            CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN,
            CNN_1D_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX / 2,
            Gna2StatusCnnErrorConvFltVolume);
    }
}

std::unique_ptr<const FiltersTensor> FiltersTensor::Create(const Gna2Tensor& filtersTensor, const LayerValidator& validatorIn)
{
    std::unique_ptr<const FiltersTensor> filters;
    const std::function<void()> command = [&]()
    {
        try // 1D CNN in new arch
        {
            auto const validator1D = LayerValidator{ validatorIn, INTEL_CONVOLUTIONAL_1D };
            filters = std::make_unique<const FiltersTensor>(
                Shape::Create(filtersTensor.Shape, GNA_TENSOR_NHWD),
                GetDataMode(filtersTensor),
                filtersTensor.Data,
                validator1D);
        }
        catch (const GnaException&) // try 2D CNN in new arch
        {
            filters = std::make_unique<const FiltersTensor>(
                Shape::Create(filtersTensor.Shape, GNA_TENSOR_NHWD),
                GetDataMode(filtersTensor),
                filtersTensor.Data,
                validatorIn);
        }
    };
    ModelErrorHelper::ExecuteForModelItem(command, FilterOperandIndex);
    return filters;
}

const FullCapabilitiesMap ConvolutionFunction::strideLimits
{
    {INTEL_CONVOLUTIONAL, {
        { GNA_1_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_W},
            { { GNA_DIM_W, { 1, CNN_N_FLT_COEFF_MAX, 1, Gna2StatusCnnErrorConvFltStride}}}))}
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
        Expect::InRange(dim.second, ui32_1, Filters->at(dim.first), Gna2StatusXnnErrorLyrCfg);
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
        input->Buffer, Filters->Buffer, Biases->Buffer, output->Buffer, Biases->Mode, Filters->Mode);
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
            static_cast<kernel_op>(INTEL_CONVOLUTIONAL), KernelMode{ input->Mode.Value }),
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
catch(GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, FilterOperandIndex);
    throw;
}

std::unique_ptr<const FiltersTensor> ConvolutionFunction::createFilters(const nn_layer_conv& cnn,
    const LayerValidator& validatorIn)
{
    return std::make_unique<const FiltersTensor>(Shape(GNA_TENSOR_NWH, cnn.nFilters, cnn.nFilterCoefficients, 0u),
        cnn.nBytesFilterCoefficient, cnn.pFilters, validatorIn);
}

std::unique_ptr<const Component> ConvolutionFunction::createStride(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
{
    std::unique_ptr<const Component> strideCreated;
    const std::function<void()> command = [&]()
    {
        const auto& strideShape = ModelWrapper::GetParameter<Gna2Shape>(apiOperation, ParameterIndexConvolutionStride);
        const Shape stride{ GNA_TENSOR_W, strideShape.Dimensions[0] };
        strideCreated = std::make_unique<const Component>(stride, Validator{ validatorIn, strideLimits });
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, ConvolutionStrideParamIndex);
    return strideCreated;
}

std::unique_ptr<const Component> ConvolutionFunction::createStride(const nn_layer_conv & cnn,
    const LayerValidator & validatorIn)
{
    const Shape stride{ GNA_TENSOR_W, cnn.nFeatureMaps * cnn.nFeatureMapColumns };
    return std::make_unique<const Component>(stride, Validator{ validatorIn, strideLimits });
}

std::unique_ptr<const BiasTensor> ConvolutionFunction::createBiases(const Gna2Operation & apiOperation,
    const LayerValidator & validatorIn)
try
{
    const auto& apiBias = ModelWrapper::GetEnabledOperand(apiOperation, BiasOperandIndex);
    return std::make_unique<const BiasTensor>(Shape::Create(apiBias.Shape, GNA_TENSOR_N),
        0, Tensor::GetDataMode(apiBias), apiBias.Data, validatorIn);
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, BiasOperandIndex);
    throw;
}

std::unique_ptr<const BiasTensor> ConvolutionFunction::createBiases(const nn_layer_conv & cnn,
    const LayerValidator & validatorIn)
{
    return  std::make_unique<const BiasTensor>(Shape(GNA_TENSOR_N, cnn.nFilters),
        0, cnn.nBytesBias, cnn.pBiases, validatorIn);
}

void ConvolutionFunction::expectValid(const Gna2Operation& apiOperation)
{
    const auto& apiInput = ModelWrapper::GetEnabledOperand(apiOperation, InputOperandIndex);
    auto const biasModeIndex = ModelWrapper::GetOperationInfo(apiOperation.Type, ParameterIndexBiasMode);
    const auto biasMode = ModelWrapper::GetOptionalParameter<Gna2BiasMode>(apiOperation, biasModeIndex, Gna2BiasModeDefault);
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault }, Gna2ItemTypeParameter);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(biasModeIndex));

    const auto featureCount = ModelWrapper::ShapeGetNumberOfElements(&apiInput.Shape);
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, Gna2StatusXnnErrorLyrCfg);
}

void ConvolutionFunction::expectValid(const nn_layer_conv& cnn)
{
    const auto featureCount = cnn.nFeatureMaps * cnn.nFeatureMapRows * cnn.nFeatureMapColumns;
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, Gna2StatusXnnErrorLyrCfg);
    Expect::InRange(cnn.nFilterRows, ui32_1, CNN_N_FLT_COEFF_MAX, Gna2StatusXnnErrorLyrCfg);
}
