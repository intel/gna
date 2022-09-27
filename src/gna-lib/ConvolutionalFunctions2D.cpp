/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ConvolutionalFunctions2D.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2D.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "KernelArguments.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"
#include "Transform.h"


#include <map>
#include <memory>
#include <utility>

using namespace GNA;

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::Create(
    const TransformFactoryConfig& config, const OperationConfig& operationConfig)

{
    switch (config.validator.Operation)
    {
    case INTEL_CONVOLUTIONAL_1D:
    case INTEL_CONVOLUTIONAL_2D:
    {
        return create(config, operationConfig);
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::create(
    const TransformFactoryConfig& config, const OperationConfig& operation)
{
    auto filters = FiltersTensor::Create(operation.FiltersTensor,
            config.validator);

    auto stride = ConvolutionalLayer2D::CreateComponentFromParameter(operation.ConvolutionStride,
        config.validator, ConvolutionStrideParamIndex);

    auto padding = ConvolutionalLayer2D::CreateComponentFromParameter(operation.ZeroPadding,
        config.validator, ZeroPaddingParamIndex);

    const Shape outputDims = GetOutputShape(config.input->Dimensions,
        filters->Dimensions, stride->Dimensions, padding->Dimensions);

    const auto biasTensor = operation.BiasesTensor;
    const auto biasMode = operation.BiasMode;

    auto biases = CreateBiasTensor(biasTensor, biasMode, filters->Count,
        outputDims, config.validator);

    return std::make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D, { config.input->Mode, filters->Mode, (biases ? biases->Mode : DataMode{}) })},
        move(filters), move(biases), move(stride), move(padding));
}

Shape ConvolutionFunction2D::CalculateBiasShape(const Gna2BiasMode mode, const uint32_t filterCount, Shape const & outputShape)
{
    switch (mode)
    {
    case Gna2BiasModeDefault:
    {
        return Shape(GNA_TENSOR_NHW, filterCount, 1u, 1u);
    }
    case Gna2BiasModePerStride:
    {
        return Shape(GNA_TENSOR_NHW,
            filterCount,
            outputShape.at(GNA_DIM_H),
            outputShape.at(GNA_DIM_W));
    }
    default:
    {
        return Shape(GNA_TENSOR_NHW, 1u, 1u, 1u);
    }
    }
}

std::unique_ptr<const BiasTensor> ConvolutionFunction2D::CreateBiasTensor(
    Gna2Tensor const & apiTensor, Gna2BiasMode biasMode, uint32_t filtersCount,
    Shape const & outputShape, const LayerValidator& validatorIn)
try
{
    Shape biasDims;

    if (apiTensor.Shape.NumberOfDimensions) // use direct dimensions when provided
    {
        biasDims = CalculateBiasShape(biasMode, apiTensor.Shape.Dimensions[0],
            Shape(GNA_TENSOR_HW, apiTensor.Shape.Dimensions[1], apiTensor.Shape.Dimensions[1]));
    }
    else // calculate from outputShape when not provided
    {
        biasDims = CalculateBiasShape(biasMode, filtersCount, outputShape);
    }

    return std::make_unique<const BiasTensor>(
        biasDims,
        0,
        BiasTensor::GetDataMode(apiTensor),
        apiTensor.Data,
        validatorIn,
        biasMode);
}
catch(GnaException&)
{
    GnaModelErrorException::DispatchAndFill(BiasOperandIndex);
    throw;
}

Shape ConvolutionFunction2D::GetOutputShape(Shape const & inputShape,
    Shape const & filerShape, Shape const & strideShape, Shape const & paddingShape)
{
    Shape outputShape;
    outputShape.LayoutOrder = GNA_TENSOR_NHWD;
    outputShape[GNA_DIM_N] = inputShape.at(GNA_DIM_N);
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    outputShape[GNA_DIM_D] = filerShape.at(GNA_DIM_N);

    for (const auto& dimPair : strideShape)
    {
        auto const dim = dimPair.first;
        outputShape[dim] =
            1 + (inputShape.at(dim) + (2 * paddingShape.at(dim)) - filerShape.at(dim))
            / dimPair.second;
    }
    return outputShape;
}

bool IsInput1D(const Shape& inputShape)
{
    try
    {
        return inputShape.at('N') == 1 && inputShape.at('H') == 1 && inputShape.at('D') == 1;
    }
    catch (...)
    {
        throw GnaException(Gna2StatusModelConfigurationInvalid);
    }
}

ConvolutionFunction2D::ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D>& config,
    std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Component> stride,
    std::unique_ptr<const Component> padding) :
    Transform{ ConvolutionalTransform2D, &config.kernels, config.input },
    Biases{ move(biases) },
    Filters{ move(filters) },
    Stride{ move(stride) },
    Padding{ move(padding) }
{
    if (KernelBiasModePerFilter == Biases->BiasMode)
    {
        ModelErrorHelper::ExpectEqual(Biases->at('H'), 1u);
        ModelErrorHelper::ExpectEqual(Biases->at('W'), 1u);
    }

    ModelErrorHelper::ExpectEqual(Filters->at('D'), Input->at('D'));
    ModelErrorHelper::ExpectBelowEq(Filters->at('W'), Input->at('W'));
    ModelErrorHelper::ExpectBelowEq(Filters->at('H'), Input->at('H'));
    ModelErrorHelper::ExpectBelowEq(Stride->at('W'), Filters->at('W'));
    ModelErrorHelper::ExpectBelowEq(Stride->at('H'), Filters->at('H'));
    ModelErrorHelper::ExpectBelowEq(Padding->at('W'), Filters->at('W') - 1);
    ModelErrorHelper::ExpectBelowEq(Padding->at('H'), Filters->at('H') - 1);

    Shape outputDims = GetOutputShape(Input->Dimensions, Filters->Dimensions,
        Stride->Dimensions, Padding->Dimensions);

    Output = std::make_unique<OutputTensor>(outputDims, DataMode{ Gna2DataTypeInt32 }, config.outputBuffer,
        config.validator, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex));

    auto out = Output->Dimensions;
    out.erase(GNA_DIM_D);

    auto kernelBiasMode = Biases->BiasMode;

    ConvolutionConfig2D kernelConvolutionConfig2D{
        Input->at(GNA_DIM_W),
        Input->at(GNA_DIM_H),
        Input->at(GNA_DIM_D),
        Filters->at(GNA_DIM_N),
        Filters->at(GNA_DIM_W),
        Filters->at(GNA_DIM_H),
        Filters->at(GNA_DIM_D),
        KernelDataMode{Filters->Mode.Size}, Filters->Buffer,
        Stride->at(GNA_DIM_W),
        Stride->at(GNA_DIM_H),
        Padding->at(GNA_DIM_W),
        Padding->at(GNA_DIM_H),
        kernelBiasMode,
        KernelDataMode{Biases->Mode.Size},
        Biases->Buffer };

    hiddenConfig = std::make_unique<KernelConfig<ConvolutionConfig2D>>(
        kernelConvolutionConfig2D,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

Tensor const & ConvolutionFunction2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case FilterOperandIndex:
    {
        return GetOperandIfExistOrThrow(Filters);
    }
    case BiasOperandIndex:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}
