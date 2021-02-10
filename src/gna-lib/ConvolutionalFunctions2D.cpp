/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ConvolutionalFunctions2D.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "Capabilities.h"
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

#include "gna-api.h"

#include <map>
#include <memory>
#include <utility>

using namespace GNA;

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::Create(
    const TransformFactoryConfig& config, const OperationConfig& operationConfig)

{
    switch (config.validator.Operation)
    {
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

    auto stride = OperationConfig::CreateCnnComponent(operation.ConvolutionStride,
        config.validator, ConvolutionalLayer2DCapabilities::GetParameters(ConvolutionStrideParamIndex));

    auto padding = OperationConfig::CreateCnnComponent(operation.ZeroPadding,
        config.validator, ConvolutionalLayer2DCapabilities::GetParameters(ZeroPaddingParamIndex));

    const Shape outputDims = GetOutputShape(config.input->Dimensions,
        filters->Dimensions, stride->Dimensions, padding->Dimensions);

    const auto biasTensor = operation.BiasesTensor;
    const auto biasMode = operation.BiasMode;

    auto biases = CreateBiasTensor(biasTensor, biasMode, filters->Count,
        outputDims, config.validator);

    return std::make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D, { config.input->Mode, filters->Mode, (biases ? static_cast<gna_data_mode>(biases->Mode) : GNA_DATA_DISABLED) })},
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
        //return Shape{};
    }
    }
}

std::unique_ptr<const BiasTensor> ConvolutionFunction2D::CreateBiasTensor(
    Gna2Tensor const & apiTensor, Gna2BiasMode biasMode, uint32_t filtersCount,
    Shape const & outputShape, const LayerValidator& validatorIn)
{
    Shape biasDims = CalculateBiasShape(biasMode, filtersCount, outputShape);
    try
    {
         // try 2D CNN in new arch
        auto const validator1D = LayerValidator{ validatorIn, INTEL_CONVOLUTIONAL_1D };
        return std::make_unique<const BiasTensor>(
            biasDims,
            0,
            DataMode{ apiTensor.Type, apiTensor.Mode },
            apiTensor.Data,
            validator1D,
            biasMode);
    }
    catch (const GnaException&)
    {
        // 2D CNN in new arch
        return std::make_unique<const BiasTensor>(
            biasDims,
            0,
            DataMode{ apiTensor.Type, apiTensor.Mode },
            apiTensor.Data,
            validatorIn,
            biasMode);
    }
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
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_H), 1, Gna2StatusXnnErrorBiasMode);
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_W), 1, Gna2StatusXnnErrorBiasMode);
    }

    auto effectiveOperation = INTEL_CONVOLUTIONAL_2D;
    if (INTEL_CONVOLUTIONAL_1D == Filters->GetEffectiveOperationType() &&
        INTEL_CONVOLUTIONAL_1D == Stride->GetEffectiveOperationType() &&
        IsInput1D(config.input->Dimensions))
    {
        is1D = true;
        effectiveOperation = INTEL_CONVOLUTIONAL_1D;
        Expect::InRange(Filters->at(GNA_DIM_W), Input->at(GNA_DIM_W),
            Gna2StatusCnnErrorConvFltVolume);
        Expect::InRange(Stride->at(GNA_DIM_W), Filters->at(GNA_DIM_W),
            Gna2StatusCnnErrorConvFltVolume);
    }

    Shape outputDims = GetOutputShape(Input->Dimensions, Filters->Dimensions,
        Stride->Dimensions, Padding->Dimensions);

    auto const validatorOut = LayerValidator{ config.validator, effectiveOperation};
    Output = std::make_unique<Tensor>(outputDims, DataMode{ GNA_INT32 }, config.outputBuffer,
        Validator{ validatorOut, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex) });

    auto out = Output->Dimensions;
    out.erase(GNA_DIM_D);

    gna_3d_dimensions input = Input->Dimensions;
    gna_3d_dimensions filter = Filters->Dimensions;
    gna_3d_dimensions convolutionStride = Stride->Dimensions;
    gna_3d_dimensions zeroPadding = Padding->Dimensions;

    auto kernelBiasMode = Biases->BiasMode;

    ConvolutionConfig2D kernelConvolutionConfig2D{ input.width, input.height, input.depth,Filters->at(GNA_DIM_N),
        filter.width, filter.height, filter.depth,
        KernelDataMode{Filters->Mode.Size}, Filters->Buffer,
        convolutionStride.width, convolutionStride.height,
        zeroPadding.width, zeroPadding.height,
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
