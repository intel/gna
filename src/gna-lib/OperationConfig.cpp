/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "OperationConfig.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"

using namespace GNA;

OperationConfig::OperationConfig(const Gna2Operation& apiOperation) :
    Operation{ &apiOperation }
{
    InitOperationConfig(apiOperation);
}

Gna2OperationType OperationConfig::GetOperationType(const Gna2Operation& apiOperation)
{
    return apiOperation.Type;
}

kernel_op OperationConfig::GetKernelOperation() const
{
    switch (OperationType)
    {
    case Gna2OperationTypeConvolution:
        return KERNEL_CONVOLUTIONAL_2D;
    case Gna2OperationTypeCopy:
        return KERNEL_COPY;
    case Gna2OperationTypeElementWiseAffine:
        return KERNEL_AFFINE_DIAGONAL;
    case Gna2OperationTypeFullyConnectedAffine:
        return KERNEL_AFFINE;
    case Gna2OperationTypeGmm:
        return KERNEL_GMM;
    case Gna2OperationTypeRecurrent:
        return KERNEL_RECURRENT;
    case Gna2OperationTypeTransposition:
        return KERNEL_TRANSPOSE;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

TransformOperation OperationConfig::GetTransformOperation() const
{
    switch (OperationType)
    {
    case Gna2OperationTypeConvolution:
        return ConvolutionalTransform2D;
    case Gna2OperationTypeCopy:
        return CopyTransform;
    case Gna2OperationTypeElementWiseAffine:
        return AffineDiagonalTransform;
    case Gna2OperationTypeFullyConnectedAffine:
        if (BiasMode == Gna2BiasModeGrouping)
        {
            return AffineMultibiasTransform;
        }
        return AffineTransform;
    case Gna2OperationTypeGmm:
        return GmmTransform;
    case Gna2OperationTypeRecurrent:
        return RecurrentTransform;
    case Gna2OperationTypeTransposition:
        return TransposeTransform;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

bool OperationConfig::IsCNN1D(const Gna2Operation & operation)
{
    return Gna2OperationTypeConvolution == operation.Type &&
        2 == operation.Operands[InputOperandIndex]->Shape.NumberOfDimensions;
}

Gna2Tensor OperationConfig::GetWeights(const Gna2Operation & operation)
{
    const auto index = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeight);
    return ModelWrapper::GetOperand(operation, index, {});
}

Gna2Tensor OperationConfig::GetFilters(const Gna2Operation & operation)
{
    auto filter = ModelWrapper::GetOperand(operation, FilterOperandIndex, {});
    if (2 == filter.Shape.NumberOfDimensions)
    {
        filter.Shape.NumberOfDimensions = 4;
        filter.Shape.Dimensions[2] = filter.Shape.Dimensions[1];
        filter.Shape.Dimensions[1] = 1;
        filter.Shape.Dimensions[3] = 1;
    }
    return filter;
}

Gna2BiasMode OperationConfig::GetBiasMode(const Gna2Operation & operation)
{
    const auto biasMode = ModelWrapper::GetOptionalParameter<Gna2BiasMode>(operation, BiasModeConvolutionParamIndex, Gna2BiasModeDefault);
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, BiasModeConvolutionParamIndex };
    ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault, Gna2BiasModePerStride, Gna2BiasModePerStride }, ctx);
    return biasMode;
}

Gna2Tensor OperationConfig::GetBiases(const Gna2Operation & operation)
{
    if (operation.Type == Gna2OperationTypeGmm)
    {
        return ModelWrapper::GetDisabledOperand();
    }
    return ModelWrapper::GetOperand(operation, BiasOperandIndex, ModelWrapper::GetDisabledOperand());
}

Shape OperationConfig::GetStride(const Gna2Operation & operation)
{
    auto parameter = GetShapeParameterOfMaximal2Dimensions(operation, ConvolutionStrideParamIndex);
    // Stride is required, dimensions must be specified
    auto const ctx = ModelItem{ Gna2ItemTypeShapeNumberOfDimensions, Gna2DisabledU32, ConvolutionStrideParamIndex };
    ModelErrorHelper::ExpectAboveEq(parameter.size(), 1, ctx);
    if (parameter.size() == 1)
    {
        parameter.LayoutOrder = Layout("HW");
        parameter['W'] = parameter['N'];
        parameter.erase(GNA_DIM_N);
        parameter['H'] = 1;
    }
    return parameter;
}

Shape OperationConfig::GetZeroPadding(const Gna2Operation& operation)
{
    auto parameter = GetShapeParameterOfMaximal2Dimensions(operation, ZeroPaddingParamIndex);
    if (parameter.size() == 1)
    {
        parameter = Shape();
    }
    return parameter;
}

Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex)
{
    const Gna2Shape shape = ModelWrapper::GetOptionalParameter<Gna2Shape>(operation, parameterIndex, {});
    return Shape::Create(shape, GNA_TENSOR_ORDER_ANY);
}

Shape OperationConfig::GetShapeParameterOfMaximal2Dimensions(const Gna2Operation & operation, const uint32_t parameterIndex)
{
    auto const command = [&]()
    {

        // when the validation happens for Component's ctor
        const auto parameter = TryGetParamShape(operation, parameterIndex);
        ModelErrorHelper::ExpectBelowEq(parameter.size(), 2, Gna2ItemTypeShapeNumberOfDimensions);
        return parameter;
    };
    return ModelErrorHelper::ExecuteForModelItem(command, Gna2DisabledU32, parameterIndex);
}

Gna2PoolingMode OperationConfig::GetPoolingMode(const Gna2Operation & operation)
{
    return ModelWrapper::GetOptionalParameter<Gna2PoolingMode>(operation, ParameterIndexPoolingMode, Gna2PoolingModeDisabled);
}

void OperationConfig::InitMultibias(const Gna2Operation& operation)
{
    auto bmIndex = ModelWrapper::GetOperationInfo(operation.Type, ParameterIndexBiasMode);
    BiasMode = ModelWrapper::GetParameter<Gna2BiasMode>(operation, bmIndex);

    auto bviIndex = ModelWrapper::GetOperationInfo(
        operation.Type, ParameterIndexBiasVectorIndex);
    BiasVectorIndex = ModelWrapper::GetParameter<uint32_t>(operation, bviIndex);

    auto wsfIndex = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeightScaleFactors);

    // GNA 2.0 backward compatibility only
    if (Gna2DataTypeInt8 == WeightsTensor.Type
        && Gna2DataTypeInt16 == operation.Operands[InputOperandIndex]->Type)
    {
        WeightScalesTensor = ModelWrapper::GetEnabledOperand(operation, wsfIndex);
        ModelWrapper::SetLayout(WeightScalesTensor, "H");
    }
}

void OperationConfig::InitPooling(const Gna2Operation & operation)
{
    PoolingWindow = GetShapeParameterOfMaximal2Dimensions(operation, PoolingWindowParamIndex);
    if (PoolingWindow.size() == 1)
    {
        PoolingWindow.LayoutOrder = Layout("HW");
        PoolingWindow['W'] = PoolingWindow['N'];
        PoolingWindow.erase(GNA_DIM_N);
        PoolingWindow['H'] = 1;
    }
    PoolingStride = GetShapeParameterOfMaximal2Dimensions(operation, PoolingStrideParamIndex);
    if (PoolingStride.size() == 1)
    {
        PoolingStride.LayoutOrder = Layout("HW");
        PoolingStride['W'] = PoolingStride['N'];
        PoolingStride.erase(GNA_DIM_N);
        PoolingStride['H'] = 1;
    }
    Mode = GetPoolingMode(operation);
    if (Mode != Gna2PoolingModeDisabled)
    {
        // required, dimensions must be specified
        auto const ctxWindow = ModelItem{ Gna2ItemTypeShapeNumberOfDimensions, Gna2DisabledU32, PoolingWindowParamIndex };
        ModelErrorHelper::ExpectAboveEq(PoolingWindow.size(), 1, ctxWindow);
        auto const ctxStride = ModelItem{ Gna2ItemTypeShapeNumberOfDimensions, Gna2DisabledU32, PoolingStrideParamIndex };
        ModelErrorHelper::ExpectAboveEq(PoolingStride.size(), 1, ctxStride);
    }
}

Gna2Tensor OperationConfig::GetEnabledOperand(uint32_t index) const
{
    Expect::NotNull(Operation);
    return ModelWrapper::GetEnabledOperand(*Operation, index);
}

bool OperationConfig::hasPooling(const Gna2Operation & operation)
{
    const auto poolingMode = ModelWrapper::GetOptionalParameter<Gna2PoolingMode>(operation, PoolingModeParamIndex, Gna2PoolingModeDisabled);
    if (poolingMode == Gna2PoolingModeDisabled)
    {
        ModelWrapper::ExpectParameterNotAvailable(operation, PoolingWindowParamIndex);
        ModelWrapper::ExpectParameterNotAvailable(operation, PoolingStrideParamIndex);
        return false;
    }
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, PoolingModeParamIndex };
    ModelErrorHelper::ExpectInSet(poolingMode, { Gna2PoolingModeMax, Gna2PoolingModeSum }, ctx);

    ModelWrapper::ExpectParameterAvailable(operation, PoolingWindowParamIndex);
    ModelWrapper::ExpectParameterAvailable(operation, PoolingStrideParamIndex);
    return true;
}

bool OperationConfig::isAffine(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeFullyConnectedAffine
        || operation.Type == Gna2OperationTypeElementWiseAffine
        || operation.Type == Gna2OperationTypeRecurrent;
}

bool OperationConfig::IsMultibias(const Gna2Operation & operation)
{
    if (operation.Type != Gna2OperationTypeFullyConnectedAffine)
    {
        return false;
    }

    if (!ModelWrapper::HasParameter(operation, BiasModeAffineParamIndex))
    {
        return false;
    }
    const auto biasMode = *static_cast<Gna2BiasMode *>(operation.Parameters[BiasModeAffineParamIndex]);

    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, BiasModeAffineParamIndex };
    ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault, Gna2BiasModeGrouping }, ctx);

    return biasMode == Gna2BiasModeGrouping;
}

bool OperationConfig::isCNN2D(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeConvolution;
}

bool OperationConfig::isRecurrent(const Gna2Operation& operation)
{
    return operation.Type == Gna2OperationTypeRecurrent;
}

uint32_t OperationConfig::GetFeedbackDelay(const Gna2Operation& operation)
{
    auto const delayIndex = ModelWrapper::GetOperationInfo(
        operation.Type, ParameterIndexDelay);
    return ModelWrapper::GetParameter<uint32_t>(operation, delayIndex);
}

std::unique_ptr<const Component> OperationConfig::CreateCnnComponent(const Shape& shape,
    const LayerValidator& validator, const FullCapabilitiesMap & caps, const uint32_t parameterIndex)
{
    auto const effectiveShape =
        shape.empty() ?
        Shape{ GNA_TENSOR_HW, 0u, 0u } :
        shape;
    return std::make_unique<const Component>(effectiveShape,
        Validator{ validator, caps }, true, parameterIndex, true);
}
