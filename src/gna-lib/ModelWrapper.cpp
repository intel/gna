/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ModelWrapper.h"

#include "DataMode.h"
#include "ExternalBuffer.h"
#include "Logger.h"
#include "ModelError.h"

#include "gna2-model-impl.h"

#include <set>

using namespace GNA;

void ModelWrapper::OperationInit(Gna2Operation& operation, const OperationType type,
    const Gna2UserAllocator userAllocator, bool initOnlyRequiredOperands)
{
    Expect::Equal(operation.Type, Gna2OperationTypeNone, Gna2StatusModelConfigurationInvalid);
    Expect::Equal(operation.NumberOfParameters, static_cast<uint32_t>(0), Gna2StatusModelConfigurationInvalid);
    Expect::Equal(operation.NumberOfOperands, static_cast<uint32_t>(0), Gna2StatusModelConfigurationInvalid);
    Expect::Null(operation.Operands);
    Expect::Null(operation.Parameters);
    Expect::NotNull(userAllocator);

    const auto numberOfOperands = GetOperationInfo(type,
        initOnlyRequiredOperands ? NumberOfOperandsRequired : NumberOfOperandsMax);
    const auto numberOfParameters = GetOperationInfo(type, NumberOfParametersMax);

    operation.Operands = AllocateAndFillZeros<Gna2Tensor const>(userAllocator, numberOfOperands);
    operation.Parameters = AllocateAndFillZeros<void>(userAllocator, numberOfParameters);
    operation.Type = type;
    operation.NumberOfOperands = numberOfOperands;
    operation.NumberOfParameters = numberOfParameters;
}

uint32_t ModelWrapper::DataTypeGetSize(DataType type)
{
    const auto dataSize = DataMode{ type }.Size;
    return dataSize;
}

uint32_t ModelWrapper::ShapeGetNumberOfElements(ApiShape const * shape)
{
    Expect::NotNull(shape);
    const auto shapeImpl = Shape::Create(*shape);
    return shapeImpl.GetNumberOfElements();
}

uint32_t ModelWrapper::GetOperationInfo(OperationType operationType, OperationInfoKey infoType)
{
    static const std::map<OperationType, std::map<OperationInfoKey, uint32_t>> metaOperationInfo =
    {
        { Gna2OperationTypeCopy,
            {
                { NumberOfOperandsMax, 2 },
                { NumberOfOperandsRequired, 2 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { ParameterIndexCopyShape, 0 },
            }
        },
        { Gna2OperationTypeConvolution,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 3 },
                { NumberOfParametersMax, 6 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { OperandIndexFilter, FilterOperandIndex },
                { OperandIndexBias, BiasOperandIndex },
                { OperandIndexActivation, PwlOperandIndex },
                { OperandIndexScratchPad, ScratchpadOperandIndex },
                { ParameterIndexConvolutionStride, 0 },
                { ParameterIndexBiasMode, 1 },
                { ParameterIndexPoolingMode, 2 },
                { ParameterIndexPoolingWindow, 3 },
                { ParameterIndexPoolingStride, 4 },
                { ParameterIndexZeroPadding, 5 },
            }
        },
        { Gna2OperationTypeElementWiseAffine,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 0 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { OperandIndexWeight, WeightOperandIndex },
                { OperandIndexBias, BiasOperandIndex },
                { OperandIndexActivation, PwlOperandIndex},
                { OperandIndexScratchPad, ScratchpadOperandIndex },
            }
        },
        { Gna2OperationTypeFullyConnectedAffine,
            {
                { NumberOfOperandsMax, 6 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 2 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { OperandIndexWeight, WeightOperandIndex },
                { OperandIndexBias, BiasOperandIndex },
                { OperandIndexActivation, PwlOperandIndex},
                { OperandIndexWeightScaleFactors, WeightScaleFactorOperandIndex },
                { ParameterIndexBiasMode, 0 },
                { ParameterIndexBiasVectorIndex, 1 },
                { OperandIndexScratchPad, ScratchpadOperandIndex },
            }
        },
        { Gna2OperationTypeGmm,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 3 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { OperandIndexMeans, GmmMeanOperandIndex },               //"flat" layout
                { OperandIndexInverseCovariances, GmmInverseCovarianceOperandIndex },  //"flat" layout
                { OperandIndexConstants, GmmGaussianConstantOperandIndex },           //"flat" layout
                { OperandIndexInterleaved, GmmInterleavedOperandIndex},         //"interleaved" layout
                { ParameterIndexMaximumScore, 0 },
            }
        },
        { Gna2OperationTypeRecurrent,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 5 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
                { OperandIndexWeight, WeightOperandIndex },
                { OperandIndexBias, BiasOperandIndex },
                { OperandIndexActivation, PwlOperandIndex},
                { ParameterIndexDelay, 0 },
                { OperandIndexScratchPad, ScratchpadOperandIndex },
            }
        },
        { Gna2OperationTypeTransposition,
            {
                { NumberOfOperandsMax, 2 },
                { NumberOfOperandsRequired, 2 },
                { NumberOfParametersMax, 0 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, InputOperandIndex },
                { OperandIndexOutput, OutputOperandIndex },
            }
        },
    };
    try
    {
        const auto & o = metaOperationInfo.at(operationType);
        try
        {
            return o.at(infoType);
        }
        catch (const std::out_of_range &)
        {
            throw GnaException(Gna2StatusUnknownError);
        }
    }
    catch (const std::out_of_range &)
    {
        throw GnaModelErrorException(
            Gna2ItemTypeOperationType,
            Gna2ErrorTypeNotInSet,
            operationType);
    }
}

bool ModelWrapper::HasEnabledOperand(const Gna2Operation & apiOperation, uint32_t operandIndex)
{
    return apiOperation.NumberOfOperands > operandIndex &&
        nullptr != apiOperation.Operands &&
        nullptr != apiOperation.Operands[operandIndex] &&
        Gna2TensorModeDisabled != apiOperation.Operands[operandIndex]->Mode;
}

bool ModelWrapper::IsOperandAvailable(const Gna2Operation & operation, uint32_t index)
{
    return nullptr != operation.Operands &&
        index < operation.NumberOfOperands &&
        nullptr != operation.Operands[index];
}

Gna2Tensor ModelWrapper::GetOperand(const Gna2Operation & operation, uint32_t index, Gna2Tensor defaultValue)
{
    if (IsOperandAvailable(operation, index))
    {
        return *(operation.Operands[index]);
    }
    return defaultValue;
}

Gna2Tensor ModelWrapper::GetOperand(const Gna2Operation & apiOperation, uint32_t operandIndex)
{
    ModelErrorHelper::ExpectNotNull(apiOperation.Operands, Gna2ItemTypeOperationOperands);
    ModelErrorHelper::ExpectAboveEq(apiOperation.NumberOfOperands, operandIndex + 1, Gna2ItemTypeOperationNumberOfOperands);
    ModelErrorHelper::ExpectNotNull(apiOperation.Operands[operandIndex], Gna2ItemTypeOperationOperands, operandIndex);
    return *apiOperation.Operands[operandIndex];
}

void ExpectOperandModeAndDataValid(const Gna2Tensor & operand, const std::set<Gna2TensorMode>& validModes, uint32_t operandIndex)
{
    ModelErrorHelper::ExpectInSet(operand.Mode, validModes, ModelItem{ Gna2ItemTypeOperandMode, operandIndex });
}

Gna2Tensor ModelWrapper::GetEnabledOperand(const Gna2Operation & apiOperation, uint32_t operandIndex)
{
    const auto& operand = GetOperand(apiOperation, operandIndex);
    auto validModes = GetValidTensorModes(apiOperation, operandIndex);
    validModes.erase(Gna2TensorModeDisabled);
    ExpectOperandModeAndDataValid(operand, validModes, operandIndex);
    return operand;
}

Gna2Tensor ModelWrapper::GetDisabledOperand()
{
    return { {}, Gna2TensorModeDisabled, {}, Gna2DataTypeNone, nullptr };
}

bool ModelWrapper::HasParameter(const Gna2Operation& operation, uint32_t parameterIndex)
{
    return operation.Parameters != nullptr &&
        parameterIndex < operation.NumberOfParameters &&
        operation.Parameters[parameterIndex] != nullptr;
}

void ModelWrapper::ExpectParameterAvailable(const Gna2Operation & operation, uint32_t parameterIndex)
{
    ModelErrorHelper::ExpectNotNull(operation.Parameters, Gna2ItemTypeOperationParameters);
    ModelErrorHelper::ExpectAboveEq(operation.NumberOfParameters, parameterIndex + 1, Gna2ItemTypeOperationNumberOfParameters);
    ModelErrorHelper::ExpectNotNull(operation.Parameters[parameterIndex], Gna2ItemTypeOperationParameters, parameterIndex, true);
}

void ModelWrapper::ExpectParameterNotAvailable(const Gna2Operation & operation, uint32_t parameterIndex)
{
    if (operation.Parameters == nullptr || operation.NumberOfParameters <= parameterIndex)
    {
        return;
    }
    ModelErrorHelper::ExpectNull(operation.Parameters[parameterIndex], Gna2ItemTypeOperationParameters, parameterIndex, true);
}

void ModelWrapper::SetLayout(Gna2Tensor& tensor, const char* layout)
{
    snprintf(tensor.Layout, sizeof(tensor.Layout), "%s", layout);
}

template<class T>
void ExpectPointerArrayValid(T ** ptr, uint32_t arraySize,
    uint32_t reqNotNull, uint32_t maxSize, const Gna2ItemType itemType = Gna2ItemTypeOperationOperands)
{
    ModelErrorHelper::ExpectAboveEq(arraySize, reqNotNull, itemType);
    ModelErrorHelper::ExpectBelowEq(arraySize, maxSize, itemType);
    const bool isParameter = itemType == Gna2ItemTypeOperationParameters;
    if (0 == arraySize && ptr != nullptr)
    {
        Log->Warning("Not null pointer provided although number of elements set to zero.\n");
    }
    else if(arraySize > 0)
    {
        ModelErrorHelper::ExpectNotNull(ptr, itemType);
        for (uint32_t i = 0; i < reqNotNull; i++)
        {
            ModelErrorHelper::ExpectNotNull(ptr[i], itemType, i, isParameter);
        }
    }
}

std::set<Gna2TensorMode> ModelWrapper::GetValidTensorModes(const Gna2Operation & operation, uint32_t operandIndex)
{
    std::set<Gna2TensorMode> validSet = { Gna2TensorModeDefault };
    if (ExternalBuffer::IsSupported(operation, operandIndex))
    {
        validSet.insert(Gna2TensorModeExternalBuffer);
    }
    const auto opRequired = GetOperationInfo(operation.Type, NumberOfOperandsRequired);
    if(operandIndex >= opRequired)
    {
        validSet.insert(Gna2TensorModeDisabled);
    }
    return validSet;
}

void ModelWrapper::ExpectOperationValid(const Gna2Operation & operation)
{
    const auto opRequired = GetOperationInfo(operation.Type, NumberOfOperandsRequired);
    const auto opMax = GetOperationInfo(operation.Type, NumberOfOperandsMax);
    ExpectPointerArrayValid(operation.Operands, operation.NumberOfOperands,
        opRequired, opMax);
    for (uint32_t index = 0; index < operation.NumberOfOperands; index++)
    {
        const auto& operand = GetOperand(operation, index, GetDisabledOperand());
        ExpectOperandModeAndDataValid(operand, GetValidTensorModes(operation, index), index);
    }

    const auto paramRequired = GetOperationInfo(operation.Type, NumberOfParametersRequired);
    const auto paramMax = GetOperationInfo(operation.Type, NumberOfParametersMax);
    ExpectPointerArrayValid(operation.Parameters, operation.NumberOfParameters,
        paramRequired, paramMax, Gna2ItemTypeOperationParameters);
}
