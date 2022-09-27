/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#ifndef __GNA2_MODEL_WRAPPER_H
#define __GNA2_MODEL_WRAPPER_H

#include "gna2-model-impl.h"

#include "Expect.h"
#include "GnaException.h"
#include "Shape.h"
#include "Tensor.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>

namespace GNA
{

enum OperationInfoKey
{
    NumberOfOperandsRequired, //must be passed from user as not null
    NumberOfOperandsMax,
    NumberOfParametersRequired, //must be passed from user as not null
    NumberOfParametersMax,
    OperandIndexInput,
    OperandIndexOutput,
    OperandIndexScratchPad,
    OperandIndexWeight,
    OperandIndexFilter,
    OperandIndexBias,
    OperandIndexActivation,
    OperandIndexWeightScaleFactors,
    OperandIndexMeans,
    OperandIndexInverseCovariances,
    OperandIndexConstants,
    OperandIndexInterleaved,

    ParameterIndexCopyShape,
    ParameterIndexConvolutionStride,
    ParameterIndexBiasMode,
    ParameterIndexPoolingMode,
    ParameterIndexPoolingWindow,
    ParameterIndexPoolingStride,
    ParameterIndexZeroPadding,
    ParameterIndexBiasVectorIndex,
    ParameterIndexMaximumScore,
    ParameterIndexDelay,
    ParameterIndexThresholdCondition,
    ParameterIndexThresholdMode,
    ParameterIndexThresholdMask,
};

class ModelWrapper
{
public:
    static void OperationInit(Gna2Operation& operation,
        OperationType type, Gna2UserAllocator userAllocator, bool initOnlyRequiredOperands = false);

    static uint32_t DataTypeGetSize(DataType type);

    static uint32_t ShapeGetNumberOfElements(ApiShape const * shape);

    static ApiShape ShapeInit()
    {
        const auto shape = Shape(GNA_TENSOR_SCALAR);
        return static_cast<ApiShape>(shape);
    }

    template<typename ... T>
    static ApiShape ShapeInit(T ... dimensions)
    {
        const auto shape = Shape(GNA_TENSOR_ORDER_ANY, static_cast<uint32_t>(dimensions)...);
        return static_cast<ApiShape>(shape);
    }

    template<typename ... T>
    static ApiTensor TensorInit(const DataType dataType, const TensorMode tensorMode,
        void const * buffer, T ... dimensions)
    {
        auto const shape = Shape(GNA_TENSOR_ORDER_ANY, static_cast<uint32_t>(dimensions)...);
        auto const tensor = std::make_unique<Tensor>(shape, DataMode{dataType, tensorMode}, buffer);
        return static_cast<ApiTensor>(*tensor);
    }

    // The first numberOfRequired pointers in source must not be nullptr, otherwise exception is thrown
    template<class T, class V>
    static void TryAssign(T ** const destination, const size_t destinationSize,
        uint32_t numberOfRequired, std::initializer_list<V*> source)
    {
        Expect::True(destinationSize >= source.size(),
            Gna2StatusModelConfigurationInvalid);
        Expect::True(numberOfRequired <= source.size(), Gna2StatusModelConfigurationInvalid);
        int i = 0;
        for (const auto& s : source)
        {
            if (0 < numberOfRequired)
            {
                Expect::NotNull(s);
                --numberOfRequired;
            }
            destination[i++] = s;
        }
        std::fill(destination + i, destination + destinationSize, nullptr);
    }

    template<class ... T>
    static void SetOperands(Gna2Operation & operation, T ... operands)
    {
        Expect::True(operation.NumberOfOperands >= GetOperationInfo(operation.Type, NumberOfOperandsRequired),
            Gna2StatusModelConfigurationInvalid);
        const auto requiredNotNull = GetOperationInfo(operation.Type, NumberOfOperandsRequired);
        TryAssign(operation.Operands, operation.NumberOfOperands,
            requiredNotNull, {std::forward<T>(operands)...});
    }

    template<class ... T>
    static void SetParameters(Gna2Operation & operation, T ... parameters)
    {
        Expect::Equal(operation.NumberOfParameters, GetOperationInfo(operation.Type, NumberOfParametersMax),
            Gna2StatusModelConfigurationInvalid);
        const auto requiredNotNull = GetOperationInfo(operation.Type, NumberOfParametersRequired);
        TryAssign(operation.Parameters, operation.NumberOfParameters,
            requiredNotNull, {static_cast<void*>(parameters)...});
    }

    static void SetLayout(Gna2Tensor& tensor, const char* layout);

    static std::set<Gna2TensorMode> GetValidTensorModes(const Gna2Operation & operation, uint32_t operandIndex);
    static void ExpectOperationValid(const Gna2Operation& operation);

    static uint32_t GetOperationInfo(OperationType operationType, OperationInfoKey infoType);
    static bool HasEnabledOperand(const Gna2Operation& apiOperation, uint32_t operandIndex);
    static bool IsOperandAvailable(const Gna2Operation & operation, uint32_t index);

    static Gna2Tensor GetOperand(const Gna2Operation & operation, uint32_t index, Gna2Tensor defaultValue);
    static Gna2Tensor GetOperand(const Gna2Operation & apiOperation, uint32_t operandIndex);
    static Gna2Tensor GetEnabledOperand(const Gna2Operation & apiOperation, uint32_t operandIndex);

    static Gna2Tensor GetDisabledOperand();

    static bool HasParameter(const Gna2Operation& operation, uint32_t parameterIndex);
    static void ExpectParameterAvailable(const Gna2Operation & operation, uint32_t index);
    static void ExpectParameterNotAvailable(const Gna2Operation & operation, uint32_t index);
    template<class T>
    static T GetParameter(const Gna2Operation & operation, OperationInfoKey parameter)
    {
        auto const index = GetOperationInfo(operation.Type, parameter);
        return GetParameter<T>(operation, index);
    }

    template<class T>
    static T GetParameter(const Gna2Operation & operation, uint32_t index)
    {
        ExpectParameterAvailable(operation, index);
        return *static_cast<T*> (operation.Parameters[index]);
    }

    template<class T>
    static T GetOptionalParameter(const Gna2Operation& operation, OperationInfoKey parameter, T defaultValue)
    {
        auto const index = GetOperationInfo(operation.Type, parameter);
        return GetOptionalParameter<T>(operation, index, defaultValue);
    }

    template<class T>
    static T GetOptionalParameter(const Gna2Operation& operation, uint32_t parameterIndex, T defaultValue)
    {
        if(HasParameter(operation, parameterIndex))
        {
            return *static_cast<const T*>(operation.Parameters[parameterIndex]);
        }
        return defaultValue;
    }

private:
    template<typename Type>
    static Type ** AllocateAndFillZeros(const Gna2UserAllocator userAllocator, uint32_t elementCount)
    {
        if (elementCount == 0)
        {
            return nullptr;
        }
        const auto size = static_cast<uint32_t>(sizeof(Type *)) * elementCount;
        const auto memory = userAllocator(size);
        Expect::NotNull(memory, Gna2StatusResourceAllocationError);
        memset(memory, 0, size);
        return static_cast<Type **>(memory);
    }
};

}

#endif //ifndef __GNA2_MODEL_WRAPPER_H
