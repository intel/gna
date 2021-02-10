/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "GnaException.h"

#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include <functional>
#include <map>
#include <set>

namespace GNA
{
class ModelValue;

class ModelErrorHelper
{
public:
    static void ExpectTrue(bool val, Gna2ModelError error);
    static void ExpectGtZero(int64_t val, Gna2ItemType valType);
    static void ExpectEqual(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectEqual(int64_t val, int64_t ref, Gna2ModelItem item);
    static void ExpectBelowEq(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectBelowEq(int64_t val, int64_t ref, Gna2ModelItem item);
    static void ExpectAboveEq(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectAboveEq(int64_t val, int64_t ref, Gna2ModelItem item);
    static void ExpectMultiplicityOf(int64_t val, int64_t factor, Gna2ItemType valType);
    static void ExpectNotNull(const void * const ptr,
        Gna2ItemType ptrType = Gna2ItemTypeOperandData,
        int32_t ptrIndex = GNA2_DISABLED,
        bool indexForParameter = false);
    static void ExpectBufferAligned(const void * const buffer, const uint32_t alignment);

    static void ExpectEqual(const ModelValue& val, const ModelValue& ref);
    static void ExpectBelowEq(const ModelValue& val, const ModelValue& ref);
    static void ExpectAboveEq(const ModelValue& val, const ModelValue& ref);

    template<class A, class B>
    static void ExpectEqual(A val, B ref, Gna2ItemType valType)
    {
        ExpectEqual(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class T>
    static void ExpectInSet(const T val, const std::set<T>& ref, Gna2ItemType itemType)
    {
        Gna2ModelError e = GetCleanedError();
        e.Source.Type = itemType;
        e.Value = static_cast<int64_t>(val);
        e.Reason = Gna2ErrorTypeNotInSet;
        ExpectTrue(ref.find(val) != ref.end(), e);
    }

    static void ExpectInSet(const Gna2DataType val, const std::set<Gna2DataType>& ref)
    {
        ExpectInSet(val, ref, Gna2ItemTypeOperandType);
    }

    static void ExpectInSet(const Gna2TensorMode val, const std::set<Gna2TensorMode>& ref)
    {
        ExpectInSet(val, ref, Gna2ItemTypeOperandMode);
    }

    template<class A, class B>
    static void ExpectBelowEq(A val, B ref, Gna2ItemType valType)
    {
        ExpectBelowEq(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class A, class B>
    static void ExpectAboveEq(A val, B ref, Gna2ItemType valType)
    {
        ExpectAboveEq(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class A, class B>
    static void ExpectMultiplicityOf(A val, B factor, Gna2ItemType valType)
    {
        ExpectMultiplicityOf(static_cast<int64_t>(val), static_cast<int64_t>(factor), valType);
    }

    static void SaveLastError(const Gna2ModelError& modelError);
    static void PopLastError(Gna2ModelError& error);
    static Gna2Status ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn);

    static Gna2ModelError GetCleanedError();
    static Gna2ModelError GetStatusError(Gna2Status status);

    static void SetOperandIndexRethrow(GnaException& e, int32_t index);

    static void ExecuteForModelItem(const std::function<void()>& command,
        int32_t operandIndexContext, int32_t parameterIndexContext = GNA2_DISABLED);

    static const std::map<enum Gna2ErrorType, std::string>& GetAllErrorTypeStrings();
    static const std::map<enum Gna2ItemType, std::string>& GetAllItemTypeStrings();
    static std::string GetErrorString(const Gna2ModelError& error);
    static uint32_t GetErrorStringMaxLength();
private:
    static void AppendNotDisabled(std::string& toAppend, int32_t index, const std::string& arrayName);
    static Gna2ModelError lastError;
};

class ModelValue
{
public:
    ModelValue(int64_t valueIn);

    ModelValue& SetOperand(int32_t index)
    {
        Source.OperandIndex = index;
        return *this;
    }
    ModelValue& SetDimension(int32_t index)
    {
        Source.Type = Gna2ItemTypeShapeDimensions;
        Source.ShapeDimensionIndex = index;
        return *this;
    }
    ModelValue& SetParameter(int32_t index)
    {
        Source.Type = Gna2ItemTypeParameter;
        Source.ParameterIndex = index;
        return *this;
    }
    ModelValue& SetItem(Gna2ItemType itemType)
    {
        Source.Type = itemType;
        return *this;
    }
    int64_t GetValue() const;
    Gna2ModelItem GetSource() const;

protected:
    int64_t Value;
    Gna2ModelItem Source = ModelErrorHelper::GetCleanedError().Source;
};

class GnaModelErrorException : public GnaException
{
public:
    GnaModelErrorException(Gna2ModelError errorIn = ModelErrorHelper::GetCleanedError()) :
        GnaException{ Gna2StatusModelConfigurationInvalid },
        error{ errorIn }
    {
    }
    GnaModelErrorException(const GnaException& e) :
        GnaException(e),
        error{ ModelErrorHelper::GetStatusError(e.GetStatus()) }
    {
    }
    GnaModelErrorException(uint32_t layerIndex, Gna2Status code = Gna2StatusUnknownError);
    GnaModelErrorException(Gna2ItemType item, Gna2ErrorType errorType, int64_t value);

    Gna2ModelError GetModelError() const
    {
        return error;
    }
    void SetLayerIndex(uint32_t index)
    {
        error.Source.OperationIndex = static_cast<int32_t>(index);
    }
    void SetOperandIndex(int32_t operandIndex)
    {
        error.Source.OperandIndex = operandIndex;
    }
    void SetParameterIndex(int32_t parameterIndex)
    {
        if(error.Source.Type == Gna2ItemTypeNone && parameterIndex != GNA2_DISABLED)
        {
            error.Source.Type = Gna2ItemTypeParameter;
        }
        error.Source.ParameterIndex = parameterIndex;
    }
    void SetDimensionIndex(int32_t dimensionIndex)
    {
        if (error.Source.Type == Gna2ItemTypeNone)
        {
            error.Source.Type = Gna2ItemTypeShapeDimensions;
        }
        error.Source.ShapeDimensionIndex = dimensionIndex;
    }

    virtual ~GnaModelErrorException() = default;

private:
    Gna2ModelError error;
};

}