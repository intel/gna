/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "GnaException.h"

#include "DataMode.h"
#include "gna2-common-api.h"
#include "gna2-model-api.h"
#include "gna2-model-impl.h"

#include <cstring>
#include <functional>
#include <map>
#include <set>

namespace GNA
{

// Gna2ModelItem wrapper
struct ModelItem : Gna2ModelItem
{
    constexpr ModelItem(Gna2ItemType type = Gna2ItemTypeNone) :
        ModelItem{ type, Gna2DisabledU32, Gna2DisabledU32 }
    {}

    constexpr ModelItem(Gna2ItemType type, uint32_t operandIndex) :
        ModelItem{ type, operandIndex, Gna2DisabledU32 }
    {}

    constexpr ModelItem(Gna2ItemType type, uint32_t operandIndex, uint32_t parameterIndex) :
        Gna2ModelItem{
            type,
            GNA2_DISABLED,
            static_cast<int32_t>(operandIndex),
            static_cast<int32_t>(parameterIndex),
            GNA2_DISABLED,
            {GNA2_DISABLED, GNA2_DISABLED, GNA2_DISABLED, GNA2_DISABLED}
    }
    {}
};

template<typename T>
class ModelValueT
{
public:
    constexpr explicit ModelValueT(T value) :
        Value{ value }
    {}

    constexpr explicit ModelValueT(T value, int32_t dimensionIndex, uint32_t operandIndex) :
        Value{ value }
    {
        Source.Type = Gna2ItemTypeOperationOperands;
        if (dimensionIndex != GNA2_DISABLED)
        {
            Source.Type = Gna2ItemTypeShapeDimensions; // most detailed information
        }
        Source.ShapeDimensionIndex = dimensionIndex;
        Source.OperandIndex = static_cast<int32_t>(operandIndex);
    }

    ModelValueT& SetParameter(uint32_t index)
    {
        Source.Type = Gna2ItemTypeParameter;
        Source.ParameterIndex = static_cast<int32_t>(index);
        return *this;
    }

    constexpr operator T() const
    {
        return Value;
    }

    constexpr operator Gna2ModelItem() const
    {
        return Source;
    }

    T Value;
    ModelItem Source;
};

using ModelValue = ModelValueT<int64_t>;

struct ModelError : Gna2ModelError
{
    using Gna2ModelError::Gna2ModelError;

    constexpr ModelError() :
        Gna2ModelError{ ModelItem(), Gna2ErrorTypeNone, 0 }
    {}

    constexpr ModelError(Gna2ModelError & e) :
        Gna2ModelError{ e }
    {}

    constexpr ModelError(Gna2Status status) :
        Gna2ModelError{ ModelItem(Gna2ItemTypeInternal), Gna2ErrorTypeOther, static_cast<int64_t>(status) }
    {}

    template<typename ValueT>
    constexpr ModelError(Gna2ErrorType reason, ValueT value, Gna2ModelItem item) :
        Gna2ModelError{ item, reason, static_cast<int64_t>(value) }
    {}

    constexpr ModelError(Gna2ErrorType reason, DataMode value, Gna2ModelItem item) :
        Gna2ModelError{ item, reason, static_cast<int64_t>(value.Type) }
    {}

    constexpr ModelError(Gna2ErrorType reason, ModelValue value) :
        Gna2ModelError{ static_cast<Gna2ModelItem>(value), reason, static_cast<int64_t>(value) }
    {}

    template<typename ValueT,
        typename = std::enable_if<!std::is_same<ValueT, ModelValue>::value>>
    constexpr ModelError(Gna2ErrorType reason, ValueT value, Gna2ItemType type) :
        Gna2ModelError{ static_cast<Gna2ModelItem>(ModelItem{ type }), reason, static_cast<int64_t>(value) }
    {}

    bool operator==(ModelError const & right) const
    {
        return memcmp(this, &right, sizeof(ModelError)) == 0;
    }
};

class GnaModelErrorException : public GnaException
{
public:
    GnaModelErrorException(ModelError errorIn = ModelError()) :
        GnaException{ Gna2StatusModelConfigurationInvalid },
        error{ errorIn }
    {}

    GnaModelErrorException(GnaException& e, uint32_t operandIndex = Gna2DisabledU32, uint32_t parameterIndex = Gna2DisabledU32);

    GnaModelErrorException(uint32_t layerIndex, Gna2Status status = Gna2StatusUnknownError);

    GnaModelErrorException(Gna2ItemType item, Gna2ErrorType errorType, int64_t value);

    static void DispatchAndFill(uint32_t operandIndex = Gna2DisabledU32, uint32_t parameterIndex = Gna2DisabledU32);

    static void DispatchAndSetLayer(uint32_t layerIndex);

    auto GetModelError() const
    {
        return error;
    }


    void SetDimensionIndex(int32_t dimensionIndex)
    {
        if (error.Source.Type == Gna2ItemTypeNone)
        {
            error.Source.Type = Gna2ItemTypeShapeDimensions;
        }
        error.Source.ShapeDimensionIndex = dimensionIndex;
    }

    GnaModelErrorException(const GnaModelErrorException&) = default;
    GnaModelErrorException& operator=(const GnaModelErrorException&) = default;
    GnaModelErrorException(GnaModelErrorException&&) = default;
    GnaModelErrorException& operator=(GnaModelErrorException&&) = default;
    virtual ~GnaModelErrorException() = default;

protected:
    void SetLayerIndex(uint32_t index)
    {
        error.Source.OperationIndex = static_cast<int32_t>(index);
    }

    void SetOperandIndex(uint32_t operandIndex)
    {
        if (operandIndex != Gna2DisabledU32)
        {
            error.Source.OperandIndex = static_cast<int32_t>(operandIndex);
        }
    }

    void SetParameterIndex(uint32_t parameterIndex)
    {
        if (parameterIndex != Gna2DisabledU32)
        {
            if (error.Source.Type == Gna2ItemTypeNone)
            {
                error.Source.Type = Gna2ItemTypeParameter;
            }
            error.Source.ParameterIndex = static_cast<int32_t>(parameterIndex);
        }
    }

    ModelError error;
};

struct Component;

class ModelErrorHelper
{
public:
    static void ExpectTrue(bool val, Gna2ModelError error);

    static void ExpectNotNull(const void * const ptr,
        Gna2ItemType ptrType = Gna2ItemTypeOperandData,
        uint32_t ptrIndex = Gna2DisabledU32,
        bool indexForParameter = false);

    static void ExpectNull(const void * const ptr,
        Gna2ItemType ptrType = Gna2ItemTypeOperandData,
        uint32_t ptrIndex = Gna2DisabledU32,
        bool indexForParameter = false);

    template<Gna2ErrorType reason, typename ValueT, typename ... ContextType>
    static void Expect(bool condition, ValueT value, ContextType ... context)
    {
        ExpectTrue(condition, ModelError(reason, value, std::forward<ContextType>(context)...));
    }

    template<typename A, typename B, typename ... ContextType>
    static void ExpectEqual(A val, B ref, ContextType ... context)
    {
        Expect<Gna2ErrorTypeNotEqual>(val == static_cast<A>(ref), val, std::forward<ContextType>(context)...);
    }

    template<typename A, typename ... ContextType>
    static void ExpectGtZero(A val, ContextType ... context)
    {
        Expect<Gna2ErrorTypeNotGtZero>(val > static_cast<A>(0), val, std::forward<ContextType>(context)...);
    }

    template<typename T, typename ... ContextType>
    static void ExpectInSet(const T val, const std::set<T>& ref, ContextType ... context)
    {
        Expect<Gna2ErrorTypeNotInSet>(ref.find(val) != ref.end(), val, std::forward<ContextType>(context)...);
    }

    static void ExpectInSet(const DataType val, const std::set<DataType>& ref)
    {
        ExpectInSet(val, ref, Gna2ItemTypeOperandType);
    }

    static void ExpectInSet(const Gna2TensorMode val, const std::set<Gna2TensorMode>& ref)
    {
        ExpectInSet(val, ref, Gna2ItemTypeOperandMode);
    }

    template<class A, class B, typename ... ContextType>
    static void ExpectBelowEq(A val, B ref, ContextType ... context)
    {
        Expect<Gna2ErrorTypeAboveRange>(val <= static_cast<A>(ref), val, std::forward<ContextType>(context)...);
    }

    template<class A, class B, typename ... ContextType>
    static void ExpectAboveEq(A val, B ref, ContextType ... context)
    {
        Expect<Gna2ErrorTypeBelowRange>(val >= static_cast<A>(ref), val, std::forward<ContextType>(context)...);
    }

    template<class A, class B, typename ... ContextType>
    static void ExpectInRange(A val, B min, B max, ContextType ... context)
    {
        ExpectAboveEq(val, min, std::forward<ContextType>(context)...);
        ExpectBelowEq(val, max, std::forward<ContextType>(context)...);
    }

    template<class A, class B, typename ... ContextType>
    static void ExpectInRange(A val, B max, ContextType ... context)
    {
        ExpectAboveEq(val, static_cast<B>(0), std::forward<ContextType>(context)...);
        ExpectBelowEq(val, max, std::forward<ContextType>(context)...);
    }

    template<class A, class B, typename ... ContextType>
    static void ExpectMultiplicityOf(A val, B factor, ContextType ... context)
    {
        Expect<Gna2ErrorTypeNotMultiplicity>(
            val == 0 || (static_cast<A>(factor) != 0 && (val % static_cast<A>(factor)) == 0),
            val, std::forward<ContextType>(context)...);
    }

    static void ExpectBufferAligned(const void * const buffer, const uint32_t alignment)
    {
        Expect<Gna2ErrorTypeNotAligned>(
            alignment != 0 && reinterpret_cast<int64_t>(buffer) % alignment == 0,
            reinterpret_cast<int64_t>(buffer), Gna2ItemTypeOperandData);
    }

    static void SaveLastError(const ModelError& modelError);
    static void PopLastError(Gna2ModelError& error);
    static Gna2Status ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn);

    template<typename CommandType>
    static auto ExecuteForModelItem(CommandType const & command,
        uint32_t operandIndexContext, uint32_t parameterIndexContext = Gna2DisabledU32)
    {
        try
        {
            return command();
        }
        catch (GnaException&)
        {
            GnaModelErrorException::DispatchAndFill(operandIndexContext, parameterIndexContext);
            throw;
        }
    }

    static const std::map<enum Gna2ErrorType, std::string>& GetAllErrorTypeStrings();
    static const std::map<enum Gna2ItemType, std::string>& GetAllItemTypeStrings();
    static std::string GetErrorString(const Gna2ModelError& error);
    static uint32_t GetErrorStringMaxLength();
private:
    static void AppendNotDisabled(std::string& toAppend, int32_t index, const std::string& arrayName);

    static Gna2ModelError GetPartiallySetError(const Gna2ItemType ptrType,
        const uint32_t ptrIndex,
        const bool indexForParameter);

    static ModelError lastError;
};

}
