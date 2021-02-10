/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ModelError.h"

#include "ApiWrapper.h"
#include "Expect.h"
#include "StringHelper.h"

using namespace GNA;

Gna2ModelError ModelErrorHelper::lastError = ModelErrorHelper::GetCleanedError();

void ModelErrorHelper::ExpectTrue(bool val, Gna2ModelError error)
{
    if (!val)
    {
        throw GnaModelErrorException(error);
    }
}

void ModelErrorHelper::ExpectGtZero(int64_t val, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotGtZero;
    ExpectTrue(val > 0, e);
}

void ModelErrorHelper::ExpectEqual(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    ExpectEqual(val, ref, e.Source);
}

void ModelErrorHelper::ExpectEqual(int64_t val, int64_t ref, Gna2ModelItem item)
{
    ExpectTrue(val == ref, { item, Gna2ErrorTypeNotEqual, val });
}

void ModelErrorHelper::ExpectEqual(const ModelValue& val, const ModelValue& ref)
{
    ExpectEqual(val.GetValue(), ref.GetValue(), val.GetSource());
}

void ModelErrorHelper::ExpectBelowEq(const ModelValue& val, const ModelValue& ref)
{
    ExpectBelowEq(val.GetValue(), ref.GetValue(), val.GetSource());
}

void ModelErrorHelper::ExpectBelowEq(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeAboveRange;
    ExpectTrue(val <= ref, e);
}

void ModelErrorHelper::ExpectBelowEq(int64_t val, int64_t ref, Gna2ModelItem item)
{
    ExpectTrue(val <= ref, { item, Gna2ErrorTypeAboveRange, val });
}

void ModelErrorHelper::ExpectAboveEq(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeBelowRange;
    ExpectTrue(val >= ref, e);
}

void ModelErrorHelper::ExpectAboveEq(int64_t val, int64_t ref, Gna2ModelItem item)
{
    ExpectTrue(val >= ref, { item, Gna2ErrorTypeBelowRange, val });
}

void ModelErrorHelper::ExpectAboveEq(const ModelValue& val, const ModelValue& ref)
{
    ExpectAboveEq(val.GetValue(), ref.GetValue(), val.GetSource());
}

void ModelErrorHelper::ExpectMultiplicityOf(int64_t val, int64_t factor, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotMultiplicity;
    ExpectTrue(val == 0 || (factor != 0 && (val % factor) == 0), e);
}

void ModelErrorHelper::ExpectNotNull(const void * const ptr,
    const Gna2ItemType ptrType,
    const int32_t ptrIndex,
    const bool indexForParameter)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = ptrType;
    if (indexForParameter)
    {
        e.Source.ParameterIndex = ptrIndex;
    }
    else
    {
        e.Source.OperandIndex = ptrIndex;
    }
    e.Value = 0;
    e.Reason = Gna2ErrorTypeNullNotAllowed;
    ExpectTrue(ptr != nullptr, e);
}

void ModelErrorHelper::ExpectBufferAligned(const void * const buffer, const uint32_t alignment)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeOperandData;
    e.Value = reinterpret_cast<int64_t>(buffer);
    e.Reason = Gna2ErrorTypeNotAligned;
    ExpectTrue(alignment != 0 && ((e.Value % alignment) == 0), e);
}

void ModelErrorHelper::SaveLastError(const Gna2ModelError& modelError)
{
    lastError = modelError;
}

void ModelErrorHelper::PopLastError(Gna2ModelError& error)
{
    Expect::True(lastError.Source.Type != Gna2ItemTypeNone, Gna2StatusUnknownError);
    error = lastError;
    lastError = GetCleanedError();
}

Gna2Status ModelErrorHelper::ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn)
{
    const std::function<ApiStatus()> command = [&]()
    {
        try
        {
            return commandIn();
        }
        catch (GnaModelErrorException& exception)
        {
            ModelErrorHelper::SaveLastError(exception.GetModelError());
            throw GnaException(Gna2StatusModelConfigurationInvalid);
        }
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2ModelError ModelErrorHelper::GetCleanedError()
{
    Gna2ModelError e = {};
    e.Reason = Gna2ErrorTypeNone;
    e.Value = 0;
    e.Source.Type = Gna2ItemTypeNone;
    e.Source.OperationIndex = GNA2_DISABLED;
    e.Source.OperandIndex = GNA2_DISABLED;
    e.Source.ParameterIndex = GNA2_DISABLED;
    e.Source.ShapeDimensionIndex = GNA2_DISABLED;
    for (auto& property : e.Source.Properties)
    {
        property = GNA2_DISABLED;
    }
    return e;
}

Gna2ModelError ModelErrorHelper::GetStatusError(Gna2Status status)
{
    auto e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Reason = Gna2ErrorTypeOther;
    e.Value = status;
    return e;
}

GnaModelErrorException::GnaModelErrorException(
    Gna2ItemType item,
    Gna2ErrorType errorType,
    int64_t value)
    : GnaException{ Gna2StatusModelConfigurationInvalid }
{
    error = ModelErrorHelper::GetCleanedError();
    error.Source.Type = item;
    error.Reason = errorType;
    error.Value = value;
}

GnaModelErrorException::GnaModelErrorException(uint32_t layerIndex, Gna2Status capturedCode)
    : GnaException{ Gna2StatusModelConfigurationInvalid }
{
    error = ModelErrorHelper::GetStatusError(capturedCode);
    SetLayerIndex(layerIndex);
}

void ModelErrorHelper::SetOperandIndexRethrow(GnaException& e, int32_t index)
{
    auto x = dynamic_cast<GnaModelErrorException*>(&e);
    if(x != nullptr)
    {
        x->SetOperandIndex(index);
        throw;
    }
    GnaModelErrorException n(e);
    n.SetOperandIndex(index);
    throw n;
}

void ModelErrorHelper::ExecuteForModelItem(const std::function<void()>& command,
    int32_t operandIndexContext, int32_t parameterIndexContext)
{
    try
    {
        command();
    }
    catch (GnaModelErrorException& e)
    {
        e.SetOperandIndex(operandIndexContext);
        e.SetParameterIndex(parameterIndexContext);
        throw;
    }
    catch (GnaException& e)
    {
        GnaModelErrorException n(e);
        n.SetOperandIndex(operandIndexContext);
        n.SetParameterIndex(parameterIndexContext);
        throw n;
    }
}

const std::map<enum Gna2ErrorType, std::string>& ModelErrorHelper::GetAllErrorTypeStrings()
{
    static const std::map<enum Gna2ErrorType, std::string> ErrorTypeStrings =
    {
        {Gna2ErrorTypeNone,            "Gna2ErrorTypeNone" },
        {Gna2ErrorTypeNotTrue,         "Gna2ErrorTypeNotTrue" },
        {Gna2ErrorTypeNotFalse,        "Gna2ErrorTypeNotFalse" },
        {Gna2ErrorTypeNullNotAllowed,  "Gna2ErrorTypeNullNotAllowed" },
        {Gna2ErrorTypeNullRequired,    "Gna2ErrorTypeNullRequired" },
        {Gna2ErrorTypeBelowRange,      "Gna2ErrorTypeBelowRange" },
        {Gna2ErrorTypeAboveRange,      "Gna2ErrorTypeAboveRange" },
        {Gna2ErrorTypeNotEqual,        "Gna2ErrorTypeNotEqual" },
        {Gna2ErrorTypeNotGtZero,       "Gna2ErrorTypeNotGtZero" },
        {Gna2ErrorTypeNotZero,         "Gna2ErrorTypeNotZero" },
        {Gna2ErrorTypeNotOne,          "Gna2ErrorTypeNotOne" },
        {Gna2ErrorTypeNotInSet,        "Gna2ErrorTypeNotInSet" },
        {Gna2ErrorTypeNotMultiplicity, "Gna2ErrorTypeNotMultiplicity" },
        {Gna2ErrorTypeNotSuccess,      "Gna2ErrorTypeNotSuccess" },
        {Gna2ErrorTypeNotAligned,      "Gna2ErrorTypeNotAligned" },
        {Gna2ErrorTypeArgumentMissing, "Gna2ErrorTypeArgumentMissing" },
        {Gna2ErrorTypeArgumentInvalid, "Gna2ErrorTypeArgumentInvalid" },
        {Gna2ErrorTypeRuntime,         "Gna2ErrorTypeRuntime" },
        {Gna2ErrorTypeOther,           "Gna2ErrorTypeOther" },
    };
    return ErrorTypeStrings;
}

const std::map<enum Gna2ItemType, std::string>& ModelErrorHelper::GetAllItemTypeStrings()
{
    static const std::map<enum Gna2ItemType, std::string> itemTypeStrings =
    {
        { Gna2ItemTypeNone,                        "Gna2ItemTypeNone"},
        { Gna2ItemTypeModelNumberOfOperations,     "Gna2ItemTypeModelNumberOfOperations"},
        { Gna2ItemTypeModelOperations,             "Gna2ItemTypeModelOperations"},
        { Gna2ItemTypeOperationType,               "Gna2ItemTypeOperationType"},
        { Gna2ItemTypeOperationOperands ,          "Gna2ItemTypeOperationOperands"},
        { Gna2ItemTypeOperationNumberOfOperands ,  "Gna2ItemTypeOperationNumberOfOperands"},
        { Gna2ItemTypeOperationParameters ,        "Gna2ItemTypeOperationParameters"},
        { Gna2ItemTypeOperationNumberOfParameters, "Gna2ItemTypeOperationNumberOfParameters"},
        { Gna2ItemTypeOperandMode ,                "Gna2ItemTypeOperandMode"},
        { Gna2ItemTypeOperandLayout,               "Gna2ItemTypeOperandLayout"},
        { Gna2ItemTypeOperandType,                 "Gna2ItemTypeOperandType"},
        { Gna2ItemTypeOperandData ,                "Gna2ItemTypeOperandData"},
        { Gna2ItemTypeParameter,                   "Gna2ItemTypeParameter"},
        { Gna2ItemTypeShapeNumberOfDimensions,     "Gna2ItemTypeShapeNumberOfDimensions"},
        { Gna2ItemTypeShapeDimensions,             "Gna2ItemTypeShapeDimensions"},
        { Gna2ItemTypeInternal,                    "Gna2ItemTypeInternal"},
        { Gna2ItemTypeOperationHardwareDescriptor, "Gna2ItemTypeOperationHardwareDescriptor"},
    };
    return itemTypeStrings;
}

// keep following define up to date if GetErrorString() changes
#define MAX_MODEL_ERROR_MESSAGE_LENGTH 256
std::string ModelErrorHelper::GetErrorString(const Gna2ModelError& error)
{
    std::string message = "Value:" + std::to_string(error.Value);
    const auto errorType = GNA::StringHelper::GetFromMap(GetAllErrorTypeStrings(), error.Reason);
    message += ";ErrorType:" + errorType;
    const auto itemType = GNA::StringHelper::GetFromMap(GetAllItemTypeStrings(), error.Source.Type);
    message += ";ItemType:" + itemType;
    message += ";Gna2Model:model";
    AppendNotDisabled(message, error.Source.OperationIndex, "Operations");
    AppendNotDisabled(message, error.Source.OperandIndex, "Operands");
    AppendNotDisabled(message, error.Source.ParameterIndex, "Parameters");
    AppendNotDisabled(message, error.Source.ShapeDimensionIndex, "Shape.Dimensions");
    return message;
}

uint32_t ModelErrorHelper::GetErrorStringMaxLength()
{
    return MAX_MODEL_ERROR_MESSAGE_LENGTH;
}

void ModelErrorHelper::AppendNotDisabled(std::string& toAppend, int32_t index, const std::string& arrayName)
{
    if (index != GNA2_DISABLED)
    {
        toAppend += "." + arrayName + "[" + std::to_string(index) + "]";
    }
}

ModelValue::ModelValue(int64_t valueIn)
    : Value{valueIn}
{
}

int64_t ModelValue::GetValue() const
{
    return Value;
}

Gna2ModelItem ModelValue::GetSource() const
{
    return Source;
}
