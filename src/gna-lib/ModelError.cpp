/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ModelError.h"

#include "ApiWrapper.h"
#include "Expect.h"
#include "StringHelper.h"

using namespace GNA;

ModelError ModelErrorHelper::lastError = {};

void ModelErrorHelper::ExpectTrue(bool val, Gna2ModelError error)
{
    if (!val)
    {
        throw GnaModelErrorException(error);
    }
}

Gna2ModelError ModelErrorHelper::GetPartiallySetError(const Gna2ItemType ptrType,
    const uint32_t ptrIndex,
    const bool indexForParameter)
{
    ModelError e;
    e.Source.Type = ptrType;
    if (indexForParameter)
    {
        e.Source.ParameterIndex = static_cast<int32_t>(ptrIndex);
    }
    else
    {
        e.Source.OperandIndex = static_cast<int32_t>(ptrIndex);
    }
    return e;
}

void ModelErrorHelper::ExpectNotNull(const void * const ptr,
    const Gna2ItemType ptrType,
    const uint32_t ptrIndex,
    const bool indexForParameter)
{
    auto e = GetPartiallySetError(ptrType, ptrIndex, indexForParameter);
    e.Reason = Gna2ErrorTypeNullNotAllowed;

    ExpectTrue(ptr != nullptr, e);
}

void ModelErrorHelper::ExpectNull(const void * const ptr,
    const Gna2ItemType ptrType,
    const uint32_t ptrIndex,
    const bool indexForParameter)
{
    auto e = GetPartiallySetError(ptrType, ptrIndex, indexForParameter);
    e.Reason = Gna2ErrorTypeNullRequired;
    ExpectTrue(ptr == nullptr, e);
}

void ModelErrorHelper::SaveLastError(const ModelError& modelError)
{
    lastError = modelError;
}

void ModelErrorHelper::PopLastError(Gna2ModelError& error)
{
    constexpr auto empty = ModelError{};
    Expect::False(lastError == empty, Gna2StatusModelErrorUnavailable);
    error = lastError;
    lastError = empty;
}

Gna2Status ModelErrorHelper::ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn)
{
    const std::function<ApiStatus()> command = [&]()
    {
        try
        {
            return commandIn();
        }
        catch (const GnaModelErrorException& exception)
        {
            SaveLastError(exception.GetModelError());
            throw GnaException(Gna2StatusModelConfigurationInvalid);
        }
    };
    return ApiWrapper::ExecuteSafely(command);
}

GnaModelErrorException::GnaModelErrorException(
    Gna2ItemType item,
    Gna2ErrorType errorType,
    int64_t value)
    : GnaException{ Gna2StatusModelConfigurationInvalid },
    error{ ModelError{ errorType, value, item } }
{
}

GnaModelErrorException::GnaModelErrorException(uint32_t layerIndex, Gna2Status status)
    : GnaException{ Gna2StatusModelConfigurationInvalid },
    error{ ModelError{ status } }
{
    SetLayerIndex(layerIndex);
}

void GnaModelErrorException::DispatchAndFill(uint32_t operandIndex, uint32_t parameterIndex)
{
    try
    {
        throw;
    }
    catch (GnaModelErrorException& e)
    {
        e.SetOperandIndex(operandIndex);
        e.SetParameterIndex(parameterIndex);
        throw;
    }
    catch (GnaException& e)
    {
        throw GnaModelErrorException(e, operandIndex, parameterIndex);
    }
}

void GnaModelErrorException::DispatchAndSetLayer(uint32_t layerIndex)
{
    try
    {
        throw;
    }
    catch (GnaModelErrorException& e)
    {
        e.SetLayerIndex(layerIndex);
        throw;
    }
    catch (const GnaException& e)
    {
        throw GnaModelErrorException(layerIndex, e.GetStatus());
    }
    catch (...)
    {
        throw GnaModelErrorException(layerIndex);
    }
}

GnaModelErrorException::GnaModelErrorException(GnaException& e, uint32_t operandIndex, uint32_t parameterIndex) :
    GnaException{ e },
    error{ ModelError(e.GetStatus()) }
{
    SetOperandIndex(operandIndex);
    SetParameterIndex(parameterIndex);
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
        {Gna2ErrorTypeNoHardwareCompliantOperation, "Gna2ErrorTypeNoHardwareCompliantOperation" },
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
    auto message = "Value:" + std::to_string(error.Value);
    const auto errorType = StringHelper::GetFromMap(GetAllErrorTypeStrings(), error.Reason);
    message += ";ErrorType:" + errorType;
    const auto itemType = StringHelper::GetFromMap(GetAllItemTypeStrings(), error.Source.Type);
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

