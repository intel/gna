/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Tensor.h"

#include "Expect.h"
#include "Macros.h"
#include "ModelError.h"
#include "Validator.h"

#include <memory>
#include <utility>
#include <vector>

using namespace GNA;

Tensor::Tensor(const ApiTensor & tensor, uint32_t operandIndex) :
    Tensor{ Shape::Create(tensor.Shape, Layout{ tensor.Layout }),
        GetDataMode(tensor), tensor.Data, operandIndex }
{
}

DataMode Tensor::GetDataMode(const Gna2Tensor& tensor)
{
    return DataMode{ tensor.Type, tensor.Mode };
}

Tensor::Tensor(const ApiTensor & tensor, gna_tensor_order order, const Validator & validatorIn, uint32_t operandIndex)
try :
    Tensor{ Shape::Create(tensor.Shape, order),
        GetDataMode(tensor),
        tensor.Data,
        validatorIn,
        operandIndex}
{
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(operandIndex);
}

Tensor::Tensor(const Shape & dimensions, const DataMode & dataMode, void const * buffer, uint32_t operandIndex) :
    Component{ dimensions, operandIndex, false },
    Mode{ dataMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{}

Tensor::Tensor(const Shape & dimensions, const DataMode & dataMode, void const * buffer,
    const Validator & validatorIn, uint32_t operandIndex)
try :
    Component{ dimensions, validatorIn, false, operandIndex, false }, // disable dimension validation as it's performed here with Mode information
    Mode{ dataMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{
    validate();
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(operandIndex);
}

Tensor::Tensor(const Tensor & tensor, const Validator & validatorIn, uint32_t operandIndex) :
    Tensor{ tensor.Dimensions, tensor.Mode, tensor.Buffer, validatorIn, operandIndex }
{}

Tensor::Tensor(const ApiTensor& apiTensor, const Validator& validatorIn, uint32_t operandIndex) :
    Tensor { Tensor{apiTensor}, validatorIn, operandIndex }
{}

void Tensor::UpdateBuffer(const BaseAddress & buffer)
{
    ValidateBuffer(buffer);
    Buffer = buffer;
}

void Tensor::ValidateBuffer(const void * const buffer) const
{
    if (validator)
    {
        auto const caps = reinterpret_cast<const TensorLimits*>(validator->Capabilities);
        validator->ValidateBuffer(buffer, Size, caps->GetAddressAlign().Value);
    }
}

void Tensor::validate() const
{
    if (validator)
    {
        auto const caps = reinterpret_cast<const TensorLimits*>(validator->Capabilities);
        try
        {
            Expect::InSet(Mode, caps->Modes);
        }
        catch(GnaException&)
        {
            throw GnaModelErrorException(
                Gna2ItemTypeOperandType,
                Gna2ErrorTypeNotInSet,
                Mode.Type);
        }
        if (Gna2TensorModeDisabled != Mode.Mode)
        {
            ValidateDimensions(Mode.Type);
            ValidateBuffer(Buffer);
        }
        else
        {
            ModelErrorHelper::ExpectNull(Buffer);
        }
    }
}

uint32_t Tensor::getEffectiveSize(const DataMode& mode, uint32_t count)
{
    return (Gna2TensorModeConstantScalar == mode.Mode) ? mode.Size : count * mode.Size;
}

std::pair<uint32_t, uint32_t> Tensor::getGroupingAndElements(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    UNREFERENCED_PARAMETER(validatorIn);
    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
    case Gna2OperationTypeElementWiseAffine:
        return {Dimensions.at('W'), Dimensions.at('H')};
    case Gna2OperationTypeRecurrent:
    case Gna2OperationTypeCopy:
    case Gna2OperationTypeGmm:
        return {Dimensions.at('H'), Dimensions.at('W')};
    case Gna2OperationTypeConvolution:
        return {1, Count}; // not applicable for 2D CNN
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

const AlignLimits* TensorLimits::overridenAlign = nullptr;

void TensorLimits::OverrideAlign(const uint32_t newAlign)
{
    static AlignLimits overriden{ 1, Gna2StatusMemoryAlignmentInvalid };
    if (newAlign == 0)
    {
        overridenAlign = nullptr;
    }
    overriden.Value = newAlign;
    overridenAlign = &overriden;
}

const AlignLimits& TensorLimits::GetAddressAlign() const
{
    if(overridenAlign != nullptr)
    {
        return *overridenAlign;
    }
    return addressAlign;
}
