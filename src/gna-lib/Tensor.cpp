/**
 @copyright (C) 2018-2021 Intel Corporation
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

Tensor::Tensor(const ApiTensor & tensor) :
    Tensor{ Shape::Create(tensor.Shape, Layout{ tensor.Layout }),
        GetDataMode(tensor).Type, GetDataMode(tensor).Mode, tensor.Data }
{
}

DataMode Tensor::GetDataMode(const Gna2Tensor& tensor)
{
    ModelErrorHelper::ExpectInSet(tensor.Mode, { Gna2TensorModeDefault });
    try
    {
        return DataMode{ tensor.Type, tensor.Mode };
    }
    catch(...)
    {
        throw GnaModelErrorException(Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, tensor.Type);
    }
}

Tensor::Tensor(const ApiTensor & tensor, gna_tensor_order order, const Validator & validatorIn) :
    Tensor{ Shape::Create(tensor.Shape, order),
        GetDataMode(tensor),
        tensor.Data,
        validatorIn}
{
}

Tensor::Tensor(const Shape & dimensions, const DataType dataType, const TensorMode tensorMode, void const * buffer) :
    Component{ dimensions },
    Mode{ dataType, tensorMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{}

Tensor::Tensor(const Shape & dimensions, const DataMode & dataMode, void const * buffer,
    const Validator & validatorIn) :
    Component{ dimensions, validatorIn, false }, // disable dimension validation as it's performed here with Mode information
    Mode{ dataMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{
    validate();
}

Tensor::Tensor(const Tensor & tensor, const Validator & validatorIn) :
    Tensor{ tensor.Dimensions, tensor.Mode, tensor.Buffer, validatorIn }
{}

Tensor::Tensor(const ApiTensor& apiTensor, const Validator& validatorIn) :
    Tensor { Tensor{apiTensor}, validatorIn }
{}

void Tensor::UpdateBuffer(const BaseAddress & buffer)
{
    ValidateBuffer(buffer);
    Buffer = buffer;
}

void Tensor::ValidateBuffer(const void * const buffer) const
{
    auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
    validator->ValidateBuffer(buffer, Size, caps->Align.Value);
}

void Tensor::validate() const
{
    if (validator)
    {
        const auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
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
        if (GNA_DATA_DISABLED != Mode)
        {
            validateDimensions();
            validator->ValidateBufferIfSet(Buffer, Size, caps->Align);
        }
        else
        {
            Expect::Null(Buffer);
        }
    }
}

void Tensor::validateDimensions() const
{
    // update Multiplier when varies for data modes
    auto caps = *validator->Capabilities;
    for (auto & dim : caps.Dimensions)
    {
        dim.second.Multipliers.SetEffective(Mode.Type);
    }
    Component::Validate(caps, true);
}

uint32_t Tensor::getEffectiveSize(const DataMode& mode, uint32_t count)
{
    return Gna2TensorModeConstantScalar == mode.Mode ? mode.Size : count * mode.Size;
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

std::pair<uint32_t, uint32_t> Tensor::getGroupingAndElements(const nn_layer& layer) const
{
    UNREFERENCED_PARAMETER(layer);
    throw GnaException(Gna2StatusNotImplemented);
}
