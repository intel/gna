/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "LayerInput.h"

#include "AffineLayerCapabilities.h"
#include "AuxiliaryCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "GmmLayerCapabilities.h"
#include "Macros.h"
#include "ModelError.h"
#include "ModelWrapper.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>


using namespace GNA;

const FullCapabilitiesMap LayerInput::capabilities = LayerCapabilities::MakeFullCaps<InputOperandIndex>();

ApiShape LayerInput::GetShape(const Gna2Operation & operation)
{
    ApiShape shape{ operation.Operands[InputOperandIndex]->Shape };
    if (Gna2OperationTypeConvolution == operation.Type &&
        shape.NumberOfDimensions < 4 &&
        !CnnLayer::IsForced(operation))
    {
        ModelErrorHelper::ExpectEqual(shape.NumberOfDimensions, 2, Gna2ItemTypeShapeNumberOfDimensions);
        shape.Dimensions[2] = shape.Dimensions[1];
        shape.Dimensions[1] = 1;
        shape.Dimensions[3] = 1;
        shape.NumberOfDimensions = 4;
    }
    return shape;
}

LayerInput::LayerInput(const Gna2Operation& operation, const LayerValidator& validatorIn)
try :
    Tensor{ Shape::Create(GetShape(operation), capabilities.GetOrder(validatorIn)),
       GetDataMode(*operation.Operands[InputOperandIndex]), operation.Operands[InputOperandIndex]->Data,
       Validator{ validatorIn, capabilities, true }, InputOperandIndex },
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(InputOperandIndex);
}

bool LayerInput::IsInputInterleave(const Gna2Tensor &apiTensor,
    const BaseValidator& validatorIn)
{
    auto const layerValidator = LayerValidator{ validatorIn, INTEL_INTERLEAVE };
    try
    {
        Tensor{
           apiTensor, capabilities.GetOrder(layerValidator),
           Validator{ layerValidator, capabilities, true } };
        return true;
    }
    catch (const GnaException&)
    {
        return false;
    }
}

std::pair<uint32_t, uint32_t> LayerInput::getGroupingAndElements(
    const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    switch (operation.Type)
    {
    case Gna2OperationTypeTransposition:
    {
        if (validatorIn.Operation == INTEL_INTERLEAVE)
        {
            return { Dimensions.at('H'), Dimensions.at('W') };
        }
        if (validatorIn.Operation == INTEL_DEINTERLEAVE)
        {
            return { Dimensions.at('W'), Dimensions.at('H') };
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Tensor::getGroupingAndElements(operation, validatorIn);
    }
}

