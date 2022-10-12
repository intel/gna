/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "LayerOutput.h"

#include "AffineLayerCapabilities.h"
#include "AffineLayers.h"
#include "AuxiliaryCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Validator.h"


#include <algorithm>
#include <memory>

using namespace GNA;
using CnnCaps = GNA::ConvolutionalLayer2DCapabilities;

static const DataModeLimits _ModesGen0_9 =
{
    {Gna2DataTypeInt16, Gna2DataTypeInt32},
    Gna2StatusXnnErrorOutputBytes
};

static const DataModeLimits _ModesGen3 =
{
    {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
    _ModesGen0_9.Error
};

const FullCapabilitiesMap LayerOutput::capabilities = LayerCapabilities::MakeFullCaps<OutputOperandIndex>();

ApiShape LayerOutput::GetShape(const Gna2Operation & operation)
{
    ApiShape s{ operation.Operands[OutputOperandIndex]->Shape };
    if (operation.Type != Gna2OperationTypeConvolution ||
        s.NumberOfDimensions >= 4)
    {
        return s;
    }
    if (!CnnLayer::IsForced(operation))
    {
        ModelErrorHelper::ExpectEqual(s.NumberOfDimensions, 3, Gna2ItemTypeShapeNumberOfDimensions);
        s.NumberOfDimensions = 4;
        s.Dimensions[3] = s.Dimensions[2];
        s.Dimensions[2] = s.Dimensions[1];
        s.Dimensions[1] = 1;

    }
    return s;
}

LayerOutput::LayerOutput(const Gna2Operation &operation, const LayerValidator& validatorIn)
try :
    Tensor{ Shape::Create(GetShape(operation), capabilities.GetOrder(validatorIn)),
        GetDataMode(*operation.Operands[OutputOperandIndex]), operation.Operands[OutputOperandIndex]->Data,
        Validator{ validatorIn, capabilities, true }, OutputOperandIndex },
    ScratchPad{ Dimensions, {Gna2DataTypeInt32, Gna2TensorModeDefault}, getScratchpadForOperation(validatorIn.Operation), ScratchpadOperandIndex },
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(OutputOperandIndex);
}

void* LayerOutput::getScratchpadForOperation(const nn_operation& operation)
{
    if (operation == INTEL_DEINTERLEAVE ||
        operation == INTEL_INTERLEAVE ||
        operation == INTEL_COPY ||
        operation == INTEL_CONVOLUTIONAL_1D ||
        operation == INTEL_CONVOLUTIONAL_2D ||
        operation == INTEL_GMM)
    {
        return nullptr;
    }

    return AffineBaseLayer::GetGlobal2MBScratchpad();
}

std::pair<uint32_t, uint32_t> LayerOutput::getGroupingAndElements(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    switch (operation.Type)
    {
    case Gna2OperationTypeTransposition:
    {
        if (validatorIn.Operation == INTEL_INTERLEAVE)
        {
            return {Dimensions.at('W'), Dimensions.at('H')};
        }
        if (validatorIn.Operation == INTEL_DEINTERLEAVE)
        {
            return {Dimensions.at('H'), Dimensions.at('W')};
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    case Gna2OperationTypeGmm:
        return {Dimensions.at('W'), Dimensions.at('H')};
    default:
        return Tensor::getGroupingAndElements(operation, validatorIn);
    }
}
