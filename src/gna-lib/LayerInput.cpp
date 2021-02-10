/**
 @copyright (C) 2018-2021 Intel Corporation
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

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>


using namespace GNA;

const FullCapabilitiesMap LayerInput::capabilities =
{
    {INTEL_AFFINE, {
        AffineLayerCapabilities::GetOperands(InputOperandIndex).at(INTEL_AFFINE)
    }},
    {INTEL_AFFINE_DIAGONAL, {
        AffineLayerCapabilities::GetOperands(InputOperandIndex).at(INTEL_AFFINE_DIAGONAL)
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        AffineLayerCapabilities::GetOperands(InputOperandIndex).at(INTEL_AFFINE_MULTIBIAS)
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetOperands(InputOperandIndex).at(INTEL_CONVOLUTIONAL)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(InputOperandIndex).at(INTEL_CONVOLUTIONAL_2D)
    }},
    {INTEL_CONVOLUTIONAL_1D, {
        ConvolutionalLayer2DCapabilities::GetOperands(InputOperandIndex).at(INTEL_CONVOLUTIONAL_1D)
    }},
    {INTEL_COPY, {
        AuxiliaryCapabilities::GetOperands(InputOperandIndex).at(INTEL_COPY)
    }},
    {INTEL_INTERLEAVE, {
        AuxiliaryCapabilities::GetOperands(InputOperandIndex).at(INTEL_INTERLEAVE)
    }},
    {INTEL_DEINTERLEAVE, {
        AuxiliaryCapabilities::GetOperands(InputOperandIndex).at(INTEL_DEINTERLEAVE)
    }},
    {INTEL_GMM, {
        GmmLayerCapabilities::GetOperands(InputOperandIndex).at(INTEL_GMM)
    }},
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(InputOperandIndex).at(INTEL_RECURRENT)
    }}
};

LayerInput::LayerInput(const nn_layer &layer, const LayerValidator& validatorIn) :
    Tensor{ GetDimensions(layer, capabilities.GetOrder(validatorIn)),
        layer.nBytesPerInput, layer.pInputs,
        Validator{ validatorIn, capabilities } },
    Grouping{ getGrouping(layer) },
    ElementCount{ getElementCount(layer) }
{
}

ApiShape GetShape(const Gna2Operation & operation)
{
    ApiShape shape{ operation.Operands[InputOperandIndex]->Shape };
    if (Gna2OperationTypeConvolution == operation.Type &&
        shape.NumberOfDimensions < 4 &&
        !CnnLayer::IsForced(operation))
    {
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
       Validator{ validatorIn, capabilities } },
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, InputOperandIndex);
}

bool LayerInput::IsInputInterleave(const Gna2Tensor &apiTensor,
    const BaseValidator& validatorIn)
{
    auto layerValidator = LayerValidator{ validatorIn, INTEL_INTERLEAVE };
    try
    {
        Tensor{
           apiTensor, capabilities.GetOrder(layerValidator),
           Validator{ layerValidator, capabilities } };
        return true;
    }
    catch (const GnaException&)
    {
        return false;
    }
}

Shape LayerInput::GetDimensions(const nn_layer& layer, gna_tensor_order order)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_COPY:
    case INTEL_DEINTERLEAVE:
    case INTEL_INTERLEAVE:
    case INTEL_RECURRENT:
    case INTEL_CONVOLUTIONAL:
        return { order, layer.nInputRows, layer.nInputColumns };
    case INTEL_GMM:
        return { order, layer.nInputColumns, layer.nInputRows };
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto const config = static_cast<nn_layer_cnn2d*>(layer.pLayerStruct);
        return { order,
            layer.nInputRows,
            config->inputDimensions.height,
            config->inputDimensions.width,
            config->inputDimensions.depth }; // GNA_TENSOR_NHWD
    }
    default:
        return {};
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

std::pair<uint32_t, uint32_t> LayerInput::getGroupingAndElements(const nn_layer& layer) const
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_DEINTERLEAVE:
        return { layer.nInputColumns, layer.nInputRows };
    case INTEL_GMM:
    case INTEL_COPY:
    case INTEL_RECURRENT:
    case INTEL_INTERLEAVE:
    case INTEL_CONVOLUTIONAL:
    case INTEL_CONVOLUTIONAL_2D:
        return { layer.nInputRows, layer.nInputColumns };
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}
