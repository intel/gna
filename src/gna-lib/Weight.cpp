/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Weight.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

using namespace GNA;

const FullCapabilitiesMap WeightTensor::capabilities =
{
    {INTEL_AFFINE, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE)
    }},
    {INTEL_AFFINE_DIAGONAL, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE_DIAGONAL)
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE_MULTIBIAS)
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL_2D)
    }},
    {INTEL_CONVOLUTIONAL_1D, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL_1D)
    }},
    {INTEL_GMM, {
        GmmLayerCapabilities::GetOperands(WeightOperandIndex).at(INTEL_GMM)
    }},
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(WeightOperandIndex).at(INTEL_RECURRENT)
    }},
};

WeightTensor::WeightTensor(const Shape& dimensions, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{validatorIn, capabilities} }
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, WeightOperandIndex);
}

WeightTensor::WeightTensor(const Gna2Tensor &apiTensor, const LayerValidator& validatorIn)
try :
    Tensor(apiTensor, capabilities.GetOrder(validatorIn), Validator{ validatorIn, capabilities })
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, WeightOperandIndex);
}
