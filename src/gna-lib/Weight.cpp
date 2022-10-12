/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Weight.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "Validator.h"


using namespace GNA;

const FullCapabilitiesMap WeightTensor::capabilities = LayerCapabilities::MakeFullCaps<WeightOperandIndex>();

WeightTensor::WeightTensor(const Shape& dimensions, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{validatorIn, capabilities}, WeightOperandIndex }
{
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(WeightOperandIndex);
}

WeightTensor::WeightTensor(const Gna2Tensor &apiTensor, const LayerValidator& validatorIn)
try :
    Tensor(apiTensor, capabilities.GetOrder(validatorIn), Validator{ validatorIn, capabilities }, WeightOperandIndex)
{
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(WeightOperandIndex);
}
