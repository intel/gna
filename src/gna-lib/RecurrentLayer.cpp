/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "RecurrentLayer.h"

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "Bias.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "Layer.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Tensor.h"
#include "Weight.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

RecurrentLayer::RecurrentLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, { RecurrentTransform }, validatorIn)
{
}

RecurrentLayer::RecurrentLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, { RecurrentTransform }, validatorIn)
{
}

DataConfig RecurrentLayer::GetDataMode() const
{
    auto affineTransform = Transforms.Get<AffineFunction>(RecurrentTransform);
    return AffineBaseLayer::getDataMode(affineTransform);
}
