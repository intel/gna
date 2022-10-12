/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "RecurrentLayer.h"

#include "DataMode.h"
#include "Layer.h"
#include "RecurrentFunction.h"

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

RecurrentLayer::RecurrentLayer(const Gna2Operation& operation, const LayerValidator& validatorIn) :
    AffineBaseLayer(operation, { RecurrentTransform }, validatorIn)
{
    auto const & affineTransform = Transforms.Get<RecurrentFunction>(RecurrentTransform);
    setDataMode(affineTransform, false);
}
