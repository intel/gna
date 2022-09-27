/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "AffineLayers.h"

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class RecurrentLayer : public AffineBaseLayer
{
public:
    RecurrentLayer(const Gna2Operation& operation, const LayerValidator& validatorIn);
    virtual ~RecurrentLayer() = default;
};

}
