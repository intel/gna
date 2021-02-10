/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "AffineLayers.h"

#include "KernelArguments.h"
#include "XnnKernel.h"

#include "common.h"

#include <map>
#include <cstdint>

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class RecurrentLayer : public AffineBaseLayer
{
public:
    RecurrentLayer(const nn_layer& layer, const BaseValidator& validatorIn);
    RecurrentLayer(const Gna2Operation& operation, const BaseValidator& validatorIn);
    virtual ~RecurrentLayer() = default;

    virtual DataConfig GetDataMode() const override;
};

}
