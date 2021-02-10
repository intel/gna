/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <map>
#include <memory>

namespace GNA
{

class LayerValidator;
struct ComponentLimits;

using OperationCapabilityMap = std::map<const gna_device_generation, std::shared_ptr<ComponentLimits>>;

class FullCapabilitiesMap : public std::map<const nn_operation, OperationCapabilityMap>
{
public:
    using std::map<const nn_operation, OperationCapabilityMap>::map;

    gna_tensor_order GetOrder(const LayerValidator& validator) const;

    ComponentLimits * GetLatestCaps(const LayerValidator& validator) const;
};

}
