/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-common-impl.h"
#include "gna2-capability-api.h"
#include "gna2-model-impl.h"

#include <map>
#include <memory>

namespace GNA
{

/** Number of input groups constraint - max */
constexpr auto BatchSizeMax = uint32_t{ 8 };

class LayerValidator;
struct ComponentLimits;

using OperationCapabilityMap = std::map<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>;

class FullCapabilitiesMap : public std::map<const nn_operation, OperationCapabilityMap>
{
public:
    using std::map<const nn_operation, OperationCapabilityMap>::map;

    gna_tensor_order GetOrder(const LayerValidator& validator) const;

    ComponentLimits * GetLatestCaps(const LayerValidator& validator) const;
};

}
