/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

struct AffineLayerCapabilities : LayerCapabilities
{
    static const FullCapabilitiesMap& GetOperands(uint32_t operandIndex);
};

}
