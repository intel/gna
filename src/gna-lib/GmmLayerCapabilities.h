/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

    struct GmmLayerCapabilities : LayerCapabilities
    {
        static const FullCapabilitiesMap& GetOperands(uint32_t operandIndex);
    };
}
