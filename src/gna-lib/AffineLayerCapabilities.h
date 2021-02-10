/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

    struct AffineLayerCapabilities : LayerCapabilities
    {
        static const FullCapabilitiesMap& GetOperands(uint32_t operandIndex);
        static const std::shared_ptr<ComponentLimits>& GetInputComponentLimits(const gna_device_generation generation);
        static const std::shared_ptr<ComponentLimits>& GetOutputComponentLimits(const gna_device_generation generation);

    private:
        /**
         MultiBias Affine Output Limits for GNA 2.0
         */
        static const std::shared_ptr<ComponentLimits>& GetMBOutputComponentLimits(const gna_device_generation generation);

    };

}