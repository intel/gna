/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Capabilities.h"
#include "DataMode.h"

#include <vector>

namespace GNA
{

using ComponentFullCapabilityMap = std::map<const uint32_t, FullCapabilitiesMap>;

struct LayerCapabilities
{
    /** Number of input groups constraint - max */
    static constexpr uint32_t BatchSizeMax = 8;

    /** Number of input groups constraint - max */
    static constexpr uint32_t InputGroupsCountMax = 8;

    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementsMultipllier = 8;

    /** Number of input groups constraint for Copy layer 3.0- max */
    static constexpr uint32_t CopyRowsMax = 255;

    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementCountMultiplier = 8;

    /** Total number of output elements constraint - must be multiple of */
    static constexpr uint32_t RecurrentOutputElementCountMultiplier = 32;

    /** Total number of input elements constraint - max elements */
    static constexpr uint32_t InputElementCountMax = UINT16_MAX;

    /** Number of pwl segments constraint - max  */
    static constexpr uint32_t ActivationFunctionSegmentCountMax = 128;

    /** Number of pwl segments constraint - min  */
    static constexpr uint32_t ActivationFunctionSegmentCountMin = 2;

    /** Weight elements size constraint - max size B */
    static constexpr uint32_t WeightElementSizeMax = 2;

    static const MultiplierMap & InputElementCountMultipliers();

    static const DataModeLimits & GetModes(uint32_t operandIndex, gna_device_generation generation);

    static const RangeLimits<>& limitsForInput();

    static const RangeLimits<>& limitsForOutput();

    static const RangeLimits<>& limitsForInputShapeLegacy();

    static const RangeLimits<>& limitsForOutputShapeLegacy();

    static const RangeLimits<>& limitsForInputGroupsMax();

    static const RangeLimits<>& limitsForOutputGroupsMax();
};

}
