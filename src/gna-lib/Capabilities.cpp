/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Capabilities.h"

#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "gna2-common-api.h"

#include <iterator>
#include <stdexcept>
#include <utility>

using namespace GNA;

gna_tensor_order FullCapabilitiesMap::GetOrder(const LayerValidator& validator) const
{
    return GetLatestCaps(validator)->Order.Value;
}

ComponentLimits * FullCapabilitiesMap::GetLatestCaps(const LayerValidator& validator) const
{
    try
    {
        auto& caps = at(validator.Operation);
        for (auto latestHW = caps.rbegin(); latestHW != caps.rend(); ++latestHW)
        {
            if (latestHW->first <= validator.HwCapabilities.GetDeviceGeneration())
            {
                return latestHW->second.get();
            }
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    // operation or device not supported at all
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}

