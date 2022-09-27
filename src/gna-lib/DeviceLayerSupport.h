/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "DataMode.h"
#include "Expect.h"

#include "gna2-capability-api.h"

#include <map>

namespace GNA
{

using Support = std::array<Gna2DeviceGeneration, 7>;

struct DataConfig
{
    DataMode Input;
    DataMode Weight;
    DataMode Bias;

    DataMode Output;
    bool IsActivationDisabled = false;

    constexpr bool operator<(const DataConfig &mode) const
    {
        if (mode.Input != Input)
        {
            return mode.Input < Input;
        }

        if (mode.Weight != Weight)
        {
            return mode.Weight < Weight;
        }

        if (mode.Bias != Bias)
        {
            return mode.Bias < Bias;
        }

        if (mode.Output != Output)
        {
            return mode.Output < Output;
        }

        if (mode.IsActivationDisabled != IsActivationDisabled)
        {
            return mode.IsActivationDisabled < IsActivationDisabled;
        }

        return false;
    }

    static bool IsOperationSupported(nn_operation operation, DataConfig config, Gna2DeviceGeneration generation);

protected:
    static const std::map<const DataConfig, std::map<const nn_operation, const Support>>& Capabilities();
};

}
