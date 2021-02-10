/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Expect.h"

#include "gna2-common-api.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <map>
#include <utility>

namespace GNA
{

typedef std::map<gna_api_version, bool> ApiSupport;
typedef std::map<gna_device_generation, bool> HwSupport;

class Support
{
public:
    Support (HwSupport const && hw) :
        Hw{hw}
    {
        for (auto const apiSupport : Api)
        {
            Expect::True(apiSupport.second, Gna2StatusNullArgumentRequired);
        }
        for (auto const hwSupport : Hw)
        {
            Expect::True(hwSupport.second, Gna2StatusNullArgumentRequired);
        }
    }

    ~Support() = default;

    const ApiSupport Api = { {GNA_API_3_0, true} };
    const HwSupport Hw;
};

struct DataConfig
{
    DataConfig(gna_data_mode input, gna_data_mode weight, gna_data_mode bias, gna_data_mode output) :
        Input{input},
        Weight{weight},
        Bias{bias},
        Output{output}
    {
    }
    ~DataConfig() = default;

    bool operator<(const DataConfig &mode) const
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

        return false;
    }

    const gna_data_mode Input;

    union
    {
        const gna_data_mode Covariance;
        const gna_data_mode Weight;
    };

    union
    {
        const gna_data_mode Const;
        const gna_data_mode Bias;
    };

    const gna_data_mode Output;

    static const std::map<const DataConfig, std::map<const gna_layer_operation, const Support>> Capabilities;
};

}
