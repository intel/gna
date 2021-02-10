/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "LayerCapabilities.h"

using namespace GNA;

const MultiplierMap& LayerCapabilities::InputElementCountMultipliers()
{
    static auto const multipliers = MultiplierMap{
        {Gna2DataTypeInt8, 2 * InputElementCountMultiplier},
        {Gna2DataTypeInt16, 1 * InputElementCountMultiplier},
        {Gna2DataTypeInt32, InputElementCountMultiplier / 2},
    };
    return multipliers;
}

const DataModeLimits& LayerCapabilities::GetModes(uint32_t operandIndex, gna_device_generation generation)
{
    static const std::map<uint32_t, std::map<gna_device_generation, DataModeLimits>> modes =
    {
        {InputOperandIndex,
            {{GNA_0_9, {{GNA_INT16}, Gna2StatusXnnErrorInputBytes}},
            {GNA_3_0, {{GNA_INT8, GNA_INT16}, Gna2StatusXnnErrorInputBytes}},}
        },
        {OutputOperandIndex,
            {{GNA_0_9, {{GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}},
            {GNA_3_0, {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}},}
        },
    };
    return modes.at(operandIndex).at(generation);
}

const RangeLimits<>& LayerCapabilities::limitsForInput()
{
    static const RangeLimits<> _limitsForInput =
    {
        1,
        InputElementCountMax,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInput;
}

const RangeLimits<>& LayerCapabilities::limitsForOutput()
{
    static const RangeLimits<> _limitsForOutput =
    {
        limitsForInput(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutput;
}

const RangeLimits<>& LayerCapabilities::limitsForInputShapeLegacy()
{
    static const RangeLimits<> _limitsForInputShapeLegacy =
    {
        InputElementCountMultiplier,
        InputElementCountMax,
        InputElementCountMultipliers(),
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputShapeLegacy;
}

const RangeLimits<>& LayerCapabilities::limitsForOutputShapeLegacy()
{
    static const RangeLimits<> _limitsForOutputShapeLegacy =
    {
        limitsForInputShapeLegacy(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputShapeLegacy;
}

const RangeLimits<>& LayerCapabilities::limitsForInputGroupsMax()
{
    static const RangeLimits<> _limitsForInputGroupsMax =
    {
        1,
        InputGroupsCountMax,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputGroupsMax;
}

const RangeLimits<>& LayerCapabilities::limitsForOutputGroupsMax()
{
    static const RangeLimits<> _limitsForInputGroupsMax =
    {
        1,
        InputGroupsCountMax,
        1,
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForInputGroupsMax;
}
