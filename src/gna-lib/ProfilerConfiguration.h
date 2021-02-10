/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once
#include "gna2-instrumentation-api.h"

#include <set>
#include <vector>

namespace GNA
{
class ProfilerConfiguration
{
public:

    static uint32_t GetMaxNumberOfInstrumentationPoints();
    static const std::set<Gna2InstrumentationPoint>& GetSupportedInstrumentationPoints();
    static const std::set<Gna2InstrumentationUnit>& GetSupportedInstrumentationUnits();
    static const std::set<Gna2InstrumentationMode>& GetSupportedInstrumentationModes();
    static void ExpectValid(Gna2InstrumentationMode encodingIn);

    ProfilerConfiguration(uint32_t configID,
        std::vector<Gna2InstrumentationPoint>&& selectedPoints,
        uint64_t* results);

    const uint32_t ID;
    const std::vector<Gna2InstrumentationPoint> Points;

    void SetUnit(Gna2InstrumentationUnit unitIn);
    Gna2InstrumentationUnit GetUnit() const;

    void SetHwPerfEncoding(Gna2InstrumentationMode encodingIn);
    uint8_t GetHwPerfEncoding() const;

    void SetResult(uint32_t index, uint64_t value) const;

private:
    uint64_t* Results = nullptr;

    Gna2InstrumentationMode HwPerfEncoding = Gna2InstrumentationModeTotalStall;
    Gna2InstrumentationUnit Unit = Gna2InstrumentationUnitMicroseconds;
};

}
