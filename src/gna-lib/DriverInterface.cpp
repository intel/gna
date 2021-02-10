/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "DriverInterface.h"

using namespace GNA;

const DriverCapabilities& DriverInterface::GetCapabilities() const
{
    return driverCapabilities;
}

void DriverInterface::convertPerfResultUnit(DriverPerfResults & driverPerf,
    Gna2InstrumentationUnit targetUnit) const
{
    auto const frequency = driverCapabilities.perfCounterFrequency;

    switch (targetUnit)
    {
    case Gna2InstrumentationUnitMicroseconds:
        return convertPerfResultUnit(driverPerf, frequency, RequestProfiler::MICROSECOND_MULTIPLIER);
    case Gna2InstrumentationUnitMilliseconds:
        return convertPerfResultUnit(driverPerf, frequency, RequestProfiler::MILLISECOND_MULTIPLIER);
    default:
        // no conversion required
        break;
    }
}

void DriverInterface::convertPerfResultUnit(DriverPerfResults& driverPerf,
    uint64_t frequency, uint64_t multiplier)
{
    if (0 == frequency || 0 == multiplier)
    {
        throw GnaException(Gna2StatusNullArgumentNotAllowed);
    }
    auto const newProcessing = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.Preprocessing, driverPerf.Processing);
    driverPerf.Preprocessing = 0;

    auto const newRequestCompleted = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.Processing, driverPerf.DeviceRequestCompleted);
    driverPerf.Processing = newProcessing;

    auto const newRequestCompletion = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.DeviceRequestCompleted, driverPerf.Completion);
    driverPerf.DeviceRequestCompleted = newProcessing + newRequestCompleted;
    driverPerf.Completion = newProcessing + newRequestCompleted + newRequestCompletion;
}

