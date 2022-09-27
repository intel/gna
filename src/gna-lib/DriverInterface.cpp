/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "DriverInterface.h"

#include "Expect.h"
#include "HardwareCapabilities.h"
#include "LinuxDriverInterface.h"
#include "WindowsDriverInterface.h"

using namespace GNA;

DeviceVersion DriverInterface::Query(uint32_t deviceIndex)
{
    auto const driverInterface = Create(deviceIndex);
    auto const drvCaps = driverInterface->GetCapabilities();
    HardwareCapabilitiesDevice caps{drvCaps};
    return caps.GetHardwareDeviceVersion();
}

std::unique_ptr<DriverInterface> DriverInterface::Create(uint32_t deviceIndex)
{
    auto driverInterface =
#if defined(_WIN32)
        std::make_unique<WindowsDriverInterface>();
#else // GNU/Linux / Android / ChromeOS
        std::make_unique<LinuxDriverInterface>();
#endif
    Expect::NotNull(driverInterface);
    driverInterface->OpenDevice(deviceIndex);
    return driverInterface;
}

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

void DriverInterface::convertPerfResultUnit(HardwarePerfResults & hardwarePerf,
    const Gna2InstrumentationUnit targetUnit) const
{
    auto frequency = RequestProfiler::DEVICE_CLOCK_FREQUENCY;

    switch (targetUnit)
    {
    case Gna2InstrumentationUnitMicroseconds:
        return convertPerfResultUnit(hardwarePerf, frequency, RequestProfiler::MICROSECOND_MULTIPLIER);
    case Gna2InstrumentationUnitMilliseconds:
        return convertPerfResultUnit(hardwarePerf, frequency, RequestProfiler::MILLISECOND_MULTIPLIER);
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

void DriverInterface::convertPerfResultUnit(HardwarePerfResults& hardwarePerf,
    const uint64_t frequency, const uint64_t multiplier)
{
    if (0 == frequency || 0 == multiplier)
    {
        throw GnaException(Gna2StatusNullArgumentNotAllowed);
    }
    auto const newStall = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        0, hardwarePerf.stall);

    auto const newTotal = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        0, hardwarePerf.total);

    hardwarePerf.stall = newStall;
    hardwarePerf.total = newTotal;
}
