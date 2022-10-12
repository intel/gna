/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "gna2-common-impl.h"
#include "gna2-instrumentation-impl.h"

#include "Logger.h"
#include "Expect.h"
#include "DeviceManager.h"
#include "ApiWrapper.h"

using namespace GNA;

Gna2Status Gna2InstrumentationConfigSetMode(uint32_t instrumentationConfigId,
    Gna2InstrumentationMode instrumentationMode)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = DeviceManager::Get().ProfilerConfigManager.GetConfiguration(instrumentationConfigId);
        config.SetHwPerfEncoding(instrumentationMode);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigSetUnit(
    uint32_t instrumentationConfigId,
    Gna2InstrumentationUnit instrumentationUnit)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = DeviceManager::Get().ProfilerConfigManager.GetConfiguration(instrumentationConfigId);
        config.SetUnit(instrumentationUnit);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t* results,
    uint32_t* instrumentationConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(instrumentationConfigId);
        Expect::NotNull(selectedInstrumentationPoints);
        Expect::NotNull(results);
        Expect::GtZero(numberOfInstrumentationPoints, Gna2StatusIdentifierInvalid);
        *instrumentationConfigId = DeviceManager::Get().ProfilerConfigManager.CreateConfiguration(
            { selectedInstrumentationPoints, selectedInstrumentationPoints + numberOfInstrumentationPoints },
            results);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().AssignProfilerConfigToRequestConfig(instrumentationConfigId, requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigRelease(uint32_t instrumentationConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().ProfilerConfigManager.ReleaseConfiguration(instrumentationConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}