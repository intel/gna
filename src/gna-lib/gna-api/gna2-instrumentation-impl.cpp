/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gna2-common-impl.h"
#include "gna2-instrumentation-impl.h"

#include "Logger.h"
#include "Expect.h"
#include "DeviceManager.h"
#include "ApiWrapper.h"

using namespace GNA;

Gna2Status Gna2InstrumentationConfigSetMode(uint32_t configId,
    Gna2InstrumentationMode hwPerfEncoding)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.SetHardwareInstrumentation(configId, hwPerfEncoding);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigSetUnit(
    uint32_t configId,
    Gna2InstrumentationUnit instrumentationUnit)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.SetInstrumentationUnit(configId, instrumentationUnit);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t* results,
    uint32_t* configId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(configId);
        Expect::NotNull(selectedInstrumentationPoints);
        Expect::NotNull(results);
        Expect::GtZero(numberOfInstrumentationPoints, Gna2StatusIdentifierInvalid);
        auto& device = DeviceManager::Get().GetDevice(0);
        *configId = device.CreateProfilerConfiguration(
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AssignProfilerConfigToRequestConfig(instrumentationConfigId, requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigRelease(uint32_t instrumentationConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseProfilerConfiguration(instrumentationConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}