/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "gna2-device-impl.h"

#include "ApiWrapper.h"
#include "DeviceManager.h"
#include "Expect.h"
#include "Logger.h"

#include "gna2-common-impl.h"

#include <functional>

using namespace GNA;

enum Gna2Status Gna2DeviceGetCount(
    uint32_t * deviceCount)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(deviceCount);
        *deviceCount = DeviceManager::Get().GetDeviceCount();
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceGetVersion(
    uint32_t deviceIndex,
    enum Gna2DeviceVersion * deviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(deviceVersion);
        *deviceVersion = DeviceManager::Get().GetDeviceVersion(deviceIndex);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& deviceManager = DeviceManager::Get();
        deviceManager.SetThreadCount(deviceIndex, numberOfThreads);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceOpen(
    uint32_t deviceIndex)
{
    try
    {
        DeviceManager::Get().OpenDevice(deviceIndex);
        return Gna2StatusSuccess;
    }
    catch (...)
    {
        return ApiWrapper::HandleAndLogExceptions();
    }
}

enum Gna2Status Gna2DeviceCreateForExport(
    Gna2DeviceVersion targetDeviceVersion,
    uint32_t * deviceIndex)
{
    try
    {
        DeviceManager::Get().CreateExportDevice(deviceIndex, targetDeviceVersion);
        return Gna2StatusSuccess;
    }
    catch (...)
    {
        return ApiWrapper::HandleAndLogExceptions();
    }
}

enum Gna2Status Gna2DeviceClose(
    uint32_t deviceIndex)
{
    try
    {
        DeviceManager::Get().CloseDevice(deviceIndex);
        return Gna2StatusSuccess;
    }
    catch (...)
    {
        return ApiWrapper::HandleAndLogExceptions();
    }
}
