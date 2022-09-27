/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "gna2-memory-api.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "DeviceManager.h"

#include "gna2-common-api.h"

#include <cstdint>
#include <functional>

using namespace GNA;

GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t * sizeGranted,
    void ** memoryAddress)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& deviceManager = DeviceManager::Get();
        deviceManager.AllocateMemory(sizeRequested, sizeGranted, memoryAddress);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().FreeMemory(memory);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2MemorySetTag(
    void * memory,
    uint32_t tag)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().TagMemory(memory, tag);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}