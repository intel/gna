/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "DeviceManager.h"

#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"
#if defined(_WIN32)
#include "WindowsDriverInterface.h"
#else // linux
#include "LinuxDriverInterface.h"
#endif

#include "common.h"
#include "gna2-common-api.h"

#include <memory>

using namespace GNA;

constexpr uint32_t DeviceManager::DefaultThreadCount;

DeviceManager::DeviceManager()
{
    for (uint8_t i = 0; i < DriverInterface::MAX_GNA_DEVICES; i++)
    {
        std::unique_ptr<DriverInterface> driverInterface =
        {
    #if defined(_WIN32)
            std::make_unique<WindowsDriverInterface>()
    #else // GNU/Linux / Android / ChromeOS
            std::make_unique<LinuxDriverInterface>()
    #endif
        };
        const auto success = driverInterface->OpenDevice(i);
        if (success ||
            i == 0)
        {
            auto caps = HardwareCapabilities{};
            caps.DiscoverHardware(driverInterface->GetCapabilities());
            if (caps.IsHardwareSupported() ||
                i == 0)
            {
                capabilities.emplace(i, caps);
            }
        }
    }
}

Device& DeviceManager::GetDevice(uint32_t deviceIndex)
{
    auto& device = *GetDeviceContext(deviceIndex);
    return device;
}

void DeviceManager::CreateDevice(uint32_t deviceIndex)
{
    if (!IsOpened(deviceIndex))
    {
        auto const emplaced = devices.emplace(deviceIndex, DeviceContext{
           std::make_unique<Device>(deviceIndex, DeviceManager::DefaultThreadCount), 0});
        MapAllToDevice(*emplaced.first->second);
    }
}

bool DeviceManager::IsOpened(uint32_t deviceIndex)
{
    return devices.end() != devices.find(deviceIndex);
}

DeviceManager::DeviceContext& DeviceManager::GetDeviceContext(uint32_t deviceIndex)
{
    try
    {
        auto & deviceContext = devices.at(deviceIndex);
        return deviceContext;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

DeviceManager::DeviceContext::DeviceContext(std::unique_ptr<Device> handle, uint32_t referenceCount) :
    std::unique_ptr<Device>{ std::move(handle) },
    ReferenceCount{referenceCount}
{
}

uint32_t DeviceManager::GetDeviceCount() const
{
    return static_cast<uint32_t>(capabilities.size());
}

DeviceVersion DeviceManager::GetDeviceVersion(uint32_t deviceIndex)
{
    if (IsOpened(deviceIndex)) // fetch opened device version
    {
        const auto& device = GetDevice(deviceIndex);
        return device.GetVersion();
    }
    else
    {
        try // fetch not yet opened device version
        {
            return capabilities.at(deviceIndex).GetHardwareDeviceVersion();
        }
        catch (std::out_of_range&)
        {
            throw GnaException(Gna2StatusIdentifierInvalid);
        }
    }
}

void DeviceManager::SetThreadCount(uint32_t deviceIndex, uint32_t threadCount)
{
    auto& device = GetDevice(deviceIndex);
    device.SetNumberOfThreads(threadCount);
}

uint32_t DeviceManager::GetThreadCount(uint32_t deviceIndex)
{
    const auto& device = GetDevice(deviceIndex);
    return device.GetNumberOfThreads();
}

void DeviceManager::OpenDevice(uint32_t deviceIndex)
{
    Expect::InRange(deviceIndex, GetDeviceCount() - 1, Gna2StatusIdentifierInvalid);

    CreateDevice(deviceIndex);

    auto & deviceRefCount = GetDeviceContext(deviceIndex).ReferenceCount;
    if (MaximumReferenceCount == deviceRefCount)
    {
        throw GnaException(Gna2StatusDeviceNotAvailable);
    }
    deviceRefCount++;
    Log->Message("Device %u opened, active handles: %u\n",
        deviceIndex, deviceRefCount);
}

void DeviceManager::CloseDevice(uint32_t deviceIndex)
{
    auto & deviceRefCount = GetDeviceContext(deviceIndex).ReferenceCount;
    if (deviceRefCount > 0)

    {
        --deviceRefCount;
        Log->Message("Device %u closed, active handles: %u\n",
            deviceIndex, deviceRefCount);

        if (deviceRefCount == 0)
        {
            UnMapAllFromDevice(GetDevice(deviceIndex));
            devices.erase(deviceIndex);
        }
    }
    else
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

Device & DeviceManager::GetDeviceForModel(uint32_t modelId)
{
    const auto device = TryGetDeviceForModel(modelId);
    Expect::NotNull(device, Gna2StatusIdentifierInvalid);
    return *device;
}

Device* DeviceManager::TryGetDeviceForModel(uint32_t modelId)
{
    for(const auto& device : devices)
    {
       if(device.second->HasModel(modelId))
       {
           return device.second.get();
       }
    }
    return nullptr;
}

void DeviceManager::AllocateMemory(uint32_t requestedSize,
    uint32_t *sizeGranted, void **memoryAddress)
{
    Expect::NotNull(sizeGranted);
    Expect::NotNull(memoryAddress);

    *sizeGranted = 0;
    auto memoryObject = createMemoryObject(requestedSize);

    MapMemoryToAll(*memoryObject);

    *memoryAddress = memoryObject->GetBuffer();
    *sizeGranted = (uint32_t)memoryObject->GetSize();
    memoryObjects.emplace_back(std::move(memoryObject));
}

std::pair<bool, std::vector<std::unique_ptr<Memory>>::const_iterator> DeviceManager::HasMemory(void * buffer) const
{
    auto memoryIterator = std::find_if(memoryObjects.cbegin(), memoryObjects.cend(),
        [buffer](const std::unique_ptr<Memory>& memory)
    {
        return memory->GetBuffer() == buffer;
    });

    return { memoryIterator != memoryObjects.end(), memoryIterator };
}

void DeviceManager::FreeMemory(void *buffer)
{
    Expect::NotNull(buffer);

    auto found = HasMemory(buffer);

    if (!found.first)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }

    memoryObjects.erase(found.second);
}

void DeviceManager::MapMemoryToAll(Memory& memoryObject)
{
    for (auto& device : devices)
    {
        device.second->MapMemory(memoryObject);
    }
}

void DeviceManager::UnMapMemoryFromAll(Memory& memoryObject)
{
    for (const auto& device : devices)
    {
        device.second->UnMapMemory(memoryObject);
    }
}

Device& DeviceManager::GetDeviceForRequestConfigId(uint32_t requestConfigId)
{
    const auto device = TryGetDeviceForRequestConfigId(requestConfigId);
    Expect::NotNull(device, Gna2StatusIdentifierInvalid);
    return *device;
}

Device* DeviceManager::TryGetDeviceForRequestConfigId(uint32_t requestConfigId)
{
    for (const auto& device : devices)
    {
        if (device.second->HasRequestConfigId(requestConfigId))
        {
            return device.second.get();
        }
    }
    return nullptr;
}

Device & DeviceManager::GetDeviceForRequestId(uint32_t requestId)
{
    for (const auto& device : devices)
    {
        if (device.second->HasRequestId(requestId))
        {
            return *device.second;
        }
    }
    throw GnaException(Gna2StatusIdentifierInvalid);
}

const std::vector<std::unique_ptr<Memory>> & DeviceManager::GetAllAllocated() const
{
    return memoryObjects;
}

void DeviceManager::UnMapAllFromDevice(Device& device)
{
    for (auto& m : memoryObjects)
    {
        device.UnMapMemory(*m);
    }
}

void DeviceManager::MapAllToDevice(Device& device)
{
    for (auto& m : memoryObjects)
    {
        device.MapMemory(*m);
    }
}

std::unique_ptr<Memory> DeviceManager::createMemoryObject(uint32_t requestedSize)
{
    return std::make_unique<Memory>(requestedSize);
}
