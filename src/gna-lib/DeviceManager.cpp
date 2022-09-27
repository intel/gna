/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define NOMINMAX 1

#include "DeviceManager.h"

#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"
#include "HybridDevice.h"

#include "gna2-common-api.h"

#include <memory>

using namespace GNA;

auto const & staticDestructionProtectionHelper1 = GNA::StatusHelper::GetStringMap();

constexpr uint32_t DeviceManager::DefaultThreadCount;

DeviceManager::DeviceManager()
{
    for (uint8_t i = 0; i < DriverInterface::MAX_GNA_DEVICES; i++)
    {
        auto const deviceVersion = DriverInterface::Query(i);
        if (deviceVersion != Gna2DeviceVersionSoftwareEmulation || i == 0)
        {
            capabilities.emplace(i, deviceVersion);
        }
    }
}

Device& DeviceManager::GetDevice(uint32_t deviceIndex)
{
    auto& device = *GetDeviceContext(deviceIndex);
    return device;
}

ExportDevice& DeviceManager::GetDeviceForExport(uint32_t deviceIndex)
{
    if (deviceIndex < GetDeviceCount())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
    return static_cast<ExportDevice&>(*GetDeviceContext(deviceIndex));
}

void DeviceManager::CreateExportDevice(uint32_t * deviceIndex, Gna2DeviceVersion targetDeviceVersion)
{
    Expect::NotNull(deviceIndex);
    Expect::True(targetDeviceVersion != Gna2DeviceVersionSoftwareEmulation, Gna2StatusDeviceVersionInvalid);

    auto const index = GetDeviceCount() + exportDevicesCount;
    Expect::InRange(index, DeviceCreateExportMaxInstances, Gna2StatusIdentifierInvalid);

    auto device = std::make_unique<ExportDevice>(targetDeviceVersion);
    *deviceIndex = index;
    auto const emplaced = devices.emplace(*deviceIndex,
        DeviceContext{ std::move(device), 0 });
    Expect::True(emplaced.second, Gna2StatusIdentifierInvalid);
    exportDevicesCount++;
    Log->Message("Export Device %u created, target version %d\n",
        deviceIndex, targetDeviceVersion);
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
    ReferenceCount{ referenceCount }
{
}

uint32_t DeviceManager::DeviceContext::operator++()
{
    if (MaximumReferenceCount == ReferenceCount)
    {
        throw GnaException(Gna2StatusDeviceNotAvailable);
    }
    ReferenceCount++;
    return ReferenceCount;
}

uint32_t DeviceManager::DeviceContext::operator--()
{
    if (ReferenceCount > 0)
    {
        --ReferenceCount;
        return ReferenceCount;
    }
    throw GnaException(Gna2StatusIdentifierInvalid);
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
    try // fetch not yet opened device version
    {
        return capabilities.at(deviceIndex);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
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

    if (!IsOpened(deviceIndex))
    {
        auto device = HybridDevice::Create(deviceIndex);
        auto const emplaced = devices.emplace(deviceIndex,
            DeviceContext{ std::move(device), 0, });
        MapAllToDevice(*emplaced.first->second);
    }

    const auto deviceRefCount = ++GetDeviceContext(deviceIndex);
    Log->Message("Device %u opened, active handles: %u\n",
        deviceIndex, deviceRefCount);
}

void DeviceManager::CloseDevice(uint32_t deviceIndex)
{
    if (deviceIndex >= GetDeviceCount())
    {
        Expect::GtZero(devices.erase(deviceIndex), Gna2StatusIdentifierInvalid);
        exportDevicesCount--;
    }
    else
    {
        const auto deviceRefCount = --GetDeviceContext(deviceIndex);

        Log->Message("Device %u closed, active handles: %u\n",
            deviceIndex, deviceRefCount);

        if (deviceRefCount == 0)
        {
            UnMapAllMemoryObjectsFromDevice(GetDevice(deviceIndex));
            devices.erase(deviceIndex);
        }
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
    for (const auto& device : devices)
    {
        if (device.second->HasModel(modelId))
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
    auto const memoryObject = CreateInternalMemory(requestedSize);

    MapMemoryToAll(*memoryObject);

    *memoryAddress = memoryObject->GetBuffer();
    *sizeGranted = static_cast<uint32_t>(memoryObject->GetSize());
}

std::pair<bool, std::vector<std::unique_ptr<Memory>>::iterator> DeviceManager::FindMemory(void * buffer)
{
    auto memoryIterator = std::find_if(memoryObjects.begin(), memoryObjects.end(),
        [buffer](const std::unique_ptr<Memory>& memory)
    {
        return memory->GetBuffer() == buffer;
    });

    return { memoryIterator != memoryObjects.end(), memoryIterator };
}

void DeviceManager::FreeMemory(void *buffer)
{
    Expect::NotNull(buffer);

    const auto found = FindMemory(buffer);

    if (!found.first)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }

    UnmapMemoryFromAllDevices(*(*found.second));
    memoryObjects.erase(found.second);
}

void DeviceManager::MapMemoryToAll(Memory& memoryObject)
{
    for (auto& device : devices)
    {
        device.second->MapMemory(memoryObject);
    }
}

void DeviceManager::UnmapMemoryFromAllDevices(Memory& memoryObject)
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

void DeviceManager::TagMemory(void* memory, uint32_t tag)
{
    const auto found = FindMemory(memory);
    Expect::True(found.first, Gna2StatusMemoryBufferInvalid);
    found.second->get()->SetTag(tag);
}

void DeviceManager::AssignProfilerConfigToRequestConfig(uint32_t instrumentationConfigId,
    uint32_t requestConfigId)
{
    auto& profilerConfig = ProfilerConfigManager.GetConfiguration(instrumentationConfigId);
    auto& deviceToAssign = GetDeviceForRequestConfigId(requestConfigId);
    deviceToAssign.AssignProfilerConfigToRequestConfig(requestConfigId, profilerConfig);
}

void DeviceManager::UnMapAllMemoryObjectsFromDevice(Device& device)
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
