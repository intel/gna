/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#define NOMINMAX 1

#include "Device.h"

#include "ExportDevice.h"
#include "gna2-common-impl.h"
#include "ProfilerConfiguration.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class DeviceManager
{
public:
    static DeviceManager& Get()
    {
        static DeviceManager deviceManager;
        return deviceManager;
    }

    ~DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;

    Device& GetDevice(uint32_t deviceIndex);

    ExportDevice& GetDeviceForExport(uint32_t deviceIndex);

    uint32_t GetDeviceCount() const;

    DeviceVersion GetDeviceVersion(uint32_t deviceIndex);

    void SetThreadCount(uint32_t deviceIndex, uint32_t threadCount);

    uint32_t GetThreadCount(uint32_t deviceIndex);

    void OpenDevice(uint32_t deviceIndex);

    void CreateExportDevice(uint32_t * deviceIndex, Gna2DeviceVersion targetDeviceVersion);

    void CloseDevice(uint32_t deviceIndex);

    Device& GetDeviceForModel(uint32_t modelId);
    Device* TryGetDeviceForModel(uint32_t modelId);

    void AllocateMemory(uint32_t deviceIndex, uint32_t requestedSize, uint32_t *sizeGranted, void **memoryAddress);

    template<typename ... T>
    Memory * CreateInternalMemory(T ... params)
    {
        auto memoryObject = std::make_unique<Memory>(std::forward<T>(params)...);
        if (!memoryObject)
        {
            throw GnaException{ Gna2StatusResourceAllocationError };
        }
        auto const ptr = memoryObject.get();
        memoryObjects.emplace_back(std::move(memoryObject));
        return ptr;
    }

    std::pair<bool, std::vector<std::unique_ptr<Memory>>::iterator> FindMemory(void * buffer);
    void FreeMemory(void * buffer);

    void MapMemoryToAll(Memory& memoryObject);
    void UnmapMemoryFromAllDevices(Memory& memoryObject);

    Device& GetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device * TryGetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device& GetDeviceForRequestId(uint32_t requestId);

    const std::vector<std::unique_ptr<Memory>>& GetAllAllocated() const;
    void TagMemory(void* memory, uint32_t tag);

    void AssignProfilerConfigToRequestConfig(uint32_t instrumentationConfigId, uint32_t requestConfigId);

    static constexpr uint32_t DefaultThreadCount = 1;

    ProfilerConfigurationManager ProfilerConfigManager;

protected:
    void UnMapAllMemoryObjectsFromDevice(Device& device);
    void MapAllToDevice(Device& device);

    static constexpr uint32_t MaximumReferenceCount = 1024;

    static constexpr uint32_t DeviceCreateExportMaxInstances = std::numeric_limits<uint32_t>::max();

    struct DeviceContext : std::unique_ptr<Device>
    {
        DeviceContext() = default;
        DeviceContext(std::unique_ptr<Device> handle, uint32_t referenceCount);

        uint32_t operator++();
        uint32_t operator--();

    private:
        uint32_t ReferenceCount = 0;
    };

    DeviceManager();
    bool IsOpened(uint32_t deviceIndex);
    inline DeviceContext& GetDeviceContext(uint32_t deviceIndex);

    std::map<uint32_t, DeviceContext> devices;

    std::map<uint32_t, DeviceVersion> capabilities;

    std::vector<std::unique_ptr<Memory>> memoryObjects;

    uint32_t exportDevicesCount = 0;
};

}
