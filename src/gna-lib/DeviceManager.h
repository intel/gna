/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Device.h"

#include "gna2-common-impl.h"

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

    DeviceManager(DeviceManager const&) = delete;
    void operator=(DeviceManager const&) = delete;

    Device& GetDevice(uint32_t deviceIndex);

    uint32_t GetDeviceCount() const;

    DeviceVersion GetDeviceVersion(uint32_t deviceIndex);

    void SetThreadCount(uint32_t deviceIndex, uint32_t threadCount);

    uint32_t GetThreadCount(uint32_t deviceIndex);

    void OpenDevice(uint32_t deviceIndex);

    void CloseDevice(uint32_t deviceIndex);

    Device& GetDeviceForModel(uint32_t modelId);
    Device* TryGetDeviceForModel(uint32_t modelId);

    void AllocateMemory(uint32_t requestedSize, uint32_t * sizeGranted, void **memoryAddress);
    std::pair<bool, std::vector<std::unique_ptr<Memory>>::const_iterator> HasMemory(void * buffer) const;
    void FreeMemory(void * memory);

    void MapMemoryToAll(Memory& memoryObject);
    void UnMapMemoryFromAll(Memory& memoryObject);

    Device& GetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device * TryGetDeviceForRequestConfigId(uint32_t requestConfigId);

    Device& GetDeviceForRequestId(uint32_t requestId);

    const std::vector<std::unique_ptr<Memory>>& GetAllAllocated() const;

    static constexpr uint32_t DefaultThreadCount = 1;

private:
    void UnMapAllFromDevice(Device& device);
    void MapAllToDevice(Device& device);

    static std::unique_ptr<Memory> createMemoryObject(const uint32_t requestedSize);

    static constexpr uint32_t MaximumReferenceCount = 1024;

    struct DeviceContext : std::unique_ptr<Device>
    {
        DeviceContext() = default;
        DeviceContext(std::unique_ptr<Device> handle, uint32_t referenceCount);

        uint32_t ReferenceCount;
    };

    DeviceManager();
    void CreateDevice(uint32_t deviceIndex);
    bool IsOpened(uint32_t deviceIndex);
    inline DeviceContext& GetDeviceContext(uint32_t deviceIndex);

    std::map<uint32_t, DeviceContext> devices;

    std::map<uint32_t, HardwareCapabilities> capabilities;

    std::vector<std::unique_ptr<Memory>> memoryObjects;
};

}
