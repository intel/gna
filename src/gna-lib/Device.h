/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "AccelerationDetector.h"
#include "CompiledModel.h"
#include "DriverInterface.h"
#include "HardwareCapabilities.h"
#include "Memory.h"
#include "RequestBuilder.h"
#include "RequestHandler.h"

#include <cstdint>
#include <map>
#include <memory>

struct Gna2ModelSueCreekHeader;

namespace GNA
{

class Device
{
public:
    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

    virtual ~Device() = default;

    DeviceVersion GetVersion() const;

    uint32_t GetNumberOfThreads() const;

    void SetNumberOfThreads(uint32_t threadCount);

    virtual uint32_t LoadModel(const ApiModel& model) = 0;

    CompiledModel const & GetModel(uint32_t modelId);

    void ReleaseModel(uint32_t modelId);

    void AttachBuffer(uint32_t configId, uint32_t operandIndex, uint32_t layerIndex, void *address);

    void CreateConfiguration(uint32_t modelId, uint32_t *configId);

    void ReleaseConfiguration(uint32_t configId);

    bool IsVersionConsistent(DeviceVersion deviceVersion) const;

    void EnforceAcceleration(uint32_t configId, Gna2AccelerationMode accelerationMode);

    void AttachActiveList(uint32_t configId, uint32_t layerIndex, uint32_t indicesCount, const uint32_t* indices);

    void PropagateRequest(uint32_t configId, uint32_t *requestId);

    Gna2Status WaitForRequest(uint32_t requestId, uint32_t milliseconds);

    void Stop();

    void AssignProfilerConfigToRequestConfig(uint32_t requestConfigId, ProfilerConfiguration& profilerConfiguration);

    bool HasModel(uint32_t modelId) const;

    bool HasRequestConfigId(uint32_t requestConfigId) const;

    bool HasRequestId(uint32_t requestId) const;

    virtual void MapMemory(Memory& memoryObject)
    {
        UNREFERENCED_PARAMETER(memoryObject);
    }

    virtual bool UnMapMemory(Memory &memoryObject)
    {
        UNREFERENCED_PARAMETER(memoryObject);
        return false;
    }

    DriverInterface* GetDriverInterface()
    {
        return driverInterface.get();
    }

protected:
    explicit Device(std::unique_ptr<HardwareCapabilities>&& hardwareCapabilitiesIn);

    uint32_t StoreModel(std::unique_ptr<CompiledModel> && compiledModel);

    std::unique_ptr<DriverInterface> driverInterface;

    static const std::map<const Gna2DeviceGeneration, const DeviceVersion> deviceDictionary;

    static uint32_t modelIdSequence;

    std::unique_ptr<HardwareCapabilities> hardwareCapabilities;

    AccelerationDetector accelerationDetector;

    RequestBuilder requestBuilder;

    RequestHandler requestHandler;

    std::map<uint32_t, std::unique_ptr<CompiledModel>> models;
};

}
